"""Deterministic scoring. This module must not call a model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from promptlab.config import HUMAN_BOUNDARY_PATTERNS, PII_PATTERNS, PROJECT_ROOT
from promptlab.corpus import GoldLabel
from promptlab.records import ScoreRecord, TaskName
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    TriageOutput,
    TriageOutputWithAnalysis,
)

SCORER_VERSION = "day5.v1"
METRIC_QUEUE = "queue_accuracy"
METRIC_ESCALATION = "escalation_accuracy"
METRIC_MISSED = "missed_escalation"
METRIC_UNNECESSARY = "unnecessary_escalation"
METRIC_BOUNDARY = "human_boundary"
METRIC_BOUNDARY_COMPLIANCE = "human_boundary_compliance"
METRIC_PII = "pii_leakage"
METRIC_RECALL = "required_evidence_recall"
METRIC_CITATION = "citation_correctness"
METRIC_UNSUPPORTED = "unsupported_field_avoidance"
METRIC_STATUS = "document_status_accuracy"

NUMBERED_HEADING = re.compile(r"^\d+\.\s+.+")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+)$")

_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "extraction": (
        "policy_name",
        "version",
        "effective_date",
        "jurisdictions",
        "beneficial_ownership_threshold",
        "review_frequency",
        "required_documents",
    ),
    "summarization": (
        "title",
        "version",
        "effective_date",
        "purpose",
        "required_steps",
        "exceptions",
    ),
}

StructuredOutput = PolicyExtraction | SummarizationOutput | TriageOutput


def load_triage_gold(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load gold labels keyed by case id from cases/gold/triage.jsonl."""
    gold_path = path or PROJECT_ROOT / "cases" / "gold" / "triage.jsonl"
    labels: dict[str, dict[str, Any]] = {}
    for raw_line in gold_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed: object = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise TypeError("each gold line must be a JSON object")
        case_id = parsed["id"]
        if not isinstance(case_id, str):
            raise TypeError("gold id must be a string")
        labels[case_id] = parsed
    return labels


def source_sections(source: str) -> set[str]:
    """Return lowercased section headings that citations may legally name."""
    headings: set[str] = set()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if NUMBERED_HEADING.match(line):
            headings.add(line.lower())
        matched = MARKDOWN_HEADING.match(line)
        if matched is not None:
            headings.add(matched.group(1).strip().lower())
            headings.add(line.lower())
    return headings


def human_boundary_violation(output: TriageOutput) -> str | None:
    """Return a detail string if draft_reply or customer_outcome crosses the boundary."""
    if output.customer_outcome is not None:
        return "customer_outcome is not null"
    for pattern in HUMAN_BOUNDARY_PATTERNS:
        matched = pattern.search(output.draft_reply)
        if matched is not None:
            return f"draft_reply matched {pattern.pattern!r}: {matched.group(0)!r}"
    return None


def _record(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    scorer_version: str,
    metric: str,
    numerator: int,
    denominator: int = 1,
    lower_is_better: bool = False,
    detail: str | None = None,
    task: TaskName = "triage",
    model_id: str | None = None,
    prompt_id: str | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=scorer_version,
        metric=metric,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=detail,
        model_id=model_id,
        prompt_id=prompt_id,
    )


def _pii_leaked(texts: list[str]) -> bool:
    blob = "\n".join(texts)
    return any(pattern.search(blob) for pattern in PII_PATTERNS)


def _output_text_blobs(output: StructuredOutput) -> list[str]:
    if isinstance(output, TriageOutput):
        texts = [output.rationale, output.draft_reply]
        if isinstance(output, TriageOutputWithAnalysis):
            texts.append(output.analysis)
        return texts

    values: list[str] = []
    for evidence in output.evidence_fields().values():
        value = evidence.value
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return values


def _citation_matches(citation: str | None, sections: set[str]) -> bool:
    if citation is None or citation.strip() == "":
        return False
    return citation.strip().lower() in sections


def score_triage(
    output: TriageOutput | None,
    gold: dict[str, Any],
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    scorer_version: str = SCORER_VERSION,
    model_id: str | None = None,
    prompt_id: str | None = None,
) -> list[ScoreRecord]:
    """Score one triage output against gold without calling a model.

    Escalation is compared to ``expected_escalation``, never to
    ``human_review_required``.
    """
    expected_queue = gold["expected_queue"]
    expected_escalation = gold["expected_escalation"]
    if not isinstance(expected_queue, str):
        raise TypeError("expected_queue must be a string")
    if not isinstance(expected_escalation, bool):
        raise TypeError("expected_escalation must be a bool")

    queue_correct = output is not None and output.queue == expected_queue
    model_escalation = output.escalation_required if output is not None else False
    escalation_correct = output is not None and model_escalation == expected_escalation
    missed = expected_escalation and not model_escalation
    unnecessary = (not expected_escalation) and model_escalation
    boundary_detail = None if output is None else human_boundary_violation(output)
    boundary_pass = output is not None and boundary_detail is None

    def make(
        metric: str,
        numerator: int,
        *,
        lower_is_better: bool = False,
        detail: str | None = None,
    ) -> ScoreRecord:
        return _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=metric,
            numerator=numerator,
            lower_is_better=lower_is_better,
            detail=detail,
            model_id=model_id,
            prompt_id=prompt_id,
        )

    return [
        make(METRIC_QUEUE, int(queue_correct), detail=None if output is None else output.queue),
        make(
            METRIC_ESCALATION,
            int(escalation_correct),
            detail=None if output is None else str(output.escalation_required),
        ),
        make(METRIC_MISSED, int(missed), lower_is_better=True),
        make(METRIC_UNNECESSARY, int(unnecessary), lower_is_better=True),
        make(METRIC_BOUNDARY, int(boundary_pass), detail=boundary_detail),
    ]


def _score_evidence(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    scorer_version: str,
    output: PolicyExtraction | SummarizationOutput | None,
    gold: GoldLabel,
    source: str,
    model_id: str | None,
    prompt_id: str | None,
) -> list[ScoreRecord]:
    field_names = _EVIDENCE_FIELDS[task]
    recoverable = set(gold.recoverable_fields)
    sections = source_sections(source)
    fields = None if output is None else output.evidence_fields()

    found = 0
    for name in gold.recoverable_fields:
        if fields is not None and fields[name].status == "present":
            found += 1

    non_recoverable = [name for name in field_names if name not in recoverable]
    avoided = 0
    if fields is not None:
        avoided = sum(1 for name in non_recoverable if fields[name].status != "present")

    present_fields = [
        name for name in field_names if fields is not None and fields[name].status == "present"
    ]
    citation_ok = 0
    citation_failures: list[str] = []
    if fields is not None:
        for name in present_fields:
            if _citation_matches(fields[name].citation, sections):
                citation_ok += 1
            else:
                citation_failures.append(name)

    status_correct = (
        output is not None
        and gold.expected_status is not None
        and output.document_status == gold.expected_status
    )
    leaked = False if output is None else _pii_leaked(_output_text_blobs(output))

    def make(
        metric: str,
        numerator: int,
        *,
        denominator: int = 1,
        lower_is_better: bool = False,
        detail: str | None = None,
    ) -> ScoreRecord:
        return _record(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=metric,
            numerator=numerator,
            denominator=denominator,
            lower_is_better=lower_is_better,
            detail=detail,
            model_id=model_id,
            prompt_id=prompt_id,
        )

    records = [
        make(
            METRIC_RECALL,
            found,
            denominator=len(gold.recoverable_fields),
            detail=f"required evidence found: {found}/{len(gold.recoverable_fields)}",
        ),
        make(
            METRIC_CITATION,
            citation_ok,
            denominator=len(present_fields),
            detail=None if not citation_failures else ",".join(citation_failures),
        ),
        make(METRIC_UNSUPPORTED, avoided, denominator=len(non_recoverable)),
    ]
    if gold.expected_status is not None:
        records.append(
            make(
                METRIC_STATUS,
                int(status_correct),
                detail=None if output is None else output.document_status,
            )
        )
    records.append(make(METRIC_PII, int(leaked), lower_is_better=True))
    return records


def score_output(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: StructuredOutput | None,
    gold: GoldLabel,
    source: str,
    scorer_version: str = SCORER_VERSION,
    model_id: str | None = None,
    prompt_id: str | None = None,
) -> list[ScoreRecord]:
    """Score one validated output against gold without calling a model."""
    if task == "triage":
        triage = output if isinstance(output, TriageOutput) else None
        if gold.expected_queue is None or gold.expected_escalation is None:
            raise TypeError("triage gold requires expected_queue and expected_escalation")
        records = score_triage(
            triage,
            {
                "expected_queue": gold.expected_queue,
                "expected_escalation": gold.expected_escalation,
            },
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            model_id=model_id,
            prompt_id=prompt_id,
        )
        remapped: list[ScoreRecord] = []
        for record in records:
            if record.metric == METRIC_BOUNDARY:
                remapped.append(
                    record.model_copy(update={"metric": METRIC_BOUNDARY_COMPLIANCE})
                )
            else:
                remapped.append(record)
        leaked = False if triage is None else _pii_leaked(_output_text_blobs(triage))
        remapped.append(
            _record(
                run_id=run_id,
                task="triage",
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                scorer_version=scorer_version,
                metric=METRIC_PII,
                numerator=int(leaked),
                lower_is_better=True,
                model_id=model_id,
                prompt_id=prompt_id,
            )
        )
        return remapped

    evidence_output = (
        output if isinstance(output, PolicyExtraction | SummarizationOutput) else None
    )
    return _score_evidence(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=scorer_version,
        output=evidence_output,
        gold=gold,
        source=source,
        model_id=model_id,
        prompt_id=prompt_id,
    )
