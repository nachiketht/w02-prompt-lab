"""Deterministic triage scoring. This module must not call a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promptlab.config import HUMAN_BOUNDARY_PATTERNS, PROJECT_ROOT
from promptlab.records import ScoreRecord
from promptlab.schemas import TriageOutput

SCORER_VERSION = "day4.v1"
METRIC_QUEUE = "queue_accuracy"
METRIC_ESCALATION = "escalation_accuracy"
METRIC_MISSED = "missed_escalation"
METRIC_UNNECESSARY = "unnecessary_escalation"
METRIC_BOUNDARY = "human_boundary"


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
    lower_is_better: bool = False,
    detail: str | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        task="triage",
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        scorer_version=scorer_version,
        metric=metric,
        numerator=numerator,
        denominator=1,
        lower_is_better=lower_is_better,
        detail=detail,
    )


def score_triage(
    output: TriageOutput | None,
    gold: dict[str, Any],
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    scorer_version: str = SCORER_VERSION,
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

    return [
        _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=METRIC_QUEUE,
            numerator=int(queue_correct),
            detail=None if output is None else output.queue,
        ),
        _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=METRIC_ESCALATION,
            numerator=int(escalation_correct),
            detail=None if output is None else str(output.escalation_required),
        ),
        _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=METRIC_MISSED,
            numerator=int(missed),
            lower_is_better=True,
        ),
        _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=METRIC_UNNECESSARY,
            numerator=int(unnecessary),
            lower_is_better=True,
        ),
        _record(
            run_id=run_id,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            scorer_version=scorer_version,
            metric=METRIC_BOUNDARY,
            numerator=int(boundary_pass),
            detail=boundary_detail,
        ),
    ]
