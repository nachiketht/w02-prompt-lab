"""Day 4: compare triage.v1 and triage.v2 through complete_structured."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import uuid
from pathlib import Path

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.errors import StructuredOutputError
from promptlab.prompts import load, render_user
from promptlab.records import ScoreRecord, append_record
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
from promptlab.scoring import (
    METRIC_BOUNDARY,
    METRIC_ESCALATION,
    METRIC_MISSED,
    METRIC_QUEUE,
    METRIC_UNNECESSARY,
    load_triage_gold,
    score_triage,
)
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

MAX_OUTPUT_TOKENS = 1024
PROMPT_ID = "triage"
LOGICAL_MODEL = "mistral"
SCHEMA_BY_VERSION: dict[str, type[TriageOutput]] = {
    "v1": TriageOutput,
    "v2": TriageOutputWithAnalysis,
}


def load_cases() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "cases" / "triage.jsonl"
    cases: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed: object = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise TypeError("each triage line must be a JSON object")
        case_id = parsed["id"]
        source = parsed["source"]
        if not isinstance(case_id, str) or not isinstance(source, str):
            raise TypeError("id and source must be strings")
        cases.append((case_id, source))
    return cases


def load_call_records(run_id: str) -> list[CallRecord]:
    path = PROJECT_ROOT / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    records: list[CallRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            records.append(CallRecord.model_validate_json(raw_line))
    return records


def _metric_total(scores: list[ScoreRecord], metric: str) -> tuple[int, int]:
    matched = [row for row in scores if row.metric == metric]
    return sum(row.numerator for row in matched), sum(row.denominator for row in matched)


def _for_version(records: list[CallRecord], version: str) -> list[CallRecord]:
    return [record for record in records if record.prompt_version == version]


def _output_tokens_by_case(records: list[CallRecord]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        totals[record.case_id] = totals.get(record.case_id, 0) + record.output_tokens
    return totals


def write_run_copy(run_id: str) -> Path:
    source = PROJECT_ROOT / "runs" / f"{run_id}.jsonl"
    destination = PROJECT_ROOT / "docs" / "day4-run.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def write_scores(path: Path, scores: list[ScoreRecord]) -> None:
    if path.exists():
        path.unlink()
    for record in scores:
        append_record(path, record)


def print_version_summary(
    version: str,
    scores: list[ScoreRecord],
    records: list[CallRecord],
    case_count: int,
) -> None:
    queue_n, queue_d = _metric_total(scores, METRIC_QUEUE)
    esc_n, esc_d = _metric_total(scores, METRIC_ESCALATION)
    missed, _ = _metric_total(scores, METRIC_MISSED)
    unnecessary, _ = _metric_total(scores, METRIC_UNNECESSARY)
    bound_n, bound_d = _metric_total(scores, METRIC_BOUNDARY)
    latencies = [record.latency_ms for record in records]
    tokens = _output_tokens_by_case(records)
    token_total = sum(tokens.values())
    median_latency = statistics.median(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    print(f"triage.{version}")
    print(f"queue correct: {queue_n}/{queue_d or case_count}")
    print(f"escalation correct: {esc_n}/{esc_d or case_count}")
    print(f"missed escalations: {missed}")
    print(f"unnecessary escalations: {unnecessary}")
    print(f"human-boundary passes: {bound_n}/{bound_d or case_count}")
    print(f"output tokens total: {token_total}")
    for case_id in sorted(tokens):
        print(f"output tokens {case_id}: {tokens[case_id]}")
    print(f"median latency: {median_latency}")
    print(f"maximum latency: {max_latency}")
    print(f"observation count: {len(records)}")
    print("provider/API cost: $0.00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 4 triage prompt comparison.")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=tuple(SCHEMA_BY_VERSION),
        default=["v1", "v2"],
        help="Prompt versions to run. Official comparison uses v1 and v2.",
    )
    return parser.parse_args()


def run_version(
    *,
    adapter: OllamaAdapter,
    run_id: str,
    version: str,
    schema: type[TriageOutput],
    cases: list[tuple[str, str]],
    gold: dict[str, dict[str, object]],
    temperature: float,
    max_repairs: int,
    model_name: str,
) -> tuple[dict[str, TriageOutput | None], list[ScoreRecord]]:
    template = load(PROMPT_ID, version)
    outputs: dict[str, TriageOutput | None] = {}
    scores: list[ScoreRecord] = []
    for case_id, source in cases:
        request = CompletionRequest(
            task="triage",
            case_id=case_id,
            prompt_id=PROMPT_ID,
            prompt_version=version,
            system=template.system,
            user_content=render_user(template, {"case_id": case_id}, source),
            temperature=temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        parsed: TriageOutput | None
        try:
            parsed = complete_structured(
                adapter,
                request,
                schema,
                run_id,
                max_repairs=max_repairs,
            )
            print(f"{version} {case_id}: succeeded=True queue={parsed.queue}")
        except StructuredOutputError as exc:
            parsed = None
            print(f"{version} {case_id}: succeeded=False error={exc}")
        outputs[case_id] = parsed
        scores.extend(
            score_triage(
                parsed,
                gold[case_id],
                run_id=run_id,
                case_id=case_id,
                model_name=model_name,
                prompt_version=version,
            )
        )
    return outputs, scores


def main() -> None:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    versions = list(dict.fromkeys(args.versions))
    settings = Settings.from_env()
    model = settings.models[LOGICAL_MODEL]
    run_id = str(uuid.uuid4())
    adapter = OllamaAdapter(model_id=model.model_id)
    cases = load_cases()
    gold = load_triage_gold()
    all_scores: list[ScoreRecord] = []
    outputs_by_version: dict[str, dict[str, TriageOutput | None]] = {}

    print(
        f"run_id={run_id} model_id={adapter.model_id} "
        f"temperature={settings.temperature} versions={','.join(versions)}"
    )

    for version in versions:
        outputs, scores = run_version(
            adapter=adapter,
            run_id=run_id,
            version=version,
            schema=SCHEMA_BY_VERSION[version],
            cases=cases,
            gold=gold,
            temperature=settings.temperature,
            max_repairs=settings.max_schema_repairs,
            model_name=model.logical_name,
        )
        outputs_by_version[version] = outputs
        all_scores.extend(scores)

    call_records = load_call_records(run_id)
    print()
    for version in versions:
        print_version_summary(
            version,
            [row for row in all_scores if row.prompt_version == version],
            _for_version(call_records, version),
            case_count=len(cases),
        )
        print()

    if "v1" in outputs_by_version and "v2" in outputs_by_version:
        changed = 0
        for case_id, _source in cases:
            left = outputs_by_version["v1"].get(case_id)
            right = outputs_by_version["v2"].get(case_id)
            if left is not None and right is not None and left.queue != right.queue:
                changed += 1
        v1_tokens = sum(row.output_tokens for row in _for_version(call_records, "v1"))
        v2_tokens = sum(row.output_tokens for row in _for_version(call_records, "v2"))
        print(f"changed-queue count: {changed}")
        print(f"output-token difference (v2-v1): {v2_tokens - v1_tokens}")
        score_path = PROJECT_ROOT / "docs" / "day4-scores.jsonl"
        write_scores(score_path, all_scores)
        evidence_path = write_run_copy(run_id)
        print(f"wrote scores to {score_path}")
        print(f"wrote records to {evidence_path}")
    else:
        print(f"skipping docs/day4 artifacts; ran versions {','.join(versions)}")
        print(f"wrote records to runs/{run_id}.jsonl")

    print("provider/API cost: $0.00")


if __name__ == "__main__":
    main()
