"""Day 5 evaluation harness: three tasks, two configured Ollama models."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, ModelConfig, Settings
from promptlab.corpus import Case, GoldLabel, load_cases, validate_corpus
from promptlab.errors import StructuredOutputError
from promptlab.prompts import load, render_user
from promptlab.records import OutputRecord, ScoreRecord, TaskName, UsageRecord, append_record
from promptlab.report import TransferKey, write_reports
from promptlab.rules import select_current_version
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    TriageOutput,
    schema_description,
)
from promptlab.scoring import candidate_from_output, score_output, score_version_selection
from promptlab.structured import complete_structured
from promptlab.usage import CallRecord

MAX_OUTPUT_TOKENS = 1024
DEFAULT_SYSTEM = "Return only a JSON object. Do not wrap it in Markdown."

TASKS: tuple[TaskName, ...] = ("summarization", "extraction", "triage")


@dataclass(frozen=True)
class TaskSpec:
    task: TaskName
    prompt_id: str
    prompt_version: str
    schema: type[BaseModel]
    developed_on: str = "mistral"


TASK_SPECS: dict[TaskName, TaskSpec] = {
    "summarization": TaskSpec(
        "summarization", "summarize", "v1", SummarizationOutput
    ),
    "extraction": TaskSpec("extraction", "extract", "v2", PolicyExtraction),
    "triage": TaskSpec("triage", "triage", "v1", TriageOutput),
}


class RecordingAdapter:
    """Capture each adapter.complete call so repairs stay distinct from transport retries."""

    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner
        self.provider = inner.provider
        self.model_id = inner.model_id
        self.calls: list[CompletionResult] = []

    def reset(self) -> None:
        self.calls = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        result = self._inner.complete(request, run_id)
        self.calls.append(result)
        return result


def is_transfer(model_name: str, spec: TaskSpec) -> bool:
    return model_name != spec.developed_on


def parse_args(
    argv: Sequence[str] | None,
    model_names: Sequence[str],
    task_names: Sequence[str] = TASKS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run summarization, extraction, and triage against configured Ollama models."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Shared identifier for call and score records.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=list(task_names),
        dest="tasks",
        help="Repeat to select tasks. Default: all three.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=list(model_names),
        dest="models",
        help="Logical model name from configuration. Repeat to select models. Default: all.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _kind(call_index: int, attempt_index: int) -> Literal[
    "primary", "transport_retry", "repair", "repair_retry"
]:
    if call_index == 0 and attempt_index == 0:
        return "primary"
    if call_index == 0:
        return "transport_retry"
    if attempt_index == 0:
        return "repair"
    return "repair_retry"


def _status(record: CallRecord, group_validated: bool) -> Literal[
    "success", "schema_invalid", "transport_error"
]:
    if record.error_type is not None:
        return "transport_error"
    if group_validated:
        return "success"
    return "schema_invalid"


def usage_from_calls(
    *,
    model_name: str,
    prompt_id: str,
    calls: Sequence[CompletionResult],
    validated: Sequence[bool],
) -> list[UsageRecord]:
    """Map adapter attempts onto UsageRecord rows with primary/retry/repair kinds."""
    rows: list[UsageRecord] = []
    for call_index, result in enumerate(calls):
        group_validated = validated[call_index] if call_index < len(validated) else False
        for attempt_index, record in enumerate(result.records):
            rows.append(
                UsageRecord(
                    run_id=record.run_id,
                    task=record.task,
                    case_id=record.case_id,
                    model_name=model_name,
                    model_id=record.model_id,
                    prompt_version=record.prompt_version,
                    attempt=record.attempt,
                    kind=_kind(call_index, attempt_index),
                    status=_status(record, group_validated),
                    prompt_tokens=record.input_tokens,
                    completion_tokens=record.output_tokens,
                    latency_ms=float(record.latency_ms),
                    cost_usd=Decimal(str(record.cost_usd)),
                    error=record.error_type,
                    prompt_id=prompt_id,
                )
            )
    return rows


def _validated_flags(call_count: int, succeeded: bool) -> list[bool]:
    if call_count == 0:
        return []
    if succeeded:
        return [False] * (call_count - 1) + [True]
    return [False] * call_count


def _render_request(
    *,
    spec: TaskSpec,
    case: Case,
    temperature: float,
) -> CompletionRequest:
    template = load(spec.prompt_id, spec.prompt_version)
    variables: dict[str, str] = {"case_id": case.id}
    if "{schema_description}" in template.user_template:
        variables["schema_description"] = schema_description(spec.schema)
    return CompletionRequest(
        task=spec.task,
        case_id=case.id,
        prompt_id=spec.prompt_id,
        prompt_version=spec.prompt_version,
        system=template.system.strip() or DEFAULT_SYSTEM,
        user_content=render_user(template, variables, case.document_text),
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def evaluate_case(
    *,
    adapter: RecordingAdapter,
    spec: TaskSpec,
    case: Case,
    gold: GoldLabel,
    run_id: str,
    model_name: str,
    temperature: float,
    max_repairs: int,
) -> tuple[list[UsageRecord], OutputRecord, list[ScoreRecord], Any]:
    adapter.reset()
    request = _render_request(spec=spec, case=case, temperature=temperature)
    started = time.perf_counter()
    parsed: Any = None
    error: str | None = None
    try:
        parsed = complete_structured(
            adapter,
            request,
            spec.schema,
            run_id,
            max_repairs=max_repairs,
        )
    except StructuredOutputError as exc:
        error = str(exc)
        parsed = None
    elapsed_ms = (time.perf_counter() - started) * 1000
    call_records = [record for result in adapter.calls for record in result.records]
    model_ms = float(sum(record.latency_ms for record in call_records))
    succeeded = parsed is not None
    validated = _validated_flags(len(adapter.calls), succeeded)
    usage = usage_from_calls(
        model_name=model_name,
        prompt_id=spec.prompt_id,
        calls=adapter.calls,
        validated=validated,
    )
    output = OutputRecord(
        run_id=run_id,
        task=spec.task,
        case_id=case.id,
        model_name=model_name,
        model_id=adapter.model_id,
        prompt_version=spec.prompt_version,
        succeeded=succeeded,
        repairs=max(0, len(adapter.calls) - 1),
        output=None if parsed is None else parsed.model_dump(),
        error=error,
        elapsed_ms=elapsed_ms,
        model_ms=model_ms,
        attempts=len(call_records),
        prompt_id=spec.prompt_id,
    )
    scores = score_output(
        run_id=run_id,
        task=spec.task,
        case_id=case.id,
        model_name=model_name,
        prompt_version=spec.prompt_version,
        output=parsed,
        gold=gold,
        source=case.document_text,
        model_id=adapter.model_id,
        prompt_id=spec.prompt_id,
    )
    return usage, output, scores, parsed


def score_version_groups(
    *,
    run_id: str,
    spec: TaskSpec,
    model_name: str,
    model_id: str,
    rows: Sequence[tuple[Case, GoldLabel, Any]],
) -> list[ScoreRecord]:
    grouped: dict[str, list[tuple[Case, GoldLabel, Any]]] = defaultdict(list)
    for case, gold, parsed in rows:
        if gold.version_group:
            grouped[gold.version_group].append((case, gold, parsed))

    scores: list[ScoreRecord] = []
    for group_rows in grouped.values():
        expected = group_rows[0][1].expected_current_case_id
        as_of = group_rows[0][1].as_of
        if expected is None or as_of is None:
            continue
        candidates = []
        for case, _gold, parsed in group_rows:
            if isinstance(parsed, PolicyExtraction | SummarizationOutput):
                candidate = candidate_from_output(case.id, parsed)
                if candidate is not None:
                    candidates.append(candidate)
        selected = select_current_version(candidates, as_of)
        scores.append(
            score_version_selection(
                run_id=run_id,
                task=spec.task,
                case_id=expected,
                model_name=model_name,
                prompt_version=spec.prompt_version,
                selected=selected,
                expected_current_case_id=expected,
                model_id=model_id,
                prompt_id=spec.prompt_id,
            )
        )
    return scores


def _write_jsonl(
    path: Path, records: Sequence[UsageRecord | OutputRecord | ScoreRecord]
) -> None:
    if path.exists():
        path.unlink()
    for record in records:
        append_record(path, record)


def _copy_call_records(run_id: str, destination: Path) -> None:
    source = PROJECT_ROOT / "runs" / f"{run_id}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, destination)
    else:
        destination.write_text("", encoding="utf-8")


def run_evaluation(
    *,
    run_id: str,
    settings: Settings,
    tasks: Sequence[TaskName],
    models: Sequence[ModelConfig],
) -> tuple[list[UsageRecord], list[OutputRecord], list[ScoreRecord], set[TransferKey]]:
    all_usage: list[UsageRecord] = []
    all_outputs: list[OutputRecord] = []
    all_scores: list[ScoreRecord] = []
    transfer_keys: set[TransferKey] = set()

    for model in models:
        adapter = RecordingAdapter(OllamaAdapter(model_id=model.model_id))
        for task_name in tasks:
            spec = TASK_SPECS[task_name]
            if is_transfer(model.logical_name, spec):
                transfer_keys.add((spec.task, model.logical_name, spec.prompt_version))
            pairs = load_cases(spec.task)
            group_rows: list[tuple[Case, GoldLabel, Any]] = []
            print(
                f"{spec.task} {spec.prompt_id}.{spec.prompt_version} "
                f"model={model.logical_name} model_id={adapter.model_id} "
                f"transfer={is_transfer(model.logical_name, spec)}"
            )
            for case, gold in pairs:
                usage, output, scores, parsed = evaluate_case(
                    adapter=adapter,
                    spec=spec,
                    case=case,
                    gold=gold,
                    run_id=run_id,
                    model_name=model.logical_name,
                    temperature=settings.temperature,
                    max_repairs=settings.max_schema_repairs,
                )
                all_usage.extend(usage)
                all_outputs.append(output)
                all_scores.extend(scores)
                group_rows.append((case, gold, parsed))
                print(
                    f"  {case.id}: succeeded={output.succeeded} "
                    f"repairs={output.repairs} elapsed_ms={output.elapsed_ms:.0f}"
                )
            all_scores.extend(
                score_version_groups(
                    run_id=run_id,
                    spec=spec,
                    model_name=model.logical_name,
                    model_id=adapter.model_id,
                    rows=group_rows,
                )
            )

    return all_usage, all_outputs, all_scores, transfer_keys


def main(argv: Sequence[str] | None = None) -> None:
    os.chdir(PROJECT_ROOT)
    settings = Settings.from_env()
    counts = validate_corpus()
    args = parse_args(argv, tuple(settings.models))
    selected_tasks = cast(list[TaskName], list(dict.fromkeys(args.tasks or list(TASKS))))
    selected_names = list(dict.fromkeys(args.models or list(settings.models)))
    selected_models = [settings.models[name] for name in selected_names]
    run_id = args.run_id

    print(
        f"run_id={run_id} tasks={','.join(selected_tasks)} "
        f"models={','.join(name for name in selected_names)} "
        f"temperature={settings.temperature} corpus={counts}"
    )

    usage, outputs, scores, transfer_keys = run_evaluation(
        run_id=run_id,
        settings=settings,
        tasks=selected_tasks,
        models=selected_models,
    )

    score_path = PROJECT_ROOT / "docs" / "day5-scores.jsonl"
    _write_jsonl(score_path, scores)
    _write_jsonl(PROJECT_ROOT / "runs" / f"{run_id}-usage.jsonl", usage)
    _write_jsonl(PROJECT_ROOT / "runs" / f"{run_id}-outputs.jsonl", outputs)
    _copy_call_records(run_id, PROJECT_ROOT / "docs" / "day5-run.jsonl")

    write_reports(
        run_id=run_id,
        models=selected_names,
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=PROJECT_ROOT / "reports" / "comparison.md",
        decision_path=PROJECT_ROOT / "docs" / "model-decision.md",
        transfer_keys=transfer_keys,
    )
    print(f"wrote scores to {score_path}")
    print("wrote records to docs/day5-run.jsonl")
    print("wrote reports/comparison.md and docs/model-decision.md")
    print("provider/API cost: $0.00")


if __name__ == "__main__":
    main()
