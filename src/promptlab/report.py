"""Reporting for the Week 2 model-comparison lab.

The reporting layer consumes the existing UsageRecord, OutputRecord, and
ScoreRecord objects.  It does not rescore model output and it does not call an
LLM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord

_ConfigKey = tuple[str, str, str]  # task, model_name, prompt_version
TransferKey = tuple[str, str, str]


def _key(record: Any) -> _ConfigKey:
    return (
        str(record.task),
        str(record.model_name),
        str(record.prompt_version),
    )


def _for_run(records: Sequence[Any], run_id: str) -> list[Any]:
    return [record for record in records if str(record.run_id) == run_id]


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _latency_text(values: Sequence[float]) -> tuple[str, str, str]:
    if not values:
        return "—", "—", "0"
    return (
        f"{_fmt_number(float(median(values)))} ms",
        f"{_fmt_number(float(max(values)))} ms",
        str(len(values)),
    )


def _aggregate_scores(
    records: Sequence[ScoreRecord],
) -> dict[str, tuple[int, int, bool | None]]:
    """Aggregate compatible score counts without averaging percentages."""

    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.metric)].append(record)

    result: dict[str, tuple[int, int, bool | None]] = {}

    for metric, rows in sorted(grouped.items()):
        numerator = sum(int(row.numerator) for row in rows)
        denominator = sum(int(row.denominator) for row in rows)
        directions = {
            bool(value)
            for value in (getattr(row, "lower_is_better", None) for row in rows)
            if value is not None
        }
        lower_is_better = next(iter(directions)) if len(directions) == 1 else None
        result[metric] = (numerator, denominator, lower_is_better)

    return result


def _metric_text(records: Sequence[ScoreRecord]) -> str:
    metrics = _aggregate_scores(records)
    if not metrics:
        return "—"

    rendered: list[str] = []
    for metric, (numerator, denominator, lower_is_better) in metrics.items():
        suffix = " ↓" if lower_is_better else ""
        rendered.append(f"{metric}: {numerator}/{denominator}{suffix}")

    return "<br>".join(rendered)


def _prompt_label(
    key: _ConfigKey,
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    usage: Sequence[UsageRecord],
    transfer_keys: Collection[TransferKey],
) -> str:
    prompt_id = next((row.prompt_id for row in outputs if row.prompt_id), None)
    if prompt_id is None:
        prompt_id = next((row.prompt_id for row in scores if row.prompt_id), None)
    if prompt_id is None:
        prompt_id = next((row.prompt_id for row in usage if row.prompt_id), None)
    prompt_version = key[2]
    label = f"{prompt_id}.{prompt_version}" if prompt_id else prompt_version
    if key in transfer_keys:
        label += " transfer"
    return label


def _token_per_case(
    usage: Sequence[UsageRecord], outputs: Sequence[OutputRecord]
) -> tuple[str, str]:
    if not usage:
        return "—", "—"
    case_count = len(outputs) or len({row.case_id for row in usage}) or 1
    prompt_tokens = sum(int(row.prompt_tokens or 0) for row in usage)
    completion_tokens = sum(int(row.completion_tokens or 0) for row in usage)
    return (
        _fmt_number(prompt_tokens / case_count),
        _fmt_number(completion_tokens / case_count),
    )


def _usage_attempt_summary(
    records: Sequence[UsageRecord],
) -> tuple[str, str, str, str]:
    if not records:
        return "—", "—", "0", "0"
    latencies = [float(row.latency_ms) for row in records]
    median_latency, max_latency, n = _latency_text(latencies)
    retries = sum(
        1
        for row in records
        if int(row.attempt or 1) > 1 and str(row.kind).lower() != "repair"
    )
    return median_latency, max_latency, n, str(retries)


def _case_latency_summary(records: Sequence[OutputRecord]) -> tuple[str, str, str]:
    latencies = [
        float(row.elapsed_ms) for row in records if row.elapsed_ms is not None
    ]
    return _latency_text(latencies)


def _output_summary(
    records: Sequence[OutputRecord],
) -> tuple[str, str, str]:
    if not records:
        return "0/0", "0/0", "0"

    total = len(records)
    succeeded = sum(1 for row in records if bool(row.succeeded))
    repairs_needed = sum(1 for row in records if int(row.repairs or 0) > 0)
    failures = total - succeeded
    return f"{succeeded}/{total}", f"{repairs_needed}/{total}", str(failures)


def _retry_stratum(records: Sequence[OutputRecord]) -> str | None:
    if not records:
        return None
    retry_cases = [
        row
        for row in records
        if int(row.repairs or 0) > 0 or int(row.attempts or 0) > 1
    ]
    if not retry_cases:
        return None
    median_text, max_text, n = _case_latency_summary(retry_cases)
    return (
        f"Retry/repair cases: {len(retry_cases)}/{len(records)} "
        f"(median case latency {median_text}, max {max_text}, n={n}). "
        "These cases are not re-weighted into the headline median."
    )


def _all_config_keys(
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
) -> list[_ConfigKey]:
    keys = {_key(row) for row in usage}
    keys.update(_key(row) for row in outputs)
    keys.update(_key(row) for row in scores)
    return sorted(keys)


def _write_report(
    *,
    run_id: str,
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    transfer_keys: Collection[TransferKey],
) -> None:
    lines: list[str] = [
        "# Model Comparison",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Counts are reported with their denominators. "
        "Headline latency is median and maximum **case** end-to-end time "
        "(`elapsed_ms`), with one observation per case. Attempt latency is "
        "HTTP-call time and uses a separate `n`. Mean latency is not used.",
        "",
        "Provider/API charge is `$0.00`.",
        "",
    ]

    keys = _all_config_keys(usage, outputs, scores)
    tasks = sorted({task for task, _model, _prompt in keys})

    if not tasks:
        lines.extend(["No records were supplied for this run.", ""])

    for task in tasks:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Model | Prompt | Valid outputs | Metrics | "
                "Input tokens/case | Output tokens/case | "
                "Median case latency | Max case latency | Case n | "
                "Median attempt latency | Max attempt latency | Attempt n | "
                "Repairs | Retries | Final failures |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        task_keys = [key for key in keys if key[0] == task]
        strata: list[str] = []

        for key in task_keys:
            u = [row for row in usage if _key(row) == key]
            o = [row for row in outputs if _key(row) == key]
            s = [row for row in scores if _key(row) == key]
            input_tokens, output_tokens = _token_per_case(u, o)
            case_median, case_max, case_n = _case_latency_summary(o)
            attempt_median, attempt_max, attempt_n, retries = _usage_attempt_summary(u)
            valid_outputs, repairs, failures = _output_summary(o)
            prompt = _prompt_label(key, o, s, u, transfer_keys)
            lines.append(
                "| "
                f"{key[1]} | {prompt} | {valid_outputs} | {_metric_text(s)} | "
                f"{input_tokens} | {output_tokens} | "
                f"{case_median} | {case_max} | {case_n} | "
                f"{attempt_median} | {attempt_max} | {attempt_n} | "
                f"{repairs} | {retries} | {failures} |"
            )
            note = _retry_stratum(o)
            if note is not None:
                strata.append(f"- {key[1]} / {prompt}: {note}")

        lines.append("")
        if strata:
            lines.extend(["Retry stratum (not mixed into the headline median):", "", *strata, ""])

    lines.extend(
        [
            "## Limits",
            "",
            "- Each task uses 12 cases. Results are directional, not production-scale estimates.",
            "- A row measures the model together with the prompt version shown in that row.",
            "- Prompt-transfer rows are labeled `transfer`. They are evidence about that "
            "transferred configuration, not proof of the model's best "
            "performance after adaptation.",
            "- Untested combinations in this harness include `triage.v2` × Qwen and any "
            "prompt version that does not appear in a table row.",
            "- Case latency `n` is 12 (one observation per case). Attempt latency `n` is "
            "HTTP calls. Do not treat those as the same observation count.",
            "- Retry/repair cases already have a larger case `elapsed_ms`. They are not "
            "weighted again into the median or max.",
            "- Local Ollama latency depends on lab hardware. No production-volume "
            "reliability claim is being made.",
            "- 11/12 versus 10/12 is not a universal model ranking.",
            "- Local provider/API charge is `$0.00`; token usage and latency still "
            "represent real operational work.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision_scaffold(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    decision_path: Path,
    transfer_keys: Collection[TransferKey],
) -> None:
    """Write an evidence scaffold, not an invented model recommendation."""

    keys = _all_config_keys(usage, outputs, scores)

    lines: list[str] = [
        "# Model Decision Record",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Use this file to record the task-level decision after reviewing the measured "
        "comparison. Do not select one universal model solely because it leads on a "
        "different task.",
        "",
        "## Evaluated models",
        "",
    ]

    evaluated_models = sorted(
        {model for _task, model, _prompt in keys} | {str(model) for model in models}
    )
    if evaluated_models:
        for model in evaluated_models:
            lines.append(f"- {model}")
    else:
        lines.append("- None")

    lines.extend(["", "## Evaluated configurations", ""])

    if keys:
        for key in keys:
            task, model, prompt_version = key
            prompt = _prompt_label(
                key,
                [row for row in outputs if _key(row) == key],
                [row for row in scores if _key(row) == key],
                [row for row in usage if _key(row) == key],
                transfer_keys,
            )
            lines.append(f"- `{task}` — {model} — `{prompt}`")
    else:
        lines.append("- No configurations supplied.")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "Every evidence row must name the prompt version. Fill this section from "
            "`reports/comparison.md` after the measured run. Do not rewrite earlier "
            "decision constraints after seeing the results.",
            "",
            "## Task decisions",
            "",
            "For each task, complete:",
            "",
            "- selected model",
            "- prompt version",
            "- measured reason",
            "- rejected alternative(s)",
            "- condition that would reopen the decision",
            "",
        ]
    )

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    decision_path: Path,
    transfer_keys: Collection[TransferKey] = (),
) -> None:
    """Generate the comparison report and decision scaffold for one run.

    Only records whose ``run_id`` matches the requested run are included.
    """

    run_usage = _for_run(usage, run_id)
    run_outputs = _for_run(outputs, run_id)
    run_scores = _for_run(scores, run_id)

    _write_report(
        run_id=run_id,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        report_path=Path(report_path),
        transfer_keys=transfer_keys,
    )

    _write_decision_scaffold(
        run_id=run_id,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        decision_path=Path(decision_path),
        transfer_keys=transfer_keys,
    )
