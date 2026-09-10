"""Day 1: three instrumented Mistral extraction calls."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, append_record, compute_cost

CASE_IDS = ("E12", "E07", "E11")
NORMAL_NUM_PREDICT = 256
TRUNCATION_NUM_PREDICT = 8


def load_sources() -> dict[str, str]:
    path = PROJECT_ROOT / "cases" / "extraction.jsonl"
    sources: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed: object = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise TypeError("each extraction line must be a JSON object")
        case_id = parsed["id"]
        source = parsed["source"]
        if not isinstance(case_id, str) or not isinstance(source, str):
            raise TypeError("id and source must be strings")
        sources[case_id] = source
    return sources


def load_prompt_template() -> str:
    return (PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md").read_text(encoding="utf-8")


def generate(
    settings: Settings,
    model_id: str,
    prompt: str,
    num_predict: int,
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": num_predict,
            },
        },
        timeout=180.0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    payload: object = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Ollama generate must return a JSON object")
    return payload, latency_ms


def as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def build_record(
    *,
    run_id: str,
    model_id: str,
    case_id: str,
    payload: dict[str, Any],
    latency_ms: int,
    max_output_tokens: int,
    error_type: str | None,
) -> CallRecord:
    input_tokens = int(payload["prompt_eval_count"])
    output_tokens = int(payload["eval_count"])
    return CallRecord(
        record_id=str(uuid.uuid4()),
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="extraction",
        case_id=case_id,
        prompt_id="baseline",
        prompt_version="v0",
        attempt=1,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        latency_ms=latency_ms,
        cost_usd=compute_cost(model_id, input_tokens, output_tokens),
        stop_reason=as_optional_str(payload.get("done_reason")),
        error_type=error_type,
        response_text=as_optional_str(payload.get("response")),
    )


def write_evidence(records: list[CallRecord]) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = docs_dir / "day1-run.jsonl"
    evidence_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    settings = Settings.from_env()
    model = settings.models["mistral"]
    sources = load_sources()
    template = load_prompt_template()
    run_id = str(uuid.uuid4())
    num_predict = NORMAL_NUM_PREDICT

    print(f"run_id={run_id} model_id={model.model_id}")

    evidence: list[CallRecord] = []
    for case_id in CASE_IDS:
        prompt = template.replace("{document_text}", sources[case_id])
        payload, latency_ms = generate(settings, model.model_id, prompt, num_predict)
        record = build_record(
            run_id=run_id,
            model_id=model.model_id,
            case_id=case_id,
            payload=payload,
            latency_ms=latency_ms,
            max_output_tokens=num_predict,
            error_type=None,
        )
        append_record(record, run_id)
        evidence.append(record)
        print(
            f"{case_id}: input_tokens={record.input_tokens} "
            f"output_tokens={record.output_tokens} latency_ms={record.latency_ms} "
            f"stop_reason={record.stop_reason}"
        )

    truncation_prompt = template.replace("{document_text}", sources["E11"])
    truncation_payload, truncation_latency_ms = generate(
        settings, model.model_id, truncation_prompt, TRUNCATION_NUM_PREDICT
    )
    truncation_stop = as_optional_str(truncation_payload.get("done_reason"))
    truncation_error = "TruncatedResponseError" if truncation_stop == "length" else None
    truncation_record = build_record(
        run_id=run_id,
        model_id=model.model_id,
        case_id="E11",
        payload=truncation_payload,
        latency_ms=truncation_latency_ms,
        max_output_tokens=TRUNCATION_NUM_PREDICT,
        error_type=truncation_error,
    )
    append_record(truncation_record, run_id)
    num_predict = NORMAL_NUM_PREDICT
    print(
        f"E11 truncation demo: num_predict={TRUNCATION_NUM_PREDICT} "
        f"stop_reason={truncation_record.stop_reason} "
        f"error_type={truncation_record.error_type} "
        f"(restored num_predict={num_predict}; not copied to docs)"
    )

    write_evidence(evidence)
    print(f"wrote {len(evidence)} successful records to docs/day1-run.jsonl")


if __name__ == "__main__":
    main()
