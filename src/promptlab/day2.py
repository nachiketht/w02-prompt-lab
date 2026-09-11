"""Day 2: run the baseline prompt against Mistral and Qwen via OllamaAdapter."""

from __future__ import annotations

import json
import uuid

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings

LOGICAL_MODELS = ("mistral", "qwen")
MAX_OUTPUT_TOKENS = 1024
PROMPT_ID = "baseline"
PROMPT_VERSION = "v0"


def load_cases() -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "cases" / "summarization.jsonl"
    cases: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed: object = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise TypeError("each summarization line must be a JSON object")
        case_id = parsed["id"]
        source = parsed["source"]
        if not isinstance(case_id, str) or not isinstance(source, str):
            raise TypeError("id and source must be strings")
        cases.append((case_id, source))
    return cases


def load_system_prompt() -> str:
    template = (PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md").read_text(encoding="utf-8")
    system, separator, _rest = template.partition("<document>")
    if separator == "":
        raise ValueError("baseline prompt is missing a <document> section")
    return system.strip()


def user_content_for(source: str) -> str:
    return f"<document>\n{source}\n</document>"


def main() -> None:
    settings = Settings.from_env()
    run_id = str(uuid.uuid4())
    cases = load_cases()
    system = load_system_prompt()

    print(f"run_id={run_id}")

    for logical_name in LOGICAL_MODELS:
        model_id = settings.models[logical_name].model_id
        adapter = OllamaAdapter(model_id=model_id)
        print(f"model={logical_name} model_id={adapter.model_id}")

        for case_id, source in cases:
            request = CompletionRequest(
                task="summarization",
                case_id=case_id,
                prompt_id=PROMPT_ID,
                prompt_version=PROMPT_VERSION,
                system=system,
                user_content=user_content_for(source),
                temperature=settings.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            result = adapter.complete(request, run_id)
            last = result.records[-1]
            print(
                f"{case_id} {adapter.model_id}: succeeded={result.succeeded} "
                f"attempts={len(result.records)} "
                f"input_tokens={last.input_tokens} output_tokens={last.output_tokens} "
                f"latency_ms={last.latency_ms} error_type={result.error_type}"
            )

    print(f"wrote records to runs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
