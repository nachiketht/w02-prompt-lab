"""Day 3: schema-validated summarization and extraction via complete_structured."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.errors import StructuredOutputError
from promptlab.schemas import PolicyExtraction, SummarizationOutput, schema_description
from promptlab.structured import complete_structured, validate_structured_text

MAX_OUTPUT_TOKENS = 1024
LEAKAGE_STRINGS = ("Northglass", "Norwyn", "Bellwater", "Larkspur", "Meadowcross")
NUMBERED_HEADING = re.compile(r"^\d+\.\s+.+")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
TaskLiteral = Literal["summarization", "extraction"]


class CountingAdapter:
    """Count adapter.complete calls so the runner can measure schema repairs."""

    def __init__(self, inner: OllamaAdapter) -> None:
        self._inner = inner
        self.provider = inner.provider
        self.model_id = inner.model_id
        self.calls = 0
        self.texts: list[str | None] = []

    def reset(self) -> None:
        self.calls = 0
        self.texts = []

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        result = self._inner.complete(request, run_id)
        self.calls += 1
        self.texts.append(result.text)
        return result


@dataclass
class TaskMetrics:
    cases: int = 0
    repairs: int = 0
    schema_successes: int = 0
    citation_failures: int = 0
    leakage_count: int = 0
    validation_errors: list[str] = field(default_factory=list)


def load_cases(filename: str) -> list[tuple[str, str]]:
    path = PROJECT_ROOT / "cases" / filename
    cases: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed: object = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise TypeError(f"each line in {filename} must be a JSON object")
        case_id = parsed["id"]
        source = parsed["source"]
        if not isinstance(case_id, str) or not isinstance(source, str):
            raise TypeError("id and source must be strings")
        cases.append((case_id, source))
    return cases


def render_prompt(template_name: str, document_text: str, schema: type[BaseModel]) -> str:
    template = (PROJECT_ROOT / "src" / "prompts" / template_name).read_text(encoding="utf-8")
    return template.replace("{schema_description}", schema_description(schema)).replace(
        "{document_text}", document_text
    )


def section_headings(source: str) -> set[str]:
    headings: set[str] = set()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if NUMBERED_HEADING.match(line):
            headings.add(line)
            headings.add(re.sub(r"^\d+\.\s+", "", line))
        matched = MARKDOWN_HEADING.match(line)
        if matched:
            headings.add(matched.group(1).strip())
            headings.add(line)
    return headings


def citation_failure_count(output: SummarizationOutput | PolicyExtraction, source: str) -> int:
    headings = section_headings(source)
    failures = 0
    for evidence in output.evidence_fields().values():
        if evidence.status != "present":
            continue
        citation = evidence.citation
        if citation is None or citation.strip() == "" or citation.strip() not in headings:
            failures += 1
    return failures


def leaked_from_examples(output: PolicyExtraction) -> bool:
    blob = output.model_dump_json()
    return any(marker in blob for marker in LEAKAGE_STRINGS)


def first_validation_error(texts: list[str | None], schema: type[BaseModel]) -> str | None:
    if not texts or texts[0] is None:
        return None
    try:
        validate_structured_text(texts[0], schema)
    except (json.JSONDecodeError, ValidationError) as exc:
        return str(exc)
    return None


def run_task(
    *,
    adapter: CountingAdapter,
    run_id: str,
    task: TaskLiteral,
    prompt_id: str,
    prompt_version: str,
    template_name: str,
    schema: type[SummarizationOutput] | type[PolicyExtraction],
    cases: list[tuple[str, str]],
    max_repairs: int,
    temperature: float,
    check_leakage: bool,
) -> TaskMetrics:
    metrics = TaskMetrics()
    for case_id, source in cases:
        adapter.reset()
        metrics.cases += 1
        request = CompletionRequest(
            task=task,
            case_id=case_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            system="Return only a JSON object. Do not wrap it in Markdown.",
            user_content=render_prompt(template_name, source, schema),
            temperature=temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        try:
            parsed = complete_structured(
                adapter,
                request,
                schema,
                run_id,
                max_repairs=max_repairs,
            )
        except StructuredOutputError as exc:
            repaired = adapter.calls > 1
            metrics.repairs += int(repaired)
            error = first_validation_error(adapter.texts, schema) or str(exc)
            metrics.validation_errors.append(error)
            print(
                f"{case_id}: succeeded=False repaired={repaired} "
                f"calls={adapter.calls} error={exc}"
            )
            continue

        repaired = adapter.calls > 1
        metrics.repairs += int(repaired)
        metrics.schema_successes += 1
        if repaired:
            initial_error = first_validation_error(adapter.texts, schema)
            if initial_error is not None:
                metrics.validation_errors.append(initial_error)

        failures = citation_failure_count(parsed, source)
        metrics.citation_failures += failures
        leaked = False
        if check_leakage and isinstance(parsed, PolicyExtraction):
            leaked = leaked_from_examples(parsed)
            metrics.leakage_count += int(leaked)
        print(
            f"{case_id}: succeeded=True repaired={repaired} "
            f"calls={adapter.calls} citation_failures={failures} leaked={leaked}"
        )
    return metrics


def write_run_copy(run_id: str) -> Path:
    source = PROJECT_ROOT / "runs" / f"{run_id}.jsonl"
    destination = PROJECT_ROOT / "docs" / "day3-run.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def main() -> None:
    os.chdir(PROJECT_ROOT)
    settings = Settings.from_env()
    model = settings.models["mistral"]
    run_id = str(uuid.uuid4())
    adapter = CountingAdapter(OllamaAdapter(model_id=model.model_id))

    print(f"run_id={run_id} model_id={adapter.model_id} temperature={settings.temperature}")

    summarization = run_task(
        adapter=adapter,
        run_id=run_id,
        task="summarization",
        prompt_id="summarize",
        prompt_version="v1",
        template_name="summarize.v1.md",
        schema=SummarizationOutput,
        cases=load_cases("summarization.jsonl"),
        max_repairs=settings.max_schema_repairs,
        temperature=settings.temperature,
        check_leakage=False,
    )
    extraction = run_task(
        adapter=adapter,
        run_id=run_id,
        task="extraction",
        prompt_id="extract",
        prompt_version="v2",
        template_name="extract.v2.md",
        schema=PolicyExtraction,
        cases=load_cases("extraction.jsonl"),
        max_repairs=settings.max_schema_repairs,
        temperature=settings.temperature,
        check_leakage=True,
    )

    evidence_path = write_run_copy(run_id)
    print(f"wrote records to {evidence_path}")
    print(
        "summarization "
        f"repair_rate={summarization.repairs}/{summarization.cases} "
        f"schema_successes={summarization.schema_successes} "
        f"citation_failures={summarization.citation_failures}"
    )
    print(
        "extraction "
        f"repair_rate={extraction.repairs}/{extraction.cases} "
        f"schema_successes={extraction.schema_successes} "
        f"citation_failures={extraction.citation_failures} "
        f"leakage_count={extraction.leakage_count}"
    )
    if summarization.validation_errors:
        print("summarization validation errors:")
        for error in summarization.validation_errors:
            print(f"- {error}")
    if extraction.validation_errors:
        print("extraction validation errors:")
        for error in extraction.validation_errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
