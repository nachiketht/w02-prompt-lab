from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.errors import StructuredOutputError


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        body = lines[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        stripped = "\n".join(body).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    return stripped[start : end + 1]


def validate_structured_text[T: BaseModel](text: str, schema: type[T]) -> T:
    """Parse model text as JSON and validate it against schema."""
    payload: object = json.loads(_extract_json_text(text))
    return schema.model_validate(payload)


def _parse_result[T: BaseModel](
    result: CompletionResult, schema: type[T]
) -> tuple[T | None, str | None, bool]:
    if not result.succeeded or result.text is None:
        return None, result.error_type or "adapter failed", True
    try:
        return validate_structured_text(result.text, schema), None, False
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc), False


def _repair_user_content(original: str, error: str, failed_text: str) -> str:
    return (
        f"{original}\n\n"
        "The previous JSON failed validation.\n"
        f"Previous output:\n{failed_text}\n\n"
        f"Validation error:\n{error}\n\n"
        "Correct only what the validation error concerns. "
        "Return the full corrected JSON object and nothing else."
    )


def complete_structured[T: BaseModel](
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[T],
    run_id: str,
    max_repairs: int = 1,
) -> T:
    """Return a schema-validated completion with a bounded semantic repair loop.

    Transport retry remains inside the adapter.
    Schema/content repair belongs here.

    On validation failure, send the failed output and the validation error text
    back to the model and instruct it to correct only what the error concerns.
    Do not perform more than max_repairs semantic repair attempts.
    """

    result = adapter.complete(request, run_id)
    parsed, error, transport_failed = _parse_result(result, schema)
    if parsed is not None:
        return parsed
    if transport_failed:
        raise StructuredOutputError(error or "adapter failed")

    for _ in range(max_repairs):
        repair_request = request.model_copy(
            update={
                "user_content": _repair_user_content(
                    request.user_content,
                    error or "",
                    result.text or "",
                )
            }
        )
        result = adapter.complete(repair_request, run_id)
        parsed, error, transport_failed = _parse_result(result, schema)
        if parsed is not None:
            return parsed
        if transport_failed:
            raise StructuredOutputError(error or "adapter failed")

    raise StructuredOutputError(error or "schema validation failed")
