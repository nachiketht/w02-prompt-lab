"""Reusable Ollama adapter for configured local models."""

from __future__ import annotations

import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord, append_record, compute_cost

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 180.0
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _response_text(payload: dict[str, Any]) -> str | None:
    text = payload.get("response")
    if isinstance(text, str):
        return text
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


class OllamaAdapter:
    def __init__(self, model_id: str) -> None:
        settings = Settings.from_env()
        if model_id not in {config.model_id for config in settings.models.values()}:
            raise UnknownModelError(model_id)
        self.provider = "ollama"
        self.model_id = model_id
        self._base_url = settings.ollama_base_url

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        records: list[CallRecord] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record = self._one_attempt(request, run_id, attempt)
            records.append(record)
            append_record(record, run_id)
            if record.error_type is None:
                return CompletionResult(
                    succeeded=True,
                    text=record.response_text,
                    error_type=None,
                    records=records,
                )
            if record.error_type != TransientProviderError.__name__ or attempt == MAX_ATTEMPTS:
                return CompletionResult(
                    succeeded=False,
                    text=record.response_text,
                    error_type=record.error_type,
                    records=records,
                )
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(
                0, BACKOFF_BASE_SECONDS
            )
            time.sleep(delay)
        return CompletionResult(
            succeeded=False,
            text=records[-1].response_text,
            error_type=records[-1].error_type,
            records=records,
        )

    def _one_attempt(
        self,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
    ) -> CallRecord:
        started = time.perf_counter()
        payload: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model_id,
                    "system": request.system,
                    "prompt": request.user_content,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_output_tokens,
                    },
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            error = TransientProviderError(str(exc))
        else:
            if response.status_code in TRANSIENT_STATUS_CODES:
                error = TransientProviderError(response.text)
            elif response.status_code >= 400:
                error = PermanentProviderError(response.text)
            else:
                raw: object = response.json()
                if not isinstance(raw, dict):
                    error = PermanentProviderError("Ollama response was not a JSON object")
                else:
                    payload = raw
                    if _as_optional_str(payload.get("done_reason")) == "length":
                        error = TruncatedResponseError("output token ceiling reached")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._build_record(
            request=request,
            run_id=run_id,
            attempt=attempt,
            payload=payload,
            error=error,
            latency_ms=latency_ms,
        )

    def _build_record(
        self,
        *,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
        payload: dict[str, Any] | None,
        error: Exception | None,
        latency_ms: int,
    ) -> CallRecord:
        if payload is None:
            input_tokens = 0
            output_tokens = 0
            stop_reason = None
            response_text = None
        else:
            input_tokens = _as_int(payload.get("prompt_eval_count"))
            output_tokens = _as_int(payload.get("eval_count"))
            stop_reason = _as_optional_str(payload.get("done_reason"))
            response_text = _response_text(payload)
        return CallRecord(
            record_id=str(uuid.uuid4()),
            run_id=run_id,
            timestamp=datetime.now(UTC),
            provider="ollama",
            model_id=self.model_id,
            task=request.task,
            case_id=request.case_id,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            attempt=attempt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            cost_usd=compute_cost(self.model_id, input_tokens, output_tokens),
            stop_reason=stop_reason,
            error_type=type(error).__name__ if error is not None else None,
            response_text=response_text,
        )
