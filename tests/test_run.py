from datetime import UTC, date, datetime
from pathlib import Path

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.corpus import Case, GoldLabel
from promptlab.run import (
    TASK_SPECS,
    RecordingAdapter,
    _render_request,
    evaluate_case,
    is_transfer,
    parse_args,
    score_version_groups,
    spec_for,
    usage_from_calls,
)
from promptlab.schemas import EvidenceField, PolicyExtraction
from promptlab.usage import CallRecord

TRIAGE_JSON = (
    '{"queue":"card_dispute","escalation_required":false,"confidence":0.8,'
    '"rationale":"Recognized duplicate charge.",'
    '"draft_reply":"A specialist will review the duplicate charge.",'
    '"human_review_required":true,"customer_outcome":null}'
)


def _call(**overrides: object) -> CallRecord:
    payload: dict[str, object] = {
        "record_id": "r1",
        "run_id": "run",
        "timestamp": datetime.now(UTC),
        "provider": "ollama",
        "model_id": "configured-model",
        "task": "triage",
        "case_id": "T01",
        "prompt_id": "triage",
        "prompt_version": "v1",
        "attempt": 1,
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_input_tokens": None,
        "latency_ms": 100,
        "cost_usd": 0.0,
        "stop_reason": "stop",
        "error_type": None,
        "response_text": TRIAGE_JSON,
    }
    payload.update(overrides)
    return CallRecord.model_validate(payload)


class FakeAdapter:
    provider = "ollama"
    model_id = "configured-model"

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        record = _call(run_id=run_id, case_id=request.case_id, prompt_id=request.prompt_id)
        return CompletionResult(
            succeeded=True,
            text=TRIAGE_JSON,
            error_type=None,
            records=[record],
        )


def test_parse_args_defaults_to_all_filters() -> None:
    args = parse_args(["--run-id", "local-comparison-01"], ["mistral", "qwen"])
    assert args.run_id == "local-comparison-01"
    assert args.tasks is None
    assert args.models is None


def test_qwen_extraction_uses_adapted_v3() -> None:
    spec = spec_for("extraction", "qwen")
    assert spec.prompt_id == "extract"
    assert spec.prompt_version == "v3"
    assert is_transfer("qwen", spec) is False
    assert is_transfer("mistral", spec_for("extraction", "mistral")) is False
    assert spec_for("extraction", "mistral").prompt_version == "v2"


def test_qwen_rows_are_prompt_transfer_except_adapted_extraction() -> None:
    assert is_transfer("qwen", TASK_SPECS["triage"]) is True
    assert is_transfer("mistral", TASK_SPECS["triage"]) is False
    assert is_transfer("qwen", TASK_SPECS["summarization"]) is True


def test_usage_maps_transport_retries_and_repairs() -> None:
    primary = CompletionResult(
        succeeded=True,
        text="{",
        error_type=None,
        records=[
            _call(attempt=1, error_type="TransientProviderError", latency_ms=50),
            _call(attempt=2, error_type=None, latency_ms=80),
        ],
    )
    repair = CompletionResult(
        succeeded=True,
        text=TRIAGE_JSON,
        error_type=None,
        records=[_call(attempt=1, latency_ms=90)],
    )
    rows = usage_from_calls(
        model_name="mistral",
        prompt_id="triage",
        calls=[primary, repair],
        validated=[False, True],
    )
    assert [row.kind for row in rows] == ["primary", "transport_retry", "repair"]
    assert rows[0].status == "transport_error"
    assert rows[1].status == "schema_invalid"
    assert rows[2].status == "success"


def test_evaluate_case_records_end_to_end_latency() -> None:
    adapter = RecordingAdapter(FakeAdapter())
    case = Case(id="T01", task="triage", document_text="I was charged twice.")
    gold = GoldLabel(
        id="T01",
        task="triage",
        expected_queue="card_dispute",
        expected_escalation=False,
    )
    usage, output, scores, parsed = evaluate_case(
        adapter=adapter,
        spec=TASK_SPECS["triage"],
        case=case,
        gold=gold,
        run_id="run",
        model_name="mistral",
        temperature=0.0,
        max_repairs=1,
    )
    assert parsed is not None
    assert output.succeeded is True
    assert output.elapsed_ms is not None
    assert output.elapsed_ms >= 0
    assert output.model_ms == 100
    assert output.attempts == 1
    assert output.repairs == 0
    assert output.retries == 0
    assert output.truncations == 0
    assert output.call_latencies_ms == [100]
    assert usage[0].kind == "primary"
    assert any(score.metric == "queue_accuracy" and score.numerator == 1 for score in scores)


def test_version_groups_score_the_python_rule() -> None:
    def extraction(case_id: str, effective: str) -> PolicyExtraction:
        return PolicyExtraction(
            document_status="valid",
            policy_name=EvidenceField(
                value="KYC", status="present", citation="1. Document Control"
            ),
            version=EvidenceField(value="1.0", status="present", citation="1. Document Control"),
            effective_date=EvidenceField(
                value=effective, status="present", citation="1. Document Control"
            ),
            jurisdictions=EvidenceField(value=None, status="absent"),
            beneficial_ownership_threshold=EvidenceField(value=None, status="absent"),
            review_frequency=EvidenceField(value=None, status="absent"),
            required_documents=EvidenceField(value=None, status="absent"),
        )

    rows = [
        (
            Case(id="E01", task="extraction", document_text="old"),
            GoldLabel(
                id="E01",
                task="extraction",
                version_group="kyc",
                expected_current_case_id="E02",
                as_of=date(2025, 6, 1),
            ),
            extraction("E01", "2025-01-01"),
        ),
        (
            Case(id="E02", task="extraction", document_text="new"),
            GoldLabel(
                id="E02",
                task="extraction",
                version_group="kyc",
                expected_current_case_id="E02",
                as_of=date(2025, 6, 1),
            ),
            extraction("E02", "2025-06-01"),
        ),
    ]
    scores = score_version_groups(
        run_id="run",
        spec=TASK_SPECS["extraction"],
        model_name="mistral",
        model_id="configured-model",
        rows=rows,
    )
    assert len(scores) == 1
    assert scores[0].metric == "version_selection_accuracy"
    assert scores[0].numerator == 1
    assert scores[0].case_id == "E02"


def test_extract_v3_fills_schema_on_the_system_layer() -> None:
    spec = spec_for("extraction", "qwen")
    case = Case(id="E01", task="extraction", document_text="1. Document Control\nPolicy")
    request = _render_request(spec=spec, case=case, temperature=0.0)
    assert "{schema_description}" not in request.system
    assert "PolicyExtraction" in request.system
    assert "{schema_description}" not in request.user_content


def test_run_module_has_no_model_id_literals() -> None:
    source = Path("src/promptlab/run.py").read_text(encoding="utf-8")
    assert "mistral:7b" not in source
    assert "qwen3:8b" not in source
    assert "httpx" not in source
    assert "OllamaProvider" not in source
