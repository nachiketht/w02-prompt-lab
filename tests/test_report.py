from decimal import Decimal
from pathlib import Path

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.report import write_reports


def test_report_is_generated_from_records(tmp_path: object) -> None:
    root = Path(str(tmp_path))
    usage = [
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="configured-model",
            prompt_version="v1",
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=100,
            completion_tokens=25,
            latency_ms=125.0,
            cost_usd=Decimal("0"),
            prompt_id="triage",
        )
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="configured-model",
            prompt_version="v1",
            succeeded=True,
            repairs=0,
            output={"queue": "card_dispute"},
            elapsed_ms=125.0,
            model_ms=125.0,
            attempts=1,
            prompt_id="triage",
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            prompt_version="v1",
            scorer_version="2.0.0",
            metric="queue_accuracy",
            numerator=1,
            denominator=1,
            prompt_id="triage",
        )
    ]
    report = root / "comparison.md"
    decision = root / "model-decision.md"
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "1/1" in text
    assert "triage.v1" in text
    assert "Median case latency" in text
    assert "125 ms" in text
    assert "Case n" in text
    assert "Attempt n" in text
    assert "12 cases" in text
    assert "mistral" in decision.read_text(encoding="utf-8")


def test_report_labels_transfer_and_keeps_retry_stratum_separate(tmp_path: object) -> None:
    root = Path(str(tmp_path))
    usage = [
        UsageRecord(
            run_id="demo",
            task="extraction",
            case_id="E01",
            model_name="qwen",
            model_id="configured-model",
            prompt_version="v2",
            attempt=1,
            kind="primary",
            status="schema_invalid",
            prompt_tokens=80,
            completion_tokens=20,
            latency_ms=200.0,
            cost_usd=Decimal("0"),
            prompt_id="extract",
        ),
        UsageRecord(
            run_id="demo",
            task="extraction",
            case_id="E01",
            model_name="qwen",
            model_id="configured-model",
            prompt_version="v2",
            attempt=1,
            kind="repair",
            status="success",
            prompt_tokens=90,
            completion_tokens=22,
            latency_ms=180.0,
            cost_usd=Decimal("0"),
            prompt_id="extract",
        ),
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="extraction",
            case_id="E01",
            model_name="qwen",
            model_id="configured-model",
            prompt_version="v2",
            succeeded=True,
            repairs=1,
            output={},
            elapsed_ms=400.0,
            model_ms=380.0,
            attempts=2,
            prompt_id="extract",
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="extraction",
            case_id="E01",
            model_name="qwen",
            prompt_version="v2",
            scorer_version="day5.v1",
            metric="required_evidence_recall",
            numerator=4,
            denominator=4,
            prompt_id="extract",
        )
    ]
    report = root / "comparison.md"
    write_reports(
        run_id="demo",
        models=["qwen"],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=root / "model-decision.md",
        transfer_keys={("extraction", "qwen", "v2")},
    )
    text = report.read_text(encoding="utf-8")
    assert "extract.v2 transfer" in text
    assert "Retry/repair cases: 1/1" in text
    assert "not re-weighted" in text
    assert "400 ms" in text
