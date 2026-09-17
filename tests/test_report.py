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
    indepth = (root / "in-depth-analysis.md").read_text(encoding="utf-8")
    assert "1/1" in text
    assert "triage.v1" in text
    assert "| Input tokens |" in text
    assert "Median latency" in text
    assert "125 ms" in text
    assert "Median case latency" in indepth
    assert "Case n" in indepth
    assert "Attempt n" in indepth
    assert "12-case" in text
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
            retries=0,
            truncations=0,
            call_latencies_ms=[200.0, 180.0],
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
    indepth = (root / "in-depth-analysis.md").read_text(encoding="utf-8")
    assert "extract.v2 transfer" in text
    assert "extract.v2 transfer" in indepth
    assert "Retry/repair/truncation cases: 1/1" in indepth
    assert "not re-weighted" in indepth
    assert "400 ms" in indepth


def test_report_appends_quality_sidecar(tmp_path: object) -> None:
    root = Path(str(tmp_path))
    (root / "comparison-quality.md").write_text(
        "## Quality\n\nDirectional reading only.\n",
        encoding="utf-8",
    )
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=[],
        outputs=[],
        scores=[],
        report_path=root / "comparison.md",
        decision_path=root / "model-decision.md",
    )
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "## Quality" in text
    assert "Directional reading only." in text


def test_filled_decision_is_not_overwritten(tmp_path: object) -> None:
    root = Path(str(tmp_path))
    decision = root / "model-decision.md"
    filled = (
        "# Model Decision Record\n\n"
        "- selected model: qwen\n"
        "- prompt version: extract.v3\n"
    )
    decision.write_text(filled, encoding="utf-8")
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=[],
        outputs=[],
        scores=[],
        report_path=root / "comparison.md",
        decision_path=decision,
    )
    assert decision.read_text(encoding="utf-8") == filled
