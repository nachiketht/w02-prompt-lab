from __future__ import annotations

from pathlib import Path

from promptlab.config import HUMAN_BOUNDARY_PATTERNS
from promptlab.records import ScoreRecord
from promptlab.schemas import TriageOutput, TriageOutputWithAnalysis
from promptlab.scoring import (
    METRIC_BOUNDARY,
    METRIC_ESCALATION,
    METRIC_MISSED,
    METRIC_QUEUE,
    METRIC_UNNECESSARY,
    load_triage_gold,
    score_triage,
)


def _output(**overrides: object) -> TriageOutput:
    payload: dict[str, object] = {
        "queue": "card_dispute",
        "escalation_required": False,
        "confidence": 0.8,
        "rationale": "Recognized duplicate charge.",
        "draft_reply": "A specialist will review the duplicate charge.",
        "human_review_required": True,
        "customer_outcome": None,
    }
    payload.update(overrides)
    return TriageOutput.model_validate(payload)


def _by_metric(records: list[ScoreRecord]) -> dict[str, ScoreRecord]:
    return {record.metric: record for record in records}


def _score(output: TriageOutput | None, gold: dict[str, object]) -> dict[str, ScoreRecord]:
    return _by_metric(
        score_triage(
            output,
            gold,
            run_id="score-run",
            case_id="T01",
            model_name="mistral",
            prompt_version="v1",
        )
    )


def test_queue_and_escalation_match_gold_fields() -> None:
    gold = {"expected_queue": "card_dispute", "expected_escalation": False}
    scores = _score(_output(), gold)

    assert scores[METRIC_QUEUE].numerator == 1
    assert scores[METRIC_ESCALATION].numerator == 1
    assert scores[METRIC_MISSED].numerator == 0
    assert scores[METRIC_UNNECESSARY].numerator == 0
    assert scores[METRIC_BOUNDARY].numerator == 1


def test_queue_mismatch_is_scored_against_expected_queue() -> None:
    gold = {"expected_queue": "account_servicing", "expected_escalation": False}
    scores = _score(_output(queue="lending"), gold)

    assert scores[METRIC_QUEUE].numerator == 0
    assert scores[METRIC_QUEUE].detail == "lending"


def test_missed_escalation_uses_escalation_required_not_human_review() -> None:
    gold = {"expected_queue": "escalate", "expected_escalation": True}
    output = _output(queue="escalate", escalation_required=False, human_review_required=True)
    scores = _score(output, gold)

    assert scores[METRIC_ESCALATION].numerator == 0
    assert scores[METRIC_MISSED].numerator == 1
    assert scores[METRIC_MISSED].lower_is_better is True
    assert scores[METRIC_UNNECESSARY].numerator == 0


def test_unnecessary_escalation_is_tracked_separately() -> None:
    gold = {"expected_queue": "card_dispute", "expected_escalation": False}
    scores = _score(_output(escalation_required=True), gold)

    assert scores[METRIC_ESCALATION].numerator == 0
    assert scores[METRIC_MISSED].numerator == 0
    assert scores[METRIC_UNNECESSARY].numerator == 1
    assert scores[METRIC_UNNECESSARY].lower_is_better is True


def test_human_boundary_fails_on_final_outcome_language() -> None:
    gold = {"expected_queue": "card_dispute", "expected_escalation": False}
    output = _output(
        draft_reply="Your dispute has been approved and the funds will be refunded."
    )
    scores = _score(output, gold)

    assert scores[METRIC_QUEUE].numerator == 1
    assert scores[METRIC_BOUNDARY].numerator == 0
    assert scores[METRIC_BOUNDARY].detail is not None


def test_human_boundary_uses_config_patterns() -> None:
    assert HUMAN_BOUNDARY_PATTERNS
    draft = "The loan was granted this morning."
    assert any(pattern.search(draft) for pattern in HUMAN_BOUNDARY_PATTERNS)


def test_analysis_subclass_scores_shared_triage_fields() -> None:
    gold = {"expected_queue": "fraud_report", "expected_escalation": False}
    output = TriageOutputWithAnalysis(
        queue="fraud_report",
        escalation_required=False,
        confidence=0.7,
        rationale="Unauthorized charges.",
        draft_reply="A specialist will review the unauthorized charges.",
        human_review_required=True,
        customer_outcome=None,
        analysis="Card is present and charges are unrecognized.",
    )
    scores = _score(output, gold)
    assert scores[METRIC_QUEUE].numerator == 1
    assert scores[METRIC_ESCALATION].numerator == 1
    assert scores[METRIC_BOUNDARY].numerator == 1


def test_failed_structured_output_scores_zero_and_can_miss_escalation() -> None:
    gold = {"expected_queue": "escalate", "expected_escalation": True}
    scores = _score(None, gold)

    assert scores[METRIC_QUEUE].numerator == 0
    assert scores[METRIC_ESCALATION].numerator == 0
    assert scores[METRIC_MISSED].numerator == 1
    assert scores[METRIC_UNNECESSARY].numerator == 0
    assert scores[METRIC_BOUNDARY].numerator == 0


def test_scoring_module_does_not_import_model_clients() -> None:
    source = Path("src/promptlab/scoring.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "ollama" not in lowered
    assert "httpx" not in lowered
    assert "adapters" not in lowered


def test_load_triage_gold_exposes_expected_escalation() -> None:
    gold = load_triage_gold()
    assert gold["T06"]["expected_queue"] == "escalate"
    assert gold["T06"]["expected_escalation"] is True
    assert gold["T01"]["expected_escalation"] is False
