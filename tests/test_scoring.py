from datetime import date

from promptlab.corpus import GoldLabel, load_cases
from promptlab.rules import VersionCandidate, select_current_version
from promptlab.schemas import EvidenceField, FieldStatus, PolicyExtraction, TriageOutput
from promptlab.scoring import (
    candidate_from_output,
    score_output,
    score_version_selection,
    source_sections,
)


def test_source_sections_reads_numbered_headings() -> None:
    assert source_sections("1. Document Control\nBody\n2. Scope\nText") == {
        "1. document control",
        "2. scope",
    }


def test_evidence_recall_citations_and_unsupported_avoidance() -> None:
    output = PolicyExtraction(
        document_status="valid",
        policy_name=EvidenceField(
            value="Test Policy", status="present", citation="1. Document Control"
        ),
        version=EvidenceField(value="1.0", status="present", citation="1. Document Control"),
        effective_date=EvidenceField(value=None, status="absent"),
        jurisdictions=EvidenceField(value="Pennsylvania", status="present", citation="2. Scope"),
        beneficial_ownership_threshold=EvidenceField(value=None, status="absent"),
        review_frequency=EvidenceField(value="12 months", status="present", citation="4. Review"),
        required_documents=EvidenceField(value=None, status="absent"),
    )
    gold = GoldLabel(
        id="E00",
        task="extraction",
        expected_status="valid",
        recoverable_fields=["policy_name", "version", "jurisdictions", "review_frequency"],
    )
    scores = score_output(
        run_id="test",
        task="extraction",
        case_id="E00",
        model_name="test",
        prompt_version="v1",
        output=output,
        gold=gold,
        source=(
            "1. Document Control\nTest Policy 1.0\n2. Scope\nPennsylvania\n"
            "4. Review\n12 months"
        ),
    )
    by_metric = {score.metric: score for score in scores}
    assert by_metric["required_evidence_recall"].numerator == 4
    assert by_metric["required_evidence_recall"].denominator == 4
    assert by_metric["citation_correctness"].numerator == 4
    assert by_metric["citation_correctness"].denominator == 4
    assert by_metric["unsupported_field_avoidance"].numerator == 3
    assert by_metric["unsupported_field_avoidance"].denominator == 3


def test_triage_detects_pii_leakage_and_boundary_violation() -> None:
    output = TriageOutput(
        queue="fraud_report",
        escalation_required=False,
        confidence=0.9,
        rationale="Unauthorized activity",
        draft_reply="We approved your claim. Call 215-555-0148.",
        human_review_required=True,
        customer_outcome=None,
    )
    gold = GoldLabel(
        id="T00",
        task="triage",
        expected_queue="fraud_report",
        expected_escalation=False,
    )
    scores = score_output(
        run_id="test",
        task="triage",
        case_id="T00",
        model_name="test",
        prompt_version="v1",
        output=output,
        gold=gold,
        source="Unauthorized purchase",
    )
    by_metric = {score.metric: score for score in scores}
    assert by_metric["pii_leakage"].numerator == 1
    assert by_metric["pii_leakage"].lower_is_better
    assert by_metric["human_boundary_compliance"].numerator == 0


def _extraction(
    *,
    version: str | None,
    version_status: FieldStatus,
    effective: str | None,
    date_status: FieldStatus,
) -> PolicyExtraction:
    return PolicyExtraction(
        document_status="valid",
        policy_name=EvidenceField(
            value="Test Policy", status="present", citation="1. Document Control"
        ),
        version=EvidenceField(value=version, status=version_status),
        effective_date=EvidenceField(value=effective, status=date_status),
        jurisdictions=EvidenceField(value=None, status="absent"),
        beneficial_ownership_threshold=EvidenceField(value=None, status="absent"),
        review_frequency=EvidenceField(value=None, status="absent"),
        required_documents=EvidenceField(value=None, status="absent"),
    )


def test_candidate_from_output_requires_present_parseable_date() -> None:
    parsed = candidate_from_output(
        "E02",
        _extraction(
            version="2.0",
            version_status="present",
            effective="2025-06-01",
            date_status="present",
        ),
    )
    assert parsed is not None
    assert parsed.case_id == "E02"
    assert parsed.effective_date == date(2025, 6, 1)

    assert (
        candidate_from_output(
            "E01",
            _extraction(
                version="1.0",
                version_status="present",
                effective=None,
                date_status="absent",
            ),
        )
        is None
    )
    assert (
        candidate_from_output(
            "E01",
            _extraction(
                version="1.0",
                version_status="present",
                effective="June 1, 2025",
                date_status="present",
            ),
        )
        is None
    )


def test_version_selection_scores_the_python_rule() -> None:
    selected = select_current_version(
        [
            VersionCandidate("E01", "1.0", date(2025, 1, 1)),
            VersionCandidate("E02", "2.0", date(2025, 6, 1)),
        ],
        date(2025, 6, 1),
    )
    score = score_version_selection(
        run_id="test",
        task="extraction",
        case_id="E02",
        model_name="test",
        prompt_version="v2",
        selected=selected,
        expected_current_case_id="E02",
    )
    assert score.metric == "version_selection_accuracy"
    assert score.numerator == 1
    assert score.denominator == 1
    assert score.detail == "E02"

    missed = score_version_selection(
        run_id="test",
        task="extraction",
        case_id="E02",
        model_name="test",
        prompt_version="v2",
        selected=None,
        expected_current_case_id="E02",
    )
    assert missed.numerator == 0
    assert missed.detail is None


def test_gold_labels_expose_version_group_fields() -> None:
    gold = {case.id: label for case, label in load_cases("extraction")}
    assert gold["E01"].version_group == "small-business-periodic-kyc"
    assert gold["E01"].expected_current_case_id == "E02"
    assert gold["E01"].as_of == date(2025, 6, 1)
    assert gold["E03"].version_group is None


