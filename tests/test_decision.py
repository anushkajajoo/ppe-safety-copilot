"""
Tests for shared/decision.py - the GO / REVIEW / STOP layer.

The most important test in this file is the last group: a system that cannot see must never
answer GO. Everything else is the ordinary business of thresholds and bands.
"""
import pytest

from shared.decision import (GO, MISSING, PRESENT, REVIEW, STOP, UNCERTAIN, UnsafePolicy,
                             classify_item, decide_frame, decide_person, system_fault,
                             validate_fail_safe, worst_of)

REQUIRED = ["helmet", "vest"]


# ------------------------------------------------------------ item classification
def test_an_item_above_the_threshold_is_present():
    assert classify_item(0.91, 0.50, 0.15) == PRESENT
    assert classify_item(0.50, 0.50, 0.15) == PRESENT          # the boundary counts as met


def test_an_item_just_below_the_threshold_is_uncertain_not_absent():
    """0.42 is not 'no helmet'. It is 'I am not sure', and that is a different decision."""
    assert classify_item(0.42, 0.50, 0.15) == UNCERTAIN
    assert classify_item(0.35, 0.50, 0.15) == UNCERTAIN         # bottom of the band


def test_an_item_well_below_the_threshold_is_missing():
    assert classify_item(0.34, 0.50, 0.15) == MISSING
    assert classify_item(0.0, 0.50, 0.15) == MISSING


def test_the_band_can_be_switched_off():
    assert classify_item(0.49, 0.50, 0.0) == MISSING


# ------------------------------------------------------------------- one person
def test_all_ppe_present_is_go():
    decision = decide_person({"helmet": 0.91, "vest": 0.77}, REQUIRED, 0.50)
    assert decision.decision == GO and decision.rule == "R-ALL-PRESENT"
    assert decision.missing == [] and decision.uncertain == []
    assert not decision.needs_human


def test_a_confident_violation_is_stop():
    decision = decide_person({"helmet": 0.12, "vest": 0.88}, REQUIRED, 0.50)
    assert decision.decision == STOP and decision.rule == "R-MISSING-PPE"
    assert decision.missing == ["helmet"]
    assert "0.12" in decision.reason and "0.50" in decision.reason
    assert decision.needs_human


def test_an_undetected_item_is_a_violation_not_a_pass():
    """The failure that matters: silence from the detector must not read as compliance."""
    decision = decide_person({}, REQUIRED, 0.50)
    assert decision.decision == STOP and decision.missing == REQUIRED


def test_a_low_confidence_item_is_review():
    decision = decide_person({"helmet": 0.42, "vest": 0.80}, REQUIRED, 0.50)
    assert decision.decision == REVIEW and decision.rule == "R-LOW-CONFIDENCE"
    assert decision.uncertain == ["helmet"] and decision.missing == []


def test_missing_beats_uncertain():
    """One item absent and another unclear is still a violation - the worse case wins."""
    decision = decide_person({"helmet": 0.42, "vest": 0.05}, REQUIRED, 0.50)
    assert decision.decision == STOP


def test_the_threshold_is_configurable_and_quoted_in_the_reason():
    strict = decide_person({"helmet": 0.60, "vest": 0.60}, REQUIRED, 0.80, review_margin=0.0)
    assert strict.decision == STOP and "0.80" in strict.reason
    lenient = decide_person({"helmet": 0.60, "vest": 0.60}, REQUIRED, 0.30)
    assert lenient.decision == GO


def test_the_evidence_explains_every_required_item():
    decision = decide_person({"helmet": 0.12, "vest": 0.88}, REQUIRED, 0.50)
    evidence = {e.item: e for e in decision.evidence}
    assert set(evidence) == set(REQUIRED)
    assert evidence["helmet"].state == MISSING and evidence["helmet"].confidence == 0.12
    assert evidence["vest"].state == PRESENT
    assert all(e.threshold == 0.50 for e in decision.evidence)


def test_requiring_nothing_is_go():
    assert decide_person({}, [], 0.50).decision == GO


def test_a_decision_serialises_with_everything_needed_to_defend_it():
    payload = decide_person({"helmet": 0.12}, ["helmet"], 0.50, person_id="Person-1",
                            source="camera:0").as_dict()
    for key in ("decision", "reason", "rule", "evidence", "required", "missing",
                "uncertain", "threshold", "person_id", "source", "at"):
        assert key in payload
    assert payload["person_id"] == "Person-1" and payload["source"] == "camera:0"


# -------------------------------------------------------------------- one frame
def test_an_empty_frame_is_go_not_a_fault():
    decision = decide_frame([], source="camera:0")
    assert decision.decision == GO and decision.rule == "R-NO-PEOPLE"


def test_the_frame_takes_the_worst_persons_decision():
    people = [decide_person({"helmet": 0.9, "vest": 0.9}, REQUIRED),
              decide_person({"helmet": 0.42, "vest": 0.9}, REQUIRED),
              decide_person({"helmet": 0.01, "vest": 0.9}, REQUIRED)]
    frame = decide_frame(people, source="camera:0")
    assert frame.decision == STOP
    assert "3 person(s)" in frame.reason and "1 with a confident violation" in frame.reason
    assert "1 uncertain" in frame.reason


def test_one_uncertain_person_makes_the_frame_review():
    people = [decide_person({"helmet": 0.9, "vest": 0.9}, REQUIRED),
              decide_person({"helmet": 0.42, "vest": 0.9}, REQUIRED)]
    assert decide_frame(people).decision == REVIEW


def test_worst_of_combines_decision_strings():
    assert worst_of([GO, GO]) == GO
    assert worst_of([GO, REVIEW]) == REVIEW
    assert worst_of([REVIEW, STOP, GO]) == STOP
    assert worst_of([]) == GO


# ------------------------------------------------------ the rule that matters most
def test_a_system_fault_never_reports_go():
    for reason in ("The detection model is not available.", "The camera did not open.",
                   "The frame could not be decoded."):
        assert system_fault(reason).decision == REVIEW
        assert system_fault(reason, fail_safe=STOP).decision == STOP


def test_a_system_fault_says_plainly_that_it_could_not_confirm():
    decision = system_fault("The detection model is not available.")
    assert decision.rule == "R-SYSTEM-FAULT"
    assert "cannot confirm compliance" in decision.reason
    assert decision.needs_human


def test_go_cannot_be_configured_as_the_fail_safe():
    """A broken system that answers GO is the one outcome this project must never produce."""
    with pytest.raises(UnsafePolicy):
        validate_fail_safe(GO)
    with pytest.raises(UnsafePolicy):
        system_fault("anything", fail_safe=GO)
    with pytest.raises(UnsafePolicy):
        validate_fail_safe("COMPLIANT")
    assert validate_fail_safe(REVIEW) == REVIEW and validate_fail_safe(STOP) == STOP
