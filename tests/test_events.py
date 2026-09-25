"""
Tests for shared/events.py - the pure rules behind the structured event record.

These are the decisions that make a history meaningful: what counts as an event, how urgent
it is, how the reason is built, when one bad frame is ignored, when a continuing problem is
the same event, and which events retention may never touch.
"""
import pytest

from shared.events import (ACKNOWLEDGED, CAMERA_FAILURE, CLOSED, CRITICAL, EVENT_TYPES, HIGH,
                           LOW, MEDIUM, MODEL_FAILURE, NEW, PPE_COMPLIANT, PPE_REVIEW,
                           PPE_VIOLATION, Debouncer, InvalidEvent, dedupe_key,
                           eligible_for_cleanup, event_type_for, expired, explain,
                           is_protected, person_label, severity_for, should_merge,
                           signature_of, to_csv)

DAY = 86400


# ------------------------------------------------------------------ vocabulary
def test_a_decision_maps_to_exactly_one_event_type():
    assert event_type_for("STOP") == PPE_VIOLATION
    assert event_type_for("REVIEW") == PPE_REVIEW
    assert event_type_for("GO") == PPE_COMPLIANT
    with pytest.raises(InvalidEvent):
        event_type_for("MAYBE")


def test_severity_follows_the_decision_and_the_zone():
    assert severity_for("GO") == LOW
    assert severity_for("REVIEW") == MEDIUM
    assert severity_for("STOP", ["helmet"]) == HIGH
    assert severity_for("STOP", ["helmet"], "RESTRICTED") == CRITICAL
    assert severity_for("STOP", ["helmet", "vest"]) == CRITICAL       # nothing worn at all


def test_a_system_failure_is_urgent_wherever_it_happens():
    """A camera that stopped is not a small problem because the area is ordinary."""
    for kind in (CAMERA_FAILURE, MODEL_FAILURE):
        assert severity_for("REVIEW", [], "GENERAL", kind) == HIGH


def test_every_event_type_is_named_in_the_vocabulary():
    for kind in (PPE_VIOLATION, PPE_REVIEW, PPE_COMPLIANT, CAMERA_FAILURE, MODEL_FAILURE):
        assert kind in EVENT_TYPES


# ------------------------------------------------------------------ person id
def test_a_person_label_is_anonymous_and_session_scoped():
    assert person_label("Person-3") == "P-03"
    assert person_label("Worker-017") == "P-17"
    assert person_label(None, 4) == "P-04"
    assert person_label("") == "P-01"


def test_a_person_label_never_carries_a_name():
    """Whatever the tracker called them, what is stored is a number."""
    for raw in ("ana silva", "employee_4471", "face_id_88"):
        label = person_label(raw, 2)
        assert label.startswith("P-") and " " not in label
        assert "ana" not in label.lower() and "employee" not in label.lower()


# ---------------------------------------------------------------- explanation
def test_the_explanation_is_built_from_the_numbers():
    reason = explain(["vest"], [], {"helmet": 0.92, "vest": 0.19}, 0.50, "STOP")
    assert "vest" in reason and "0.19" in reason and "0.50" in reason


def test_the_explanation_of_an_uncertain_case_says_so():
    reason = explain([], ["vest"], {"helmet": 0.92, "vest": 0.42}, 0.50, "REVIEW")
    assert "0.42" in reason and "person should look" in reason


def test_the_explanation_is_deterministic():
    args = (["vest"], [], {"helmet": 0.92, "vest": 0.19}, 0.50, "STOP")
    assert explain(*args) == explain(*args)


def test_a_compliant_explanation_lists_what_was_seen():
    reason = explain([], [], {"helmet": 0.91, "vest": 0.77}, 0.50, "GO")
    assert "0.91" in reason and "0.77" in reason


# ------------------------------------------------------------------- debounce
def test_one_bad_frame_does_not_make_an_event():
    debouncer = Debouncer(frames=3)
    signature = signature_of(PPE_VIOLATION, ["helmet"], [])
    assert debouncer.confirm("camera:0", "P-01", signature) is False
    assert debouncer.confirm("camera:0", "P-01", signature) is False
    assert debouncer.confirm("camera:0", "P-01", signature) is True


def test_the_streak_keeps_confirming_while_the_problem_continues():
    """The store decides new-or-extend; the debouncer only says 'this is real'."""
    debouncer = Debouncer(frames=2)
    signature = signature_of(PPE_VIOLATION, ["helmet"], [])
    results = [debouncer.confirm("camera:0", "P-01", signature) for _ in range(5)]
    assert results == [False, True, True, True, True]


def test_becoming_compliant_resets_the_streak():
    debouncer = Debouncer(frames=3)
    signature = signature_of(PPE_VIOLATION, ["helmet"], [])
    debouncer.confirm("camera:0", "P-01", signature)
    debouncer.confirm("camera:0", "P-01", signature)
    debouncer.reset("camera:0", "P-01")
    assert debouncer.confirm("camera:0", "P-01", signature) is False


def test_a_different_problem_starts_its_own_streak():
    debouncer = Debouncer(frames=2)
    helmet = signature_of(PPE_VIOLATION, ["helmet"], [])
    vest = signature_of(PPE_VIOLATION, ["vest"], [])
    debouncer.confirm("camera:0", "P-01", helmet)
    assert debouncer.confirm("camera:0", "P-01", vest) is False
    assert debouncer.streak("camera:0", "P-01", helmet) == 0


def test_two_people_are_counted_separately():
    debouncer = Debouncer(frames=2)
    signature = signature_of(PPE_VIOLATION, ["helmet"], [])
    debouncer.confirm("camera:0", "P-01", signature)
    assert debouncer.confirm("camera:0", "P-02", signature) is False


def test_a_debounce_of_one_confirms_immediately():
    assert Debouncer(frames=1).confirm("camera:0", "P-01", "sig") is True


# -------------------------------------------------------------- deduplication
def test_the_signature_ignores_confidence_but_not_the_items():
    assert signature_of(PPE_VIOLATION, ["helmet"], []) == signature_of(PPE_VIOLATION, ["helmet"], [])
    assert signature_of(PPE_VIOLATION, ["helmet"], []) != signature_of(PPE_VIOLATION, ["vest"], [])
    assert signature_of(PPE_VIOLATION, ["helmet"], []) != signature_of(PPE_REVIEW, ["helmet"], [])


def test_the_dedupe_key_separates_cameras_and_people():
    base = dedupe_key("CAM-01", "P-01", "sig")
    assert base != dedupe_key("CAM-02", "P-01", "sig")
    assert base != dedupe_key("CAM-01", "P-02", "sig")


def test_a_continuing_problem_merges_inside_the_window():
    assert should_merge(1000.0, 1020.0, 30.0) is True
    assert should_merge(1000.0, 1060.0, 30.0) is False


def test_a_clock_that_goes_backwards_starts_a_new_event():
    assert should_merge(1000.0, 900.0, 30.0) is False


# ------------------------------------------------------------------ retention
def test_events_waiting_on_a_person_are_protected():
    assert is_protected(NEW) and is_protected(ACKNOWLEDGED)
    assert not is_protected(CLOSED)


def test_age_is_measured_from_when_it_happened():
    now = 100 * DAY
    assert expired(now - 40 * DAY, now, 30) is True
    assert expired(now - 20 * DAY, now, 30) is False


def test_retention_never_removes_an_unreviewed_event():
    """The rule that stops a cleanup job from erasing the alerts nobody answered."""
    now = 100 * DAY
    old = now - 365 * DAY
    assert eligible_for_cleanup(NEW, old, now, 30) is False
    assert eligible_for_cleanup(ACKNOWLEDGED, old, now, 30) is False
    assert eligible_for_cleanup(CLOSED, old, now, 30) is True


def test_a_recent_closed_event_is_kept():
    now = 100 * DAY
    assert eligible_for_cleanup(CLOSED, now - 2 * DAY, now, 30) is False


def test_a_soft_deleted_event_expires_even_if_it_was_never_closed():
    now = 100 * DAY
    assert eligible_for_cleanup(NEW, now - 90 * DAY, now, 30, deleted=True) is True


# --------------------------------------------------------------------- export
def test_csv_export_has_a_header_and_flattens_lists():
    rows = [{"event_id": "EVT-000001", "occurred_at": "2026-09-25T10:00:00+00:00",
             "camera_id": "CAM-01", "person_id": "P-03", "event_type": PPE_VIOLATION,
             "decision": "STOP", "severity": HIGH, "status": NEW,
             "required_ppe": ["helmet", "vest"], "missing_ppe": ["vest"],
             "uncertain_ppe": [], "confidence": 0.93, "rule_id": "R-MISSING-PPE",
             "reason": "Required vest not detected.", "acknowledged_by": None,
             "acknowledged_at": None, "sync_status": "LOCAL"}]
    csv_text = to_csv(rows)
    assert csv_text.splitlines()[0].startswith("event_id,occurred_at,camera_id")
    assert "helmet vest" in csv_text and "EVT-000001" in csv_text


def test_the_export_never_contains_an_evidence_path():
    """A spreadsheet of events must not double as a list of images to go and fetch."""
    rows = [{"event_id": "EVT-000001", "snapshot_path": "/data/snapshots/masked/x.jpg",
             "recording_path": "/data/recordings/x.mp4", "decision": "STOP"}]
    csv_text = to_csv(rows)
    assert "snapshot" not in csv_text and ".jpg" not in csv_text and ".mp4" not in csv_text
