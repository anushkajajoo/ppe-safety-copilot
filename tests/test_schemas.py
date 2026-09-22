"""Contract tests: the event format both teammates agreed on."""
import pytest
from pydantic import ValidationError

from shared.schemas import ComplianceStatus, EventCreate, ReasonCode


def valid_event(**overrides):
    base = dict(
        device_id="EDGE-01", camera_id="CAM-01", session_id="2026-09-22-a",
        worker_display_id="Worker-017", zone_id="welding-a",
        status="POTENTIAL_VIOLATION", reasons=["MISSING_MASK"], confidence=0.87,
        frames_in_window=15, frames_missing=12, model_version="ppe4-yolo26n-v1", rules_version="1.0",
        detections=[{"object_class": "helmet", "confidence": 0.91,
                     "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 40}}],
    )
    base.update(overrides)
    return base


def test_valid_event_parses_and_gets_an_id():
    e = EventCreate(**valid_event())
    assert e.status is ComplianceStatus.POTENTIAL_VIOLATION
    assert e.reasons == [ReasonCode.MISSING_MASK]
    assert len(e.event_id) >= 8


def test_two_events_get_different_ids():
    assert EventCreate(**valid_event()).event_id != EventCreate(**valid_event()).event_id


@pytest.mark.parametrize("field,value", [
    ("status", "VIOLATON"),                 # typo -> rejected
    ("reasons", ["NOT_A_REASON"]),
    ("confidence", 1.5),
    ("worker_display_id", "John Smith"),    # real names can never enter the system
    ("frames_missing", 99),                 # more than the window
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        EventCreate(**valid_event(**{field: value}))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        EventCreate(**valid_event(face_embedding=[0.1, 0.2]))


def test_bad_bbox_rejected():
    with pytest.raises(ValidationError):
        EventCreate(**valid_event(detections=[{"object_class": "vest", "confidence": 0.5,
                                               "bbox": {"x1": 50, "y1": 10, "x2": 20, "y2": 40}}]))


def test_duplicate_reasons_removed():
    e = EventCreate(**valid_event(reasons=["MISSING_MASK", "MISSING_MASK"]))
    assert e.reasons == [ReasonCode.MISSING_MASK]
