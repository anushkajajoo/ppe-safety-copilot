"""
Event / zone / audit API tests (FastAPI TestClient + temporary SQLite file).
Covers: auth, zone sync, idempotent event ingestion, conflict detection,
filters, review status, and audit hash-chain tamper detection.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from server.config import Settings
from server.main import create_app

KEY = "test-key-123"
HDR = {"X-API-Key": KEY}

ZONES = [
    {"zone_id": "general-a", "zone_name": "General-A", "zone_type": "GENERAL", "camera_id": "CAM-01",
     "polygon": [[0, 0], [1279, 0], [1279, 719], [0, 719]], "required_ppe": ["helmet", "vest"]},
    {"zone_id": "welding-a", "zone_name": "Welding-A", "zone_type": "CAUTION", "camera_id": "CAM-01",
     "polygon": [[660, 150], [1100, 135], [1130, 570], [640, 575]], "required_ppe": ["helmet", "vest", "mask"],
     "priority": 2},
]


def event(**kw):
    e = {"event_id": "evt-0000-0001", "device_id": "EDGE-01", "camera_id": "CAM-01", "session_id": "s1",
         "worker_display_id": "Worker-017", "zone_id": "welding-a", "status": "POTENTIAL_VIOLATION",
         "reasons": ["MISSING_MASK", "ZONE_REQUIREMENT_NOT_MET"], "confidence": 0.87,
         "frames_in_window": 15, "frames_missing": 12, "model_version": "test", "rules_version": "1.0",
         "occurred_at": "2026-09-22T10:00:00Z",
         "detections": [{"object_class": "person", "confidence": 0.9, "bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 90}}]}
    e.update(kw)
    return e


@pytest.fixture()
def client(tmp_path):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}", evidence_dir=tmp_path / "ev",
                 edge_api_key=KEY)
    with TestClient(create_app(s)) as c:
        assert c.post("/api/v1/zones/sync", json=ZONES, headers=HDR).status_code == 200
        yield c


def test_requires_api_key(client):
    assert client.post("/api/v1/events", json=event()).status_code == 401
    assert client.post("/api/v1/events", json=event(), headers={"X-API-Key": "wrong"}).status_code == 401


def test_create_then_duplicate_is_idempotent(client):
    r1 = client.post("/api/v1/events", json=event(), headers=HDR)
    assert r1.status_code == 201 and r1.json()["duplicate"] is False
    assert r1.json()["review_status"] == "PENDING"
    r2 = client.post("/api/v1/events", json=event(), headers=HDR)      # network retry
    assert r2.status_code == 200 and r2.json()["duplicate"] is True
    assert client.get("/api/v1/events").json()["total"] == 1           # stored once


def test_same_id_different_content_conflicts(client):
    client.post("/api/v1/events", json=event(), headers=HDR)
    r = client.post("/api/v1/events", json=event(confidence=0.10), headers=HDR)
    assert r.status_code == 409


def test_unknown_zone_rejected(client):
    r = client.post("/api/v1/events", json=event(event_id="evt-x-000001", zone_id="nowhere"), headers=HDR)
    assert r.status_code == 422


def test_malformed_event_rejected(client):
    bad = event(event_id="evt-x-000002", status="VIOLATON")
    assert client.post("/api/v1/events", json=bad, headers=HDR).status_code == 422
    extra = event(event_id="evt-x-000003", face_embedding=[0.1])
    assert client.post("/api/v1/events", json=extra, headers=HDR).status_code == 422


def test_uncertain_needs_no_review(client):
    r = client.post("/api/v1/events", json=event(event_id="evt-u-000001", status="UNCERTAIN",
                                                 reasons=["LOW_CONFIDENCE"], frames_missing=7), headers=HDR)
    assert r.json()["review_status"] == "NOT_REQUIRED"


def test_list_filters_and_get(client):
    client.post("/api/v1/events", json=event(), headers=HDR)
    client.post("/api/v1/events", json=event(event_id="evt-0000-0002", zone_id="general-a", status="UNCERTAIN",
                                             reasons=["LOW_CONFIDENCE"], frames_missing=6,
                                             occurred_at="2026-09-22T11:00:00Z"), headers=HDR)
    assert client.get("/api/v1/events").json()["total"] == 2
    v = client.get("/api/v1/events", params={"status": "POTENTIAL_VIOLATION"}).json()
    assert v["total"] == 1 and v["items"][0]["zone_id"] == "welding-a"
    assert client.get("/api/v1/events", params={"zone_id": "general-a"}).json()["total"] == 1
    one = client.get("/api/v1/events/evt-0000-0001").json()
    assert one["worker_display_id"] == "Worker-017" and one["detections"][0]["object_class"] == "person"
    assert client.get("/api/v1/events/nope").status_code == 404


def test_zone_resync_bumps_version(client):
    changed = [dict(ZONES[1], required_ppe=["helmet", "vest"])]
    r = client.post("/api/v1/zones/sync", json=changed, headers=HDR).json()
    welding = next(z for z in r if z["zone_id"] == "welding-a")
    assert welding["config_version"] == 2


def test_audit_chain_detects_tampering(client):
    client.post("/api/v1/events", json=event(), headers=HDR)
    client.post("/api/v1/events", json=event(event_id="evt-0000-0003"), headers=HDR)
    ok = client.get("/api/v1/audit/verify").json()
    assert ok["ok"] is True and ok["rows_checked"] >= 3                # zones sync + 2 events
    with client.app.state.engine.begin() as conn:                       # attacker edits the DB directly
        conn.execute(text("UPDATE audit_logs SET actor='someone-else' WHERE audit_id = 2"))
    bad = client.get("/api/v1/audit/verify").json()
    assert bad["ok"] is False and bad["first_bad_audit_id"] == 2
