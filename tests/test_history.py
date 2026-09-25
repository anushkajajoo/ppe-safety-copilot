"""
Event history over HTTP: create, read, filter, acknowledge, close, delete, purge, clear,
retention, statistics and export.

The store is exercised through the real app with a temporary database, because the parts
worth testing are exactly the ones involving rows and files: deduplication, soft versus
permanent deletion, which files a purge may touch, and which events retention must refuse
to remove.
"""
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server import event_store
from server.config import Settings
from server.main import create_app
from shared.events import ACKNOWLEDGED, CLOSED, NEW, PPE_VIOLATION

PASSWORD = "history-test-2026"
PERSON = Detection(cls="person", conf=0.93, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.88, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.85, box=(110, 200, 190, 300))


class FakeService:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, frame, conf=None):
        return list(self.detections), 9.0


def jpeg():
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), 70, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def write_users(path):
    import yaml
    from server.auth import hash_password
    entries = []
    for username, role, display in (("ana", "supervisor", "Ana Silva"),
                                    ("root", "admin", "Site Admin"),
                                    ("vic", "viewer", "Vic Watcher")):
        entry = {"username": username, "role": role, "display_name": display}
        entry.update(hash_password(PASSWORD))
        entries.append(entry)
    path.write_text(yaml.safe_dump({"users": entries}), encoding="utf-8")


@pytest.fixture()
def client(tmp_path):
    write_users(tmp_path / "users.yaml")
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                        evidence_dir=tmp_path / "snapshots",
                        snapshots_masked_dir=tmp_path / "snapshots",
                        recordings_dir=tmp_path / "recordings",
                        violation_log=tmp_path / "violations.jsonl",
                        storage_root=tmp_path, logs_dir=tmp_path / "logs",
                        reports_dir=tmp_path / "reports", events_dir=tmp_path / "events",
                        users_file=tmp_path / "users.yaml", session_secret="t",
                        edge_api_key="k", store_snapshots=True)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.predict_service = FakeService([PERSON, HELMET])     # no vest -> STOP
        yield c


def sign_in(client, username="ana"):
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": PASSWORD}).status_code == 200


def analyse(client, name="site.jpg"):
    return client.post("/api/v1/predict", files={"image": (name, jpeg(), "image/jpeg")},
                       data={"required": "helmet,vest", "annotate": "false"})


def new_session(client):
    return client.app.state.session_factory()


# ============================================================ create and read
def test_an_analysed_violation_becomes_one_structured_event(client):
    analyse(client)
    body = client.get("/api/v1/history").json()
    assert body["count"] == 1
    event = body["events"][0]
    assert event["event_id"] == "EVT-000001"
    assert event["event_type"] == PPE_VIOLATION and event["decision"] == "STOP"
    assert event["person_id"] == "P-01" and event["camera_id"] == "image:site.jpg"
    assert event["missing_ppe"] == ["vest"]
    assert event["detected_ppe"] == [{"type": "helmet", "confidence": 0.88}]
    assert event["severity"] == "HIGH" and event["status"] == NEW
    assert event["rule_id"] == "R-MISSING-PPE"
    assert "0.50" in event["reason"] and "vest" in event["reason"]
    assert event["snapshot"] is not None and event["snapshot"]["available"] is True


def test_the_detail_view_carries_everything_needed_to_explain_the_decision(client):
    analyse(client)
    event = client.get("/api/v1/history/EVT-000001").json()
    for field in ("event_id", "occurred_at", "camera_id", "person_id", "event_type",
                  "required_ppe", "detected_ppe", "missing_ppe", "uncertain_ppe",
                  "decision", "severity", "rule_id", "reason", "confidence", "threshold",
                  "snapshot", "recording", "status", "acknowledged", "sync_status",
                  "occurrences", "duration_s"):
        assert field in event, field
    assert event["sync_status"] == "LOCAL"


def test_a_compliant_analysis_is_history_but_not_an_alert(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    analyse(client, "clean.jpg")
    body = client.get("/api/v1/history").json()
    assert body["count"] == 1 and body["events"][0]["decision"] == "GO"
    assert client.get("/api/v1/alerts").json()["count"] == 0


def test_an_unknown_event_is_a_404(client):
    assert client.get("/api/v1/history/EVT-999999").status_code == 404


# =============================================================== deduplication
def test_the_same_continuing_violation_is_one_event_not_many(client):
    """Four observations of the same worker without a vest must not be four rows."""
    for _ in range(4):
        analyse(client)
    body = client.get("/api/v1/history").json()
    assert body["count"] == 1
    assert body["events"][0]["occurrences"] == 4


def test_a_different_missing_item_is_a_different_event(client):
    analyse(client)                                             # missing vest
    client.app.state.predict_service = FakeService([PERSON, VEST])   # now missing helmet
    analyse(client)
    body = client.get("/api/v1/history").json()
    assert body["count"] == 2
    assert {tuple(e["missing_ppe"]) for e in body["events"]} == {("vest",), ("helmet",)}


def test_an_old_event_does_not_absorb_a_new_one(client):
    """Outside the merge window it is a new occurrence, not the same one continuing."""
    session = new_session(client)
    try:
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        for when in (old, None):
            event_store.record_event(
                session, camera_id="CAM-01", person_id="P-01", decision="STOP",
                required=["helmet"], detected={}, missing=["helmet"], uncertain=[],
                threshold=0.5, rule_id="R-MISSING-PPE", now=when, merge_window_s=30)
        session.commit()
    finally:
        session.close()
    assert client.get("/api/v1/history?camera_id=CAM-01").json()["count"] == 2


# ==================================================================== filters
def seed(client):
    """A small, deliberately varied history."""
    session = new_session(client)
    try:
        now = datetime.now(timezone.utc)
        rows = [
            ("CAM-01", "P-01", "STOP", ["helmet"], [], now - timedelta(minutes=5)),
            ("CAM-01", "P-02", "REVIEW", [], ["vest"], now - timedelta(hours=2)),
            ("CAM-02", "P-03", "STOP", ["vest"], [], now - timedelta(days=3)),
            ("CAM-02", "P-04", "GO", [], [], now - timedelta(days=10)),
        ]
        for camera, person, decision, missing, uncertain, when in rows:
            event_store.record_event(
                session, camera_id=camera, person_id=person, decision=decision,
                required=["helmet", "vest"], detected={"helmet": 0.9}, missing=missing,
                uncertain=uncertain, threshold=0.5, rule_id="R-TEST", now=when)
        session.commit()
    finally:
        session.close()


def test_history_filters_by_camera_decision_severity_and_type(client):
    seed(client)
    assert client.get("/api/v1/history?camera_id=CAM-01").json()["count"] == 2
    assert client.get("/api/v1/history?decision=STOP").json()["count"] == 2
    assert client.get("/api/v1/history?event_type=PPE_REVIEW").json()["count"] == 1
    assert client.get("/api/v1/history?severity=LOW").json()["count"] == 1
    assert client.get("/api/v1/history?person_id=P-03").json()["count"] == 1


def test_history_filters_by_time(client):
    seed(client)
    assert client.get("/api/v1/history?hours=1").json()["count"] == 1
    assert client.get("/api/v1/history?hours=24").json()["count"] == 2
    assert client.get("/api/v1/history?hours=168").json()["count"] == 3


def test_history_filters_by_acknowledgement(client):
    seed(client)
    sign_in(client, "ana")
    first = client.get("/api/v1/history").json()["events"][0]["event_id"]
    client.post(f"/api/v1/history/{first}/acknowledge", json={})
    assert client.get("/api/v1/history?unacknowledged_only=true").json()["count"] == 3


def test_a_nonsense_filter_is_refused_rather_than_ignored(client):
    assert client.get("/api/v1/history?decision=PROBABLY").status_code == 422
    assert client.get("/api/v1/history?severity=EXTREME").status_code == 422


# =========================================================== alerts vs history
def test_acknowledging_moves_an_event_out_of_alerts_and_keeps_it_in_history(client):
    analyse(client)
    assert client.get("/api/v1/alerts").json()["unacknowledged"] == 1

    sign_in(client, "ana")
    body = client.post("/api/v1/history/EVT-000001/acknowledge",
                       json={"note": "spoke to the worker"}).json()
    assert body["status"] == ACKNOWLEDGED and body["acknowledged_by"] == "Ana Silva"

    assert client.get("/api/v1/alerts").json()["unacknowledged"] == 0
    assert client.get("/api/v1/history").json()["count"] == 1        # still in history
    assert client.get("/api/v1/history/EVT-000001").json()["note"] == "spoke to the worker"


def test_only_a_supervisor_can_acknowledge(client):
    analyse(client)
    assert client.post("/api/v1/history/EVT-000001/acknowledge", json={}).status_code == 401
    sign_in(client, "vic")
    assert client.post("/api/v1/history/EVT-000001/acknowledge", json={}).status_code == 403


def test_closing_an_event_is_what_makes_it_retention_eligible(client):
    analyse(client)
    sign_in(client, "ana")
    body = client.post("/api/v1/history/EVT-000001/close", json={}).json()
    assert body["status"] == CLOSED and body["acknowledged"] is True


# ================================================================== deleting
def test_a_normal_delete_is_soft_and_keeps_the_evidence(client, tmp_path):
    analyse(client)
    snapshots = list((tmp_path / "snapshots").glob("*.jpg"))
    assert len(snapshots) == 1

    sign_in(client, "ana")
    body = client.delete("/api/v1/history/EVT-000001").json()
    assert body["deleted"] is True and body["permanent"] is False
    assert client.get("/api/v1/history").json()["count"] == 0
    assert client.get("/api/v1/history/EVT-000001").status_code == 404
    assert snapshots[0].exists(), "a soft delete must not touch evidence"


def test_a_purge_is_permanent_and_takes_the_evidence_with_it(client, tmp_path):
    analyse(client)
    snapshot = list((tmp_path / "snapshots").glob("*.jpg"))[0]

    sign_in(client, "ana")
    assert client.post("/api/v1/history/EVT-000001/purge", json={}).status_code == 403

    sign_in(client, "root")
    body = client.post("/api/v1/history/EVT-000001/purge", json={}).json()
    assert body["removed"] is True and body["permanent"] is True
    assert snapshot.name in body["files_deleted"] and not snapshot.exists()
    assert client.get("/api/v1/history").json()["count"] == 0


def test_the_audit_row_outlives_the_event_it_describes(client):
    analyse(client)
    sign_in(client, "root")
    client.post("/api/v1/history/EVT-000001/purge", json={})
    actions = [row["action"] for row in client.get("/api/v1/audit").json()]
    assert "EVENT_PURGED" in actions
    assert client.get("/api/v1/audit/verify").json()["ok"] is True


def test_a_purge_will_not_delete_a_file_outside_the_evidence_folders(client, tmp_path):
    """The important one: a path in the database must not become arbitrary file deletion."""
    outsider = tmp_path.parent / "important_file.jpg"
    outsider.write_bytes(b"do not delete me")
    session = new_session(client)
    try:
        event, _ = event_store.record_event(
            session, camera_id="CAM-01", person_id="P-09", decision="STOP",
            required=["helmet"], detected={}, missing=["helmet"], uncertain=[],
            threshold=0.5, rule_id="R-TEST",
            snapshot_path=str(tmp_path / "snapshots" / ".." / ".." / "important_file.jpg"))
        session.commit()
        event_id = event.event_id
    finally:
        session.close()

    sign_in(client, "root")
    body = client.post(f"/api/v1/history/{event_id}/purge", json={}).json()
    assert body["files_deleted"] == []
    assert outsider.exists(), "path traversal must never reach outside the evidence folders"
    assert body["files_skipped"], "and the refusal is reported, not silent"


def test_clearing_history_insists_on_confirmation(client):
    analyse(client)
    sign_in(client, "root")
    assert client.post("/api/v1/history/clear", json={"confirm": "yes"}).status_code == 422
    assert client.get("/api/v1/history").json()["count"] == 1


def test_clearing_history_removes_only_closed_events_by_default(client):
    seed(client)
    sign_in(client, "ana")
    first = client.get("/api/v1/history").json()["events"][0]["event_id"]
    client.post(f"/api/v1/history/{first}/close", json={})

    sign_in(client, "root")
    body = client.post("/api/v1/history/clear", json={"confirm": "CLEAR"}).json()
    assert body["removed_count"] == 1
    assert client.get("/api/v1/history").json()["count"] == 3


def test_only_an_admin_can_clear_history(client):
    sign_in(client, "ana")
    assert client.post("/api/v1/history/clear", json={"confirm": "CLEAR"}).status_code == 403


# ================================================================= retention
def test_retention_removes_closed_events_and_protects_the_rest(client, tmp_path):
    session = new_session(client)
    try:
        old = datetime.now(timezone.utc) - timedelta(days=120)
        for person, status in (("P-01", NEW), ("P-02", ACKNOWLEDGED), ("P-03", CLOSED)):
            event, _ = event_store.record_event(
                session, camera_id="CAM-01", person_id=person, decision="STOP",
                required=["helmet"], detected={}, missing=["helmet"], uncertain=[],
                threshold=0.5, rule_id="R-TEST", now=old)
            event.status = status
            event.acknowledged = status != NEW
        session.commit()

        result = event_store.cleanup(session, [tmp_path / "snapshots"], retention_days=30)
        session.commit()
    finally:
        session.close()

    assert result["removed_count"] == 1, "only the CLOSED one may go"
    assert result["protected"] == 2
    remaining = {e["person_id"] for e in client.get("/api/v1/history").json()["events"]}
    assert remaining == {"P-01", "P-02"}


def test_retention_keeps_a_recent_closed_event(client, tmp_path):
    session = new_session(client)
    try:
        event, _ = event_store.record_event(
            session, camera_id="CAM-01", person_id="P-01", decision="STOP",
            required=["helmet"], detected={}, missing=["helmet"], uncertain=[],
            threshold=0.5, rule_id="R-TEST")
        event.status = CLOSED
        session.commit()
        result = event_store.cleanup(session, [tmp_path], retention_days=30)
    finally:
        session.close()
    assert result["removed_count"] == 0


def test_a_dry_run_reports_what_it_would_remove_and_removes_nothing(client, tmp_path):
    seed(client)
    sign_in(client, "ana")
    first = client.get("/api/v1/history").json()["events"][0]["event_id"]
    client.post(f"/api/v1/history/{first}/close", json={})     # only CLOSED is eligible

    session = new_session(client)
    try:
        result = event_store.cleanup(session, [tmp_path], retention_days=0, dry_run=True)
    finally:
        session.close()

    assert result["dry_run"] is True
    assert result["removed_count"] == 1, "only the closed event is eligible"
    assert result["protected"] == 3, "the three unreviewed ones are protected"
    assert client.get("/api/v1/history").json()["count"] == 4, "a dry run deletes nothing"


# ================================================================ statistics
def test_statistics_are_counted_from_the_rows(client):
    seed(client)
    stats = client.get("/api/v1/history/stats").json()
    assert stats["total_events"] == 4
    assert stats["stop"] == 2 and stats["review"] == 1 and stats["go"] == 1
    assert stats["compliance_pct"] == 25.0
    assert stats["violations_by_item"] == {"helmet": 1, "vest": 1, "mask": 0}
    assert stats["by_camera"] == {"CAM-01": 2, "CAM-02": 2}
    assert stats["unacknowledged"] == 4 and stats["open_alerts"] == 4


def test_statistics_of_an_empty_history_do_not_invent_a_rate(client):
    stats = client.get("/api/v1/history/stats").json()
    assert stats["total_events"] == 0 and stats["compliance_pct"] is None


# ==================================================================== export
def test_history_exports_as_csv(client):
    seed(client)
    response = client.get("/api/v1/history/export?fmt=csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("event_id,occurred_at")
    assert len(lines) == 5                              # header + 4 events


def test_history_exports_as_json_and_respects_filters(client):
    seed(client)
    body = client.get("/api/v1/history/export?fmt=json&decision=STOP").json()
    assert body["count"] == 2
    assert all(row["decision"] == "STOP" for row in body["events"])


def test_an_export_never_contains_evidence_paths(client):
    """
    No snapshot or recording column, and no path to one.

    Checking for ".jpg" anywhere was too crude: `camera_id` is legitimately
    "image:site.jpg" for an uploaded photo, which is the name of the SOURCE, not a path to
    stored evidence. What must not appear is a column of evidence, or a filesystem path.
    """
    analyse(client)
    text = client.get("/api/v1/history/export?fmt=csv").text
    header = text.splitlines()[0]
    assert "snapshot" not in header and "recording" not in header
    for pointer in ("snapshots/", "snapshots\\", "recordings/", "recordings\\",
                    "data/", "C:\\", "/home/"):
        assert pointer not in text, f"the export leaks a path: {pointer}"
