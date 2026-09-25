"""
End-to-end integration: one image goes in, and everything the brief asks for comes out.

    IMAGE -> detection -> confidence filter -> rule engine -> GO/REVIEW/STOP -> event
          -> masked snapshot -> violation log -> alert -> acknowledgement
          -> analytics -> audit trail -> dashboard data

The detector is faked (a fixed set of boxes) so the test needs no GPU, no model file and no
camera - but every other component is the real one, wired the way the application wires it.
"""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app

PASSWORD = "integration-2026"

# One worker with a helmet and no vest: the case the whole pipeline exists for.
PERSON = Detection(cls="person", conf=0.93, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.88, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.81, box=(110, 200, 190, 300))


class FakeService:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, frame, conf=None):
        return list(self.detections), 9.9


class BrokenService:
    def detect(self, frame, conf=None):
        from server.predict_service import ModelNotAvailable
        raise ModelNotAvailable("Model not found. Train it first.")


def jpeg():
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), 70, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def write_users(path):
    import yaml
    from server.auth import hash_password
    entries = []
    for username, role, display in (("ana", "supervisor", "Ana Silva"),
                                    ("root", "admin", "Site Admin")):
        entry = {"username": username, "role": role, "display_name": display}
        entry.update(hash_password(PASSWORD))
        entries.append(entry)
    path.write_text(yaml.safe_dump({"users": entries}), encoding="utf-8")


@pytest.fixture()
def client(tmp_path):
    write_users(tmp_path / "users.yaml")
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                        evidence_dir=tmp_path / "snapshots",
                        violation_log=tmp_path / "violations.jsonl",
                        storage_root=tmp_path, logs_dir=tmp_path / "logs",
                        recordings_dir=tmp_path / "recordings",
                        reports_dir=tmp_path / "reports",
                        users_file=tmp_path / "users.yaml", session_secret="t",
                        edge_api_key="k", store_snapshots=True)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.predict_service = FakeService([PERSON, HELMET])
        yield c


def sign_in(client, username="ana"):
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": PASSWORD}).status_code == 200


def analyse(client, name="site_east.jpg"):
    return client.post("/api/v1/predict",
                       files={"image": (name, jpeg(), "image/jpeg")},
                       data={"required": "helmet,vest", "annotate": "false"})


# ===================================================================== the flow
def test_the_whole_pipeline_from_one_image(client, tmp_path):
    response = analyse(client)
    assert response.status_code == 200
    body = response.json()

    # --- detection and the confidence filter --------------------------------
    person = body["people"][0]
    assert person["wearing"]["helmet"] == 0.88
    assert "vest" in person["missing"]

    # --- the rule engine and the decision ------------------------------------
    assert body["decision"] == "STOP"
    assert body["rule"] == "R-MISSING-PPE"
    assert "vest" in body["reason"] and str(body["threshold"]) in body["reason"]
    assert person["decision"] == "STOP"
    evidence = {item["item"]: item for item in person["evidence"]}
    assert evidence["helmet"]["state"] == "present"
    assert evidence["vest"]["state"] == "missing"

    # --- the masked snapshot -------------------------------------------------
    snapshots = list((tmp_path / "snapshots").glob("*.jpg"))
    assert len(snapshots) == 1, "an event with snapshots enabled must produce one"
    assert body["image_stored"] is False, "the UPLOAD is never written to disk"
    assert body["event_snapshot_stored"] is True, "the masked event snapshot was kept"
    assert body["privacy_status"] == "FACE_BLUR_OK"

    # --- the violation log ---------------------------------------------------
    violations = client.get("/api/v1/violations").json()
    assert violations["count"] == 1
    entry = violations["violations"][0]
    assert entry["person_id"] == "Person-1" and entry["privacy_status"] == "FACE_BLUR_OK"
    assert entry["snapshot"] and "base64" not in json.dumps(entry)

    # --- the alert and its acknowledgement -----------------------------------
    alerts = client.get("/api/v1/alerts").json()
    assert alerts["count"] == 1 and alerts["unacknowledged"] == 1
    alert_id = alerts["alerts"][0]["alert_id"]
    sign_in(client, "ana")
    ack = client.post(f"/api/v1/alerts/{alert_id}/ack", json={}).json()
    assert ack["state"] == "ACKNOWLEDGED" and ack["acknowledged_by"] == "Ana Silva"

    # --- the copilot proposal, still waiting for a person ---------------------
    queue = client.get("/api/v1/copilot/suggestions").json()
    assert queue["count"] == 1 and queue["suggestions"][0]["status"] == "PENDING"

    # --- compliance analytics, counted ---------------------------------------
    analytics = client.get("/api/v1/analytics/compliance").json()
    assert analytics["total_evaluations"] == 1
    assert analytics["violations"] == 1 and analytics["compliance_pct"] == 0.0
    assert analytics["by_item"]["vest"] == 1

    # --- the audit trail still verifies --------------------------------------
    assert client.get("/api/v1/audit/verify").json()["ok"] is True

    # --- system health sees the same world -----------------------------------
    health = client.get("/api/v1/health/system").json()
    assert health["evidence"]["snapshots_enabled"] is True
    assert health["queue"]["mode"] in ("ONLINE", "OFFLINE")


def test_a_compliant_image_produces_go_and_no_alert(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    body = analyse(client, "clean.jpg").json()
    assert body["decision"] == "GO" and body["people"][0]["decision"] == "GO"
    assert client.get("/api/v1/violations").json()["count"] == 0
    assert client.get("/api/v1/alerts").json()["count"] == 0
    assert list((tmp_path / "snapshots").glob("*.jpg")) == []

    analytics = client.get("/api/v1/analytics/compliance").json()
    assert analytics["compliant"] == 1 and analytics["compliance_pct"] == 100.0


def test_a_low_confidence_item_produces_review_not_a_violation(client):
    """The uncertainty band in action: 0.42 is neither compliance nor a violation."""
    unsure_vest = Detection(cls="vest", conf=0.42, box=(110, 200, 190, 300))
    client.app.state.predict_service = FakeService([PERSON, HELMET, unsure_vest])
    body = analyse(client, "unsure.jpg").json()
    assert body["decision"] == "REVIEW" and body["rule"] == "R-LOW-CONFIDENCE"
    assert body["people"][0]["uncertain"] == ["vest"]

    analytics = client.get("/api/v1/analytics/compliance").json()
    assert analytics["review"] == 1 and analytics["violations"] == 0


def test_the_threshold_decides_and_it_is_the_configured_one(client):
    """Raising the policy threshold turns the same picture from GO into STOP."""
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    assert analyse(client, "a.jpg").json()["decision"] == "GO"

    client.app.state.settings.ppe_confidence_threshold = 0.95
    client.app.state.settings.review_margin = 0.0
    body = analyse(client, "b.jpg").json()
    assert body["decision"] == "STOP" and "0.95" in body["reason"]


def test_an_empty_frame_is_go_with_nothing_logged(client):
    client.app.state.predict_service = FakeService([])
    body = analyse(client, "empty.jpg").json()
    assert body["decision"] == "GO" and body["people_count"] == 0
    assert client.get("/api/v1/violations").json()["count"] == 0


def test_a_missing_model_never_reports_compliance(client):
    """The failure the whole fail-safe design exists for."""
    client.app.state.predict_service = BrokenService()
    response = analyse(client, "x.jpg")
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]

    # nothing was recorded as compliant, and no evaluation was counted
    assert client.get("/api/v1/analytics/compliance").json()["total_evaluations"] == 0
    assert client.get("/api/v1/violations").json()["count"] == 0


def test_two_people_take_the_worse_decision(client):
    other = Detection(cls="person", conf=0.9, box=(300, 100, 400, 400))
    other_helmet = Detection(cls="helmet", conf=0.9, box=(320, 90, 380, 140))
    other_vest = Detection(cls="vest", conf=0.9, box=(310, 200, 390, 300))
    client.app.state.predict_service = FakeService([PERSON, HELMET, other, other_helmet, other_vest])
    body = analyse(client, "two.jpg").json()
    assert body["people_count"] == 2
    assert body["decision"] == "STOP"
    assert sorted(p["decision"] for p in body["people"]) == ["GO", "STOP"]


def test_the_dashboard_can_read_everything_it_displays(client):
    analyse(client)
    for path in ("/api/v1/violations", "/api/v1/alerts", "/api/v1/analytics/compliance",
                 "/api/v1/health/system", "/api/v1/copilot/suggestions",
                 "/api/v1/copilot/catalog", "/api/v1/tools", "/api/v1/auth/me"):
        assert client.get(path).status_code == 200, path
