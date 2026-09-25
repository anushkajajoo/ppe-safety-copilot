"""
The chain, link by link:

    camera/video -> detection -> confidence -> rules -> GO/REVIEW/STOP -> alert
                 -> masked snapshot -> recorded evidence -> audit log -> dashboard

`test_integration.py` proves the image path end to end. This file proves the CAMERA path,
which is the one that needs a webcam and therefore never gets exercised by an ordinary test
run: the camera is replaced by a fake that yields a fixed number of frames, and everything
downstream of it is the real thing, including the recorder and the audit chain.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app

PASSWORD = "chain-test-2026"
PERSON = Detection(cls="person", conf=0.93, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.88, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.85, box=(110, 200, 190, 300))


class FakeCamera:
    """Yields `frames` frames and then reports end-of-stream, like a finished video would."""

    def __init__(self, frames=24, size=(480, 640)):
        self.remaining = frames
        self.size = size
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.full((*self.size, 3), 80, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeService:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, frame, conf=None):
        return list(self.detections), 7.5


def write_users(path):
    import yaml
    from server.auth import hash_password
    entry = {"username": "ana", "role": "supervisor", "display_name": "Ana Silva"}
    entry.update(hash_password(PASSWORD))
    path.write_text(yaml.safe_dump({"users": [entry]}), encoding="utf-8")


def build(tmp_path, **overrides):
    write_users(tmp_path / "users.yaml")
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                        evidence_dir=tmp_path / "snapshots",
                        violation_log=tmp_path / "violations.jsonl",
                        storage_root=tmp_path, logs_dir=tmp_path / "logs",
                        recordings_dir=tmp_path / "recordings",
                        reports_dir=tmp_path / "reports",
                        users_file=tmp_path / "users.yaml", session_secret="t",
                        edge_api_key="k", **overrides)
    return create_app(settings)


def sign_in(client, username="ana"):
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": PASSWORD}).status_code == 200


def drain(client, frames=24, camera=0):
    """Open the stream and consume it, which runs the loop exactly `frames` times."""
    with client.stream("GET", f"/live/stream?camera={camera}&fps=30&required=helmet,vest") as r:
        assert r.status_code == 200
        for _ in r.iter_bytes():
            pass


@pytest.fixture()
def client(tmp_path):
    app = build(tmp_path, store_snapshots=True, record_events=True,
                clip_seconds_before=0.3, clip_seconds_after=0.2, clip_fps=10)
    with TestClient(app) as c:
        c.app.state.camera_factory = lambda index: FakeCamera(frames=24)
        c.app.state.predict_service = FakeService([PERSON, HELMET])     # no vest -> STOP
        yield c


# ============================================================== the whole chain
def test_one_camera_event_travels_the_entire_chain(client, tmp_path):
    drain(client)

    # --- decision --------------------------------------------------------
    status = client.get("/live/status").json()
    assert status["decision"] == "STOP"
    assert status["rule"] == "R-MISSING-PPE" and "vest" in status["reason"]
    assert status["frames"] == 24 and status["dropped"] == 0

    # --- alert (from the violation log, not invented) --------------------
    alerts = client.get("/api/v1/alerts").json()
    assert alerts["count"] >= 1
    alert = alerts["alerts"][0]
    assert alert["source"] == "camera:0" and alert["state"] == "NEW"

    # --- masked snapshot --------------------------------------------------
    snapshots = list((tmp_path / "snapshots").glob("*.jpg"))
    assert len(snapshots) == 1, "a STOP with snapshots on must write exactly one"
    assert snapshots[0].stat().st_size > 0

    # --- recorded evidence -------------------------------------------------
    clips = list((tmp_path / "recordings").rglob("*.mp4"))
    assert clips, "the rolling buffer must produce a clip for the event"

    # --- the structured event record -----------------------------------------
    history = client.get("/api/v1/history").json()
    assert history["count"] == 1, "a continuing violation is ONE event, not one per frame"
    event = history["events"][0]
    assert event["event_type"] == "PPE_VIOLATION"
    assert event["decision"] == "STOP"
    assert event["camera_id"] == "camera:0"
    assert event["person_id"].startswith("P-")
    assert event["occurrences"] > 1, "the repeat frames extended the same event"
    assert event["missing_ppe"] == ["vest"]
    assert event["rule_id"] == "R-MISSING-PPE"
    assert event["snapshot"] is not None
    assert event["snapshot"]["available"] is True

    # --- audit log ----------------------------------------------------------
    rows = client.get("/api/v1/audit").json()
    events = [r for r in rows if r["action"] == "SAFETY_EVENT"]
    assert events, "a camera event must reach the tamper-evident chain"
    assert events[0]["details"]["decision"] == "STOP"
    assert events[0]["details"]["source"] == "camera:0"
    assert client.get("/api/v1/audit/verify").json()["ok"] is True

    # --- the event detail panel reads back -----------------------------------
    detail = client.get(f"/api/v1/history/{event['event_id']}").json()
    assert detail["reason"] == event["reason"] and detail["threshold"] == 0.5

    # --- acknowledging moves it out of alerts, not out of history -------------
    sign_in(client)
    acknowledged = client.post(
        f"/api/v1/history/{event['event_id']}/acknowledge", json={}).json()
    assert acknowledged["status"] == "ACKNOWLEDGED"
    assert client.get("/api/v1/alerts").json()["unacknowledged"] == 0
    assert client.get("/api/v1/history").json()["count"] == 1

    # --- dashboard data ------------------------------------------------------
    analytics = client.get("/api/v1/analytics/compliance").json()
    assert analytics["total_evaluations"] >= 1 and analytics["violations"] >= 1
    health = client.get("/api/v1/health/system").json()
    assert health["camera"]["frames"] == 24


def test_a_compliant_camera_produces_no_evidence_at_all(tmp_path):
    """The quiet shift: nothing is written, which is the whole point of event-based storage."""
    app = build(tmp_path, store_snapshots=True, record_events=True)
    with TestClient(app) as client:
        client.app.state.camera_factory = lambda index: FakeCamera(frames=12)
        client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
        drain(client, frames=12)

        assert client.get("/live/status").json()["decision"] == "GO"
        assert list((tmp_path / "snapshots").glob("*.jpg")) == []
        assert list((tmp_path / "recordings").rglob("*.mp4")) == []
        assert client.get("/api/v1/alerts").json()["count"] == 0
        assert [r for r in client.get("/api/v1/audit").json()
                if r["action"] == "SAFETY_EVENT"] == []
        assert client.get("/api/v1/history").json()["count"] == 0


def test_evidence_stays_off_unless_it_is_switched_on(tmp_path):
    """Default settings: a violation still decides and logs, but writes no image or clip."""
    app = build(tmp_path)                       # store_snapshots and record_events default False
    with TestClient(app) as client:
        client.app.state.camera_factory = lambda index: FakeCamera(frames=12)
        client.app.state.predict_service = FakeService([PERSON, HELMET])
        drain(client, frames=12)

        assert client.get("/live/status").json()["decision"] == "STOP"
        assert client.get("/api/v1/alerts").json()["count"] >= 1      # the event is recorded
        assert not (tmp_path / "snapshots").exists() or \
               list((tmp_path / "snapshots").glob("*.jpg")) == []
        assert list((tmp_path / "recordings").rglob("*.mp4")) == []


def test_the_camera_is_released_when_the_stream_ends(client):
    cameras = []
    client.app.state.camera_factory = lambda index: cameras.append(FakeCamera(8)) or cameras[-1]
    drain(client, frames=8)
    assert cameras[0].released is True


def test_a_camera_that_will_not_open_is_a_503_not_a_crash(client):
    class DeadCamera(FakeCamera):
        def isOpened(self):
            return False

    client.app.state.camera_factory = lambda index: DeadCamera()
    response = client.get("/live/stream?camera=3")
    assert response.status_code == 503
    assert "Could not open camera" in response.json()["detail"]


def test_a_missing_model_makes_the_live_view_say_review_not_compliant(client):
    """The fail-safe, on the camera path: it must never look like everything is fine."""
    from server.predict_service import ModelNotAvailable

    class BrokenService:
        def detect(self, frame, conf=None):
            raise ModelNotAvailable("Model not found. Train it first.")

    client.app.state.predict_service = BrokenService()
    drain(client, frames=5)

    status = client.get("/live/status").json()
    assert status["decision"] in ("REVIEW", "STOP")
    assert status["rule"] == "R-SYSTEM-FAULT"
    assert "cannot confirm compliance" in status["reason"]
