"""
Tests for the live webcam endpoints.

No real camera is touched: app.state.camera_factory is replaced by a fake capture
that returns a fixed number of blank frames. That is why the factory exists.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app

PERSON = Detection(cls="person", conf=0.90, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.81, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.77, box=(110, 200, 190, 300))


class FakeService:
    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, frame, conf=None):
        return list(self.detections), 5.0


class FakeCapture:
    """Stands in for cv2.VideoCapture: hands out `frames` blank images, then stops."""

    def __init__(self, frames=3, opened=True, size=(240, 320)):
        self.remaining = frames
        self._opened = opened
        self.size = size
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.full((self.size[0], self.size[1], 3), 90, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture()
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        evidence_dir=tmp_path / "evidence",
                        violation_log=tmp_path / "violations.jsonl")
    with TestClient(create_app(settings)) as c:
        c.app.state.predict_service = FakeService([PERSON, HELMET])
        yield c


def test_status_starts_idle(client):
    body = client.get("/live/status").json()
    assert body["running"] is False and body["people"] == 0


def test_stream_returns_mjpeg_frames(client):
    captures = []

    def factory(index):
        cap = FakeCapture(frames=3)
        captures.append(cap)
        return cap

    client.app.state.camera_factory = factory
    response = client.get("/live/stream?fps=30")

    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
    body = response.content
    assert body.count(b"--frame") == 3                 # one part per frame
    assert b"Content-Type: image/jpeg" in body
    assert b"\xff\xd8" in body                          # a real JPEG header


def test_status_reflects_the_last_frame_and_camera_is_released(client):
    captures = []

    def factory(index):
        cap = FakeCapture(frames=2)
        captures.append(cap)
        return cap

    client.app.state.camera_factory = factory
    client.get("/live/stream?fps=30").content                 # consume the whole stream

    assert captures[0].released is True                        # camera freed at the end
    after = client.get("/live/status").json()
    assert after["running"] is False                           # and the state says so


def test_a_camera_that_will_not_open_gives_503(client):
    client.app.state.camera_factory = lambda index: FakeCapture(frames=0, opened=False)
    response = client.get("/live/stream")
    assert response.status_code == 503
    assert "Could not open camera" in response.json()["detail"]


def test_required_ppe_is_validated(client):
    client.app.state.camera_factory = lambda index: FakeCapture(frames=1)
    assert client.get("/live/stream?required=helmet,boots").status_code == 400


def test_conf_out_of_range_is_rejected(client):
    client.app.state.camera_factory = lambda index: FakeCapture(frames=1)
    assert client.get("/live/stream?conf=1.4").status_code == 400


def test_live_page_is_served_and_calls_the_stream(client):
    page = client.get("/live").text
    assert "/live/stream" in page and "/live/status" in page


def test_root_lists_the_live_page(client):
    assert client.get("/").json()["live"] == "/live"


def test_dashboard_links_to_the_live_page(client):
    assert 'href="/live"' in client.get("/dashboard").text


def test_live_violations_are_logged_once_per_cooldown(client):
    """10 frames of the same violation must not write 10 lines."""
    client.app.state.camera_factory = lambda index: FakeCapture(frames=10)
    client.get("/live/stream?fps=30").content
    body = client.get("/api/v1/violations").json()
    assert body["count"] == 1                       # throttled, not one per frame
    assert body["violations"][0]["source"] == "camera:0"
    assert body["violations"][0]["status"] == "MISSING_VEST"
