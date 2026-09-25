"""
Tests for POST /detect/video.

A tiny video is generated with OpenCV, and the detector is replaced by a fake that
returns fixed boxes - so these tests need no model, no GPU and no real footage.
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app
from server.predict_service import ModelNotAvailable

PERSON = Detection(cls="person", conf=0.90, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.81, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.77, box=(110, 200, 190, 300))


class FakeService:
    def __init__(self, detections=(), error=None):
        self.detections = list(detections)
        self.error = error
        self.calls = 0

    def detect(self, frame, conf=None):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.detections), 5.0


@pytest.fixture()
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        evidence_dir=tmp_path / "evidence",
                        violation_log=tmp_path / "violations.jsonl")
    with TestClient(create_app(settings)) as c:
        yield c


def make_video(tmp_path: Path, frames: int = 10, size=(320, 240), fps: float = 10.0) -> bytes:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 files in this environment")
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()
    data = path.read_bytes()
    assert data, "the test video is empty"
    return data


def post_video(client, data, name="clip.mp4", **form):
    return client.post("/detect/video", files={"video": (name, data, "video/mp4")}, data=form)


# ------------------------------------------------------------------ happy paths
def test_video_summary_counts_violations(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON, HELMET])     # no vest
    body = post_video(client, make_video(tmp_path), every_nth=2).json()

    assert body["frames_analysed"] >= 1
    assert body["people_max"] == 1
    assert body["violation_rate"] == 1.0                      # every frame non-compliant
    assert body["status_counts"] == {"MISSING_VEST": body["frames_analysed"]}
    assert body["frames_with_violation"] == body["frames_analysed"]
    assert body["video_stored"] is False


def test_compliant_video_has_no_violations(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    body = post_video(client, make_video(tmp_path), every_nth=5).json()
    assert body["violation_rate"] == 0.0
    assert body["frames_with_violation"] == 0
    assert set(body["status_counts"]) == {"COMPLIANT"}


def test_every_nth_controls_how_many_frames_are_analysed(client, tmp_path):
    data = make_video(tmp_path, frames=10)

    fake = FakeService([PERSON])
    client.app.state.predict_service = fake
    dense = post_video(client, data, every_nth=1).json()

    fake2 = FakeService([PERSON])
    client.app.state.predict_service = fake2
    sparse = post_video(client, data, every_nth=5).json()

    assert dense["frames_analysed"] > sparse["frames_analysed"]
    assert dense["every_nth"] == 1 and sparse["every_nth"] == 5
    assert fake.calls == dense["frames_analysed"]             # one model call per analysed frame


def test_max_frames_caps_the_work(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON])
    body = post_video(client, make_video(tmp_path, frames=20), every_nth=1, max_frames=3).json()
    assert body["frames_analysed"] == 3


def test_worst_frame_image_is_returned_only_when_asked(client, tmp_path):
    data = make_video(tmp_path)
    client.app.state.predict_service = FakeService([PERSON])
    assert post_video(client, data).json()["worst_frame_image"].startswith("data:image/jpeg;base64,")

    client.app.state.predict_service = FakeService([PERSON])
    assert post_video(client, data, annotate="false").json()["worst_frame_image"] is None


def test_timeline_entries_have_frame_numbers_and_times(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    body = post_video(client, make_video(tmp_path), every_nth=2).json()
    frames = [entry["frame"] for entry in body["timeline"]]
    assert frames == sorted(frames)
    assert all(entry["time_s"] >= 0 for entry in body["timeline"])
    assert body["timeline"][0]["people"] == 1


# ---------------------------------------------------------------- privacy check
def test_the_uploaded_clip_is_deleted_from_the_temp_folder(client, tmp_path):
    temp_dir = Path(tempfile.gettempdir())
    before = {p.name for p in temp_dir.glob("*.mp4")}
    client.app.state.predict_service = FakeService([PERSON])
    post_video(client, make_video(tmp_path))
    after = {p.name for p in temp_dir.glob("*.mp4")}
    assert after - before == set(), "the uploaded video was left in the temp folder"


# ---------------------------------------------------------------- error handling
def test_a_file_that_is_not_a_video(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON])
    response = post_video(client, b"this is not a video")
    assert response.status_code == 415


def test_unsupported_extension(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON])
    response = post_video(client, make_video(tmp_path), name="clip.gif")
    assert response.status_code == 415
    assert "Unsupported video type" in response.json()["detail"]


def test_empty_upload(client):
    client.app.state.predict_service = FakeService([PERSON])
    assert post_video(client, b"").status_code == 400


def test_every_nth_must_be_positive(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON])
    assert post_video(client, make_video(tmp_path), every_nth=0).status_code == 400


def test_missing_model_gives_a_clear_503(client, tmp_path):
    client.app.state.predict_service = FakeService(error=ModelNotAvailable("Model not found: models/x.pt"))
    response = post_video(client, make_video(tmp_path))
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]


def test_root_lists_the_video_endpoint(client):
    assert client.get("/").json()["detect_video"] == "/detect/video"


def test_a_video_with_violations_writes_one_summary_line(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON])              # no PPE at all
    post_video(client, make_video(tmp_path), every_nth=5)
    body = client.get("/api/v1/violations").json()
    assert body["count"] == 1
    entry = body["violations"][0]
    assert entry["kind"] == "video" and entry["source"].startswith("video:")
    assert entry["frames_with_violation"] == entry["frames_analysed"]


def test_a_compliant_video_logs_nothing(client, tmp_path):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    post_video(client, make_video(tmp_path), every_nth=5)
    assert client.get("/api/v1/violations").json()["count"] == 0
