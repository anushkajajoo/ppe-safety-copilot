"""
Tests for POST /api/v1/predict and the dashboard page.

No model and no GPU: app.state.predict_service is swapped for a fake that returns
fixed detections. That is the point of keeping the detector behind a small service
class - the API can be tested for free.
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app
from server.predict_service import ModelNotAvailable

# person box (100,100)-(200,400): head region ~y 70-205, torso ~y 160-325
PERSON = Detection(cls="person", conf=0.90, box=(100, 100, 200, 400))
HELMET = Detection(cls="helmet", conf=0.81, box=(120, 90, 180, 140))
VEST = Detection(cls="vest", conf=0.77, box=(110, 200, 190, 300))


class FakeService:
    """Stands in for PredictService: returns whatever detections the test wants."""
    def __init__(self, detections=(), error=None):
        self.detections = list(detections)
        self.error = error
        self.calls = []

    def detect(self, frame, conf=None):
        self.calls.append({"shape": frame.shape, "conf": conf})
        if self.error:
            raise self.error
        return list(self.detections), 12.3


def jpeg_bytes(width=640, height=480):
    image = np.full((height, width, 3), 60, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        evidence_dir=tmp_path / "evidence",
                        violation_log=tmp_path / "violations.jsonl")
    with TestClient(create_app(settings)) as c:
        yield c


def post_image(client, data=None, **form):
    files = {"image": ("frame.jpg", data if data is not None else jpeg_bytes(), "image/jpeg")}
    return client.post("/api/v1/predict", files=files, data=form)


# ------------------------------------------------------------------ happy paths
def test_compliant_person(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    body = post_image(client).json()

    assert body["people_count"] == 1
    assert body["people"][0]["status"] == "COMPLIANT"
    assert body["people"][0]["missing"] == []
    assert body["violations"] == 0 and body["compliant"] == 1
    assert body["counts_by_class"] == {"person": 1, "helmet": 1, "vest": 1}
    assert body["required"] == ["helmet", "vest"]
    assert body["width"] == 640 and body["height"] == 480


def test_missing_vest_is_reported(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET])
    body = post_image(client).json()
    assert body["people"][0]["status"] == "MISSING_VEST"
    assert body["people"][0]["missing"] == ["vest"]
    assert body["violations"] == 1


def test_person_with_no_ppe(client):
    client.app.state.predict_service = FakeService([PERSON])
    body = post_image(client).json()
    assert body["people"][0]["status"] == "MISSING_HELMET_AND_VEST"
    assert body["people"][0]["wearing"] == {}


def test_empty_frame_is_not_an_error(client):
    client.app.state.predict_service = FakeService([])
    body = post_image(client).json()
    assert body["people_count"] == 0 and body["people"] == []


def test_ppe_not_on_anybody_is_listed_separately(client):
    stray = Detection(cls="helmet", conf=0.6, box=(500, 400, 560, 440))
    client.app.state.predict_service = FakeService([PERSON, stray])
    body = post_image(client).json()
    assert [d["cls"] for d in body["unassigned_ppe"]] == ["helmet"]
    assert body["people"][0]["status"] == "MISSING_HELMET_AND_VEST"


def test_required_list_can_be_changed(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    body = post_image(client, required="helmet").json()
    assert body["required"] == ["helmet"] and body["people"][0]["status"] == "COMPLIANT"

    body = post_image(client, required="helmet,vest,mask").json()
    assert body["people"][0]["status"] == "MISSING_MASK"


def test_annotated_image_is_returned_only_when_asked(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    assert post_image(client).json()["annotated_image"].startswith("data:image/jpeg;base64,")
    assert post_image(client, annotate="false").json()["annotated_image"] is None


def test_conf_is_passed_to_the_detector(client):
    fake = FakeService([PERSON])
    client.app.state.predict_service = fake
    post_image(client, conf="0.6")
    assert fake.calls[-1]["conf"] == pytest.approx(0.6)


def test_the_image_is_never_stored(client):
    client.app.state.predict_service = FakeService([PERSON])
    assert post_image(client).json()["image_stored"] is False


# ---------------------------------------------------------------- error handling
def test_a_file_that_is_not_an_image(client):
    client.app.state.predict_service = FakeService([PERSON])
    response = post_image(client, data=b"this is not a picture")
    assert response.status_code == 415
    assert "not a readable image" in response.json()["detail"]


def test_an_empty_upload(client):
    client.app.state.predict_service = FakeService([PERSON])
    assert post_image(client, data=b"").status_code == 400


def test_missing_file_is_a_validation_error(client):
    assert client.post("/api/v1/predict").status_code == 422


def test_unknown_required_item(client):
    client.app.state.predict_service = FakeService([PERSON])
    response = post_image(client, required="helmet,boots")
    assert response.status_code == 400 and "boots" in response.json()["detail"]


def test_conf_out_of_range(client):
    client.app.state.predict_service = FakeService([PERSON])
    assert post_image(client, conf="1.5").status_code == 400


def test_missing_model_gives_a_clear_503(client):
    client.app.state.predict_service = FakeService(error=ModelNotAvailable("Model not found: models/x.pt"))
    response = post_image(client)
    assert response.status_code == 503
    assert "Model not found" in response.json()["detail"]


# --------------------------------------------------------------------- the page
def test_dashboard_page_is_served(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Edge Vision Safety Copilot" in response.text
    assert "/api/v1/predict" in response.text          # the page really calls our API


def test_root_lists_the_dashboard(client):
    body = client.get("/").json()
    assert body["dashboard"] == "/dashboard"
    assert body["predict"] == "/api/v1/predict"


# ------------------------------------------------------------------- the alias
def test_detect_image_alias_is_the_same_handler(client):
    """POST /detect/image and POST /api/v1/predict must give identical results."""
    client.app.state.predict_service = FakeService([PERSON, HELMET])
    files = {"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")}
    alias = client.post("/detect/image", files=files).json()
    files = {"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")}
    canonical = client.post("/api/v1/predict", files=files).json()
    assert alias["people"] == canonical["people"]
    [person] = canonical["people"]
    assert person["id"] == "Person-1" and person["conf"] == 0.9
    assert person["box"] == [100.0, 100.0, 200.0, 400.0]
    assert person["wearing"] == {"helmet": 0.81} and person["missing"] == ["vest"]
    assert person["status"] == "MISSING_VEST"
    # the same picture also produces the same operational decision on both routes
    assert person["decision"] == alias["people"][0]["decision"] == "STOP"
    assert alias["decision"] == canonical["decision"] == "STOP"


def test_root_lists_the_detect_image_alias(client):
    assert client.get("/").json()["detect_image"] == "/detect/image"


def test_dashboard_has_the_video_panel(client):
    page = client.get("/dashboard").text
    assert "/detect/video" in page          # the page really calls the video API
    assert "vdrop" in page and "renderVideo" in page


# --------------------------------------------------------------- violation log
def test_a_violation_is_written_to_the_log(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET])      # missing vest
    post_image(client)
    body = client.get("/api/v1/violations").json()
    assert body["count"] == 1
    entry = body["violations"][0]
    assert entry["status"] == "MISSING_VEST" and entry["person_id"] == "Person-1"
    assert entry["source"].startswith("image:")


def test_a_compliant_image_logs_nothing(client):
    client.app.state.predict_service = FakeService([PERSON, HELMET, VEST])
    post_image(client)
    assert client.get("/api/v1/violations").json()["count"] == 0


def test_the_violations_endpoint_reports_the_log_file(client):
    body = client.get("/api/v1/violations").json()
    assert body["log_file"].endswith("violations.jsonl")


def test_root_lists_the_violations_endpoint(client):
    assert client.get("/").json()["violations"] == "/api/v1/violations"


def test_dashboard_has_the_violation_log_panel(client):
    page = client.get("/dashboard").text
    assert "/api/v1/violations" in page and "vlrows" in page


def test_the_backend_picks_the_device_automatically():
    """auto = GPU only when it has free VRAM, else CPU (shared/device.py, D-022a)."""
    from server.config import Settings
    assert Settings().detect_device == "auto"


def test_the_pages_are_served_from_the_frontend_folder(client):
    """The HTML lives in frontend/, not inside the server package (D-023)."""
    from server.config import FRONTEND_DIR
    assert FRONTEND_DIR.name == "frontend"
    assert (FRONTEND_DIR / "dashboard.html").exists()
    assert (FRONTEND_DIR / "live.html").exists()
    assert client.get("/dashboard").status_code == 200
    assert client.get("/live").status_code == 200


def test_the_shared_stylesheet_is_served(client):
    """Both pages link /assets/theme.css - the mount must work."""
    response = client.get("/assets/theme.css")
    assert response.status_code == 200
    assert "--brand" in response.text
    assert 'href="/assets/theme.css"' in client.get("/dashboard").text


# ------------------------------------------------- snapshots are off by default
def test_no_snapshot_is_stored_by_default(client):
    from server.config import Settings
    assert Settings().store_snapshots is False

    client.app.state.predict_service = FakeService([PERSON, HELMET])     # a violation
    body = post_image(client).json()
    assert body["image_stored"] is False              # the upload, always
    assert body["event_snapshot_stored"] is False     # and no masked snapshot either
    assert body["privacy_status"] == "NO_EVIDENCE"
    assert client.get("/api/v1/violations").json()["violations"][0]["snapshot"] is None


def test_when_snapshots_are_on_the_stored_image_is_masked(tmp_path):
    """Switching snapshots on must still never store an unmasked frame."""
    from server.config import Settings
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
                        evidence_dir=tmp_path / "evidence",
                        violation_log=tmp_path / "violations.jsonl",
                        store_snapshots=True)
    with TestClient(create_app(settings)) as c:
        c.app.state.predict_service = FakeService([PERSON, HELMET])
        body = post_image(c).json()

        # the UPLOAD is still never stored - only the masked event snapshot is
        assert body["image_stored"] is False
        assert body["event_snapshot_stored"] is True
        assert body["privacy_status"] == "FACE_BLUR_OK"
        files = list((tmp_path / "evidence").glob("*.jpg"))
        assert len(files) == 1

        entry = c.get("/api/v1/violations").json()["violations"][0]
        assert entry["privacy_status"] == "FACE_BLUR_OK"
        assert entry["snapshot"].endswith(".jpg")      # a PATH, never image data
