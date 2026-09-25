"""
Security tests: the things an attacker or a careless user would try.

Each test is an attempt, not a demonstration: it tries the bad thing and asserts it did not
work. Lightweight by design - this is a prototype's threat surface, not a bank's.
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.types import Detection
from server.config import Settings
from server.main import create_app
from shared.safe_names import safe_filename

PASSWORD = "security-test-2026"
PERSON = Detection(cls="person", conf=0.90, box=(100, 100, 200, 400))


class FakeService:
    def detect(self, frame, conf=None):
        return [PERSON], 8.0


def jpeg(width=320, height=240):
    ok, buffer = cv2.imencode(".jpg", np.full((height, width, 3), 60, dtype=np.uint8))
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
                        violation_log=tmp_path / "violations.jsonl",
                        storage_root=tmp_path, logs_dir=tmp_path / "logs",
                        recordings_dir=tmp_path / "recordings",
                        reports_dir=tmp_path / "reports",
                        users_file=tmp_path / "users.yaml", session_secret="t",
                        edge_api_key="k")
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.predict_service = FakeService()
        yield c


def sign_in(client, username="ana"):
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": PASSWORD}).status_code == 200


# ------------------------------------------------------------------ file names
@pytest.mark.parametrize("hostile", [
    "../../etc/passwd.jpg",
    "..\\..\\windows\\system32\\evil.jpg",
    "/etc/shadow",
    "photo\x00.jpg",
    "; rm -rf / .jpg",
    "a" * 400 + ".jpg",
])
def test_a_hostile_upload_name_never_becomes_a_path(client, hostile, tmp_path):
    response = client.post("/api/v1/predict",
                           files={"image": (hostile, jpeg(), "image/jpeg")},
                           data={"required": "helmet,vest", "annotate": "false"})
    assert response.status_code == 200

    log_text = (tmp_path / "violations.jsonl").read_text(encoding="utf-8")
    assert ".." not in log_text and "\\" not in log_text and "\x00" not in log_text
    # nothing was created outside the folders the settings name
    assert not (tmp_path.parent / "etc").exists()


def test_the_name_cleaner_keeps_only_a_leaf_name():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\cmd.exe") == "cmd.exe"
    assert safe_filename("") == "upload" and safe_filename("...") == "upload"
    assert "/" not in safe_filename("/a/b/c.jpg") and len(safe_filename("x" * 500)) <= 80


# --------------------------------------------------------------------- uploads
def test_an_oversized_upload_is_refused(client):
    huge = b"\xff\xd8" + b"0" * (11 * 1024 * 1024)
    response = client.post("/api/v1/predict", files={"image": ("big.jpg", huge, "image/jpeg")},
                           data={"annotate": "false"})
    assert response.status_code == 413


def test_an_empty_upload_is_refused(client):
    response = client.post("/api/v1/predict", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_a_file_that_is_not_an_image_is_refused(client):
    """415, not 400: the upload arrived intact, it just is not a picture."""
    response = client.post("/api/v1/predict",
                           files={"image": ("notes.txt", b"this is not a picture", "text/plain")},
                           data={"annotate": "false"})
    assert response.status_code == 415
    assert "not a readable image" in response.json()["detail"]


def test_an_unsupported_video_type_is_refused(client):
    response = client.post("/detect/video",
                           files={"video": ("clip.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert response.status_code == 415


def test_a_corrupt_video_is_refused_not_crashed(client):
    response = client.post("/detect/video",
                           files={"video": ("clip.mp4", b"not really a video", "video/mp4")})
    assert response.status_code in (400, 415)


def test_a_confidence_outside_the_range_is_refused(client):
    for bad in ("1.4", "0", "-0.2"):
        response = client.post("/api/v1/predict", files={"image": ("a.jpg", jpeg(), "image/jpeg")},
                               data={"conf": bad, "annotate": "false"})
        assert response.status_code == 400


# ------------------------------------------------------------- authorisation
def test_an_unauthenticated_caller_cannot_decide_or_change_anything(client):
    assert client.post("/api/v1/copilot/suggestions/x/decide",
                       json={"decision": "approve"}).status_code == 401
    assert client.post("/api/v1/approvals", json={"kind": "change_threshold",
                                                  "summary": "x"}).status_code == 401
    assert client.post("/api/v1/alerts/abc/ack", json={}).status_code == 401


def test_a_viewer_cannot_escalate_by_asking_nicely(client):
    sign_in(client, "vic")
    assert client.post("/api/v1/approvals", json={"kind": "change_threshold",
                                                  "summary": "please"}).status_code == 403
    assert client.post("/api/v1/tools/acknowledge_alert", json={"alert_id": "x"}).status_code == 403


def test_a_supervisor_cannot_approve_their_own_configuration_change(client):
    """Proposing and approving a settings change are deliberately different roles."""
    sign_in(client, "ana")
    request_id = client.post("/api/v1/approvals", json={
        "kind": "change_threshold", "summary": "lower it",
        "changes": {"ppe_confidence_threshold": 0.1}}).json()["request_id"]
    assert client.post(f"/api/v1/approvals/{request_id}/decide",
                       json={"decision": "approve"}).status_code == 403
    assert client.app.state.settings.ppe_confidence_threshold == 0.5


def test_there_is_no_endpoint_that_writes_settings_directly(client):
    """Every settings change has to go through the approval gate, or there is no gate."""
    paths = {getattr(route, "path", "") for route in client.app.routes}
    for dangerous in ("/api/v1/config", "/api/v1/settings", "/api/v1/config/set"):
        assert dangerous not in paths


def test_the_api_key_is_not_exposed_by_any_public_endpoint(client):
    for path in ("/", "/health", "/api/v1/health/system", "/api/v1/tools",
                 "/api/v1/analytics/compliance"):
        body = client.get(path).text
        assert "edge_api_key" not in body and "change-me" not in body


def test_a_forged_session_cookie_is_refused(client):
    sign_in(client, "vic")
    cookie = client.cookies.get("ppe_session")
    client.cookies.set("ppe_session", cookie.split(".")[0] + ".deadbeef")
    assert client.get("/api/v1/auth/me").json()["signed_in"] is False


# -------------------------------------------------------------- tool surface
def test_arbitrary_tools_are_denied_and_recorded(client):
    for tool in ("run_command", "read_file", "write_file", "query_database", "http_request"):
        assert client.post(f"/api/v1/tools/{tool}", json={}).status_code == 403
    denied = client.get("/api/v1/tools/log?denied_only=true").json()["calls"]
    assert len(denied) >= 5


def test_snapshots_stay_inside_the_folder_they_were_given(tmp_path):
    """Masking writes where it is told and nowhere else, whatever the prefix says."""
    from shared.privacy import save_masked_snapshot
    frame = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    path, status = save_masked_snapshot(frame, [(50.0, 50.0, 150.0, 200.0)],
                                        tmp_path / "evidence", name_prefix="../../escape")
    assert path is not None
    assert tmp_path in path.parents
    assert ".." not in str(path.relative_to(tmp_path))
