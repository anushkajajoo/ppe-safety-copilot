"""
Tests for the human-governed copilot: shared/copilot.py (the rules) and
server/routes/copilot.py (the gate, the queue and the audit trail).

These tests are the evidence for the three claims the project makes about this layer:
allow-listed, human-governed, traceable. Each one is checked by code, not asserted in a
document.
"""
import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from shared.copilot import ACTIONS, BY_ID, NotAllowed, catalog, check_action, suggest

VIOLATION, UNCERTAIN, COMPLIANT = "POTENTIAL_VIOLATION", "UNCERTAIN", "COMPLIANT"


PEOPLE = {"ana": ("supervisor", "Ana Silva"), "sam": ("supervisor", "Sam Okoye"),
          "vic": ("viewer", "Vic Watcher")}
PASSWORD = "test-password-2026"


def write_users(path):
    """A real users file, hashed the real way - the tests sign in like a person would."""
    import yaml
    from server.auth import hash_password
    entries = []
    for username, (role, display) in PEOPLE.items():
        entry = {"username": username, "role": role, "display_name": display}
        entry.update(hash_password(PASSWORD))
        entries.append(entry)
    path.write_text(yaml.safe_dump({"users": entries}), encoding="utf-8")


def sign_in(client, username="ana"):
    response = client.post("/api/v1/auth/login",
                           json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # keep the outbox inside the test's own folder - never the real data/ directory
    from server.routes import copilot as copilot_routes
    monkeypatch.setattr(copilot_routes, "OUTBOX", tmp_path / "outbox.jsonl")
    write_users(tmp_path / "users.yaml")
    s = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                 evidence_dir=tmp_path / "ev", edge_api_key="k",
                 violation_log=tmp_path / "violations.jsonl",
                 users_file=tmp_path / "users.yaml", session_secret="test-secret")
    with TestClient(create_app(s)) as c:
        yield c


def propose(client, **kw):
    body = {"source": "image:site.jpg", "person_id": "Worker-1", "compliance_status": VIOLATION,
            "missing": ["helmet"], "zone_type": "GENERAL", "confidence": 0.9}
    body.update(kw)
    return client.post("/api/v1/copilot/suggestions", json=body)


# ---------------------------------------------------------------- the rules (pure)
def test_a_compliant_worker_gets_no_suggestion():
    """The copilot stays quiet when there is nothing to decide."""
    assert suggest(COMPLIANT, []) == []


def test_an_unclear_case_asks_for_another_look_instead_of_raising_an_alarm():
    [s] = suggest(UNCERTAIN, ["vest"])
    assert s.action_id == "request_recheck" and s.rule == "R2"


def test_a_restricted_zone_violation_escalates():
    [s] = suggest(VIOLATION, ["helmet"], "RESTRICTED", "vault-c", "Worker-9")
    assert s.action_id == "escalate_restricted_zone" and s.rule == "R3"
    assert "vault-c" in s.reason and s.severity == 3


def test_a_repeated_violation_notifies_the_supervisor():
    [s] = suggest(VIOLATION, ["vest"], repeat_count=4)
    assert s.action_id == "notify_supervisor" and s.rule == "R4"


def test_a_first_violation_is_only_flagged():
    [s] = suggest(VIOLATION, ["vest"], repeat_count=0)
    assert s.action_id == "flag_for_review" and s.rule == "R5"


def test_the_rules_only_ever_produce_allow_listed_actions():
    """The property that matters: no input can talk the copilot into a new action."""
    for status in (COMPLIANT, UNCERTAIN, VIOLATION, "NONSENSE"):
        for zone in ("GENERAL", "CAUTION", "RESTRICTED", "made-up-zone"):
            for repeats in (0, 3, 99):
                for s in suggest(status, ["helmet", "vest"], zone, repeat_count=repeats):
                    assert s.action_id in BY_ID


def test_the_rules_are_deterministic():
    args = (VIOLATION, ["helmet"], "CAUTION", "weld-a", "Worker-2", 0.8, 2)
    assert [s.as_dict() for s in suggest(*args)] == [s.as_dict() for s in suggest(*args)]


def test_at_most_one_suggestion_at_a_time():
    for status in (COMPLIANT, UNCERTAIN, VIOLATION):
        assert len(suggest(status, ["helmet", "vest", "mask"], "RESTRICTED")) <= 1


def test_an_unknown_action_is_refused():
    with pytest.raises(NotAllowed):
        check_action("shut_down_the_site")
    assert check_action("log_only").id == "log_only"


def test_the_catalog_matches_the_allow_list():
    assert len(catalog()) == len(ACTIONS)
    assert {a["id"] for a in catalog()} == set(BY_ID)


# ---------------------------------------------------------------- the API
def test_the_catalog_is_published(client):
    body = client.get("/api/v1/copilot/catalog").json()
    assert body["count"] == len(ACTIONS)
    assert all({"id", "title", "effect", "reversible"} <= set(a) for a in body["actions"])


def test_a_suggestion_starts_pending_and_does_nothing(client, tmp_path):
    created = propose(client).json()["created"]
    assert len(created) == 1
    assert created[0]["status"] == "PENDING" and created[0]["outcome"] is None
    # nothing has been carried out while it waits for a human
    assert client.get("/api/v1/copilot/outbox").json()["count"] == 0


def test_the_same_situation_does_not_pile_up_duplicates(client):
    propose(client)
    again = propose(client).json()
    assert again["created"] == [] and again["duplicates_skipped"] == 1
    assert client.get("/api/v1/copilot/suggestions").json()["count"] == 1


def test_a_compliant_report_creates_nothing(client):
    body = propose(client, compliance_status=COMPLIANT, missing=[]).json()
    assert body["created"] == []


def test_approving_records_who_decided_and_carries_the_action_out(client):
    sign_in(client, "ana")
    sid = propose(client, zone_type="RESTRICTED", zone_name="vault-c").json()["created"][0]["suggestion_id"]
    out = client.post(f"/api/v1/copilot/suggestions/{sid}/decide",
                      json={"decision": "approve", "note": "checked the camera"}).json()
    assert out["status"] == "APPROVED"
    # the name comes from the SESSION, not from the request body
    assert out["decided_by"] == "Ana Silva"
    assert out["decided_at"] is not None and "outbox" in out["outcome"]

    messages = client.get("/api/v1/copilot/outbox").json()["messages"]
    assert len(messages) == 1
    assert messages[0]["priority"] == "HIGH" and messages[0]["approved_by"] == "Ana Silva"


def test_a_decision_cannot_be_attributed_to_someone_else(client):
    """The old design trusted a typed name. Sending one now changes nothing (threat T1)."""
    sign_in(client, "ana")
    sid = propose(client).json()["created"][0]["suggestion_id"]
    out = client.post(f"/api/v1/copilot/suggestions/{sid}/decide",
                      json={"decision": "approve", "actor": "somebody_else"}).json()
    assert out["decided_by"] == "Ana Silva"


def test_nobody_signed_in_cannot_decide(client):
    sid = propose(client).json()["created"][0]["suggestion_id"]
    r = client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "approve"})
    assert r.status_code == 401
    assert client.get("/api/v1/copilot/suggestions").json()["count"] == 1   # still pending


def test_a_viewer_may_look_but_not_decide(client):
    """The role-based journey: a viewer sees the queue and is refused at the decision."""
    sign_in(client, "vic")
    sid = propose(client).json()["created"][0]["suggestion_id"]
    assert client.get("/api/v1/copilot/suggestions").status_code == 200
    r = client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "approve"})
    assert r.status_code == 403 and "supervisor" in r.json()["detail"]


def test_rejecting_carries_nothing_out(client):
    sign_in(client, "ana")
    sid = propose(client, repeat_count=5).json()["created"][0]["suggestion_id"]
    out = client.post(f"/api/v1/copilot/suggestions/{sid}/decide",
                      json={"decision": "reject"}).json()
    assert out["status"] == "REJECTED" and out["outcome"] is None
    assert client.get("/api/v1/copilot/outbox").json()["count"] == 0


def test_a_suggestion_can_only_be_decided_once(client):
    sign_in(client, "ana")
    sid = propose(client).json()["created"][0]["suggestion_id"]
    first = client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "approve"})
    sign_in(client, "sam")                       # a different supervisor tries the same row
    second = client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "approve"})
    assert first.status_code == 200
    assert second.status_code == 409 and "Ana Silva" in second.json()["detail"]


def test_an_invalid_decision_word_is_refused(client):
    sign_in(client, "ana")
    sid = propose(client).json()["created"][0]["suggestion_id"]
    r = client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "maybe"})
    assert r.status_code == 422


def test_deciding_something_that_does_not_exist_is_a_404(client):
    sign_in(client, "ana")
    r = client.post("/api/v1/copilot/suggestions/no-such-id/decide", json={"decision": "approve"})
    assert r.status_code == 404


def test_the_queue_puts_urgent_work_first(client):
    propose(client, person_id="Worker-1")                                   # severity 1
    propose(client, person_id="Worker-9", zone_type="RESTRICTED", zone_name="vault-c")  # severity 3
    queue = client.get("/api/v1/copilot/suggestions").json()["suggestions"]
    assert [s["subject"] for s in queue] == ["Worker-9", "Worker-1"]


def test_every_proposal_and_decision_reaches_the_audit_chain(client):
    sign_in(client, "ana")
    sid = propose(client).json()["created"][0]["suggestion_id"]
    client.post(f"/api/v1/copilot/suggestions/{sid}/decide", json={"decision": "approve"})

    rows = client.get("/api/v1/audit").json()
    actions = [r["action"] for r in rows]
    assert "SUGGESTION_CREATED" in actions and "SUGGESTION_APPROVED" in actions
    assert any(r["actor"] == "copilot" for r in rows)          # the machine proposed
    decision = next(r for r in rows if r["action"] == "SUGGESTION_APPROVED")
    assert decision["actor"] == "Ana Silva"                    # the human decided
    assert decision["details"]["role"] == "supervisor"         # and with what authority
    assert client.get("/api/v1/audit/verify").json()["ok"] is True


# ---------------------------------------------------------------- the dashboard
def test_the_dashboard_shows_the_review_queue_and_the_allow_list(client):
    """The governance is visible to the user, not buried in the API."""
    page = client.get("/dashboard").text
    assert "Copilot review queue" in page
    assert "cprows" in page and "cpcatalog" in page
    assert "/api/v1/copilot/suggestions" in page
    assert "a person decides" in page


def test_the_dashboard_asks_who_is_deciding_rather_than_taking_a_typed_name(client):
    page = client.get("/dashboard").text
    assert "/api/v1/auth/login" in page and "/api/v1/auth/me" in page
    assert "session.can_decide" in page            # the buttons follow the role
    assert "cpactor.value" not in page             # the old typed-name field is gone


def test_an_image_upload_produces_a_proposal_but_no_action(client, monkeypatch):
    """
    End to end: analysing a picture with a bare-headed worker puts a suggestion in the
    queue and leaves it there. This is the whole design in one test.
    """
    import cv2
    import numpy as np
    from edge.types import Detection

    class FakeService:
        def detect(self, frame, conf=None):
            return [Detection(cls="person", conf=0.9, box=(100, 100, 200, 400))], 10.0

    client.app.state.predict_service = FakeService()
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), 60, dtype=np.uint8))
    assert ok
    response = client.post("/api/v1/predict",
                           files={"image": ("site.jpg", buffer.tobytes(), "image/jpeg")},
                           data={"required": "helmet,vest", "annotate": "false"})
    assert response.status_code == 200
    assert response.json()["violations"] == 1

    queue = client.get("/api/v1/copilot/suggestions").json()
    assert queue["count"] == 1
    assert queue["suggestions"][0]["status"] == "PENDING"
    assert client.get("/api/v1/copilot/outbox").json()["count"] == 0     # nothing done yet
