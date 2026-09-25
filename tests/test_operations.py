"""
API tests for the operations layer: tools over HTTP, alerts and acknowledgement, compliance
analytics, system health, and the approval gate on configuration changes.

The pure-logic half lives in tests/test_operations_logic.py.
"""
import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from shared.approvals import APPROVED, PENDING, REJECTED
from shared.tools import ALLOWED, DENIED, TOOLS

PASSWORD = "operations-test-2026"

# ==================================================================== the API
def write_users(path):
    import yaml
    from server.auth import hash_password
    people = [("ana", "supervisor", "Ana Silva"), ("root", "admin", "Site Admin"),
              ("vic", "viewer", "Vic Watcher")]
    entries = []
    for username, role, display in people:
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
    with TestClient(create_app(settings)) as c:
        yield c


def sign_in(client, username="ana"):
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": PASSWORD}).status_code == 200


def test_the_tool_allow_list_is_published(client):
    body = client.get("/api/v1/tools").json()
    assert body["count"] == len(TOOLS)
    assert "no shell" in body["note"]


def test_a_read_tool_runs_and_is_logged(client):
    response = client.post("/api/v1/tools/get_camera_status", json={})
    assert response.status_code == 200
    assert response.json()["outcome"] == ALLOWED
    calls = client.get("/api/v1/tools/log").json()["calls"]
    assert calls[0]["tool"] == "get_camera_status" and calls[0]["outcome"] == ALLOWED


def test_an_unknown_tool_is_refused_and_the_denial_is_recorded(client):
    response = client.post("/api/v1/tools/run_command", json={"cmd": "rm -rf /"})
    assert response.status_code == 403
    denials = client.get("/api/v1/tools/log?denied_only=true").json()["calls"]
    assert denials and denials[0]["tool"] == "run_command"
    assert denials[0]["outcome"] == DENIED


def test_a_viewer_cannot_call_a_writing_tool(client):
    sign_in(client, "vic")
    response = client.post("/api/v1/tools/acknowledge_alert", json={"alert_id": "x"})
    assert response.status_code == 403


def test_evidence_producing_tools_cannot_be_called_on_demand(client):
    """A tool that can manufacture a snapshot is a tool that can manufacture a violation."""
    sign_in(client, "ana")
    for name in ("create_event", "save_snapshot", "save_recording"):
        assert client.post(f"/api/v1/tools/{name}", json={}).status_code == 409


def test_compliance_analytics_start_empty_and_stay_honest(client):
    body = client.get("/api/v1/analytics/compliance").json()
    assert body["total_evaluations"] == 0 and body["compliance_pct"] is None


def test_system_health_reports_the_parts_that_can_fail(client):
    body = client.get("/api/v1/health/system").json()
    for section in ("camera", "process", "storage", "queue", "model", "evidence", "decision"):
        assert section in body
    assert body["queue"]["mode"] in ("ONLINE", "OFFLINE")


def make_event(client, person="Person-1", decision="STOP", missing=("helmet",)):
    """Create a structured event the way the pipeline does - alerts are a view of these."""
    from server import event_store
    session = client.app.state.session_factory()
    try:
        event, _ = event_store.record_event(
            session, camera_id="camera:0", person_id=person, decision=decision,
            required=["helmet", "vest"], detected={"vest": 0.9}, missing=list(missing),
            uncertain=[], threshold=0.5, rule_id="R-MISSING-PPE", confidence=0.91)
        session.commit()
        return event.event_id
    finally:
        session.close()


def test_alerts_are_derived_from_the_structured_events(client):
    make_event(client)
    body = client.get("/api/v1/alerts").json()
    assert body["count"] == 1 and body["unacknowledged"] == 1
    alert = body["alerts"][0]
    assert alert["state"] == "NEW" and alert["decision"] == "STOP"
    assert alert["person"] == "P-01" and alert["missing"] == ["helmet"]


def test_an_alert_can_be_acknowledged_once_by_a_named_person(client):
    event_id = make_event(client, missing=("vest",))

    assert client.post(f"/api/v1/alerts/{event_id}/ack", json={}).status_code == 401
    sign_in(client, "ana")
    body = client.post(f"/api/v1/alerts/{event_id}/ack", json={}).json()
    assert body["state"] == "ACKNOWLEDGED" and body["acknowledged_by"] == "Ana Silva"

    after = client.get("/api/v1/alerts").json()
    assert after["unacknowledged"] == 0
    # acknowledging takes it off the ACTIVE list; it stays in history
    assert after["alerts"][0]["state"] == "ACKNOWLEDGED"
    assert client.get("/api/v1/history").json()["count"] == 1


def test_a_configuration_change_needs_a_request_and_an_admin(client):
    sign_in(client, "ana")                                   # supervisor may propose
    created = client.post("/api/v1/approvals", json={
        "kind": "change_threshold", "summary": "Raise the PPE threshold to 0.6",
        "reason": "too many uncertain calls", "changes": {"ppe_confidence_threshold": 0.6}})
    assert created.status_code == 201
    request_id = created.json()["request_id"]
    assert created.json()["status"] == PENDING
    assert client.app.state.settings.ppe_confidence_threshold == 0.5   # nothing applied yet

    # a supervisor may not decide it
    assert client.post(f"/api/v1/approvals/{request_id}/decide",
                       json={"decision": "approve"}).status_code == 403

    sign_in(client, "root")                                  # admin decides
    decided = client.post(f"/api/v1/approvals/{request_id}/decide",
                          json={"decision": "approve", "note": "agreed"}).json()
    assert decided["status"] == APPROVED and decided["applied"] is True
    assert client.app.state.settings.ppe_confidence_threshold == 0.6


def test_a_rejected_change_is_not_applied(client):
    sign_in(client, "ana")
    request_id = client.post("/api/v1/approvals", json={
        "kind": "change_privacy", "summary": "Turn on snapshot storage",
        "changes": {"store_snapshots": True}}).json()["request_id"]
    sign_in(client, "root")
    body = client.post(f"/api/v1/approvals/{request_id}/decide",
                       json={"decision": "reject", "note": "no lawful basis yet"}).json()
    assert body["status"] == REJECTED and body["applied"] is False
    assert client.app.state.settings.store_snapshots is False


def test_a_change_request_cannot_smuggle_in_another_setting(client):
    sign_in(client, "ana")
    response = client.post("/api/v1/approvals", json={
        "kind": "change_threshold", "summary": "innocent looking",
        "changes": {"database_url": "sqlite:///elsewhere.db"}})
    assert response.status_code == 400


def test_a_change_is_decided_only_once(client):
    sign_in(client, "ana")
    request_id = client.post("/api/v1/approvals", json={
        "kind": "change_retention", "summary": "Keep clips for 3 days",
        "changes": {"video_retention_days": 3}}).json()["request_id"]
    sign_in(client, "root")
    first = client.post(f"/api/v1/approvals/{request_id}/decide", json={"decision": "approve"})
    second = client.post(f"/api/v1/approvals/{request_id}/decide", json={"decision": "reject"})
    assert first.status_code == 200 and second.status_code == 409


def test_every_change_reaches_the_audit_chain(client):
    sign_in(client, "ana")
    request_id = client.post("/api/v1/approvals", json={
        "kind": "change_threshold", "summary": "0.55 please",
        "changes": {"ppe_confidence_threshold": 0.55}}).json()["request_id"]
    sign_in(client, "root")
    client.post(f"/api/v1/approvals/{request_id}/decide", json={"decision": "approve"})

    actions = [row["action"] for row in client.get("/api/v1/audit").json()]
    assert "CHANGE_REQUESTED" in actions and "CHANGE_APPROVED" in actions
    assert client.get("/api/v1/audit/verify").json()["ok"] is True
