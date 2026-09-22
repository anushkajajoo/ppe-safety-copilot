"""Backend smoke tests: app starts, tables exist, /health works. Uses a temporary DB file."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from server.config import Settings
from server.main import create_app

EXPECTED_TABLES = {"workers", "zones", "events", "detections", "reviews", "audit_logs"}


@pytest.fixture()
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        evidence_dir=tmp_path / "evidence")
    app = create_app(settings)
    with TestClient(app) as c:  # "with" runs startup (creates tables)
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["contract_version"]


def test_tables_created(client):
    tables = set(inspect(client.app.state.engine).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_root_points_to_docs(client):
    assert client.get("/").json()["docs"] == "/docs"
