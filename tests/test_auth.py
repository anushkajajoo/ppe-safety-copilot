"""
Tests for sign-in and roles (server/auth.py, server/routes/auth.py).

The point of this layer is that a decision in the audit log names a person who was
authenticated, not a string someone typed. These tests are the evidence for that, plus the
usual unglamorous things: passwords are never stored, a forged cookie is refused, and an
expired session stops working.
"""
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from server.auth import (ADMIN, AuthError, COOKIE_NAME, SUPERVISOR, VIEWER, authenticate,
                         hash_password, load_users, make_token, password_matches,
                         read_token, role_allows)
from server.config import Settings
from server.main import create_app

PASSWORD = "correct-horse-2026"
SECRET = "unit-test-secret"


def users_file(tmp_path, people=(("ana", SUPERVISOR, "Ana Silva"),
                                 ("vic", VIEWER, "Vic Watcher"),
                                 ("root", ADMIN, "Site Admin"))):
    entries = []
    for username, role, display in people:
        entry = {"username": username, "role": role, "display_name": display}
        entry.update(hash_password(PASSWORD))
        entries.append(entry)
    path = tmp_path / "users.yaml"
    path.write_text(yaml.safe_dump({"users": entries}), encoding="utf-8")
    return path


@pytest.fixture()
def client(tmp_path):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                 evidence_dir=tmp_path / "ev", edge_api_key="k",
                 violation_log=tmp_path / "v.jsonl",
                 users_file=users_file(tmp_path), session_secret=SECRET)
    with TestClient(create_app(s)) as c:
        yield c


# ----------------------------------------------------------------- passwords
def test_a_password_is_never_stored_only_a_hash(tmp_path):
    path = users_file(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert PASSWORD not in text
    entry = yaml.safe_load(text)["users"][0]
    assert set(entry) >= {"salt", "hash"} and len(entry["hash"]) == 64


def test_the_same_password_hashes_differently_for_two_people():
    """Per-user salt: two people with one bad habit do not share a hash."""
    assert hash_password(PASSWORD)["hash"] != hash_password(PASSWORD)["hash"]


def test_a_hash_verifies_only_the_right_password():
    made = hash_password(PASSWORD)
    assert password_matches(PASSWORD, made["salt"], made["hash"])
    assert not password_matches(PASSWORD + "!", made["salt"], made["hash"])
    assert not password_matches("", made["salt"], made["hash"])


def test_a_corrupt_salt_fails_closed():
    assert not password_matches(PASSWORD, "not-hex", "0" * 64)


# --------------------------------------------------------------------- users
def test_a_missing_users_file_means_nobody_can_sign_in(tmp_path):
    """Fail closed: the site is still visible, but nothing can be approved."""
    assert load_users(tmp_path / "nothing.yaml") == {}


def test_an_unknown_role_is_ignored_rather_than_trusted(tmp_path):
    path = tmp_path / "u.yaml"
    entry = {"username": "x", "role": "superuser"}
    entry.update(hash_password(PASSWORD))
    path.write_text(yaml.safe_dump({"users": [entry]}), encoding="utf-8")
    assert load_users(path) == {}


def test_wrong_user_and_wrong_password_give_the_same_answer(tmp_path):
    """Telling them apart is how someone learns which usernames exist."""
    users = load_users(users_file(tmp_path))
    with pytest.raises(AuthError) as no_user:
        authenticate(users, "nobody", PASSWORD)
    with pytest.raises(AuthError) as bad_password:
        authenticate(users, "ana", "wrong")
    assert str(no_user.value) == str(bad_password.value)


# ------------------------------------------------------------------ sessions
def test_a_token_round_trips():
    users = {"ana": None}
    from server.auth import User
    token = make_token(User("ana", SUPERVISOR, "aa", "bb", "Ana Silva"), SECRET)
    data = read_token(token, SECRET)
    assert data["u"] == "ana" and data["r"] == SUPERVISOR and data["n"] == "Ana Silva"


def test_a_forged_token_is_refused():
    from server.auth import User
    token = make_token(User("vic", VIEWER, "aa", "bb"), SECRET)
    body, _, signature = token.partition(".")
    # keep the signature, swap the payload for an admin one
    admin = make_token(User("vic", ADMIN, "aa", "bb"), "a-different-secret").split(".")[0]
    with pytest.raises(AuthError):
        read_token(f"{admin}.{signature}", SECRET)


def test_a_token_signed_with_another_secret_is_refused():
    from server.auth import User
    token = make_token(User("ana", SUPERVISOR, "aa", "bb"), "someone-elses-secret")
    with pytest.raises(AuthError):
        read_token(token, SECRET)


def test_an_expired_session_stops_working():
    from server.auth import User
    token = make_token(User("ana", SUPERVISOR, "aa", "bb"), SECRET, now=time.time() - 86400)
    with pytest.raises(AuthError):
        read_token(token, SECRET)


def test_garbage_is_refused_without_crashing():
    for rubbish in ("", "not-a-token", "a.b", "...."):
        with pytest.raises(AuthError):
            read_token(rubbish, SECRET)


def test_roles_are_ordered():
    assert role_allows(ADMIN, SUPERVISOR) and role_allows(SUPERVISOR, SUPERVISOR)
    assert not role_allows(VIEWER, SUPERVISOR)
    assert not role_allows("made-up", VIEWER)


# ----------------------------------------------------------------- the API
def test_signing_in_and_out(client):
    assert client.get("/api/v1/auth/me").json()["signed_in"] is False

    body = client.post("/api/v1/auth/login",
                       json={"username": "ana", "password": PASSWORD}).json()
    assert body["role"] == SUPERVISOR and body["display_name"] == "Ana Silva"

    me = client.get("/api/v1/auth/me").json()
    assert me["signed_in"] and me["can_decide"] and not me["can_administer"]

    client.post("/api/v1/auth/logout")
    assert client.get("/api/v1/auth/me").json()["signed_in"] is False


def test_a_wrong_password_is_a_401_and_sets_no_cookie(client):
    r = client.post("/api/v1/auth/login", json={"username": "ana", "password": "nope"})
    assert r.status_code == 401
    assert COOKIE_NAME not in r.cookies
    assert client.get("/api/v1/auth/me").json()["signed_in"] is False


def test_the_session_cookie_is_not_readable_by_scripts(client):
    r = client.post("/api/v1/auth/login", json={"username": "ana", "password": PASSWORD})
    cookie_header = r.headers["set-cookie"].lower()
    assert "httponly" in cookie_header          # a page script cannot steal it
    assert "samesite=strict" in cookie_header   # and another site cannot ride it


def test_an_admin_can_do_what_a_supervisor_can(client):
    client.post("/api/v1/auth/login", json={"username": "root", "password": PASSWORD})
    me = client.get("/api/v1/auth/me").json()
    assert me["can_decide"] and me["can_administer"]


def test_the_login_response_never_echoes_the_password(client):
    r = client.post("/api/v1/auth/login", json={"username": "ana", "password": PASSWORD})
    assert PASSWORD not in r.text
