"""
Who is using this system, and what they are allowed to do.

WHY IT EXISTS
    Until now the copilot recorded the name a supervisor typed into a box. That is a label,
    not an identity: anyone could type "supervisor_ana" and approve an escalation in her
    name. For a system whose value is a traceable human decision, that is the weakest link -
    threat T1 in docs/threat_model.md. Signing in turns "the name on the record" into "the
    person who was authenticated when the record was written".

THREE ROLES, BECAUSE THE WORK HAS THREE SHAPES
    viewer      - see the site's state. Cannot decide anything.
    supervisor  - everything a viewer can do, plus approve or reject copilot proposals.
    admin       - everything, plus configuration and maintenance.
    Roles are ordered, so `require_role("supervisor")` also admits an admin. A flat set of
    permissions would be more flexible and, at three roles, more code than it is worth.

HOW IT WORKS - AND WHAT IT DELIBERATELY DOES NOT USE
    * Passwords are stored as PBKDF2-HMAC-SHA256 hashes with a per-user salt and 200 000
      iterations (hashlib, standard library). Never plaintext, never a plain SHA-256, which
      a GPU walks through in seconds.
    * The session is a signed token in an HttpOnly cookie: base64(payload).HMAC-SHA256. The
      server stores no session state, and a tampered payload fails the signature.
    * The signing secret is generated fresh at start-up unless one is configured. That means
      signing in again after a restart - the right trade for a project that must never have
      a secret committed to the repository.
    * No JWT library, no OAuth, no password-reset email flow. Each would be more moving
      parts than a single-site console needs, and the brief's scope is a prototype.

HONEST LIMITS (also in the threat model)
    There is no rate limiting on sign-in, no account lockout, and no TLS: the server binds
    127.0.0.1, so the cookie never crosses a network. On a real site this would run behind
    HTTPS with a proper identity provider, and this module is the seam where that attaches.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

VIEWER, SUPERVISOR, ADMIN = "viewer", "supervisor", "admin"
ROLE_ORDER = {VIEWER: 0, SUPERVISOR: 1, ADMIN: 2}

COOKIE_NAME = "ppe_session"
SESSION_HOURS = 12
PBKDF2_ROUNDS = 200_000


class AuthError(Exception):
    """Sign-in failed, or the session is not valid. Never says which - see verify()."""


@dataclass(frozen=True)
class User:
    username: str
    role: str
    salt: str
    hash: str
    display_name: str = ""

    def label(self) -> str:
        return self.display_name or self.username


# ------------------------------------------------------------------ passwords
def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Returns {salt, hash}. A new random salt per user, so equal passwords differ."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), PBKDF2_ROUNDS)
    return {"salt": salt, "hash": digest.hex()}


def password_matches(password: str, salt: str, expected_hash: str) -> bool:
    try:
        candidate = hash_password(password, salt)["hash"]
    except ValueError:
        return False
    # compare_digest: constant time, so a wrong password takes as long as a right one
    return hmac.compare_digest(candidate, expected_hash)


# ---------------------------------------------------------------------- users
def load_users(path: Path) -> Dict[str, User]:
    """
    Read configs/users.yaml. A missing file means nobody can sign in, which is the safe
    direction: the system still runs and shows the site, but nothing can be approved.
    """
    path = Path(path)
    if not path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    users: Dict[str, User] = {}
    for entry in raw.get("users", []):
        role = str(entry.get("role", VIEWER)).lower()
        if role not in ROLE_ORDER:
            continue                                   # an unknown role grants nothing
        username = str(entry.get("username", "")).strip().lower()
        if not username or not entry.get("hash") or not entry.get("salt"):
            continue
        users[username] = User(username=username, role=role, salt=str(entry["salt"]),
                               hash=str(entry["hash"]),
                               display_name=str(entry.get("display_name", "")))
    return users


def authenticate(users: Dict[str, User], username: str, password: str) -> User:
    """
    Check a sign-in. Raises AuthError with the SAME message whether the user does not
    exist or the password is wrong - telling them apart is how account lists get harvested.
    """
    user = users.get((username or "").strip().lower())
    if user is None:
        # still do the work, so a missing user does not answer faster than a wrong password
        hash_password(password or "", "00" * 16)
        raise AuthError("incorrect username or password")
    if not password_matches(password or "", user.salt, user.hash):
        raise AuthError("incorrect username or password")
    return user


# ------------------------------------------------------------------- sessions
def _sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def make_token(user: User, secret: str, now: Optional[float] = None,
               hours: float = SESSION_HOURS) -> str:
    now = time.time() if now is None else now
    payload = json.dumps({"u": user.username, "r": user.role, "n": user.label(),
                          "exp": now + hours * 3600}, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{encoded}.{_sign(payload, secret)}"


def read_token(token: str, secret: str, now: Optional[float] = None) -> dict:
    """Returns the payload, or raises AuthError. Signature first, then expiry."""
    now = time.time() if now is None else now
    try:
        encoded, signature = (token or "").split(".", 1)
        padding = "=" * (-len(encoded) % 4)
        payload = base64.urlsafe_b64decode(encoded + padding)
    except (ValueError, TypeError, base64.binascii.Error):
        raise AuthError("malformed session")
    if not hmac.compare_digest(_sign(payload, secret), signature):
        raise AuthError("session signature does not match")     # edited or forged
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise AuthError("malformed session")
    if float(data.get("exp", 0)) < now:
        raise AuthError("session expired")
    if data.get("r") not in ROLE_ORDER:
        raise AuthError("unknown role")
    return data


def role_allows(role: str, required: str) -> bool:
    return ROLE_ORDER.get(role, -1) >= ROLE_ORDER.get(required, 99)
