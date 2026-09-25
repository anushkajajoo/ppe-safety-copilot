"""
Tamper-evident audit log (hash chain).

Every audit row stores:
    prev_hash = row_hash of the previous row ("000...0" for the first row)
    row_hash  = SHA-256(prev_hash + time + actor + action + event_id + details)

If anyone edits or deletes an old row directly in the database, its hash no longer
matches, and every row after it breaks too. verify_chain() finds the first bad row.
This doesn't PREVENT tampering (whoever owns the DB file can rewrite everything),
but it makes silent tampering DETECTABLE. That's the honest claim to make in the viva.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.models import AuditLog

GENESIS = "0" * 64
# One writer at a time, so two simultaneous requests can't both chain onto the same previous row.
WRITE_LOCK = threading.Lock()


def _ts(dt: datetime) -> str:
    # SQLite returns naive datetimes; normalise so insert-time and verify-time hashes match.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="microseconds")


def _hash(prev: str, occurred_at: datetime, actor: str, action: str, event_id: Optional[str], details: dict) -> str:
    body = json.dumps({"prev": prev, "t": _ts(occurred_at), "actor": actor, "action": action,
                       "event_id": event_id, "details": details}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def append_audit(db: Session, actor: str, action: str, event_id: Optional[str] = None,
                 details: Optional[dict] = None) -> AuditLog:
    """Adds an audit row to the session (caller commits, so it's in the same transaction as the change)."""
    details = details or {}
    last = db.execute(select(AuditLog).order_by(AuditLog.audit_id.desc()).limit(1)).scalar_one_or_none()
    # rows added earlier in this same (uncommitted) transaction:
    pending = [o for o in db.new if isinstance(o, AuditLog)]
    prev = pending[-1].row_hash if pending else (last.row_hash if last else GENESIS)
    now = datetime.now(timezone.utc)
    row = AuditLog(occurred_at=now, actor=actor, action=action, event_id=event_id, details=details,
                   prev_hash=prev, row_hash=_hash(prev, now, actor, action, event_id, details))
    db.add(row)
    return row


def verify_chain(db: Session) -> Tuple[bool, int, Optional[int]]:
    """Returns (ok, rows_checked, first_bad_audit_id)."""
    prev = GENESIS
    n = 0
    for row in db.execute(select(AuditLog).order_by(AuditLog.audit_id)).scalars():
        n += 1
        expected = _hash(prev, row.occurred_at, row.actor, row.action, row.event_id, row.details or {})
        if row.prev_hash != prev or row.row_hash != expected:
            return False, n, row.audit_id
        prev = row.row_hash
    return True, n, None
