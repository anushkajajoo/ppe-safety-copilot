"""GET /api/v1/audit (latest rows) and GET /api/v1/audit/verify (hash-chain check)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.audit import verify_chain
from server.database import get_db
from server.models import AuditLog

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("")
def list_audit(db: Session = Depends(get_db), limit: int = Query(default=100, ge=1, le=1000)):
    rows = db.execute(select(AuditLog).order_by(AuditLog.audit_id.desc()).limit(limit)).scalars()
    return [{"audit_id": r.audit_id, "occurred_at": r.occurred_at, "actor": r.actor, "action": r.action,
             "event_id": r.event_id, "details": r.details, "row_hash": r.row_hash, "prev_hash": r.prev_hash}
            for r in rows]


@router.get("/verify")
def verify(db: Session = Depends(get_db)):
    ok, n, bad = verify_chain(db)
    return {"ok": ok, "rows_checked": n, "first_bad_audit_id": bad}
