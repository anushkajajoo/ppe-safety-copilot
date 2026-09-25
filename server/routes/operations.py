"""
Operations API: tools, alerts, analytics, system health and configuration approvals.

    GET  /api/v1/tools                     the allow-list, published
    POST /api/v1/tools/{name}              call an allow-listed tool (anything else: DENIED)
    GET  /api/v1/tools/log                 every tool request, allowed and denied
    GET  /api/v1/analytics/compliance      counts and compliance % from recorded evaluations
    GET  /api/v1/health/system             frame rate, latency, memory, queue, storage
    GET  /api/v1/alerts                    alerts derived from the violation log
    POST /api/v1/alerts/{alert_id}/ack     acknowledge one (supervisor)
    GET  /api/v1/approvals                 configuration changes awaiting a person
    POST /api/v1/approvals                 propose one
    POST /api/v1/approvals/{id}/decide     approve or reject it (admin), then apply it

Nothing in here makes a safety decision. The rule engine does that; these endpoints report
what it decided and record what people did about it.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from server.http_status import UNPROCESSABLE
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.audit import WRITE_LOCK, append_audit
from server.auth import ADMIN, SUPERVISOR
from server.database import get_db
from server import event_store
from server.models import ApprovalRequest
from server.routes.auth import current_user, require_role
from shared.approvals import (APPROVED, PENDING, REJECTED, NotApprovable, catalog as approval_catalog,
                              check_changes, check_kind)
from shared.tools import ALLOWED, DENIED, ToolDenied, catalog as tool_catalog, check_tool

router = APIRouter(prefix="/api/v1", tags=["operations"])



def tool_log(request: Request):
    return request.app.state.tool_log


def decision_log(request: Request):
    return request.app.state.decision_log


# ================================================================== tools
@router.get("/tools")
def list_tools():
    """Published on purpose: the copilot's whole reach, readable in one screen."""
    return {"tools": tool_catalog(), "count": len(tool_catalog()),
            "note": "Anything not in this list is denied and the denial is logged. There is "
                    "no shell, filesystem, database or arbitrary HTTP tool."}


@router.post("/tools/{name}")
def call_tool(name: str, request: Request, payload: Dict = Body(default={}),
              db: Session = Depends(get_db)):
    user = current_user(request)
    actor = (user or {}).get("n") or (user or {}).get("u")
    can_write = bool(user) and user.get("r") in (SUPERVISOR, ADMIN)

    try:
        tool = check_tool(name, can_write=can_write)
    except ToolDenied as exc:
        tool_log(request).record(name, DENIED, str(exc), actor)
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=str(exc))

    result = _run_tool(tool.name, request, db, payload, actor)
    tool_log(request).record(name, ALLOWED, "executed", actor)
    return {"tool": name, "outcome": ALLOWED, "result": result}


def _run_tool(name: str, request: Request, db: Session, payload: Dict, actor: Optional[str]):
    """
    Every allow-listed tool, in one place. Each one reads or records; none decides.
    """
    state = getattr(request.app.state, "live_state", None)
    live = state.snapshot() if state is not None else {}

    if name == "get_camera_status":
        return {"running": live.get("running", False), "camera": live.get("camera"),
                "fps": live.get("fps", 0.0), "frames": live.get("frames", 0),
                "dropped": live.get("dropped", 0)}
    if name == "get_latest_detection":
        return {"people": live.get("people", 0), "statuses": live.get("statuses", []),
                "decision": live.get("decision"), "reason": live.get("reason"),
                "updated": live.get("updated")}
    if name == "get_compliance_status":
        return decision_log(request).summary()
    if name == "get_system_health":
        return system_health(request)
    if name == "list_recent_events":
        log = getattr(request.app.state, "violation_log", None)
        return {"events": log.recent(limit=int(payload.get("limit", 20))) if log else []}
    if name == "acknowledge_alert":
        alert_id = str(payload.get("alert_id", "")).strip()
        if not alert_id:
            raise HTTPException(UNPROCESSABLE, detail="alert_id is required")
        return _acknowledge(db, alert_id, actor or "unknown", payload.get("note"))
    if name == "generate_report":
        return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "compliance": decision_log(request).summary(),
                "recent_violations": (request.app.state.violation_log.recent(limit=20)
                                      if getattr(request.app.state, "violation_log", None) else [])}
    if name in ("create_event", "save_snapshot", "save_recording"):
        # These are performed by the pipeline itself when a decision fires; exposing them as
        # callable tools would let the assistant layer manufacture evidence.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail=f"'{name}' is produced by the detection pipeline, not "
                                   f"called on demand - a tool that can invent evidence is "
                                   f"a tool that can invent violations")
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=f"'{name}' has no handler")


@router.get("/tools/log")
def tool_requests(request: Request, limit: int = Query(default=50, ge=1, le=500),
                  denied_only: bool = False):
    log = tool_log(request)
    rows = log.denials(limit) if denied_only else log.recent(limit)
    return {"calls": rows, "count": len(rows)}


# ============================================================== analytics
@router.get("/analytics/compliance")
def compliance(request: Request, limit: Optional[int] = Query(default=None, ge=1, le=100000)):
    """Counted from `data/logs/decisions.jsonl`. No figure here is assumed or hard-coded."""
    summary = decision_log(request).summary(limit)
    summary["note"] = ("compliance_pct is compliant / total evaluations. It is null when "
                       "nothing has been evaluated yet - which is not the same as 0 %.")
    return summary


# ============================================================ system health
def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@router.get("/health/system")
def system_health(request: Request):
    """Lightweight application telemetry: no Prometheus, no agent, no extra dependency."""
    settings = request.app.state.settings
    state = getattr(request.app.state, "live_state", None)
    live = state.snapshot() if state is not None else {}

    process: Dict[str, object] = {}
    try:
        import psutil
        proc = psutil.Process()
        process = {"memory_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
                   "cpu_pct": proc.cpu_percent(interval=0.05),
                   "threads": proc.num_threads()}
    except Exception:
        process = {"memory_mb": None, "cpu_pct": None, "threads": None}

    storage_root = Path(getattr(settings, "storage_root", "data"))
    try:
        usage = shutil.disk_usage(storage_root if storage_root.exists() else Path("."))
        free_gb = round(usage.free / (1024 ** 3), 2)
    except OSError:
        free_gb = None

    outbox = Path(getattr(settings, "storage_root", Path("data"))) / "edge_outbox.jsonl"
    queued = 0
    if outbox.exists():
        queued = sum(1 for line in outbox.read_text(encoding="utf-8", errors="ignore").splitlines()
                     if line.strip())

    model_ready = getattr(getattr(request.app.state, "predict_service", None), "ready", None)
    return {
        "camera": {"running": live.get("running", False), "camera": live.get("camera"),
                   "fps": live.get("fps", 0.0), "frames": live.get("frames", 0),
                   "dropped": live.get("dropped", 0),
                   "latency_ms": live.get("latency_ms", 0.0)},
        "decision": {"current": live.get("decision"), "reason": live.get("reason")},
        "process": process,
        "storage": {"root": str(storage_root), "free_gb": free_gb,
                    "recordings_mb": round(_directory_size(Path(getattr(settings, "recordings_dir", "data/recordings"))) / (1024 * 1024), 2),
                    "snapshots_mb": round(_directory_size(Path(getattr(settings, "evidence_dir", "data/snapshots/masked"))) / (1024 * 1024), 2)},
        "queue": {"pending_events": queued, "mode": "OFFLINE" if queued else "ONLINE"},
        "model": {"weights": str(getattr(settings, "weights", "")), "loaded": model_ready},
        "evidence": {"snapshots_enabled": bool(getattr(settings, "store_snapshots", False)),
                     "recording_enabled": bool(getattr(settings, "record_events", False))},
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ================================================================= alerts
# An "active alert" is not a separate thing: it is a structured event whose status is still
# NEW. Deriving alerts from the event store (rather than from the flat violation log, as an
# earlier version did) is what makes acknowledging one move it from the alert list into
# history instead of destroying it.
@router.get("/alerts")
def list_alerts(request: Request, db: Session = Depends(get_db),
                limit: int = Query(default=50, ge=1, le=500),
                unacknowledged_only: bool = False):
    events = event_store.list_events(db, limit=limit,
                                     unacknowledged_only=unacknowledged_only)
    alerts = []
    for event in events:
        if event.decision == "GO":
            continue                       # compliance is history, not an alert
        record = event_store.to_dict(event)
        alerts.append({
            "alert_id": event.event_id, "event_id": event.event_id,
            "at": event.occurred_at, "source": event.camera_id, "person": event.person_id,
            "status": event.event_type, "decision": event.decision,
            "severity": event.severity, "reason": event.reason,
            "missing": record["missing_ppe"], "confidence": event.confidence,
            "snapshot": record["snapshot"], "recording": record["recording"],
            "state": "NEW" if event.status == "NEW" else "ACKNOWLEDGED",
            "acknowledged_by": event.acknowledged_by,
            "acknowledged_at": event.acknowledged_at,
            "occurrences": event.occurrences, "duration_s": event.duration_s,
        })
    return {"alerts": alerts, "count": len(alerts),
            "unacknowledged": sum(1 for a in alerts if a["state"] == "NEW")}


def _acknowledge(db: Session, alert_id: str, actor: str, note: Optional[str]) -> dict:
    event = event_store.acknowledge(db, alert_id, actor, note)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such alert")
    append_audit(db, actor=actor, action="ALERT_ACKNOWLEDGED", event_id=alert_id,
                 details={"decision": event.decision, "note": note})
    db.commit()
    return {"alert_id": alert_id, "event_id": alert_id, "state": "ACKNOWLEDGED",
            "acknowledged_by": event.acknowledged_by, "already": False}


@router.post("/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str, db: Session = Depends(get_db),
                      user: dict = Depends(require_role(SUPERVISOR)),
                      note: Optional[str] = Body(default=None, embed=True)):
    with WRITE_LOCK:
        return _acknowledge(db, alert_id, user.get("n") or user["u"], note)


# ============================================================== approvals
class ChangeRequest(BaseModel):
    kind: str
    summary: str = Field(..., max_length=300)
    reason: Optional[str] = Field(default=None, max_length=500)
    changes: Dict[str, object] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    decision: str
    note: Optional[str] = Field(default=None, max_length=500)


def approval_to_dict(row: ApprovalRequest) -> dict:
    return {"request_id": row.request_id, "created_at": row.created_at, "kind": row.kind,
            "summary": row.summary, "payload": row.payload, "requested_by": row.requested_by,
            "reason": row.reason, "status": row.status, "decided_by": row.decided_by,
            "decided_at": row.decided_at, "decision_note": row.decision_note,
            "applied": row.applied, "outcome": row.outcome}


@router.get("/approvals/catalog")
def approvals_catalog():
    return {"kinds": approval_catalog(),
            "note": "These settings cannot be changed without a named person approving it."}


@router.post("/approvals", status_code=status.HTTP_201_CREATED)
def request_change(body: ChangeRequest, db: Session = Depends(get_db),
                   user: dict = Depends(require_role(SUPERVISOR))):
    try:
        check_kind(body.kind)
        changes = check_changes(body.kind, body.changes) if body.changes else {}
    except NotApprovable as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    with WRITE_LOCK:
        row = ApprovalRequest(kind=body.kind, summary=body.summary, payload=changes,
                              requested_by=user.get("n") or user["u"], reason=body.reason,
                              status=PENDING)
        db.add(row)
        db.flush()
        append_audit(db, actor=row.requested_by, action="CHANGE_REQUESTED",
                     details={"request_id": row.request_id, "kind": row.kind,
                              "changes": changes})
        db.commit()
        db.refresh(row)
    return approval_to_dict(row)


@router.get("/approvals")
def list_approvals(db: Session = Depends(get_db),
                   request_status: str = Query(default=PENDING, alias="status"),
                   limit: int = Query(default=50, ge=1, le=500)):
    query = select(ApprovalRequest)
    if request_status.lower() != "all":
        query = query.where(ApprovalRequest.status == request_status.upper())
    rows = db.execute(query.order_by(ApprovalRequest.created_at.desc()).limit(limit)).scalars().all()
    return {"approvals": [approval_to_dict(row) for row in rows], "count": len(rows)}


@router.post("/approvals/{request_id}/decide")
def decide_change(request_id: str, request: Request, body: ApprovalDecision = Body(...),
                  db: Session = Depends(get_db),
                  user: dict = Depends(require_role(ADMIN))):
    """
    Only an admin decides a configuration change, and only once. On approval the change is
    applied to the running settings - but only to the keys its kind is allowed to touch.
    """
    choice = body.decision.strip().lower()
    if choice not in ("approve", "reject"):
        raise HTTPException(UNPROCESSABLE,
                            detail="decision must be 'approve' or 'reject'")
    with WRITE_LOCK:
        row = db.get(ApprovalRequest, request_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such request")
        if row.status != PENDING:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                detail=f"already {row.status.lower()} by {row.decided_by}")

        actor = user.get("n") or user["u"]
        row.status = APPROVED if choice == "approve" else REJECTED
        row.decided_by = actor
        row.decided_at = datetime.now(timezone.utc)
        row.decision_note = body.note

        if choice == "approve" and row.payload:
            try:
                changes = check_changes(row.kind, dict(row.payload))
            except NotApprovable as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
            settings = request.app.state.settings
            for key, value in changes.items():
                setattr(settings, key, value)
            row.applied = True
            row.outcome = "applied: " + ", ".join(f"{k}={v}" for k, v in changes.items())
        elif choice == "approve":
            row.outcome = "approved; no settings change to apply"

        append_audit(db, actor=actor,
                     action="CHANGE_APPROVED" if choice == "approve" else "CHANGE_REJECTED",
                     details={"request_id": row.request_id, "kind": row.kind,
                              "changes": dict(row.payload or {}), "outcome": row.outcome,
                              "role": user.get("r")})
        db.commit()
        db.refresh(row)
    return approval_to_dict(row)
