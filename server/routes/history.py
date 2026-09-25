"""
Event history: the structured record, and everything a person does with it.

    GET    /api/v1/history                 filtered list (the History table)
    GET    /api/v1/history/stats           counts, computed from the rows
    GET    /api/v1/history/export          CSV or JSON of the same filtered set
    GET    /api/v1/history/{event_id}      one event, in full (the detail panel)
    POST   /api/v1/history/{event_id}/acknowledge    supervisor
    POST   /api/v1/history/{event_id}/close          supervisor - makes it retention-eligible
    DELETE /api/v1/history/{event_id}                supervisor - SOFT delete
    POST   /api/v1/history/{event_id}/purge          admin - permanent, with its evidence
    POST   /api/v1/history/clear                     admin - bulk, and it insists on confirm

ACTIVE ALERTS ARE NOT HISTORY
    They are the same rows seen through a different filter: `status=NEW`. Acknowledging one
    takes it off the alert list and leaves it in history, which is the distinction that makes
    "clear the alerts" safe.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server import event_store
from server.audit import WRITE_LOCK, append_audit
from server.auth import ADMIN, SUPERVISOR
from server.database import get_db
from server.http_status import UNPROCESSABLE
from server.routes.auth import require_role
from shared.events import EVENT_TYPES, SEVERITIES, STATUSES, to_csv

router = APIRouter(prefix="/api/v1/history", tags=["history"])


def evidence_roots(request: Request) -> List[Path]:
    """The only folders a delete may ever touch."""
    settings = request.app.state.settings
    return [Path(settings.evidence_dir), Path(settings.recordings_dir),
            Path(settings.snapshots_masked_dir)]


class Decision(BaseModel):
    note: Optional[str] = Field(default=None, max_length=500)


class ClearRequest(BaseModel):
    confirm: str = Field(..., description="must be the word CLEAR")
    status: Optional[str] = Field(default="CLOSED",
                                  description="which events to clear; CLOSED by default")
    before: Optional[datetime] = None
    camera_id: Optional[str] = None
    remove_evidence: bool = True


def parse_since(hours: Optional[float], since: Optional[datetime]) -> Optional[datetime]:
    if since is not None:
        return since
    if hours:
        return datetime.now(timezone.utc) - timedelta(hours=float(hours))
    return None


@router.get("")
def history(request: Request, db: Session = Depends(get_db),
            camera_id: Optional[str] = None,
            event_type: Optional[str] = Query(default=None),
            decision: Optional[str] = None, severity: Optional[str] = None,
            event_status: Optional[str] = Query(default=None, alias="status"),
            person_id: Optional[str] = None,
            hours: Optional[float] = Query(default=None, ge=0, le=24 * 365),
            since: Optional[datetime] = None, until: Optional[datetime] = None,
            unacknowledged_only: bool = False,
            limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0)):
    for value, allowed, name in ((event_type, EVENT_TYPES, "event_type"),
                                 (severity, SEVERITIES, "severity"),
                                 (event_status, STATUSES, "status"),
                                 (decision, ("GO", "REVIEW", "STOP"), "decision")):
        if value and value.upper() not in allowed:
            raise HTTPException(UNPROCESSABLE,
                                detail=f"{name} must be one of {sorted(allowed)}")

    events = event_store.list_events(
        db, camera_id=camera_id, event_type=event_type, decision=decision, severity=severity,
        status=event_status, person_id=person_id, since=parse_since(hours, since),
        until=until, unacknowledged_only=unacknowledged_only, limit=limit, offset=offset)
    return {"events": [event_store.to_dict(event) for event in events],
            "count": len(events), "offset": offset, "limit": limit}


@router.get("/stats")
def stats(request: Request, db: Session = Depends(get_db),
          hours: Optional[float] = Query(default=None, ge=0, le=24 * 365)):
    return event_store.statistics(db, since=parse_since(hours, None))


@router.get("/export")
def export(request: Request, db: Session = Depends(get_db),
           fmt: str = Query(default="csv", pattern="^(csv|json)$"),
           camera_id: Optional[str] = None, decision: Optional[str] = None,
           event_status: Optional[str] = Query(default=None, alias="status"),
           hours: Optional[float] = Query(default=None, ge=0, le=24 * 365),
           limit: int = Query(default=5000, ge=1, le=50000)):
    """
    History as data. Evidence files are NOT part of an export: a spreadsheet of events is a
    record, a folder of pictures of people is a different decision with different rules.
    """
    events = event_store.list_events(db, camera_id=camera_id, decision=decision,
                                     status=event_status, since=parse_since(hours, None),
                                     limit=limit)
    rows = event_store.export_rows(events)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if fmt == "json":
        return Response(content=json.dumps({"events": rows, "count": len(rows)}, indent=2),
                        media_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="events_export_{stamp}.json"'})
    return Response(content=to_csv(rows), media_type="text/csv",
                    headers={"Content-Disposition":
                             f'attachment; filename="events_export_{stamp}.csv"'})


@router.get("/{event_id}")
def detail(event_id: str, db: Session = Depends(get_db)):
    event = event_store.get_event(db, event_id)
    if event is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such event")
    return event_store.to_dict(event)


@router.post("/{event_id}/acknowledge")
def acknowledge(event_id: str, body: Decision = Body(default=Decision()),
                db: Session = Depends(get_db),
                user: dict = Depends(require_role(SUPERVISOR))):
    actor = user.get("n") or user["u"]
    with WRITE_LOCK:
        event = event_store.acknowledge(db, event_id, actor, body.note)
        if event is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such event")
        append_audit(db, actor=actor, action="EVENT_ACKNOWLEDGED", event_id=event_id,
                     details={"decision": event.decision, "note": body.note})
        db.commit()
        db.refresh(event)
    return event_store.to_dict(event)


@router.post("/{event_id}/close")
def close(event_id: str, body: Decision = Body(default=Decision()),
          db: Session = Depends(get_db), user: dict = Depends(require_role(SUPERVISOR))):
    """Closing says a person has dealt with it - and only then may retention remove it."""
    actor = user.get("n") or user["u"]
    with WRITE_LOCK:
        event = event_store.close(db, event_id, actor)
        if event is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such event")
        append_audit(db, actor=actor, action="EVENT_CLOSED", event_id=event_id,
                     details={"note": body.note})
        db.commit()
        db.refresh(event)
    return event_store.to_dict(event)


@router.delete("/{event_id}")
def remove(event_id: str, db: Session = Depends(get_db),
           user: dict = Depends(require_role(SUPERVISOR))):
    """Soft delete: it leaves the views, the record and its evidence stay."""
    actor = user.get("n") or user["u"]
    with WRITE_LOCK:
        event = event_store.soft_delete(db, event_id)
        if event is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND,
                                detail="no such event, or it is already deleted")
        append_audit(db, actor=actor, action="EVENT_DELETED", event_id=event_id,
                     details={"kind": "soft", "recoverable": True})
        db.commit()
    return {"event_id": event_id, "deleted": True, "permanent": False,
            "note": "The record is hidden from history. Evidence files are untouched."}


@router.post("/{event_id}/purge")
def purge(event_id: str, request: Request, db: Session = Depends(get_db),
          user: dict = Depends(require_role(ADMIN)),
          remove_evidence: bool = Body(default=True, embed=True)):
    """Permanent: the row and its own evidence files. Admin only, and it is not reversible."""
    actor = user.get("n") or user["u"]
    with WRITE_LOCK:
        event = event_store.get_event(db, event_id, include_deleted=True)
        if event is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such event")
        summary = {"decision": event.decision, "camera_id": event.camera_id,
                   "occurred_at": str(event.occurred_at)}
        result = event_store.purge(db, event_id, evidence_roots(request),
                                   remove_evidence=remove_evidence)
        # the audit row survives the event it describes - that is the point of an audit row
        append_audit(db, actor=actor, action="EVENT_PURGED", event_id=event_id,
                     details={**summary, "files_deleted": result["files_deleted"],
                              "files_skipped": result["files_skipped"]})
        db.commit()
    return {**result, "permanent": True}


@router.post("/clear")
def clear(body: ClearRequest, request: Request, db: Session = Depends(get_db),
          user: dict = Depends(require_role(ADMIN))):
    """
    Bulk removal, and it argues with you first: the word CLEAR must be in the body, and by
    default only CLOSED events are eligible. Clearing what nobody has reviewed is possible
    but has to be asked for explicitly, by status.
    """
    if body.confirm.strip().upper() != "CLEAR":
        raise HTTPException(UNPROCESSABLE,
                            detail="send confirm='CLEAR' to remove history permanently")
    actor = user.get("n") or user["u"]
    events = event_store.list_events(db, status=body.status, camera_id=body.camera_id,
                                     until=body.before, include_deleted=True, limit=10 ** 6)
    removed, files = [], []
    with WRITE_LOCK:
        for event in events:
            result = event_store.purge(db, event.event_id, evidence_roots(request),
                                       remove_evidence=body.remove_evidence)
            removed.append(event.event_id)
            files.extend(result["files_deleted"])
        append_audit(db, actor=actor, action="HISTORY_CLEARED",
                     details={"removed": len(removed), "status": body.status,
                              "camera_id": body.camera_id,
                              "before": str(body.before) if body.before else None,
                              "files_deleted": len(files)})
        db.commit()
    return {"removed": removed, "removed_count": len(removed), "files_deleted": files,
            "permanent": True}
