"""
Device telemetry: how the edge tells the server it is alive, and what it could not send.

    POST /api/v1/telemetry     one heartbeat from a device (needs X-API-Key)
    GET  /api/v1/telemetry     the most recent heartbeat per device, newest first

WHY THIS EXISTS
    A safety camera that stops sending events looks exactly like a site where nobody breaks
    the rules. Heartbeats are how the two are told apart: if a device stops reporting, the
    dashboard can say "no data from CAM-01 for 12 minutes" instead of implying all is well.

WHAT IT CARRIES - AND WHAT IT DOES NOT
    Device health only: frame rate, queue depth, dropped count, model and rules version,
    uptime, last error. No frames, no detections, no worker labels. A heartbeat is about the
    machine, not about anybody in front of it.

    Heartbeats travel through the same store-and-forward outbox as events, so a device that
    was offline reports the gap itself once it reconnects - the outage becomes a record
    rather than a silence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.database import get_db
from server.models import DeviceTelemetry
from server.security import require_edge_key

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


class Heartbeat(BaseModel):
    device_id: str = Field(..., max_length=64)
    camera_id: Optional[str] = Field(default=None, max_length=64)
    session_id: Optional[str] = Field(default=None, max_length=64)
    fps: float = Field(default=0.0, ge=0)
    frames_processed: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0, description="events waiting in the edge outbox")
    queue_dropped: int = Field(default=0, ge=0, description="events dropped because the queue was full")
    uptime_s: float = Field(default=0.0, ge=0)
    model_version: Optional[str] = Field(default=None, max_length=64)
    rules_version: Optional[str] = Field(default=None, max_length=32)
    device_label: Optional[str] = Field(default=None, max_length=64, description="cpu / cuda:0")
    last_error: Optional[str] = Field(default=None, max_length=255)
    occurred_at: Optional[datetime] = None


@router.post("", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_edge_key)])
def post_heartbeat(beat: Heartbeat, db: Session = Depends(get_db)):
    row = DeviceTelemetry(
        device_id=beat.device_id, camera_id=beat.camera_id, session_id=beat.session_id,
        fps=beat.fps, frames_processed=beat.frames_processed, queue_depth=beat.queue_depth,
        queue_dropped=beat.queue_dropped, uptime_s=beat.uptime_s,
        model_version=beat.model_version, rules_version=beat.rules_version,
        device_label=beat.device_label, last_error=beat.last_error,
        occurred_at=beat.occurred_at or datetime.now(timezone.utc))
    db.add(row)
    db.commit()
    return {"stored": True, "device_id": row.device_id, "telemetry_id": row.telemetry_id}


@router.get("")
def list_heartbeats(db: Session = Depends(get_db),
                    limit: int = Query(default=50, ge=1, le=500),
                    stale_after_s: float = Query(default=120.0, gt=0)):
    """
    Newest first, with `stale` computed here rather than stored: staleness is a question
    about *now*, and a row that was fresh when written is not fresh an hour later.
    """
    rows = db.execute(select(DeviceTelemetry)
                      .order_by(DeviceTelemetry.telemetry_id.desc())
                      .limit(limit)).scalars().all()
    now = datetime.now(timezone.utc)
    out = []
    for row in rows:
        occurred = row.occurred_at
        if occurred is not None and occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=timezone.utc)
        age = (now - occurred).total_seconds() if occurred else None
        out.append({"telemetry_id": row.telemetry_id, "device_id": row.device_id,
                    "camera_id": row.camera_id, "session_id": row.session_id,
                    "fps": row.fps, "frames_processed": row.frames_processed,
                    "queue_depth": row.queue_depth, "queue_dropped": row.queue_dropped,
                    "uptime_s": row.uptime_s, "model_version": row.model_version,
                    "rules_version": row.rules_version, "device_label": row.device_label,
                    "last_error": row.last_error, "occurred_at": row.occurred_at,
                    "age_s": round(age, 1) if age is not None else None,
                    "stale": bool(age is not None and age > stale_after_s)})
    return {"telemetry": out, "count": len(out), "stale_after_s": stale_after_s}
