"""
Event API.

POST /api/v1/events            edge -> server (needs X-API-Key). IDEMPOTENT:
      new event_id                    -> 201 Created, stored
      same event_id + same content    -> 200 OK, duplicate=true (nothing stored twice)
      same event_id + DIFFERENT content -> 409 Conflict (possible tampering or a bug)
GET  /api/v1/events            list with filters + pagination
GET  /api/v1/events/{event_id} one event
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from server.audit import WRITE_LOCK, append_audit
from server.database import get_db
from server.models import Detection, Event, Worker, Zone
from server.security import require_edge_key
from shared.schemas import (ComplianceStatus, EventAck, EventCreate, EventList, EventRead, ReviewStatus)

router = APIRouter(prefix="/api/v1/events", tags=["events"])


def payload_hash(ev: EventCreate) -> str:
    body = ev.model_dump(mode="json")
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def to_read(e: Event) -> EventRead:
    return EventRead(
        event_id=e.event_id, device_id=e.device_id, camera_id=e.camera_id, session_id=e.session_id,
        worker_display_id=e.worker_display_id, zone_id=e.zone_id, status=e.status, reasons=e.reasons or [],
        confidence=e.confidence, frames_in_window=e.frames_in_window, frames_missing=e.frames_missing,
        model_version=e.model_version, rules_version=e.rules_version, occurred_at=e.occurred_at,
        privacy_status=e.privacy_status,
        detections=[{"object_class": d.object_class, "confidence": d.confidence, "bbox": d.bbox,
                     "assigned_to_worker": d.assigned_to_worker} for d in e.detections],
        received_at=e.received_at, review_status=e.review_status, evidence_available=bool(e.evidence_path),
    )


@router.post("", response_model=EventAck, status_code=status.HTTP_201_CREATED)
def create_event(ev: EventCreate, response: Response, db: Session = Depends(get_db),
                 actor: str = Depends(require_edge_key)) -> EventAck:
    with WRITE_LOCK:
        return _create_event(ev, response, db)


def _create_event(ev: EventCreate, response: Response, db: Session) -> EventAck:
    digest = payload_hash(ev)
    existing = db.get(Event, ev.event_id)
    if existing is not None:
        if existing.payload_sha256 != digest:
            raise HTTPException(status.HTTP_409_CONFLICT, "event_id already exists with different content")
        response.status_code = status.HTTP_200_OK
        return EventAck(event_id=ev.event_id, duplicate=True, review_status=existing.review_status)

    if ev.zone_id is not None and db.get(Zone, ev.zone_id) is None:
        raise HTTPException(422,
                            f"unknown zone_id '{ev.zone_id}' - sync zones first (POST /api/v1/zones/sync)")

    # anonymous worker row for (camera, session, display id)
    worker = db.execute(select(Worker).where(Worker.camera_id == ev.camera_id, Worker.session_id == ev.session_id,
                                             Worker.display_id == ev.worker_display_id)).scalar_one_or_none()
    if worker is None:
        worker = Worker(display_id=ev.worker_display_id, camera_id=ev.camera_id, session_id=ev.session_id,
                        first_seen=ev.occurred_at)
        db.add(worker)
        db.flush()
    worker.last_seen = ev.occurred_at
    worker.last_status = ev.status.value

    review = ReviewStatus.PENDING if ev.status == ComplianceStatus.POTENTIAL_VIOLATION else ReviewStatus.NOT_REQUIRED
    row = Event(
        event_id=ev.event_id, device_id=ev.device_id, camera_id=ev.camera_id, session_id=ev.session_id,
        worker_uid=worker.worker_uid, worker_display_id=ev.worker_display_id, zone_id=ev.zone_id,
        status=ev.status.value, reasons=[r.value for r in ev.reasons], confidence=ev.confidence,
        frames_in_window=ev.frames_in_window, frames_missing=ev.frames_missing,
        model_version=ev.model_version, rules_version=ev.rules_version, occurred_at=ev.occurred_at,
        received_at=datetime.now(timezone.utc), payload_sha256=digest,
        privacy_status=ev.privacy_status.value, review_status=review.value,
    )
    row.detections = [Detection(object_class=d.object_class.value, confidence=d.confidence,
                                bbox=d.bbox.model_dump(), assigned_to_worker=d.assigned_to_worker)
                      for d in ev.detections]
    db.add(row)
    append_audit(db, actor=ev.device_id, action="EVENT_CREATED", event_id=ev.event_id,
                 details={"status": ev.status.value, "reasons": [r.value for r in ev.reasons],
                          "zone_id": ev.zone_id, "payload_sha256": digest})
    db.commit()
    return EventAck(event_id=ev.event_id, duplicate=False, review_status=review)


@router.get("", response_model=EventList)
def list_events(db: Session = Depends(get_db),
                status_: Optional[ComplianceStatus] = Query(default=None, alias="status"),
                review_status: Optional[ReviewStatus] = None,
                zone_id: Optional[str] = None,
                worker: Optional[str] = Query(default=None, description="e.g. Worker-017"),
                since: Optional[datetime] = None,
                limit: int = Query(default=50, ge=1, le=500),
                offset: int = Query(default=0, ge=0)) -> EventList:
    q = select(Event)
    if status_:
        q = q.where(Event.status == status_.value)
    if review_status:
        q = q.where(Event.review_status == review_status.value)
    if zone_id:
        q = q.where(Event.zone_id == zone_id)
    if worker:
        q = q.where(Event.worker_display_id == worker)
    if since:
        q = q.where(Event.occurred_at >= since)
    total = db.execute(select(func.count()).select_from(q.subquery())).scalar_one()
    rows = db.execute(q.options(selectinload(Event.detections)).order_by(Event.occurred_at.desc())
                      .limit(limit).offset(offset)).scalars().all()
    return EventList(total=total, limit=limit, offset=offset, items=[to_read(r) for r in rows])


@router.get("/{event_id}", response_model=EventRead)
def get_event(event_id: str, db: Session = Depends(get_db)) -> EventRead:
    e = db.get(Event, event_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "event not found")
    return to_read(e)
