"""
Database tables (SQLAlchemy ORM models).

Design decisions (say these in the viva):
  * NO real identity anywhere. A worker row is an anonymous track inside one edge
    session: "Worker-017 of session 2026-09-22-a". Worker numbers restart every run,
    so the primary key is a generated UUID, not "Worker-017".
  * event_id is generated on the edge and is the PRIMARY KEY -> sending the same
    event twice cannot create two rows (idempotent sync).
  * AI decision (events.status) and human decision (reviews.decision /
    events.review_status) are separate columns. A reviewer never overwrites what
    the AI said, so we can later measure how often the AI was wrong.
  * Detections are stored ONLY for event frames, not for every video frame
    (data minimisation).
  * audit_logs is append-only and hash-chained (hash = SHA-256 of this row + previous hash).
  * The sync queue lives on the EDGE in its own SQLite file (Day 3), not here.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Worker(Base):
    __tablename__ = "workers"
    __table_args__ = (UniqueConstraint("camera_id", "session_id", "display_id", name="uq_worker_session"),)

    worker_uid: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    display_id: Mapped[str] = mapped_column(String(16))          # "Worker-017"
    camera_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(64))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    events: Mapped[List["Event"]] = relationship(back_populates="worker")


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String(64), primary_key=True)   # "welding-a"
    zone_name: Mapped[str] = mapped_column(String(128))
    zone_type: Mapped[str] = mapped_column(String(16))                   # GENERAL / CAUTION / RESTRICTED
    camera_id: Mapped[str] = mapped_column(String(64))
    polygon: Mapped[list] = mapped_column(JSON)                          # [[x, y], ...] in pixels
    required_ppe: Mapped[list] = mapped_column(JSON, default=list)       # ["helmet", "vest"]
    authorization_required: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)            # overlapping zones: higher wins
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    config_version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_occurred_at", "occurred_at"),
        Index("ix_events_status_review", "status", "review_status"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # from the edge = idempotency key
    device_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(64))
    worker_uid: Mapped[Optional[str]] = mapped_column(ForeignKey("workers.worker_uid"), nullable=True)
    worker_display_id: Mapped[str] = mapped_column(String(16))
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.zone_id"), nullable=True)

    # ---- AI decision (never edited after insert) ----
    status: Mapped[str] = mapped_column(String(32))
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float)
    frames_in_window: Mapped[int] = mapped_column(Integer)
    frames_missing: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(64))
    rules_version: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    payload_sha256: Mapped[str] = mapped_column(String(64))   # detects "same id, different content"

    # ---- Evidence / privacy ----
    privacy_status: Mapped[str] = mapped_column(String(32))
    evidence_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    evidence_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # ---- Human decision summary (details live in reviews) ----
    review_status: Mapped[str] = mapped_column(String(16), default="PENDING")

    worker: Mapped[Optional[Worker]] = relationship(back_populates="events")
    detections: Mapped[List["Detection"]] = relationship(back_populates="event", cascade="all, delete-orphan")
    reviews: Mapped[List["Review"]] = relationship(back_populates="event")


class Detection(Base):
    __tablename__ = "detections"

    detection_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id", ondelete="CASCADE"))
    object_class: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[dict] = mapped_column(JSON)              # {"x1":..,"y1":..,"x2":..,"y2":..}
    assigned_to_worker: Mapped[bool] = mapped_column(Boolean, default=True)

    event: Mapped[Event] = relationship(back_populates="detections")


class Review(Base):
    __tablename__ = "reviews"

    review_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"))
    reviewer: Mapped[str] = mapped_column(String(64))       # username, never the worker's name
    decision: Mapped[str] = mapped_column(String(16))       # APPROVED / DISMISSED
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    event: Mapped[Event] = relationship(back_populates="reviews")


class CopilotSuggestion(Base):
    """
    One action the copilot PROPOSED and a human decided on (shared/copilot.py).

    Deliberately separate from `reviews`: a review is a judgement about whether the AI
    was right, a suggestion is a proposal about what to DO next. Keeping them apart means
    we can later measure how often supervisors reject the copilot's advice - which is the
    number that tells you whether the copilot is worth having.

    `status` moves PENDING -> APPROVED or REJECTED, once. There is no path that sets it
    without an actor, and no path that executes anything while it is PENDING.
    """
    __tablename__ = "copilot_suggestions"
    __table_args__ = (Index("ix_copilot_status", "status", "created_at"),)

    suggestion_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[str] = mapped_column(String(128))         # "image:site.jpg", "camera:0"
    subject: Mapped[str] = mapped_column(String(32))         # the anonymous person label
    event_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    action_id: Mapped[str] = mapped_column(String(48))       # ALWAYS one of shared.copilot.ACTIONS
    rule: Mapped[str] = mapped_column(String(8))             # which rule fired: R2..R5
    reason: Mapped[str] = mapped_column(Text)
    severity: Mapped[int] = mapped_column(Integer, default=1)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    decided_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # what executing it did


class DeviceTelemetry(Base):
    """
    A heartbeat from an edge device. Health only - never a frame, a detection or a worker.

    Why it is stored at all: a camera that stops sending events looks exactly like a site
    where nobody breaks the rules. `queue_depth` and `queue_dropped` come from the edge
    outbox, so an outage reports itself once the device reconnects.
    """
    __tablename__ = "device_telemetry"
    __table_args__ = (Index("ix_telemetry_device_time", "device_id", "occurred_at"),)

    telemetry_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    fps: Mapped[float] = mapped_column(Float, default=0.0)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    queue_dropped: Mapped[int] = mapped_column(Integer, default=0)
    uptime_s: Mapped[float] = mapped_column(Float, default=0.0)

    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rules_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    device_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class ApprovalRequest(Base):
    """
    A change that a person must approve before it takes effect.

    The copilot's five safety actions have their own table; this one is for the settings a
    site runs on - PPE requirements, the confidence threshold, privacy switches, retention,
    deleting evidence, exporting footage. Those are not safety decisions, they are the
    configuration that safety decisions are made against, which is exactly why they should
    not change quietly.

    PENDING -> APPROVED or REJECTED, once, by a named person, with the reason kept.
    """
    __tablename__ = "approval_requests"
    __table_args__ = (Index("ix_approval_status", "status", "created_at"),)

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    kind: Mapped[str] = mapped_column(String(48))          # one of shared.approvals.KINDS
    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    requested_by: Mapped[str] = mapped_column(String(64))
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    decided_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class SafetyEvent(Base):
    """
    The structured record of one meaningful thing that happened.

    This is the source of truth for alerts, history, the event detail panel, analytics and
    export. Everything else is either raw input (frames, which are discarded) or a flat
    append-only trail (violations.jsonl, decisions.jsonl) kept for evidence and counting.

    Why not reuse `events`: that table is the edge->server SYNC contract, keyed by an id the
    edge generates, with the fields the sync protocol needs. This one is the operational
    record the dashboard works with. Conflating them would mean one schema serving two
    masters, and the sync contract would change every time the interface wanted a column.

    Deduplication lives in the columns: `dedupe_key` identifies "the same problem, same
    person, same camera", and first_seen/last_seen/occurrences extend one record instead of
    writing a hundred.

    Deletion is soft by default (`deleted_at`), because an audit record that one click can
    destroy is not an audit record. Permanent removal is a separate, explicit action.
    """
    __tablename__ = "safety_events"
    __table_args__ = (
        Index("ix_safety_events_time", "occurred_at"),
        Index("ix_safety_events_open", "status", "deleted_at"),
        Index("ix_safety_events_dedupe", "dedupe_key", "status"),
    )

    event_id: Mapped[str] = mapped_column(String(32), primary_key=True)   # EVT-000001
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now,
                                                 onupdate=_now)

    camera_id: Mapped[str] = mapped_column(String(64))       # "CAM-01" / "image:site.jpg"
    person_id: Mapped[str] = mapped_column(String(16))       # anonymous, session-scoped
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    event_type: Mapped[str] = mapped_column(String(24))      # shared.events.EVENT_TYPES
    decision: Mapped[str] = mapped_column(String(8))         # GO / REVIEW / STOP
    severity: Mapped[str] = mapped_column(String(8))         # LOW / MEDIUM / HIGH / CRITICAL
    rule_id: Mapped[str] = mapped_column(String(48))
    reason: Mapped[str] = mapped_column(Text)

    required_ppe: Mapped[list] = mapped_column(JSON, default=list)
    detected_ppe: Mapped[list] = mapped_column(JSON, default=list)   # [{type, confidence}]
    missing_ppe: Mapped[list] = mapped_column(JSON, default=list)
    uncertain_ppe: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # evidence: PATHS only, and only inside the configured folders
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    recording_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # deduplication of a continuing problem
    dedupe_key: Mapped[str] = mapped_column(String(160), default="")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    # the human side
    status: Mapped[str] = mapped_column(String(16), default="NEW")   # NEW/ACKNOWLEDGED/CLOSED
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True),
                                                               nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sync_status: Mapped[str] = mapped_column(String(16), default="LOCAL")
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True),
                                                          nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    actor: Mapped[str] = mapped_column(String(64))          # "edge-01", "reviewer_1", "copilot", "system"
    action: Mapped[str] = mapped_column(String(48))         # EVENT_CREATED, REVIEW_SUBMITTED, POLICY_CHANGED ...
    event_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64))
    row_hash: Mapped[str] = mapped_column(String(64), unique=True)
