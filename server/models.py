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
