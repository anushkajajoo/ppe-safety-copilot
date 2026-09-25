"""
Reading and writing structured safety events.

This is the thin database layer over `shared/events.py`: the rules live there, the rows live
here. Everything the dashboard shows - active alerts, history, the detail panel, statistics,
export - comes through this module, so there is one place where "what happened" is decided.

THREE THINGS IT IS CAREFUL ABOUT
    1. Deduplication. A continuing violation extends the open record (last_seen, duration,
       occurrences) instead of creating another one. Without this, ten seconds of a worker
       without a helmet is a hundred rows and a useless history.
    2. Deletion. Soft by default - `deleted_at` is set and the row leaves the normal views.
       Permanent removal is a separate call, and it only ever unlinks files that are inside
       the configured evidence folders AND recorded on the event itself.
    3. Protection. Retention never touches an event somebody still has to look at.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from server.models import SafetyEvent
from shared.events import (ACKNOWLEDGED, CLOSED, LOCAL, NEW, dedupe_key, eligible_for_cleanup,
                           event_type_for, explain, is_protected, person_label, severity_for,
                           should_merge, signature_of)

EVIDENCE_FIELDS = ("snapshot_path", "recording_path")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes; comparisons need them tz-aware."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def next_event_id(db: Session) -> str:
    """EVT-000001, EVT-000002 ... Sequential so a person can read and say them aloud."""
    total = db.execute(select(func.count()).select_from(SafetyEvent)).scalar() or 0
    return f"EVT-{total + 1:06d}"


# ------------------------------------------------------------------- writing
def record_event(db: Session, *, camera_id: str, person_id: Optional[str], decision: str,
                 required: Sequence[str], detected: Dict[str, float],
                 missing: Sequence[str], uncertain: Sequence[str], threshold: float,
                 rule_id: str, confidence: float = 0.0, zone_id: Optional[str] = None,
                 zone_type: str = "GENERAL", session_id: Optional[str] = None,
                 snapshot_path: Optional[str] = None, recording_path: Optional[str] = None,
                 event_type: Optional[str] = None, reason: Optional[str] = None,
                 merge_window_s: float = 30.0, person_index: int = 1,
                 now: Optional[datetime] = None) -> Tuple[SafetyEvent, bool]:
    """
    Store one observation. Returns (event, created) - `created` is False when an existing
    open event was extended instead.

    The caller decides WHETHER something is worth recording (debounce, policy); this decides
    whether it is a new record or more of an old one.
    """
    moment = now or _now()
    kind = event_type or event_type_for(decision)
    label = person_label(person_id, person_index)
    signature = signature_of(kind, missing, uncertain)
    key = dedupe_key(camera_id, label, signature)

    open_event = db.execute(
        select(SafetyEvent)
        .where(SafetyEvent.dedupe_key == key,
               SafetyEvent.deleted_at.is_(None),
               SafetyEvent.status != CLOSED)
        .order_by(SafetyEvent.last_seen_at.desc())
        .limit(1)).scalars().first()

    if open_event is not None and should_merge(_aware(open_event.last_seen_at).timestamp(),
                                               moment.timestamp(), merge_window_s):
        open_event.last_seen_at = moment
        open_event.occurrences += 1
        open_event.duration_s = round(
            moment.timestamp() - _aware(open_event.first_seen_at).timestamp(), 1)
        open_event.updated_at = moment
        # keep the strongest evidence we have, never overwrite a path with nothing
        if snapshot_path and not open_event.snapshot_path:
            open_event.snapshot_path = snapshot_path
        if recording_path and not open_event.recording_path:
            open_event.recording_path = recording_path
        db.flush()
        return open_event, False

    event = SafetyEvent(
        event_id=next_event_id(db), occurred_at=moment, created_at=moment, updated_at=moment,
        camera_id=camera_id, person_id=label, session_id=session_id,
        event_type=kind, decision=decision,
        severity=severity_for(decision, missing, zone_type, kind),
        rule_id=rule_id,
        reason=reason or explain(missing, uncertain, detected, threshold, decision),
        required_ppe=list(required),
        detected_ppe=[{"type": name, "confidence": round(float(value), 3)}
                      for name, value in sorted(detected.items())],
        missing_ppe=list(missing), uncertain_ppe=list(uncertain),
        confidence=round(float(confidence), 3), threshold=float(threshold), zone_id=zone_id,
        snapshot_path=snapshot_path, recording_path=recording_path,
        dedupe_key=key, first_seen_at=moment, last_seen_at=moment, occurrences=1,
        duration_s=0.0, status=NEW, acknowledged=False, sync_status=LOCAL)
    db.add(event)
    db.flush()
    return event, True


def attach_evidence(db: Session, event_id: str, snapshot_path: Optional[str] = None,
                    recording_path: Optional[str] = None) -> Optional[SafetyEvent]:
    """A clip finishes a few seconds after the event that triggered it."""
    event = db.get(SafetyEvent, event_id)
    if event is None:
        return None
    if snapshot_path:
        event.snapshot_path = snapshot_path
    if recording_path:
        event.recording_path = recording_path
    event.updated_at = _now()
    db.flush()
    return event


def acknowledge(db: Session, event_id: str, actor: str,
                note: Optional[str] = None) -> Optional[SafetyEvent]:
    event = db.get(SafetyEvent, event_id)
    if event is None or event.deleted_at is not None:
        return None
    if event.acknowledged:
        return event                       # already seen; acknowledging twice changes nothing
    event.acknowledged = True
    event.status = ACKNOWLEDGED
    event.acknowledged_by = actor
    event.acknowledged_at = _now()
    event.note = note or event.note
    event.updated_at = _now()
    db.flush()
    return event


def close(db: Session, event_id: str, actor: str) -> Optional[SafetyEvent]:
    """Closing is what makes an event eligible for retention. It is a human act."""
    event = db.get(SafetyEvent, event_id)
    if event is None or event.deleted_at is not None:
        return None
    event.status = CLOSED
    event.updated_at = _now()
    if not event.acknowledged:
        event.acknowledged = True
        event.acknowledged_by = actor
        event.acknowledged_at = _now()
    db.flush()
    return event


# ------------------------------------------------------------------- reading
def list_events(db: Session, *, camera_id: Optional[str] = None,
                event_type: Optional[str] = None, decision: Optional[str] = None,
                severity: Optional[str] = None, status: Optional[str] = None,
                person_id: Optional[str] = None, since: Optional[datetime] = None,
                until: Optional[datetime] = None, unacknowledged_only: bool = False,
                include_deleted: bool = False, limit: int = 100,
                offset: int = 0) -> List[SafetyEvent]:
    query = select(SafetyEvent)
    if not include_deleted:
        query = query.where(SafetyEvent.deleted_at.is_(None))
    if camera_id:
        query = query.where(SafetyEvent.camera_id == camera_id)
    if event_type:
        query = query.where(SafetyEvent.event_type == event_type.upper())
    if decision:
        query = query.where(SafetyEvent.decision == decision.upper())
    if severity:
        query = query.where(SafetyEvent.severity == severity.upper())
    if status:
        query = query.where(SafetyEvent.status == status.upper())
    if person_id:
        query = query.where(SafetyEvent.person_id == person_id)
    if since:
        query = query.where(SafetyEvent.occurred_at >= since)
    if until:
        query = query.where(SafetyEvent.occurred_at <= until)
    if unacknowledged_only:
        query = query.where(SafetyEvent.acknowledged.is_(False))
    query = query.order_by(SafetyEvent.occurred_at.desc()).limit(limit).offset(offset)
    return list(db.execute(query).scalars().all())


def get_event(db: Session, event_id: str, include_deleted: bool = False) -> Optional[SafetyEvent]:
    event = db.get(SafetyEvent, event_id)
    if event is None:
        return None
    if event.deleted_at is not None and not include_deleted:
        return None
    return event


def to_dict(event: SafetyEvent, evidence_root: Optional[Path] = None) -> dict:
    """
    The event as the API and the dashboard see it.

    Evidence is reported as "is there a file" plus its name - never a full filesystem path,
    which is both a privacy leak and an invitation to try fetching it directly.
    """
    def evidence(path: Optional[str]) -> Optional[dict]:
        if not path:
            return None
        name = Path(path).name
        return {"name": name, "available": Path(path).exists()}

    return {
        "event_id": event.event_id,
        "occurred_at": event.occurred_at, "created_at": event.created_at,
        "updated_at": event.updated_at,
        "camera_id": event.camera_id, "person_id": event.person_id,
        "session_id": event.session_id,
        "event_type": event.event_type, "decision": event.decision,
        "severity": event.severity, "rule_id": event.rule_id, "reason": event.reason,
        "required_ppe": event.required_ppe or [], "detected_ppe": event.detected_ppe or [],
        "missing_ppe": event.missing_ppe or [], "uncertain_ppe": event.uncertain_ppe or [],
        "confidence": event.confidence, "threshold": event.threshold,
        "zone_id": event.zone_id,
        "snapshot": evidence(event.snapshot_path), "recording": evidence(event.recording_path),
        "first_seen_at": event.first_seen_at, "last_seen_at": event.last_seen_at,
        "occurrences": event.occurrences, "duration_s": event.duration_s,
        "status": event.status, "acknowledged": event.acknowledged,
        "acknowledged_by": event.acknowledged_by, "acknowledged_at": event.acknowledged_at,
        "note": event.note, "sync_status": event.sync_status,
        "deleted": event.deleted_at is not None,
    }


def export_rows(events: Iterable[SafetyEvent]) -> List[dict]:
    """Flat rows for CSV/JSON export - no evidence paths, by design."""
    rows = []
    for event in events:
        rows.append({
            "event_id": event.event_id,
            "occurred_at": _aware(event.occurred_at).isoformat(timespec="seconds"),
            "camera_id": event.camera_id, "person_id": event.person_id,
            "event_type": event.event_type, "decision": event.decision,
            "severity": event.severity, "status": event.status,
            "required_ppe": list(event.required_ppe or []),
            "missing_ppe": list(event.missing_ppe or []),
            "uncertain_ppe": list(event.uncertain_ppe or []),
            "confidence": event.confidence, "rule_id": event.rule_id,
            "reason": event.reason, "acknowledged_by": event.acknowledged_by,
            "acknowledged_at": (_aware(event.acknowledged_at).isoformat(timespec="seconds")
                                if event.acknowledged_at else None),
            "sync_status": event.sync_status,
        })
    return rows


def statistics(db: Session, since: Optional[datetime] = None) -> dict:
    """Every number counted from the rows. Nothing here has a default of "looks good"."""
    events = list_events(db, since=since, limit=10 ** 6)
    by_decision: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_camera: Dict[str, int] = {}
    by_date: Dict[str, int] = {}
    by_item = {"helmet": 0, "vest": 0, "mask": 0}

    for event in events:
        by_decision[event.decision] = by_decision.get(event.decision, 0) + 1
        by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
        by_camera[event.camera_id] = by_camera.get(event.camera_id, 0) + 1
        day = _aware(event.occurred_at).date().isoformat()
        by_date[day] = by_date.get(day, 0) + 1
        for item in (event.missing_ppe or []):
            if item in by_item:
                by_item[item] += 1

    total = len(events)
    compliant = by_decision.get("GO", 0)
    return {
        "total_events": total,
        "go": compliant, "review": by_decision.get("REVIEW", 0),
        "stop": by_decision.get("STOP", 0),
        "compliance_pct": round(compliant / total * 100, 1) if total else None,
        "by_type": by_type, "by_camera": by_camera, "by_date": by_date,
        "violations_by_item": by_item,
        "acknowledged": sum(1 for e in events if e.acknowledged),
        "unacknowledged": sum(1 for e in events if not e.acknowledged),
        "open_alerts": sum(1 for e in events if e.status == NEW),
    }


# ------------------------------------------------------------------ deleting
def soft_delete(db: Session, event_id: str) -> Optional[SafetyEvent]:
    """The normal delete: the row leaves the views, the record still exists."""
    event = db.get(SafetyEvent, event_id)
    if event is None or event.deleted_at is not None:
        return None
    event.deleted_at = _now()
    event.updated_at = _now()
    db.flush()
    return event


def safe_evidence_path(candidate: Optional[str], roots: Sequence[Path]) -> Optional[Path]:
    """
    Turn a stored path into one that may actually be deleted, or None.

    Two conditions, both required: the path is recorded on the event itself, and it resolves
    INSIDE one of the configured evidence folders. A path from anywhere else - a traversal,
    a symlink out, an absolute path someone wrote into the database - is refused. This is the
    only place in the project that deletes a file, so this is the only place that has to be
    right.
    """
    if not candidate:
        return None
    try:
        path = Path(candidate).resolve()
    except OSError:
        return None
    if not path.is_file():
        return None
    for root in roots:
        try:
            resolved_root = Path(root).resolve()
        except OSError:
            continue
        if path == resolved_root:
            continue
        if resolved_root in path.parents:
            return path
    return None


def purge(db: Session, event_id: str, roots: Sequence[Path],
          remove_evidence: bool = True) -> dict:
    """
    Permanent removal: the row and, if asked, its own evidence files. Explicit by design -
    nothing in the system calls this except an admin action and the retention job.
    """
    event = db.get(SafetyEvent, event_id)
    if event is None:
        return {"event_id": event_id, "removed": False, "files_deleted": [],
                "files_skipped": [], "reason": "no such event"}

    deleted, skipped = [], []
    if remove_evidence:
        for field in EVIDENCE_FIELDS:
            stored = getattr(event, field)
            path = safe_evidence_path(stored, roots)
            if path is None:
                if stored:
                    skipped.append(stored)     # outside the evidence folders, or already gone
                continue
            try:
                os.remove(path)
                deleted.append(path.name)
            except OSError as exc:
                skipped.append(f"{path.name}: {exc}")

    db.delete(event)
    db.flush()
    return {"event_id": event_id, "removed": True, "files_deleted": deleted,
            "files_skipped": skipped, "reason": "permanently removed"}


def cleanup(db: Session, roots: Sequence[Path], retention_days: float,
            now: Optional[datetime] = None, dry_run: bool = False) -> dict:
    """
    Retention: remove CLOSED and already soft-deleted events past their period, and their
    evidence. NEW and ACKNOWLEDGED events are protected however old they are - an event
    nobody has looked at is not stale, it is overdue.
    """
    moment = (now or _now()).timestamp()
    candidates = db.execute(select(SafetyEvent)).scalars().all()
    removed, protected, kept = [], 0, 0
    files_deleted: List[str] = []

    for event in candidates:
        occurred = _aware(event.occurred_at).timestamp()
        if eligible_for_cleanup(event.status, occurred, moment, retention_days,
                                deleted=event.deleted_at is not None):
            if dry_run:
                removed.append(event.event_id)
                continue
            result = purge(db, event.event_id, roots)
            removed.append(event.event_id)
            files_deleted.extend(result["files_deleted"])
        elif is_protected(event.status) and event.deleted_at is None:
            protected += 1
        else:
            kept += 1

    return {"removed": removed, "removed_count": len(removed), "files_deleted": files_deleted,
            "protected": protected, "kept": kept, "retention_days": retention_days,
            "dry_run": dry_run}
