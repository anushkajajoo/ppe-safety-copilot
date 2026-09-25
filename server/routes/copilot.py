"""
The human-governed copilot API.

    GET  /api/v1/copilot/catalog        every action the copilot may EVER propose
    POST /api/v1/copilot/suggestions    create proposals from a compliance verdict
    GET  /api/v1/copilot/suggestions    the queue (default: still awaiting a decision)
    POST /api/v1/copilot/suggestions/{id}/decide   a named person approves or rejects
    GET  /api/v1/copilot/outbox         what approved actions actually produced

THE RULE THIS MODULE ENFORCES
    Nothing happens without a human. A suggestion is inert until `decide` is called with
    an actor's name; only then is the action executed, and only if it is on the allow-list
    (shared/copilot.check_action). Both the proposal and the decision are appended to the
    hash-chained audit log, so the trail survives even if this table is edited later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from server.http_status import UNPROCESSABLE
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.audit import WRITE_LOCK, append_audit
from server.auth import SUPERVISOR
from server.config import PROJECT_ROOT
from server.database import get_db
from server.models import CopilotSuggestion
from server.routes.auth import require_role
from shared.copilot import NotAllowed, check_action, catalog, suggest

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])

PENDING, APPROVED, REJECTED = "PENDING", "APPROVED", "REJECTED"

OUTBOX = PROJECT_ROOT / "data" / "outbox.jsonl"


# ----------------------------------------------------------------- request bodies
class SuggestRequest(BaseModel):
    source: str = Field(..., max_length=128)
    person_id: str = Field(..., max_length=32)
    compliance_status: str
    missing: List[str] = []
    zone_type: str = "GENERAL"
    zone_name: Optional[str] = None
    confidence: float = 0.0
    event_id: Optional[str] = None
    # NOTE: there is deliberately no `repeat_count` here. Rule R4 escalates a worker seen
    # repeatedly, so a caller able to set that number could talk its way up to a supervisor
    # alert. The count is counted, not claimed.


class Decision(BaseModel):
    decision: str = Field(..., description="approve or reject")
    note: Optional[str] = Field(default=None, max_length=500)
    # NOTE: there is deliberately no `actor` field. Who decided is taken from the signed-in
    # session, never from the request body - otherwise the audit trail records a claim
    # rather than an identity (threat T1).


def to_dict(row: CopilotSuggestion) -> dict:
    action = check_action(row.action_id)
    return {"suggestion_id": row.suggestion_id, "created_at": row.created_at,
            "source": row.source, "subject": row.subject, "event_id": row.event_id,
            "action_id": row.action_id, "action_title": action.title,
            "action_effect": action.effect, "reversible": action.reversible,
            "rule": row.rule, "reason": row.reason, "severity": row.severity,
            "evidence": row.evidence, "status": row.status, "decided_by": row.decided_by,
            "decided_at": row.decided_at, "note": row.note, "outcome": row.outcome}


# ----------------------------------------------------------------- the allow-list
@router.get("/catalog")
def get_catalog():
    """Published on purpose: a user can read the copilot's entire vocabulary."""
    return {"actions": catalog(), "count": len(catalog()),
            "note": "The copilot can propose nothing outside this list, and executes "
                    "nothing without a recorded human decision."}


# ----------------------------------------------------------------- proposing
def record_suggestions(db: Session, source: str, person_id: str, compliance_status: str,
                       missing: List[str], zone_type: str = "GENERAL",
                       zone_name: Optional[str] = None, confidence: float = 0.0,
                       event_id: Optional[str] = None):
    """
    Run the rules and store what they propose. Shared by this API and the predict route,
    so a suggestion is produced the same way however the verdict arrived.

    How often this person has been seen before is COUNTED from the table, never taken from
    the request: rule R4 ("seen repeatedly") escalates to a supervisor alert, so a number a
    caller can set is a number an attacker can set.

    Duplicate-proof: if the same person already has a PENDING suggestion of the same
    action from the same source, nothing new is created. A camera looking at one
    unprotected worker for a minute must not produce sixty identical review cards.
    """
    seen_before = db.execute(
        select(CopilotSuggestion).where(CopilotSuggestion.source == source,
                                        CopilotSuggestion.subject == person_id)).scalars().all()
    proposals = suggest(compliance_status, missing, zone_type, zone_name, person_id,
                        confidence, len(seen_before))
    created, skipped = [], 0
    for proposal in proposals:
        check_action(proposal.action_id)              # the gate, even for our own rules
        if any(row.action_id == proposal.action_id and row.status == PENDING
               for row in seen_before):
            skipped += 1
            continue
        row = CopilotSuggestion(
            source=source, subject=person_id, event_id=event_id,
            action_id=proposal.action_id, rule=proposal.rule, reason=proposal.reason,
            severity=proposal.severity, evidence=proposal.evidence, status=PENDING)
        db.add(row)
        db.flush()
        append_audit(db, actor="copilot", action="SUGGESTION_CREATED", event_id=event_id,
                     details={"suggestion_id": row.suggestion_id, "action": row.action_id,
                              "rule": row.rule, "subject": row.subject, "source": source})
        created.append(row)
    return created, skipped


@router.post("/suggestions", status_code=status.HTTP_201_CREATED)
def create_suggestions(body: SuggestRequest, db: Session = Depends(get_db)):
    """Propose actions for one compliance verdict. Nothing is carried out here."""
    with WRITE_LOCK:
        created, skipped = record_suggestions(
            db, body.source, body.person_id, body.compliance_status, body.missing,
            body.zone_type, body.zone_name, body.confidence, body.event_id)
        db.commit()
    return {"created": [to_dict(r) for r in created], "duplicates_skipped": skipped}


# ----------------------------------------------------------------- the queue
@router.get("/suggestions")
def list_suggestions(db: Session = Depends(get_db),
                     suggestion_status: str = Query(default=PENDING, alias="status"),
                     limit: int = Query(default=50, ge=1, le=500)):
    query = select(CopilotSuggestion)
    if suggestion_status.lower() != "all":
        query = query.where(CopilotSuggestion.status == suggestion_status.upper())
    # urgent first, then oldest first: a queue a supervisor can work top to bottom
    query = query.order_by(CopilotSuggestion.severity.desc(),
                           CopilotSuggestion.created_at.asc()).limit(limit)
    rows = db.execute(query).scalars().all()
    return {"suggestions": [to_dict(r) for r in rows], "count": len(rows)}


# ----------------------------------------------------------------- deciding
def execute(action_id: str, row: CopilotSuggestion, actor: str) -> str:
    """
    Carry out an APPROVED action. Called from exactly one place, after a human decision.

    Everything stays on this machine: the two alerting actions append to a local outbox
    file, which is where a real deployment would hand off to a siren, an SMS gateway or a
    site system. Nothing is sent anywhere from here, which is the honest thing to build
    for a privacy-preserving project and the honest thing to say in the viva.
    """
    action = check_action(action_id)
    if action_id in ("notify_supervisor", "escalate_restricted_zone"):
        OUTBOX.parent.mkdir(parents=True, exist_ok=True)
        message = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "priority": "HIGH" if action.severity >= 3 else "NORMAL",
                   "action": action_id, "subject": row.subject, "source": row.source,
                   "reason": row.reason, "approved_by": actor,
                   "suggestion_id": row.suggestion_id}
        with OUTBOX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")
        return f"added to the site alert queue ({OUTBOX.name})"
    if action_id == "flag_for_review":
        return "flagged for supervisor review"
    if action_id == "request_recheck":
        return "marked for another observation"
    return "recorded, no further action"


@router.post("/suggestions/{suggestion_id}/decide")
def decide(suggestion_id: str, body: Decision = Body(...), db: Session = Depends(get_db),
           user: dict = Depends(require_role(SUPERVISOR))):
    """
    Approve or reject one proposal. Requires a signed-in supervisor or admin: this is the
    single point where the machine's suggestion becomes a human decision, so it is the one
    place where knowing *who* is not optional.
    """
    actor = user.get("n") or user["u"]
    choice = body.decision.strip().lower()
    if choice not in ("approve", "reject"):
        raise HTTPException(UNPROCESSABLE,
                            detail="decision must be 'approve' or 'reject'")

    with WRITE_LOCK:
        row = db.get(CopilotSuggestion, suggestion_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such suggestion")
        if row.status != PENDING:
            # Decided once, by one person. A second click must not re-run the action.
            raise HTTPException(status.HTTP_409_CONFLICT,
                                detail=f"already {row.status.lower()} by {row.decided_by}")
        try:
            check_action(row.action_id)
        except NotAllowed as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

        row.status = APPROVED if choice == "approve" else REJECTED
        row.decided_by = actor
        row.decided_at = datetime.now(timezone.utc)
        row.note = body.note
        row.outcome = execute(row.action_id, row, actor) if choice == "approve" else None

        append_audit(db, actor=actor,
                     action="SUGGESTION_APPROVED" if choice == "approve" else "SUGGESTION_REJECTED",
                     event_id=row.event_id,
                     details={"suggestion_id": row.suggestion_id, "action": row.action_id,
                              "rule": row.rule, "subject": row.subject, "note": body.note,
                              "outcome": row.outcome, "role": user.get("r")})
        db.commit()
        db.refresh(row)
    return to_dict(row)


# ----------------------------------------------------------------- the outbox
@router.get("/outbox")
def read_outbox(limit: int = Query(default=50, ge=1, le=500)):
    """What approved alerts produced. Newest first; a corrupt line is skipped, not fatal."""
    if not OUTBOX.exists():
        return {"messages": [], "count": 0, "path": str(OUTBOX)}
    rows = []
    for line in OUTBOX.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = list(reversed(rows))[:limit]
    return {"messages": rows, "count": len(rows), "path": str(OUTBOX)}
