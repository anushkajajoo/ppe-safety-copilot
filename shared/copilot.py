"""
The safety copilot: what the system may PROPOSE, and what only a human may DECIDE.

WHY THIS LAYER EXISTS
    A detector that quietly takes action on a construction site is a bad idea, and the
    capstone brief asks for the opposite: a system that assists a human supervisor rather
    than replacing them. So the copilot never acts. It proposes, from a fixed list, with
    the reason attached; a person approves or rejects; and both the proposal and the
    decision are written to the tamper-evident audit chain (server/audit.py).

THREE PROPERTIES, EACH ENFORCED BY CODE AND TESTED
    1. ALLOW-LISTED. `ACTIONS` below is the complete set of things the copilot can ever
       propose. Anything else is rejected by `check_action`, including an action id that
       arrives in an API request. The list is also served to the dashboard, so a user can
       see the copilot's whole vocabulary rather than being told to trust it.
    2. HUMAN-GOVERNED. Nothing executes without a recorded decision by a named actor.
       There is no "auto-approve" path and no confidence threshold that bypasses review.
    3. TRACEABLE. Every suggestion carries the rule that produced it and the evidence it
       saw, so a decision can be reconstructed months later.

WHY THERE IS NO LANGUAGE MODEL HERE
    The mapping from "a worker in a restricted zone is missing a helmet" to "propose
    escalating to the zone supervisor" is a policy, not a judgement call. A rule table is
    auditable, runs in microseconds on a laptop, gives the same answer every time, and
    cannot be talked into proposing something that is not on the list. An LLM would give
    up all four properties in exchange for nicer wording.

SEVERITY is advisory - it orders the review queue. It never decides anything by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

COMPLIANT = "COMPLIANT"
UNCERTAIN = "UNCERTAIN"
VIOLATION = "POTENTIAL_VIOLATION"

# Zone types, as used by edge/zones.py and configs/zones.yaml.
GENERAL, CAUTION, RESTRICTED = "GENERAL", "CAUTION", "RESTRICTED"


@dataclass(frozen=True)
class Action:
    """One thing the copilot is permitted to propose. Nothing outside this list exists."""
    id: str
    title: str
    description: str          # plain English, shown in the dashboard next to the button
    effect: str               # what approving it actually does, on this machine
    severity: int             # 1 = routine, 3 = urgent. Orders the queue; decides nothing.
    reversible: bool


# ------------------------------------------------------------------ the allow-list
ACTIONS: Tuple[Action, ...] = (
    Action(
        id="log_only",
        title="Record and take no action",
        description="Keep the event in the violation log for the shift report.",
        effect="Nothing beyond the log line that already exists.",
        severity=1, reversible=True,
    ),
    Action(
        id="flag_for_review",
        title="Flag for supervisor review",
        description="Mark this event so a supervisor looks at it before the shift ends.",
        effect="Adds a review flag to the event record.",
        severity=1, reversible=True,
    ),
    Action(
        id="notify_supervisor",
        title="Notify the area supervisor",
        description="Queue a message about a worker missing required PPE.",
        effect="Adds a message to the site alert queue on this machine.",
        severity=2, reversible=False,
    ),
    Action(
        id="escalate_restricted_zone",
        title="Escalate: unprotected worker in a restricted zone",
        description="The highest-priority case - required PPE missing inside a restricted area.",
        effect="Raises a high-priority alert in the site queue on this machine.",
        severity=3, reversible=False,
    ),
    Action(
        id="request_recheck",
        title="Ask for a second look",
        description="The evidence was not clear enough to judge. Re-observe before deciding.",
        effect="Marks the event as needing another observation. No alert is raised.",
        severity=1, reversible=True,
    ),
)

BY_ID: Dict[str, Action] = {action.id: action for action in ACTIONS}


class NotAllowed(ValueError):
    """Raised when something asks for an action the copilot may not perform."""


def check_action(action_id: str) -> Action:
    """The single gate. Every path that executes anything goes through here."""
    action = BY_ID.get(action_id)
    if action is None:
        raise NotAllowed(f"'{action_id}' is not an allow-listed copilot action")
    return action


@dataclass(frozen=True)
class Suggestion:
    """A proposal awaiting a human. `rule` is why it exists; `evidence` is what was seen."""
    action_id: str
    rule: str
    reason: str
    severity: int
    evidence: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"action_id": self.action_id, "rule": self.rule, "reason": self.reason,
                "severity": self.severity, "evidence": dict(self.evidence)}


# ------------------------------------------------------------------ the rules
def suggest(status: str, missing: Sequence[str], zone_type: str = GENERAL,
            zone_name: Optional[str] = None, person_id: str = "person",
            confidence: float = 0.0, repeat_count: int = 0) -> List[Suggestion]:
    """
    Turn one compliance verdict into proposals. Pure: same input, same output, no I/O.

    The rules, in the order they are checked:
      R1  compliant                      -> propose nothing at all
      R2  uncertain                      -> ask for a second look, never raise an alarm
      R3  violation in a restricted zone -> escalate (the one urgent case)
      R4  violation, seen repeatedly     -> notify the supervisor
      R5  violation, first sighting      -> flag for review
    Every branch returns at most one suggestion. A copilot that proposes three things at
    once is a copilot nobody reads.
    """
    missing = list(missing)
    evidence = {"person": person_id, "missing": missing, "status": status,
                "zone": zone_name or zone_type, "zone_type": zone_type,
                "confidence": round(float(confidence), 3), "times_seen": int(repeat_count)}

    if status == COMPLIANT:
        return []                                                             # R1

    if status == UNCERTAIN:
        return [Suggestion("request_recheck", "R2",
                           f"{person_id} could not be judged reliably "
                           f"({', '.join(missing) or 'no items'} unclear).",
                           BY_ID["request_recheck"].severity, evidence)]      # R2

    items = ", ".join(missing) or "required PPE"
    if zone_type == RESTRICTED:
        return [Suggestion("escalate_restricted_zone", "R3",
                           f"{person_id} is missing {items} inside "
                           f"{zone_name or 'a restricted zone'}.",
                           BY_ID["escalate_restricted_zone"].severity, evidence)]   # R3

    if repeat_count >= 3:
        return [Suggestion("notify_supervisor", "R4",
                           f"{person_id} has been seen missing {items} "
                           f"{repeat_count} times.",
                           BY_ID["notify_supervisor"].severity, evidence)]    # R4

    return [Suggestion("flag_for_review", "R5",
                       f"{person_id} is missing {items}.",
                       BY_ID["flag_for_review"].severity, evidence)]          # R5


def catalog() -> List[Dict[str, object]]:
    """The allow-list, for the dashboard. Transparency is part of the design."""
    return [{"id": a.id, "title": a.title, "description": a.description, "effect": a.effect,
             "severity": a.severity, "reversible": a.reversible} for a in ACTIONS]
