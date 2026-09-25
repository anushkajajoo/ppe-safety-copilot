"""
The safety decision layer: GO / REVIEW / STOP.

WHY THIS EXISTS
    Detection produces classes and confidences. Compliance produces "missing helmet". Neither
    of those tells a supervisor what to DO. This module turns a compliance verdict into one of
    three operational decisions, with the reason, the evidence and the rule that produced it -
    so a decision can be defended months later without re-running anything.

        GO      required PPE present, every item above the threshold
        REVIEW  something is uncertain: a low-confidence item, a degraded frame, or the
                system itself is not healthy. A human looks.
        STOP    a confident safety violation. Intervene.

THE ONE RULE THAT MATTERS MOST
    A system that cannot see must never answer GO. Every failure path in here - no model, no
    frame, a detector exception, an unreadable image - lands on REVIEW (or STOP under a
    stricter policy), never on GO. `fail_safe_decision` picks which, and it cannot be set to
    GO: the setter refuses.

DETERMINISTIC, AND NO MODEL IN THE LOOP
    Pure functions over numbers. No LLM, no learned thresholds, no randomness. Given the same
    detections and the same configuration this returns the same decision, which is what makes
    it testable and what makes it auditable.

THE UNCERTAINTY BAND
    A detection just under the threshold is not the same as no detection at all. Between
    `threshold - review_margin` and `threshold` an item counts as UNCERTAIN, and one uncertain
    required item makes the whole decision REVIEW rather than STOP. Below that band the item
    is absent and the decision is STOP. This is the difference between "I think he might not
    be wearing a helmet" and "he is not wearing a helmet".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Mapping, Optional, Sequence

GO, REVIEW, STOP = "GO", "REVIEW", "STOP"
ORDER = {GO: 0, REVIEW: 1, STOP: 2}          # worst wins when several people are in frame

PRESENT, UNCERTAIN, MISSING = "present", "uncertain", "missing"

DEFAULT_THRESHOLD = 0.50
DEFAULT_REVIEW_MARGIN = 0.15
FAIL_SAFE_CHOICES = (REVIEW, STOP)           # GO is deliberately not an option


class UnsafePolicy(ValueError):
    """Raised if anyone tries to configure GO as the fail-safe. That is not a policy."""


def validate_fail_safe(value: str) -> str:
    if value not in FAIL_SAFE_CHOICES:
        raise UnsafePolicy(
            f"fail-safe must be one of {FAIL_SAFE_CHOICES}; {value!r} would let a broken "
            f"system report that a worker is protected")
    return value


@dataclass
class ItemEvidence:
    """One required PPE item and what the detector had to say about it."""
    item: str
    confidence: float
    threshold: float
    state: str

    def as_dict(self) -> dict:
        return {"item": self.item, "confidence": round(self.confidence, 3),
                "threshold": self.threshold, "state": self.state}


@dataclass
class Decision:
    decision: str
    reason: str
    rule: str
    evidence: List[ItemEvidence] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD
    person_id: Optional[str] = None
    source: Optional[str] = None
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def needs_human(self) -> bool:
        return self.decision in (REVIEW, STOP)

    def as_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "rule": self.rule,
                "evidence": [e.as_dict() for e in self.evidence],
                "required": list(self.required), "missing": list(self.missing),
                "uncertain": list(self.uncertain), "threshold": self.threshold,
                "person_id": self.person_id, "source": self.source, "at": self.at}


def classify_item(confidence: float, threshold: float, review_margin: float) -> str:
    """present / uncertain / missing for one item, from its confidence alone."""
    if confidence >= threshold:
        return PRESENT
    if confidence >= max(0.0, threshold - review_margin):
        return UNCERTAIN
    return MISSING


def decide_person(wearing: Mapping[str, float], required: Sequence[str],
                  threshold: float = DEFAULT_THRESHOLD,
                  review_margin: float = DEFAULT_REVIEW_MARGIN,
                  person_id: Optional[str] = None,
                  source: Optional[str] = None) -> Decision:
    """
    One worker. `wearing` maps a PPE class to the confidence it was detected with; an item
    the detector never reported is simply absent from the mapping, which is a confidence of
    zero for this purpose.
    """
    evidence: List[ItemEvidence] = []
    missing: List[str] = []
    uncertain: List[str] = []

    for item in required:
        confidence = float(wearing.get(item, 0.0))
        state = classify_item(confidence, threshold, review_margin)
        evidence.append(ItemEvidence(item, confidence, threshold, state))
        if state == MISSING:
            missing.append(item)
        elif state == UNCERTAIN:
            uncertain.append(item)

    common = dict(evidence=evidence, required=list(required), missing=missing,
                  uncertain=uncertain, threshold=threshold, person_id=person_id, source=source)

    if missing:
        worst = ", ".join(missing)
        confidences = ", ".join(f"{e.item} {e.confidence:.2f}" for e in evidence
                                if e.state == MISSING)
        return Decision(STOP, rule="R-MISSING-PPE",
                        reason=(f"Required {worst} not detected above the configured "
                                f"threshold of {threshold:.2f} ({confidences})."),
                        **common)
    if uncertain:
        items = ", ".join(uncertain)
        confidences = ", ".join(f"{e.item} {e.confidence:.2f}" for e in evidence
                                if e.state == UNCERTAIN)
        return Decision(REVIEW, rule="R-LOW-CONFIDENCE",
                        reason=(f"{items} detected but below the threshold of "
                                f"{threshold:.2f} ({confidences}) - too close to call."),
                        **common)
    return Decision(GO, rule="R-ALL-PRESENT",
                    reason=f"All required PPE ({', '.join(required) or 'none'}) detected "
                           f"at or above {threshold:.2f}.",
                    **common)


def system_fault(reason: str, fail_safe: str = REVIEW, source: Optional[str] = None,
                 required: Optional[Sequence[str]] = None) -> Decision:
    """
    The system could not see. Never GO.

    Used for: the model is missing or failed to load, the camera did not open, a frame could
    not be decoded, or the detector raised. The caller supplies the human-readable cause.
    """
    return Decision(validate_fail_safe(fail_safe), rule="R-SYSTEM-FAULT",
                    reason=f"{reason} The system cannot confirm compliance, so it does not.",
                    required=list(required or []), source=source)


def decide_frame(people: Sequence[Decision], source: Optional[str] = None,
                 fail_safe: str = REVIEW) -> Decision:
    """
    One frame's overall decision: the worst of everyone in it.

    An empty frame is GO - there is nobody to be unsafe. That is not the same as a frame the
    system failed to read, which is a system fault and goes through `system_fault`.
    """
    if not people:
        return Decision(GO, rule="R-NO-PEOPLE", reason="No person detected in the frame.",
                        source=source)
    worst = max(people, key=lambda d: ORDER.get(d.decision, 1))
    count_stop = sum(1 for d in people if d.decision == STOP)
    count_review = sum(1 for d in people if d.decision == REVIEW)
    summary = f"{len(people)} person(s) in frame"
    if count_stop:
        summary += f"; {count_stop} with a confident violation"
    if count_review:
        summary += f"; {count_review} uncertain"
    return Decision(worst.decision, rule=worst.rule,
                    reason=f"{summary}. Worst case: {worst.reason}",
                    evidence=worst.evidence, required=worst.required, missing=worst.missing,
                    uncertain=worst.uncertain, threshold=worst.threshold,
                    person_id=worst.person_id, source=source)


def worst_of(decisions: Sequence[str], default: str = GO) -> str:
    """Combine decision strings (used by the video and analytics paths)."""
    if not decisions:
        return default
    return max(decisions, key=lambda d: ORDER.get(d, 1))
