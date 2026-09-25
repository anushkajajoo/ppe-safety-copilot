"""
Structured safety events: what gets remembered when something matters.

WHY A SEPARATE RECORD
    A frame is not a memory. The camera path already produces a decision for every frame,
    but almost every one of those decisions is "nothing is happening" and keeping them would
    be surveillance with extra steps. A SafetyEvent is what survives: one structured record
    per meaningful thing that happened, carrying enough evidence to explain itself later
    without anyone re-watching video.

WHAT THIS MODULE IS
    The pure half of the event system - types, severity, the deterministic explanation, the
    debounce counter and the deduplication key. No database, no I/O, no framework, so every
    rule in here is testable on its own. `server/event_store.py` is the thin layer that puts
    these records in SQLite.

TWO GUARDS AGAINST A USELESS HISTORY
    debounce      one bad frame is not an event. A violation must be seen on N consecutive
                  inference frames before a record is created.
    deduplication a worker who is still not wearing a helmet ten seconds later is the SAME
                  event, extended - not the hundredth alert. Records carry first_seen,
                  last_seen, a duration and an occurrence count instead.

EXPLANATIONS ARE BUILT, NOT WRITTEN
    `explain()` composes the reason from the numbers in the record: the item, the confidence
    the detector gave it, and the threshold it had to clear. No language model is involved,
    and the same event always produces the same sentence, which is what makes it evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ----------------------------------------------------------------- vocabulary
PPE_VIOLATION = "PPE_VIOLATION"
PPE_REVIEW = "PPE_REVIEW"
PPE_COMPLIANT = "PPE_COMPLIANT"
CAMERA_FAILURE = "CAMERA_FAILURE"
MODEL_FAILURE = "MODEL_FAILURE"
STORAGE_FAILURE = "STORAGE_FAILURE"
SYSTEM_WARNING = "SYSTEM_WARNING"

EVENT_TYPES = (PPE_VIOLATION, PPE_REVIEW, PPE_COMPLIANT, CAMERA_FAILURE, MODEL_FAILURE,
               STORAGE_FAILURE, SYSTEM_WARNING)

LOW, MEDIUM, HIGH, CRITICAL = "LOW", "MEDIUM", "HIGH", "CRITICAL"
SEVERITIES = (LOW, MEDIUM, HIGH, CRITICAL)

NEW, ACKNOWLEDGED, CLOSED = "NEW", "ACKNOWLEDGED", "CLOSED"
STATUSES = (NEW, ACKNOWLEDGED, CLOSED)

LOCAL, SYNCED, SYNC_FAILED = "LOCAL", "SYNCED", "FAILED"

# Statuses that retention must not touch: nobody has looked at these yet.
PROTECTED_STATUSES = (NEW, ACKNOWLEDGED)

GO, REVIEW, STOP = "GO", "REVIEW", "STOP"

TYPE_FOR_DECISION = {GO: PPE_COMPLIANT, REVIEW: PPE_REVIEW, STOP: PPE_VIOLATION}


class InvalidEvent(ValueError):
    """A field was given a value outside its vocabulary."""


def event_type_for(decision: str) -> str:
    kind = TYPE_FOR_DECISION.get(decision)
    if kind is None:
        raise InvalidEvent(f"'{decision}' is not a decision this system produces")
    return kind


def severity_for(decision: str, missing: Sequence[str] = (), zone_type: str = "GENERAL",
                 event_type: Optional[str] = None) -> str:
    """
    How urgent this is. Advisory: it orders the queue and colours the row, and decides
    nothing by itself - the decision already did that.

    A system failure is HIGH whatever the zone: a camera that stopped is not a small problem
    just because the area is ordinary.
    """
    if event_type in (CAMERA_FAILURE, MODEL_FAILURE, STORAGE_FAILURE):
        return HIGH
    if event_type == SYSTEM_WARNING:
        return MEDIUM
    if decision == GO:
        return LOW
    if decision == REVIEW:
        return MEDIUM
    restricted = str(zone_type).upper() == "RESTRICTED"
    if restricted or len(missing) >= 2:
        return CRITICAL              # unprotected in a restricted area, or nothing worn
    return HIGH


def person_label(raw: Optional[str], index: int = 1) -> str:
    """
    An anonymous, session-scoped label: P-01.

    It means "the person the tracker was following in this session", and nothing else. There
    is no name, no face, no employee number, and the numbering restarts every run - so two
    P-01s from different sessions are not the same human and the schema cannot pretend they
    are.
    """
    if raw:
        digits = "".join(character for character in str(raw) if character.isdigit())
        if digits:
            return f"P-{int(digits):02d}"
    return f"P-{int(index):02d}"


def explain(missing: Sequence[str], uncertain: Sequence[str],
            detected: Mapping[str, float], threshold: float, decision: str) -> str:
    """
    The reason, composed from the record's own numbers. Deterministic by construction.
    """
    def confidences(items: Sequence[str]) -> str:
        return ", ".join(f"{item} {float(detected.get(item, 0.0)):.2f}" for item in items)

    if decision == STOP and missing:
        return (f"Required {', '.join(missing)} not detected above the configured threshold "
                f"of {threshold:.2f}. Detected confidence: {confidences(missing)}.")
    if decision == REVIEW and uncertain:
        return (f"{', '.join(uncertain)} detected but below the threshold of {threshold:.2f}. "
                f"Detected confidence: {confidences(uncertain)}. Too close to call, so a "
                f"person should look.")
    if decision == GO:
        worn = ", ".join(f"{item} {value:.2f}" for item, value in sorted(detected.items()))
        return (f"All required PPE detected at or above {threshold:.2f}"
                + (f" ({worn})." if worn else "."))
    return f"Decision {decision} with no PPE item to explain."


# ------------------------------------------------------------------- debounce
@dataclass
class Debouncer:
    """
    One bad frame is not an event.

    A violation must be seen on `frames` consecutive inference frames for the SAME person
    with the SAME missing items before `confirm` returns True. Anything else resets that
    counter - including the person becoming compliant, which is the case that matters: a
    detector that blinks for one frame must not raise an alert.

    Deliberately not the 15-frame temporal filter in edge/temporal.py: that one decides
    whether a violation is real, this one decides whether to write a record about it. Both
    exist because they answer different questions at different costs.
    """
    frames: int = 3
    _counts: Dict[Tuple[str, str, str], int] = field(default_factory=dict)

    def key(self, source: str, person: str, signature: str) -> Tuple[str, str, str]:
        return (str(source), str(person), str(signature))

    def confirm(self, source: str, person: str, signature: str) -> bool:
        """
        True from the frame where the streak reaches `frames`, and on every frame after it
        while the same problem continues.

        Returning True repeatedly is deliberate: the store decides whether that means a new
        record or an extension of the open one (see `should_merge`). Making this method
        return True exactly once would move that decision here, where there is no history
        to make it with.
        """
        identity = self.key(source, person, signature)
        # anything else this person was doing is no longer true
        for existing in [k for k in self._counts if k[0] == source and k[1] == person
                         and k[2] != signature]:
            del self._counts[existing]
        count = self._counts.get(identity, 0) + 1
        self._counts[identity] = count
        return count >= max(1, int(self.frames))

    def reset(self, source: str, person: Optional[str] = None) -> None:
        """Called when a person becomes compliant or leaves the frame."""
        for existing in [k for k in self._counts
                         if k[0] == source and (person is None or k[1] == person)]:
            del self._counts[existing]

    def streak(self, source: str, person: str, signature: str) -> int:
        return self._counts.get(self.key(source, person, signature), 0)


# -------------------------------------------------------------- deduplication
def signature_of(event_type: str, missing: Sequence[str], uncertain: Sequence[str]) -> str:
    """
    What makes two observations "the same problem": the same kind of event about the same
    items. Confidence changes frame to frame and must not split one event into two.
    """
    items = ",".join(sorted(missing)) or ",".join(sorted(uncertain)) or "none"
    return f"{event_type}:{items}"


def dedupe_key(camera_id: str, person_id: str, signature: str) -> str:
    return f"{camera_id}|{person_id}|{signature}"


def should_merge(last_seen_epoch: float, now_epoch: float, window_s: float) -> bool:
    """
    Same problem, still going: extend the open record instead of writing a new one.

    The window is a gap tolerance, not a maximum event length - a worker without a helmet
    for an hour is one event lasting an hour, as long as they are seen at least every
    `window_s` seconds.
    """
    if now_epoch < last_seen_epoch:
        return False                          # a clock that went backwards starts a new event
    return (now_epoch - last_seen_epoch) <= max(0.0, float(window_s))


# --------------------------------------------------------------- retention
def is_protected(status: str) -> bool:
    """NEW and ACKNOWLEDGED events are waiting on a person. Retention leaves them alone."""
    return str(status).upper() in PROTECTED_STATUSES


def expired(occurred_epoch: float, now_epoch: float, retention_days: float) -> bool:
    """Age is measured from when the event HAPPENED, not from a file's timestamp."""
    return (now_epoch - float(occurred_epoch)) > float(retention_days) * 86400


def eligible_for_cleanup(status: str, occurred_epoch: float, now_epoch: float,
                         retention_days: float, deleted: bool = False) -> bool:
    if deleted:
        return expired(occurred_epoch, now_epoch, retention_days)
    if is_protected(status):
        return False
    return expired(occurred_epoch, now_epoch, retention_days)


# ------------------------------------------------------------------- export
EXPORT_COLUMNS = ("event_id", "occurred_at", "camera_id", "person_id", "event_type",
                  "decision", "severity", "status", "required_ppe", "missing_ppe",
                  "uncertain_ppe", "confidence", "rule_id", "reason", "acknowledged_by",
                  "acknowledged_at", "sync_status")


def to_csv(rows: Iterable[Mapping[str, object]]) -> str:
    """
    History as CSV. Evidence paths are deliberately NOT columns: exporting a spreadsheet
    should never be a way to hand someone a list of image files to go and fetch.
    """
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EXPORT_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat = dict(row)
        for key in ("required_ppe", "missing_ppe", "uncertain_ppe"):
            value = flat.get(key)
            if isinstance(value, (list, tuple)):
                flat[key] = " ".join(str(item) for item in value)
        writer.writerow(flat)
    return buffer.getvalue()
