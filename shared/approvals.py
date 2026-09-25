"""
Which settings changes need a human, and what a change is allowed to touch.

THE RULE
    A safety system whose thresholds can be changed silently is a safety system with no
    thresholds. Every entry below is a change that alters what the system will report, hide,
    keep or delete - so each one is proposed, reviewed by a named person, and applied only
    after approval. The applying code can only touch the keys listed here: an approved
    request cannot become a way to set any attribute on the settings object.
"""
from __future__ import annotations

from typing import Dict, List

PENDING, APPROVED, REJECTED = "PENDING", "APPROVED", "REJECTED"

# kind -> (what it changes, which settings keys it may write)
KINDS: Dict[str, dict] = {
    "change_required_ppe": {
        "what": "Change which PPE items are required on site",
        "keys": [],                      # applied per-request at call time, not stored here
        "why": "It decides what counts as a violation at all.",
    },
    "change_threshold": {
        "what": "Change the confidence threshold or the review band",
        "keys": ["ppe_confidence_threshold", "review_margin"],
        "why": "Lowering it turns uncertain detections into confident ones.",
    },
    "change_privacy": {
        "what": "Turn snapshot or recording storage on or off",
        "keys": ["store_snapshots", "record_events", "store_raw_snapshots"],
        "why": "It decides whether the system holds pictures of people.",
    },
    "change_retention": {
        "what": "Change how long evidence is kept",
        "keys": ["video_retention_days", "snapshot_retention_days", "log_retention_days"],
        "why": "Shortening it destroys evidence; lengthening it holds personal data longer.",
    },
    "delete_evidence": {
        "what": "Delete stored evidence before its retention period ends",
        "keys": [],
        "why": "Deleting evidence early is exactly the action that must not be quiet.",
    },
    "export_footage": {
        "what": "Export an evidence clip or snapshot out of the system",
        "keys": [],
        "why": "Export is how personal data leaves the machine.",
    },
    "external_notification": {
        "what": "Send a notification outside this machine",
        "keys": [],
        "why": "Nothing should leave the site without someone deciding it should.",
    },
}

# Keys that may EVER be written by an approved request, gathered from the table above.
WRITABLE_KEYS = sorted({key for entry in KINDS.values() for key in entry["keys"]})


class NotApprovable(ValueError):
    """The kind is unknown, or the change would touch a setting it may not."""


def check_kind(kind: str) -> dict:
    entry = KINDS.get(kind)
    if entry is None:
        raise NotApprovable(f"'{kind}' is not a change this system knows how to approve")
    return entry


def check_changes(kind: str, changes: Dict[str, object]) -> Dict[str, object]:
    """Only the keys this kind declares, and only keys on the global writable list."""
    entry = check_kind(kind)
    allowed = set(entry["keys"])
    for key in changes:
        if key not in allowed or key not in WRITABLE_KEYS:
            raise NotApprovable(f"'{kind}' may not change '{key}'")
    return dict(changes)


def catalog() -> List[dict]:
    return [{"kind": kind, "what": entry["what"], "why": entry["why"],
             "settings_keys": entry["keys"]} for kind, entry in KINDS.items()]
