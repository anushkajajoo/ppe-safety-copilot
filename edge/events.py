"""
Event builder: decides WHEN a decision becomes a stored safety event.

We do NOT send one event per frame (that would be ~25 events per second per worker).
An event is created only when:
  * a worker's status CHANGES to POTENTIAL_VIOLATION (or UNCERTAIN, if enabled), or
  * the reasons change (e.g. was MISSING_MASK, now MISSING_MASK + MISSING_HELMET), and
  * the same worker + same reasons has not produced an event within `cooldown_s`.

Output is a dict matching shared.schemas.EventCreate (validated before it leaves).
Day 2: events are appended to a JSONL file. Day 3: they go into the offline outbox -> API.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from edge.types import Decision, Worker
from shared.schemas import EventCreate

ALERT_STATES = {"POTENTIAL_VIOLATION", "UNCERTAIN"}


class EventBuilder:
    def __init__(self, device_id: str, camera_id: str, session_id: str, model_version: str, rules_version: str,
                 cooldown_s: float = 30.0, emit_uncertain: bool = True):
        self.device_id, self.camera_id, self.session_id = device_id, camera_id, session_id
        self.model_version, self.rules_version = model_version, rules_version
        self.cooldown_s = cooldown_s
        self.emit_uncertain = emit_uncertain
        self._last_status: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        self._last_emit: Dict[Tuple[str, Tuple[str, ...]], float] = {}

    def _should_emit(self, d: Decision, now: float) -> bool:
        key = (d.status, tuple(sorted(d.reasons)))
        changed = self._last_status.get(d.display_id) != key
        self._last_status[d.display_id] = key
        if d.status not in ALERT_STATES or not changed:
            return False
        if d.status == "UNCERTAIN" and (not self.emit_uncertain or d.reasons == ["INSUFFICIENT_EVIDENCE"]):
            return False  # "still warming up" is not worth an event
        ck = (d.display_id, key[1])
        if now - self._last_emit.get(ck, -1e9) < self.cooldown_s:
            return False
        self._last_emit[ck] = now
        return True

    def maybe_build(self, d: Decision, worker: Worker, now: Optional[float] = None) -> Optional[dict]:
        now = time.monotonic() if now is None else now
        if not self._should_emit(d, now):
            return None
        detections = [{"object_class": "person", "confidence": round(worker.conf, 3),
                       "bbox": dict(zip(("x1", "y1", "x2", "y2"), map(float, worker.box)))}]
        for cls, det in worker.ppe.items():
            detections.append({"object_class": cls, "confidence": round(det.conf, 3),
                               "bbox": dict(zip(("x1", "y1", "x2", "y2"), map(float, det.box)))})
        event = EventCreate(
            device_id=self.device_id, camera_id=self.camera_id, session_id=self.session_id,
            worker_display_id=d.display_id, zone_id=d.zone_id, status=d.status, reasons=d.reasons,
            confidence=d.confidence, frames_in_window=d.frames_in_window, frames_missing=d.frames_missing,
            model_version=self.model_version, rules_version=self.rules_version,
            privacy_status="NO_EVIDENCE",   # Day 4: blurred snapshot -> FACE_BLUR_OK / EVIDENCE_WITHHELD
            detections=detections,
        )
        return event.model_dump(mode="json")

    def forget(self, display_id: str) -> None:
        self._last_status.pop(display_id, None)


class JsonlSink:
    """Simple sink: one JSON event per line. Used when there is no server to talk to."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, events: List[dict]) -> None:
        if not events:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        self.count += len(events)


class OutboxSink:
    """
    Durable sink: every event is written to the store-and-forward queue immediately.

    Same tiny interface as JsonlSink (`write`, `count`), so `run_edge` does not care which
    one it has - the difference is entirely in what happens when the network is missing.
    """

    def __init__(self, outbox) -> None:
        self.outbox = outbox
        self.count = 0

    def write(self, events: List[dict]) -> None:
        for event in events:
            # queued before anything is attempted: an event survives a crash one line later
            self.outbox.add(event)
            self.count += 1
