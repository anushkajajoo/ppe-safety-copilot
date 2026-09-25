"""
A small local violation log - one JSON object per line (JSONL).

WHY A FILE AND NOT A DATABASE
    A violation record is written once and read in order. That is exactly what an
    append-only text file is good at: no schema migration, no server, and you can
    open it in Notepad during a demo. The SQLite database in server/ is for the
    tracked edge pipeline's reviewed events; this log is the simple record the
    project brief asks for.

WHAT IS STORED - AND WHAT IS NOT
    Stored: timestamp, source (image name / video name / camera index), the
    anonymous person label inside that one image, the status, which items were
    missing, the detector's confidence for that person, and the required-PPE list.
    NOT stored: the picture, the frame, or anything identifying a real person.

FLOOD PROTECTION
    A live camera at 10 FPS would otherwise write 600 identical lines a minute, so
    the same (source, person, status) is only written once every `cooldown_s`
    seconds (default 30, matching configs/policies.yaml). This is a logging
    cooldown, not a safety decision - the temporal filter in edge/temporal.py is
    what decides whether a violation is real.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

COMPLIANT = "COMPLIANT"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ViolationLog:
    def __init__(self, path: Path, cooldown_s: float = 30.0) -> None:
        self.path = Path(path)
        self.cooldown_s = float(cooldown_s)
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}          # (source, person, status) -> monotonic-ish time

    # ------------------------------------------------------------------ writing
    def _append(self, record: dict) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return True

    def _allowed(self, key: str, now: float) -> bool:
        """True if this exact (source, person, status) has not been logged recently."""
        with self._lock:
            last = self._last.get(key)
            if last is not None and (now - last) < self.cooldown_s:
                return False
            self._last[key] = now
            return True

    def log_person(self, source: str, person_id: str, status: str, missing: Sequence[str],
                   confidence: float, required: Sequence[str], now: Optional[float] = None,
                   snapshot: Optional[str] = None, privacy_status: str = "NO_EVIDENCE") -> bool:
        """
        Record one non-compliant person. Returns True if a line was written.

        `snapshot` is a PATH to a masked image, never image data, and only when
        snapshots are switched on. `privacy_status` says what happened to it:
        NO_EVIDENCE (none kept), FACE_BLUR_OK, or EVIDENCE_WITHHELD (masking failed).
        """
        if status == COMPLIANT:
            return False
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        if not self._allowed(f"{source}|{person_id}|{status}", now):
            return False
        return self._append({
            "time": _now_iso(), "kind": "person", "source": source, "person_id": person_id,
            "status": status, "missing": list(missing), "confidence": round(float(confidence), 3),
            "required": list(required), "privacy_status": privacy_status, "snapshot": snapshot,
        })

    def log_people(self, source: str, people: Sequence[dict], required: Sequence[str],
                   now: Optional[float] = None, snapshot: Optional[str] = None,
                   privacy_status: str = "NO_EVIDENCE") -> int:
        """Record every non-compliant person from one image. Returns how many lines were written."""
        written = 0
        for person in people:
            if self.log_person(source, person["id"], person["status"], person.get("missing", []),
                               person.get("conf", 0.0), required, now=now,
                               snapshot=snapshot, privacy_status=privacy_status):
                written += 1
        return written

    def log_video(self, source: str, frames_analysed: int, frames_with_violation: int,
                  violation_rate: float, people_max: int, status_counts: Dict[str, int],
                  required: Sequence[str]) -> bool:
        """One summary line per analysed clip - not one per frame, which would flood the log."""
        if frames_with_violation <= 0:
            return False
        return self._append({
            "time": _now_iso(), "kind": "video", "source": source,
            "frames_analysed": frames_analysed, "frames_with_violation": frames_with_violation,
            "violation_rate": round(float(violation_rate), 3), "people_max": people_max,
            "status_counts": dict(status_counts), "required": list(required),
        })

    # ------------------------------------------------------------------ reading
    def recent(self, limit: int = 50) -> List[dict]:
        """The newest entries first. A bad line is skipped rather than crashing the page."""
        if not self.path.exists():
            return []
        rows: List[dict] = []
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(reversed(rows))[:max(1, limit)]

    def count(self) -> int:
        return len(self.recent(limit=10 ** 9))
