"""
Anonymous worker IDs.

The tracker (ByteTrack, inside Ultralytics) gives every person a numeric track_id that
stays the same from frame to frame as long as it keeps matching the box (by motion
prediction + box overlap). We turn those raw ids into friendly, sequential,
ANONYMOUS labels: Worker-001, Worker-002, ...

  * No faces, no embeddings, no names: "Worker-017" only means "the 17th person
    this edge session has tracked".
  * Numbers restart every session (session_id changes), so the backend stores
    (session_id, display_id) — never the display id alone.
  * If a person is hidden for a while and comes back, they may get a NEW number
    (an "ID switch"). That's a documented limitation, not a bug.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional


class WorkerRegistry:
    def __init__(self, forget_after_s: float = 3.0):
        self.forget_after_s = forget_after_s
        self._display: Dict[int, str] = {}
        self._last_seen: Dict[int, float] = {}
        self._counter = 0

    def display_id(self, track_id: int, now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        if track_id not in self._display:
            self._counter += 1
            self._display[track_id] = f"Worker-{self._counter:03d}"
        self._last_seen[track_id] = now
        return self._display[track_id]

    def forget_stale(self, now: Optional[float] = None) -> List[str]:
        """Drop tracks unseen for `forget_after_s`. Returns the display ids that were dropped."""
        now = time.monotonic() if now is None else now
        stale = [t for t, seen in self._last_seen.items() if now - seen > self.forget_after_s]
        dropped = []
        for t in stale:
            dropped.append(self._display.pop(t))
            self._last_seen.pop(t, None)
        return dropped

    @property
    def active_count(self) -> int:
        return len(self._display)

    @property
    def total_seen(self) -> int:
        return self._counter
