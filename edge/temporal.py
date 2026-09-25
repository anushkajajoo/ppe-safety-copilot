"""
Temporal filtering: remember the last N frames for every worker so one bad frame
(motion blur, a hand in front of the helmet, a missed detection) does not create
a violation.

For each worker we keep a sliding window (deque) per PPE item:
    1 = item detected on that worker in that frame
    0 = item not detected

Example with window 15:
    mask: 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0   -> missing 12 of 15 -> strong evidence of no mask
    mask: 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1   -> missing 2 of 15  -> treat as worn (detector blinked)

When a worker changes zone, the window is reset, because the requirements changed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional

ITEMS = ("helmet", "vest", "mask")


@dataclass
class WorkerHistory:
    zone_id: Optional[str]
    window: int
    seen: Dict[str, Deque[int]] = field(default_factory=dict)
    in_zone: Deque[int] = field(default_factory=deque)          # frames observed in this zone
    person_conf: Deque[float] = field(default_factory=deque)

    def __post_init__(self):
        self.seen = {k: deque(maxlen=self.window) for k in ITEMS}
        self.in_zone = deque(maxlen=self.window)
        self.person_conf = deque(maxlen=self.window)

    @property
    def n(self) -> int:
        return len(self.in_zone)

    def missing(self, item: str) -> int:
        d = self.seen[item]
        return len(d) - sum(d)


class TemporalFilter:
    def __init__(self, window_frames: int = 15):
        self.window = window_frames
        self._h: Dict[str, WorkerHistory] = {}

    def update(self, display_id: str, zone_id: Optional[str], present: Iterable[str], person_conf: float) -> WorkerHistory:
        h = self._h.get(display_id)
        if h is None or h.zone_id != zone_id:        # new worker OR zone changed -> fresh evidence
            h = WorkerHistory(zone_id=zone_id, window=self.window)
            self._h[display_id] = h
        present = set(present)
        for item in ITEMS:
            h.seen[item].append(1 if item in present else 0)
        h.in_zone.append(1)
        h.person_conf.append(person_conf)
        return h

    def forget(self, display_id: str) -> None:
        self._h.pop(display_id, None)
