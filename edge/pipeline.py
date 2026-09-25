"""
The edge decision pipeline for ONE frame, without the camera or the model:

  persons + PPE boxes
     -> anonymous Worker IDs            (workers.py)
     -> PPE assigned to each worker      (association.py)
     -> zone from the foot point         (zones.py)
     -> sliding-window evidence          (temporal.py)
     -> COMPLIANT / UNCERTAIN / VIOLATION (compliance.py)
     -> event only on change + cooldown  (events.py)

Keeping this separate from run_edge.py means the whole logic is unit-tested with
fake detections (tests/test_edge_pipeline.py) and the SAME code runs live.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from edge.association import associate
from edge.compliance import ComplianceEngine
from edge.events import EventBuilder
from edge.temporal import TemporalFilter
from edge.types import Decision, Detection, Worker
from edge.workers import WorkerRegistry
from edge.zones import ZoneManager


class EdgePipeline:
    def __init__(self, zones: ZoneManager, builder: EventBuilder, mode: str = "proposed"):
        if mode not in ("proposed", "baseline"):
            raise ValueError("mode must be 'proposed' or 'baseline'")
        p = zones.policies
        self.zones = zones
        self.mode = mode
        self.engine = ComplianceEngine.from_policies(p)
        self.temporal = TemporalFilter(self.engine.window)
        self.registry = WorkerRegistry(float(p["events"].get("forget_worker_after_s", 3.0)))
        self.builder = builder
        self.min_ioa = float(p["detection"].get("assoc_min_ioa", 0.5))

    def step(self, persons: List[Detection], ppe: List[Detection], frame_w: int, frame_h: int,
             now: Optional[float] = None, wall: Optional[datetime] = None) -> Tuple[List[Worker], Dict[str, Decision], List[dict]]:
        now = time.monotonic() if now is None else now

        workers = [Worker(track_id=p.track_id, display_id=self.registry.display_id(p.track_id, now),
                          box=p.box, conf=p.conf) for p in persons if p.track_id is not None]
        associate(workers, ppe, self.min_ioa)

        decisions: Dict[str, Decision] = {}
        events: List[dict] = []
        for w in workers:
            fx, fy = w.foot_point
            zone = self.zones.zone_at(fx, min(fy, frame_h - 1), frame_w, frame_h)
            if self.mode == "baseline":
                d = self.engine.decide_baseline(w, zone, wall)
            else:
                hist = self.temporal.update(w.display_id, zone.zone_id if zone else None, w.ppe.keys(), w.conf)
                d = self.engine.decide(w, zone, hist, wall)
            decisions[w.display_id] = d
            ev = self.builder.maybe_build(d, w, now)
            if ev:
                events.append(ev)

        for gone in self.registry.forget_stale(now):
            self.temporal.forget(gone)
            self.builder.forget(gone)
        return workers, decisions, events
