"""
Compliance engine: turns evidence into a decision.

It is deliberately SEPARATE from YOLO:
    YOLO says WHAT is visible.  This module decides WHETHER that is a problem, using
    the zone's policy and the temporal evidence. It is deterministic, so every
    decision can be explained and unit-tested.

Two modes (used for the baseline comparison on Day 6):
    proposed : zone rules + temporal window + UNCERTAIN state
    baseline : one frame only, no uncertainty -> missing now = violation now
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from edge.temporal import WorkerHistory
from edge.types import Decision, Worker
from edge.zones import Zone

COMPLIANT, UNCERTAIN, VIOLATION = "COMPLIANT", "UNCERTAIN", "POTENTIAL_VIOLATION"
MISSING_REASON = {"helmet": "MISSING_HELMET", "vest": "MISSING_VEST", "mask": "MISSING_MASK"}


class ComplianceEngine:
    def __init__(self, window_frames: int = 15, violate_at: int = 10, uncertain_at: int = 5):
        if not (0 < uncertain_at <= violate_at <= window_frames):
            raise ValueError("need 0 < uncertain_at <= violate_at <= window_frames")
        self.window = window_frames
        self.violate_at = violate_at
        self.uncertain_at = uncertain_at

    @classmethod
    def from_policies(cls, policies: dict) -> "ComplianceEngine":
        t = policies["temporal"]
        return cls(int(t["window_frames"]), int(t["violate_at"]), int(t["uncertain_at"]))

    # ------------------------------------------------------------------ proposed
    def decide(self, worker: Worker, zone: Optional[Zone], hist: WorkerHistory,
               now: Optional[datetime] = None) -> Decision:
        if zone is None:
            return self._decision(worker, None, UNCERTAIN, ["OUT_OF_CONFIGURED_ZONE"], 0.0, hist.n, 0, [], {})

        required = list(zone.required_ppe)
        item_state = {i: "N/A" for i in ("helmet", "vest", "mask")}
        n = hist.n
        mean_conf = sum(hist.person_conf) / len(hist.person_conf) if hist.person_conf else 0.0

        # Not enough frames yet -> we honestly don't know.
        if n < self.window:
            for item in required:
                item_state[item] = "OK" if item in worker.ppe else "UNSURE"
            return self._decision(worker, zone, UNCERTAIN, ["INSUFFICIENT_EVIDENCE"], 0.0, n, 0, required, item_state)

        reasons: List[str] = []
        uncertain_items: List[str] = []
        worst_missing = 0
        support = []  # fraction of frames supporting the decision, per relevant item

        for item in required:
            m = hist.missing(item)
            worst_missing = max(worst_missing, m)
            if m >= self.violate_at:
                item_state[item] = "MISSING"
                reasons.append(MISSING_REASON[item])
                support.append(m / n)
            elif m >= self.uncertain_at:
                item_state[item] = "UNSURE"
                uncertain_items.append(item)
            else:
                item_state[item] = "OK"

        if zone.authorization_required and not zone.is_authorized_now(now):
            reasons.insert(0, "RESTRICTED_ZONE_ENTRY")
            support.append(1.0)   # the person was in the zone for the whole window

        if reasons:
            if any(r.startswith("MISSING_") for r in reasons):
                reasons.append("ZONE_REQUIREMENT_NOT_MET")
            conf = round(mean_conf * (sum(support) / len(support)), 3)
            return self._decision(worker, zone, VIOLATION, reasons, conf, n, worst_missing, required, item_state)
        if uncertain_items:
            conf = round(mean_conf * 0.5, 3)
            return self._decision(worker, zone, UNCERTAIN, ["LOW_CONFIDENCE"], conf, n, worst_missing, required, item_state)

        present_ratio = [1 - hist.missing(i) / n for i in required] or [1.0]
        conf = round(mean_conf * (sum(present_ratio) / len(present_ratio)), 3)
        return self._decision(worker, zone, COMPLIANT, [], conf, n, worst_missing, required, item_state)

    # ------------------------------------------------------------------ baseline
    def decide_baseline(self, worker: Worker, zone: Optional[Zone], now: Optional[datetime] = None) -> Decision:
        """Single-frame rule: anything missing in THIS frame = violation. No uncertainty, no memory."""
        if zone is None:
            return self._decision(worker, None, VIOLATION, ["OUT_OF_CONFIGURED_ZONE"], worker.conf, 1, 1, [], {})
        required = list(zone.required_ppe)
        item_state = {i: "N/A" for i in ("helmet", "vest", "mask")}
        reasons = []
        for item in required:
            if item in worker.ppe:
                item_state[item] = "OK"
            else:
                item_state[item] = "MISSING"
                reasons.append(MISSING_REASON[item])
        if zone.authorization_required and not zone.is_authorized_now(now):
            reasons.insert(0, "RESTRICTED_ZONE_ENTRY")
        status = VIOLATION if reasons else COMPLIANT
        return self._decision(worker, zone, status, reasons, round(worker.conf, 3), 1, 1 if reasons else 0,
                              required, item_state)

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _decision(worker: Worker, zone: Optional[Zone], status: str, reasons: List[str], conf: float, n: int,
                  missing: int, required: List[str], item_state: Dict[str, str]) -> Decision:
        return Decision(display_id=worker.display_id, zone_id=zone.zone_id if zone else None,
                        zone_name=zone.zone_name if zone else None, status=status, reasons=reasons,
                        confidence=max(0.0, min(1.0, conf)), frames_in_window=max(1, n),
                        frames_missing=min(missing, max(1, n)), required=required, item_state=item_state)
