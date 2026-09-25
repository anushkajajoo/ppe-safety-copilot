"""
Person <-> PPE association: decide WHICH worker each helmet / vest / mask belongs to.

Idea (explain this in the viva):
  A helmet belongs on a head, a vest on a torso, a mask on a face. So for every person
  box we build body REGIONS, and a PPE box is given to the person whose matching region
  contains most of it (IoA = fraction of the PPE box inside the region).

      person box height h
      ┌───────────┐  y1 - 0.10h   <- helmet may stick out above the person box
      │   HEAD    │
      │           │  y1 + 0.35h
      ├───────────┤  y1 + 0.20h   (regions overlap a little on purpose)
      │   TORSO   │
      │           │  y1 + 0.75h
      └───────────┘  y2

  Each PPE box is assigned to at most ONE person (the best match), so one helmet can't
  make two people compliant.

Limitations: if two people overlap heavily, a PPE box can land on the wrong person.
This is why we use temporal filtering and human review instead of trusting one frame.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from edge.geometry import Box, center, ioa
from edge.types import Detection, Worker

REGION_FOR = {"helmet": "head", "mask": "head", "vest": "torso"}


def body_regions(box: Box) -> Dict[str, Box]:
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    pad = 0.10 * w   # allow slightly wider than the person box
    return {
        "head": (x1 - pad, y1 - 0.10 * h, x2 + pad, y1 + 0.35 * h),
        "torso": (x1 - pad, y1 + 0.20 * h, x2 + pad, y1 + 0.75 * h),
    }


def associate(workers: List[Worker], ppe: List[Detection], min_ioa: float = 0.5) -> Tuple[List[Worker], List[Detection]]:
    """
    Fills worker.ppe in place. Returns (workers, unassigned_ppe).
    If a worker gets two helmets, the one with higher confidence is kept.
    """
    regions = {w.track_id: body_regions(w.box) for w in workers}
    unassigned: List[Detection] = []
    for det in ppe:
        region_name = REGION_FOR.get(det.cls)
        if region_name is None:
            continue
        best, best_score, best_dist = None, 0.0, float("inf")
        cx, cy = center(det.box)
        for w in workers:
            score = ioa(det.box, regions[w.track_id][region_name])
            if score < min_ioa:
                continue
            wx, wy = center(regions[w.track_id][region_name])
            dist = (cx - wx) ** 2 + (cy - wy) ** 2
            # higher overlap wins; if equal, the closer region centre wins
            if score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and dist < best_dist):
                best, best_score, best_dist = w, score, dist
        if best is None:
            unassigned.append(det)
            continue
        current = best.ppe.get(det.cls)
        if current is None or det.conf > current.conf:
            best.ppe[det.cls] = det
    return workers, unassigned
