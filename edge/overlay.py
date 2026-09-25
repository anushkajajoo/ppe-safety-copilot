"""
Drawing: zones, worker boxes, and a small label card per worker.
(OpenCV's built-in fonts are ASCII only, so we write OK / NO / ? / -- instead of ticks.)
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from edge.types import Decision, Worker
from edge.zones import ZoneManager

# BGR colours
STATUS_COLOR = {"COMPLIANT": (80, 185, 63), "UNCERTAIN": (36, 165, 245), "POTENTIAL_VIOLATION": (73, 81, 248)}
ZONE_COLOR = {"GENERAL": (80, 185, 63), "CAUTION": (36, 165, 245), "RESTRICTED": (73, 81, 248)}
ITEM_TEXT = {"OK": "OK", "MISSING": "NO", "UNSURE": "?", "N/A": "--"}
STATUS_TEXT = {"COMPLIANT": "COMPLIANT", "UNCERTAIN": "UNCERTAIN", "POTENTIAL_VIOLATION": "POTENTIAL VIOLATION"}


def draw_zones(frame, zm: ZoneManager) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for z in reversed(zm.zones):  # low priority first, so high-priority zones are drawn on top
        pts = np.array(zm.polygon_px(z, w, h), dtype=np.int32)
        color = ZONE_COLOR[z.zone_type]
        if z.zone_type != "GENERAL":
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        label = f"{z.zone_name} [{'+'.join(z.required_ppe) or 'authorized only'}]"
        cv2.putText(frame, label, (int(x) + 6, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)


def draw_workers(frame, workers: List[Worker], decisions: Dict[str, Decision]) -> None:
    for wk in workers:
        d = decisions.get(wk.display_id)
        if d is None:
            continue
        color = STATUS_COLOR[d.status]
        x1, y1, x2, y2 = map(int, wk.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        fx, fy = map(int, wk.foot_point)
        cv2.circle(frame, (fx, fy), 5, color, -1)
        for det in wk.ppe.values():
            a, b, c, e = map(int, det.box)
            cv2.rectangle(frame, (a, b), (c, e), (230, 230, 230), 1)

        lines = [f"{d.display_id}  {d.zone_name or 'no zone'}",
                 "  ".join(f"{k[0].upper()}:{ITEM_TEXT[d.item_state.get(k, 'N/A')]}" for k in ("helmet", "vest", "mask")),
                 STATUS_TEXT[d.status]]
        if d.reasons:
            lines.append(",".join(r.replace("MISSING_", "NO_") for r in d.reasons[:2]))
        card_w, line_h = 250, 18
        cy = max(0, y1 - line_h * len(lines) - 8)
        cv2.rectangle(frame, (x1, cy), (x1 + card_w, cy + line_h * len(lines) + 6), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, cy), (x1 + card_w, cy + line_h * len(lines) + 6), color, 1)
        for i, text in enumerate(lines):
            c = color if i == 2 else (235, 235, 235)
            cv2.putText(frame, text, (x1 + 6, cy + 16 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.47, c, 1, cv2.LINE_AA)


def draw_hud(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (15, 15, 15), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
