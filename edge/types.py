"""
Plain data structures passed between edge stages.

Each stage takes structured data in and gives structured data out, so every
stage can be unit-tested without a camera or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Detection:
    """One box from the detector, already mapped to OUR class names."""
    cls: str                 # "person" | "helmet" | "vest" | "mask"
    conf: float
    box: Box
    track_id: Optional[int] = None   # only meaningful for persons


@dataclass
class Worker:
    """A tracked person in the current frame, with the PPE assigned to them."""
    track_id: int
    display_id: str                          # "Worker-017"
    box: Box
    conf: float
    ppe: Dict[str, Detection] = field(default_factory=dict)   # best box per PPE class

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box = where the person stands on the floor."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


@dataclass
class Decision:
    """Output of the compliance engine for one worker in one frame."""
    display_id: str
    zone_id: Optional[str]
    zone_name: Optional[str]
    status: str                               # COMPLIANT | UNCERTAIN | POTENTIAL_VIOLATION
    reasons: List[str]
    confidence: float
    frames_in_window: int
    frames_missing: int
    required: List[str]
    item_state: Dict[str, str]                # {"helmet": "OK" | "MISSING" | "UNSURE" | "N/A"}
