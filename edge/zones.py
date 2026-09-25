"""
Zone manager: loads configs/zones.yaml + configs/policies.yaml and answers
"which zone is this point in, and what does that zone require?".

Zones are polygons in the IMAGE, not GPS. The worker's position is the foot point
(bottom-centre of the person box), because that is where the person stands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from edge.geometry import point_in_polygon

VALID_TYPES = {"GENERAL", "CAUTION", "RESTRICTED"}
VALID_PPE = {"helmet", "vest", "mask"}


@dataclass
class Zone:
    zone_id: str
    zone_name: str
    zone_type: str
    polygon: List[Tuple[float, float]]
    required_ppe: List[str]
    authorization_required: bool
    priority: int = 0
    active: bool = True
    authorization_windows: List[dict] = field(default_factory=list)

    def is_authorized_now(self, now: Optional[datetime] = None) -> bool:
        """True if a supervisor-approved access window is open right now."""
        if not self.authorization_required:
            return True
        now_t = (now or datetime.now()).time()
        for w in self.authorization_windows:
            start = time.fromisoformat(str(w["start"]))
            end = time.fromisoformat(str(w["end"]))
            if start <= now_t <= end:
                return True
        return False

    def to_api(self, camera_id: str) -> dict:
        """Payload for the backend's /zones/sync endpoint (matches shared.schemas.ZoneIn)."""
        return {"zone_id": self.zone_id, "zone_name": self.zone_name, "zone_type": self.zone_type,
                "camera_id": camera_id, "polygon": [list(p) for p in self.polygon],
                "required_ppe": self.required_ppe, "authorization_required": self.authorization_required,
                "priority": self.priority, "active": self.active}


class ZoneManager:
    def __init__(self, camera_id: str, frame_size: Tuple[int, int], zones: List[Zone], rules_version: str,
                 policies: dict):
        self.camera_id = camera_id
        self.frame_size = frame_size
        self.zones = sorted([z for z in zones if z.active], key=lambda z: -z.priority)
        self.rules_version = rules_version
        self.policies = policies
        self._scaled_for: Optional[Tuple[int, int]] = None
        self._scaled: Dict[str, List[Tuple[float, float]]] = {}

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_files(cls, zones_path: Path, policies_path: Path) -> "ZoneManager":
        zcfg = yaml.safe_load(Path(zones_path).read_text(encoding="utf-8"))
        pcfg = yaml.safe_load(Path(policies_path).read_text(encoding="utf-8"))
        types = pcfg["zone_types"]
        zones, seen = [], set()
        for z in zcfg["zones"]:
            zt = str(z["zone_type"]).upper()
            if zt not in VALID_TYPES:
                raise ValueError(f"zone {z['zone_id']}: unknown zone_type {zt}")
            if z["zone_id"] in seen:
                raise ValueError(f"duplicate zone_id {z['zone_id']}")
            seen.add(z["zone_id"])
            poly = [tuple(map(float, p)) for p in z["polygon"]]
            if len(poly) < 3:
                raise ValueError(f"zone {z['zone_id']}: polygon needs at least 3 points")
            req = z.get("required_ppe", types[zt]["required_ppe"])
            bad = set(req) - VALID_PPE
            if bad:
                raise ValueError(f"zone {z['zone_id']}: unknown PPE {bad}")
            zones.append(Zone(
                zone_id=z["zone_id"], zone_name=z.get("zone_name", z["zone_id"]), zone_type=zt, polygon=poly,
                required_ppe=list(req),
                authorization_required=bool(z.get("authorization_required", types[zt].get("authorization_required", False))),
                priority=int(z.get("priority", 0)), active=bool(z.get("active", True)),
                authorization_windows=list(z.get("authorization_windows") or []),
            ))
        return cls(zcfg["camera_id"], tuple(zcfg.get("frame_size", [1280, 720])), zones,
                   str(pcfg.get("rules_version", "1.0")), pcfg)

    # ---------------------------------------------------------------- queries
    def _polygons_for(self, width: int, height: int) -> Dict[str, List[Tuple[float, float]]]:
        if self._scaled_for != (width, height):
            sx, sy = width / self.frame_size[0], height / self.frame_size[1]
            self._scaled = {z.zone_id: [(x * sx, y * sy) for x, y in z.polygon] for z in self.zones}
            self._scaled_for = (width, height)
        return self._scaled

    def polygon_px(self, zone: Zone, width: int, height: int) -> List[Tuple[float, float]]:
        return self._polygons_for(width, height)[zone.zone_id]

    def zone_at(self, x: float, y: float, width: int, height: int) -> Optional[Zone]:
        """Highest-priority active zone containing the point, or None."""
        polys = self._polygons_for(width, height)
        for z in self.zones:                     # already sorted by priority (high first)
            if point_in_polygon(x, y, polys[z.zone_id]):
                return z
        return None
