"""
Zone API.

POST /api/v1/zones/sync   edge uploads its zones.yaml at startup (needs X-API-Key).
                          Upsert: new zones are created, changed zones get config_version + 1.
GET  /api/v1/zones        list zones (dashboard uses this to draw zone analytics).
"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.audit import WRITE_LOCK, append_audit
from server.database import get_db
from server.models import Zone
from server.security import require_edge_key
from shared.schemas import ZoneIn, ZoneRead

router = APIRouter(prefix="/api/v1/zones", tags=["zones"])
FIELDS = ("zone_name", "zone_type", "camera_id", "polygon", "required_ppe", "authorization_required", "priority", "active")


def _values(z: ZoneIn) -> dict:
    d = z.model_dump(mode="json")
    return {k: d[k] for k in FIELDS}


@router.post("/sync", response_model=List[ZoneRead])
def sync_zones(zones: List[ZoneIn], db: Session = Depends(get_db), _: str = Depends(require_edge_key)):
    with WRITE_LOCK:
        return _sync(zones, db)


def _sync(zones: List[ZoneIn], db: Session):
    changed = []
    for z in zones:
        vals = _values(z)
        row = db.get(Zone, z.zone_id)
        if row is None:
            db.add(Zone(zone_id=z.zone_id, config_version=1, **vals))
            changed.append({"zone_id": z.zone_id, "change": "created"})
        elif any(getattr(row, k) != v for k, v in vals.items()):
            for k, v in vals.items():
                setattr(row, k, v)
            row.config_version += 1
            changed.append({"zone_id": z.zone_id, "change": f"updated to v{row.config_version}"})
    if changed:
        append_audit(db, actor="edge", action="ZONES_SYNCED", details={"changes": changed})
    db.commit()
    return [ZoneRead.model_validate(r) for r in db.execute(select(Zone).order_by(Zone.zone_id)).scalars()]


@router.get("", response_model=List[ZoneRead])
def list_zones(db: Session = Depends(get_db)):
    return [ZoneRead.model_validate(r) for r in db.execute(select(Zone).order_by(Zone.zone_id)).scalars()]
