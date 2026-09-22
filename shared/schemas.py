"""
Shared data contract between the EDGE (laptop pipeline) and the SERVER (FastAPI).

WHY THIS FILE EXISTS
    Person 1 (edge) produces events, Person 2 (server) stores them. If each person
    writes their own version of "what an event looks like", integration on Day 3
    breaks. So both sides import the SAME Pydantic models from here.

RULE
    Changing this file = changing the API contract. Discuss it with your teammate,
    bump CONTRACT_VERSION, and update tests/test_schemas.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Enums: the fixed vocabulary of the system. Using enums (not free text)
# means a typo like "VIOLATON" is rejected by validation instead of silently
# stored in the database.
# --------------------------------------------------------------------------- #
class PPEClass(str, Enum):
    PERSON = "person"
    HELMET = "helmet"
    VEST = "vest"
    MASK = "mask"


class ZoneType(str, Enum):
    GENERAL = "GENERAL"          # e.g. helmet + vest
    CAUTION = "CAUTION"          # e.g. helmet + vest + mask
    RESTRICTED = "RESTRICTED"    # authorized access only


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    UNCERTAIN = "UNCERTAIN"
    POTENTIAL_VIOLATION = "POTENTIAL_VIOLATION"


class ReasonCode(str, Enum):
    MISSING_HELMET = "MISSING_HELMET"
    MISSING_VEST = "MISSING_VEST"
    MISSING_MASK = "MISSING_MASK"
    ZONE_REQUIREMENT_NOT_MET = "ZONE_REQUIREMENT_NOT_MET"
    RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"
    OUT_OF_CONFIGURED_ZONE = "OUT_OF_CONFIGURED_ZONE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CAMERA_OBSTRUCTED = "CAMERA_OBSTRUCTED"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DISMISSED = "DISMISSED"
    NOT_REQUIRED = "NOT_REQUIRED"


class PrivacyStatus(str, Enum):
    FACE_BLUR_OK = "FACE_BLUR_OK"            # evidence stored, faces blurred
    EVIDENCE_WITHHELD = "EVIDENCE_WITHHELD"  # blur failed -> NO image stored (fail-safe)
    NO_EVIDENCE = "NO_EVIDENCE"              # event type needs no image


class SyncStatus(str, Enum):  # used on the EDGE outbox (Day 3)
    SYNC_PENDING = "SYNC_PENDING"
    SYNCED = "SYNCED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Detection evidence attached to an event (only for the event frame, never
# for every frame -> data minimisation).
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    """Pixel coordinates in the ORIGINAL camera frame: (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @field_validator("x2")
    @classmethod
    def _x2_after_x1(cls, v: float, info):
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be greater than x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_after_y1(cls, v: float, info):
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be greater than y1")
        return v


class DetectionIn(BaseModel):
    object_class: PPEClass
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    assigned_to_worker: bool = True


# --------------------------------------------------------------------------- #
# The EVENT: what the edge sends to the server.
# --------------------------------------------------------------------------- #
class EventCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422, not silently stored

    # Generated ON THE EDGE. Re-sending the same event_id must not create a
    # duplicate row -> this is the idempotency key for offline sync.
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=8, max_length=64)

    device_id: str = Field(min_length=1, max_length=64, examples=["EDGE-01"])
    camera_id: str = Field(min_length=1, max_length=64, examples=["CAM-01"])
    session_id: str = Field(min_length=1, max_length=64,
                            description="Edge run id. Worker IDs restart each run, so "
                                        "(session_id, worker_display_id) is unique, not the display id alone.")
    worker_display_id: str = Field(pattern=r"^Worker-\d{3,6}$", examples=["Worker-017"])
    zone_id: Optional[str] = Field(default=None, max_length=64, examples=["welding-a"])

    status: ComplianceStatus
    reasons: List[ReasonCode] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)

    frames_in_window: int = Field(ge=1, le=10_000)
    frames_missing: int = Field(ge=0, le=10_000)

    model_version: str = Field(max_length=64, examples=["ppe4-yolo26n-v1"])
    rules_version: str = Field(max_length=32, examples=["1.0"])

    occurred_at: datetime = Field(default_factory=utc_now)
    privacy_status: PrivacyStatus = PrivacyStatus.NO_EVIDENCE
    detections: List[DetectionIn] = Field(default_factory=list, max_length=50)

    @field_validator("frames_missing")
    @classmethod
    def _missing_le_window(cls, v: int, info):
        if "frames_in_window" in info.data and v > info.data["frames_in_window"]:
            raise ValueError("frames_missing cannot exceed frames_in_window")
        return v

    @field_validator("reasons")
    @classmethod
    def _unique_reasons(cls, v: List[ReasonCode]):
        return list(dict.fromkeys(v))  # remove duplicates, keep order


class EventRead(EventCreate):
    model_config = ConfigDict(from_attributes=True, extra="ignore")
    received_at: datetime
    review_status: ReviewStatus
    evidence_available: bool = False


class HealthResponse(BaseModel):
    status: str
    database: str
    api_version: str
    contract_version: str = CONTRACT_VERSION
    server_time: datetime
