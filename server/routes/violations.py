"""
Read the local violation log.

    GET /api/v1/violations?limit=50     -> newest entries first

The log is written by the image, video and live-camera paths through
shared/violation_log.py. It holds metadata only - no pictures - which is what makes
the privacy claim in the project title defensible.
"""
from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["violations"])


class ViolationsResponse(BaseModel):
    count: int = Field(description="how many entries are returned")
    log_file: str
    violations: List[Dict] = Field(default_factory=list, description="newest first")


@router.get("/api/v1/violations", response_model=ViolationsResponse,
            summary="Recent violations recorded on this machine")
def list_violations(request: Request, limit: int = Query(50, ge=1, le=500)) -> ViolationsResponse:
    log = request.app.state.violation_log
    rows = log.recent(limit=limit)
    return ViolationsResponse(count=len(rows), log_file=str(log.path), violations=rows)
