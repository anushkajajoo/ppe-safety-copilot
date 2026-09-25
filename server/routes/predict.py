"""
Prediction API - upload one image, get the PPE compliance verdict back.

    POST /api/v1/predict      (aliases: POST /detect/image, POST /predict)
        image      : the picture to analyse (multipart file upload)
        required   : comma list of PPE every person must wear   (default "helmet,vest")
        conf       : detection confidence threshold             (default from settings)
        annotate   : true -> also return the drawn image as a data URI (default true)

The same code path as `python -m edge.detect`: YOLO detects, edge.association assigns
each PPE box to a person, and the simple rule produces
COMPLIANT / MISSING_HELMET / MISSING_VEST / MISSING_HELMET_AND_VEST.

PRIVACY: the uploaded image is decoded in memory, analysed and discarded. It is never
written to disk and never sent anywhere. Only the JSON summary below leaves this
function, and `image_stored` says so explicitly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from server.http_status import TOO_LARGE
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from edge.detect import COMPLIANT, REQUIRED_PPE, build_workers, draw, status_for
from server.audit import WRITE_LOCK, append_audit
from server import event_store
from server.database import get_db
from server.predict_service import ModelNotAvailable
from server.routes.copilot import record_suggestions
from shared.decision import decide_frame, decide_person
from shared.safe_names import safe_filename
from shared.privacy import NO_EVIDENCE, save_masked_snapshot

router = APIRouter(tags=["predict"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024        # 10 MB is plenty for a camera frame

VALID_PPE = {"helmet", "vest", "mask"}


# ------------------------------------------------------------------ response shape
class DetectedItem(BaseModel):
    cls: str
    conf: float
    box: List[float] = Field(description="x1, y1, x2, y2 in pixels")


class PersonResult(BaseModel):
    id: str = Field(description="Person-1, Person-2, ... within this image")
    conf: float
    box: List[float]
    wearing: Dict[str, float] = Field(default_factory=dict, description="PPE class -> confidence")
    missing: List[str] = Field(default_factory=list)
    status: str = Field(description="COMPLIANT | MISSING_HELMET | MISSING_VEST | MISSING_HELMET_AND_VEST")
    # The operational decision for this worker, with the rule and evidence behind it.
    decision: str = Field(default="GO", description="GO | REVIEW | STOP")
    reason: str = ""
    rule: str = ""
    uncertain: List[str] = Field(default_factory=list)
    evidence: List[Dict] = Field(default_factory=list)


class PredictResponse(BaseModel):
    filename: Optional[str] = None
    width: int
    height: int
    required: List[str]
    people: List[PersonResult]
    people_count: int
    # The frame's overall decision: the worst of everyone in it.
    decision: str = "GO"
    reason: str = ""
    rule: str = ""
    threshold: float = 0.5
    violations: int
    compliant: int
    counts_by_class: Dict[str, int]
    unassigned_ppe: List[DetectedItem]
    inference_ms: float
    image_stored: bool = Field(default=False,
                               description="the UPLOADED image is never written to disk")
    event_snapshot_stored: bool = Field(
        default=False,
        description="a face-masked snapshot of a violation was stored (off unless "
                    "PPE_STORE_SNAPSHOTS=true)")
    privacy_status: str = Field(default="NO_EVIDENCE",
                                description="NO_EVIDENCE | FACE_BLUR_OK | EVIDENCE_WITHHELD")
    annotated_image: Optional[str] = Field(
        default=None,
        description="data:image/jpeg;base64,... of the drawn result, only when annotate=true")


def parse_required(raw: str) -> List[str]:
    items = [piece.strip().lower() for piece in raw.split(",") if piece.strip()]
    unknown = [i for i in items if i not in VALID_PPE]
    if unknown:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown PPE item(s): {unknown}. Choose from {sorted(VALID_PPE)}.")
    return items or list(REQUIRED_PPE)


@router.post("/api/v1/predict", response_model=PredictResponse, summary="PPE compliance for one image")
@router.post("/detect/image", response_model=PredictResponse, summary="PPE compliance for one image (alias)")
@router.post("/predict", response_model=PredictResponse, include_in_schema=False)
async def predict(request: Request,
                  image: UploadFile = File(..., description="jpg / png frame to analyse"),
                  required: str = Form("helmet,vest"),
                  conf: Optional[float] = Form(None),
                  annotate: bool = Form(True),
                  db: Session = Depends(get_db)) -> PredictResponse:
    import cv2                                  # imported here to keep app startup light

    required_items = parse_required(required)
    if conf is not None and not (0.0 < conf < 1.0):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="conf must be between 0 and 1.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(TOO_LARGE,
                            detail=f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="That file is not a readable image (expected jpg, png, bmp or webp).")

    service = request.app.state.predict_service
    try:
        detections, inference_ms = service.detect(frame, conf=conf)
    except ModelNotAvailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    workers, unassigned = build_workers(detections)

    settings = getattr(request.app.state, "settings", None)
    threshold = getattr(settings, "ppe_confidence_threshold", 0.50)
    margin = getattr(settings, "review_margin", 0.15)
    # the uploaded name is attacker-controlled text: it is a label here, never a path
    safe_name = safe_filename(image.filename or "upload")
    source_label = f"image:{safe_name}"

    people: List[PersonResult] = []
    decisions = []
    for worker in workers:
        verdict, missing = status_for(worker, required_items)
        wearing = {name: round(det.conf, 3) for name, det in sorted(worker.ppe.items())}
        # GO / REVIEW / STOP from the confidences themselves, so the threshold that
        # produced the decision is the one quoted in its reason (shared/decision.py).
        call = decide_person(wearing, required_items, threshold, margin,
                             person_id=worker.display_id, source=source_label)
        decisions.append(call)
        people.append(PersonResult(
            id=worker.display_id, conf=round(worker.conf, 3), box=[round(v, 1) for v in worker.box],
            wearing=wearing, missing=missing, status=verdict,
            decision=call.decision, reason=call.reason, rule=call.rule,
            uncertain=call.uncertain, evidence=[e.as_dict() for e in call.evidence]))

    frame_call = decide_frame(decisions, source=source_label)

    # One line per evaluation - including the clean ones. Without the clean ones there is
    # no denominator, and "compliance %" becomes a number nobody can defend.
    analytics = getattr(request.app.state, "decision_log", None)
    if analytics is not None:
        analytics.record(source_label, frame_call.decision, people=len(people),
                         violations=sum(1 for p in people if p.decision != "GO"),
                         missing=sorted({item for p in people for item in p.missing}),
                         required=required_items, throttle=False)

    counts: Dict[str, int] = {}
    for det in detections:
        counts[det.cls] = counts.get(det.cls, 0) + 1

    # ---- evidence, log, structured record, proposal -------------------------
    # Order matters: the masked snapshot is produced first so the event can carry its
    # path, then the flat log line, then the structured event (the source of truth for
    # alerts, history and analytics), and last the copilot's proposal about it.
    log = getattr(request.app.state, "violation_log", None)
    snapshot_path, privacy_status = None, NO_EVIDENCE
    offenders = [person for person in people if person.status != COMPLIANT]

    if offenders and settings is not None and getattr(settings, "store_snapshots", False):
        # keep_boxes: the PPE evidence is restored after masking, so blurring a face can
        # never erase the helmet or vest the violation is about.
        keep = [tuple(det.box) for det in detections if det.cls in ("helmet", "vest")]
        saved, privacy_status = save_masked_snapshot(
            frame, [tuple(worker.box) for worker in workers], settings.evidence_dir,
            name_prefix="image", keep_boxes=keep)
        snapshot_path = str(saved) if saved else None

    if log is not None:
        log.log_people(source_label,
                       [person.model_dump() for person in people], required_items,
                       snapshot=snapshot_path, privacy_status=privacy_status)

    # The structured record does NOT depend on the flat log existing: it is what the
    # alert list, the history table, the detail panel and the statistics are built from.
    with WRITE_LOCK:
        for index, (person, call) in enumerate(zip(people, decisions), start=1):
            event, created = event_store.record_event(
                db, camera_id=source_label, person_id=person.id,
                decision=call.decision, required=required_items,
                detected=dict(person.wearing), missing=call.missing,
                uncertain=call.uncertain, threshold=threshold, rule_id=call.rule,
                confidence=person.conf, person_index=index,
                merge_window_s=getattr(settings, "event_merge_window_s", 30.0),
                snapshot_path=snapshot_path if call.decision != "GO" else None)
            if created and call.decision != "GO":
                append_audit(db, actor="system", action="SAFETY_EVENT",
                             event_id=event.event_id,
                             details={"source": source_label, "decision": call.decision,
                                      "rule": call.rule, "person": event.person_id,
                                      "missing": list(call.missing)})
        db.commit()

    # The copilot PROPOSES what to do about each offender. It never acts: every
    # suggestion waits for a named person to approve or reject it in the dashboard
    # (shared/copilot.py, server/routes/copilot.py).
    if offenders:
        with WRITE_LOCK:
            for person in offenders:
                record_suggestions(db, source_label, person.id, person.status,
                                   person.missing, confidence=person.conf)
            db.commit()

    annotated = None
    if annotate:
        drawn = draw(frame.copy(), workers, unassigned, required_items)
        ok, buffer = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            import base64
            annotated = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()

    height, width = frame.shape[:2]
    return PredictResponse(
        filename=image.filename, width=width, height=height, required=required_items,
        people=people, people_count=len(people),
        decision=frame_call.decision, reason=frame_call.reason, rule=frame_call.rule,
        threshold=threshold,
        violations=sum(1 for p in people if p.status != COMPLIANT),
        compliant=sum(1 for p in people if p.status == COMPLIANT),
        counts_by_class=counts,
        unassigned_ppe=[DetectedItem(cls=d.cls, conf=round(d.conf, 3),
                                     box=[round(v, 1) for v in d.box]) for d in unassigned],
        inference_ms=round(inference_ms, 1),
        # The upload itself is decoded in memory and dropped - always. Whether a
        # MASKED event snapshot was kept is a separate fact and says so separately.
        image_stored=False, event_snapshot_stored=snapshot_path is not None,
        privacy_status=privacy_status, annotated_image=annotated)
