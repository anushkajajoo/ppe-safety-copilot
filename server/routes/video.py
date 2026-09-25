"""
Video analysis API - upload a clip, get a PPE compliance summary back.

    POST /detect/video   (alias: POST /api/v1/detect/video)
        video        : the clip to analyse (multipart upload, <= 50 MB)
        required     : comma list of PPE every person must wear  (default "helmet,vest")
        conf         : detection confidence threshold            (default from settings)
        every_nth    : analyse every Nth frame                   (default 5)
        max_frames   : stop after this many ANALYSED frames      (default 60)
        annotate     : return the worst frame as a picture       (default true)

WHY SAMPLE FRAMES?
    A 30-second clip at 30 fps is 900 frames. At ~40 ms each that is 36 seconds of
    GPU time for a demo that has to feel live. Every 5th frame gives the same
    verdicts in a fifth of the time. The response always states how many frames
    were analysed, so nothing is hidden.

WHY NO TEMPORAL FILTER HERE?
    The 15-frame filter in edge/temporal.py needs a tracker following the same
    person across CONSECUTIVE frames. Sampling breaks that chain, so this endpoint
    reports per-frame verdicts and a summary. The live pipeline (edge/run_edge.py)
    is where temporal confirmation belongs.

PRIVACY: OpenCV cannot read a video from memory, so the upload is written to the
operating system's temporary folder, read, and then DELETED in a finally block -
it is gone before the response is sent. No frames are stored. The only picture
that can come back is the single worst frame, inside the JSON response, and only
when annotate=true.
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from server.http_status import TOO_LARGE
from pydantic import BaseModel, Field

from edge.detect import COMPLIANT, build_workers, draw, status_for
from server.predict_service import ModelNotAvailable
from shared.decision import GO, decide_frame, decide_person, worst_of
from shared.safe_names import safe_filename
from server.routes.predict import parse_required

router = APIRouter(tags=["predict"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024          # 50 MB
MAX_ANALYSED_FRAMES = 300                    # hard ceiling, whatever the request asks for
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


class FrameResult(BaseModel):
    frame: int = Field(description="frame number in the original video")
    time_s: float = Field(description="position in the clip, seconds")
    people: int
    violations: int
    statuses: List[str] = Field(default_factory=list)


class VideoResponse(BaseModel):
    filename: Optional[str] = None
    width: int
    height: int
    source_fps: float
    source_frames: int
    duration_s: float
    required: List[str]
    every_nth: int
    frames_analysed: int
    people_max: int = Field(description="most people seen in any single analysed frame")
    frames_with_violation: int
    violation_rate: float = Field(description="frames_with_violation / frames_analysed, 0-1")
    status_counts: Dict[str, int] = Field(description="how many person-verdicts of each kind")
    timeline: List[FrameResult]
    worst_frame: Optional[FrameResult] = None
    worst_frame_image: Optional[str] = Field(default=None, description="data:image/jpeg;base64,...")
    processing_s: float
    video_stored: bool = Field(default=False, description="always false - the temp file is deleted")
    # The clip's overall decision: the worst any analysed frame produced.
    decision: str = "GO"
    reason: str = ""
    rule: str = ""
    threshold: float = 0.5


@router.post("/detect/video", response_model=VideoResponse, summary="PPE compliance for a video clip")
@router.post("/api/v1/detect/video", response_model=VideoResponse, include_in_schema=False)
async def detect_video(request: Request,
                       video: UploadFile = File(..., description="mp4 / avi / mov clip"),
                       required: str = Form("helmet,vest"),
                       conf: Optional[float] = Form(None),
                       every_nth: int = Form(5),
                       max_frames: int = Form(60),
                       annotate: bool = Form(True)) -> VideoResponse:
    import cv2

    required_items = parse_required(required)
    if conf is not None and not (0.0 < conf < 1.0):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="conf must be between 0 and 1.")
    if every_nth < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="every_nth must be 1 or more.")
    max_frames = max(1, min(int(max_frames), MAX_ANALYSED_FRAMES))

    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"Unsupported video type '{suffix}'. Use {sorted(VIDEO_SUFFIXES)}.")

    raw = await video.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(TOO_LARGE,
                            detail=f"Video is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    # OpenCV needs a real file path, so the clip goes to the OS temp folder and is
    # deleted in the finally block below - before this function returns.
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    service = request.app.state.predict_service
    started = time.perf_counter()

    settings_now = getattr(request.app.state, "settings", None)
    threshold = getattr(settings_now, "ppe_confidence_threshold", 0.50)
    margin = getattr(settings_now, "review_margin", 0.15)
    safe_name = safe_filename(video.filename or "upload")
    source_label = f"video:{safe_name}"

    try:
        with os.fdopen(handle, "wb") as f:
            f.write(raw)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                detail="That file could not be opened as a video.")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        timeline: List[FrameResult] = []
        status_counts: Dict[str, int] = {}
        people_max = 0
        frames_with_violation = 0
        worst: Optional[FrameResult] = None
        worst_image: Optional[str] = None
        frame_calls = []
        frame_index = -1

        while len(timeline) < max_frames:
            ok = capture.grab()                      # cheap: decodes nothing
            if not ok:
                break
            frame_index += 1
            if frame_index % every_nth:
                continue
            ok, frame = capture.retrieve()           # only now do we decode
            if not ok:
                break

            try:
                detections, _ = service.detect(frame, conf=conf)
            except ModelNotAvailable as exc:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

            workers, unassigned = build_workers(detections)
            statuses = [status_for(w, required_items)[0] for w in workers]
            calls = [decide_person({name: det.conf for name, det in w.ppe.items()},
                                   required_items, threshold, margin,
                                   person_id=w.display_id, source=source_label)
                     for w in workers]
            frame_calls.append(decide_frame(calls, source=source_label))
            for verdict in statuses:
                status_counts[verdict] = status_counts.get(verdict, 0) + 1
            violations = sum(1 for verdict in statuses if verdict != COMPLIANT)

            result = FrameResult(
                frame=frame_index,
                time_s=round(frame_index / source_fps, 2) if source_fps > 0 else 0.0,
                people=len(workers), violations=violations, statuses=statuses)
            timeline.append(result)

            people_max = max(people_max, len(workers))
            if violations:
                frames_with_violation += 1
            if worst is None or violations > worst.violations:
                worst = result
                if annotate:
                    drawn = draw(frame.copy(), workers, unassigned, required_items)
                    ok_enc, buffer = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    worst_image = ("data:image/jpeg;base64," +
                                   base64.b64encode(buffer.tobytes()).decode()) if ok_enc else None

            if width == 0 or height == 0:            # some containers do not report size
                height, width = frame.shape[:2]

        capture.release()
        if not timeline:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                detail="No frames could be read from that video.")

        # Phase 12: ONE summary line per clip - one per frame would flood the log.
        log = getattr(request.app.state, "violation_log", None)
        if log is not None:
            log.log_video(source_label, len(timeline), frames_with_violation,
                          frames_with_violation / len(timeline), people_max, status_counts,
                          required_items)

        # The clip's decision is the worst any analysed frame produced.
        clip_decision = worst_of([call.decision for call in frame_calls])
        worst_call = next((call for call in frame_calls if call.decision == clip_decision), None)

        analytics = getattr(request.app.state, "decision_log", None)
        if analytics is not None:
            analytics.record(source_label, clip_decision, people=people_max,
                             violations=frames_with_violation,
                             missing=sorted({item for call in frame_calls
                                             for item in call.missing}),
                             required=required_items, throttle=False)

        return VideoResponse(
            decision=clip_decision,
            reason=(worst_call.reason if worst_call is not None
                    else "No frames produced a decision."),
            rule=worst_call.rule if worst_call is not None else "R-NO-PEOPLE",
            threshold=threshold,
            filename=video.filename, width=width, height=height,
            source_fps=round(source_fps, 2), source_frames=source_frames,
            duration_s=round(source_frames / source_fps, 2) if source_fps > 0 else 0.0,
            required=required_items, every_nth=every_nth, frames_analysed=len(timeline),
            people_max=people_max, frames_with_violation=frames_with_violation,
            violation_rate=round(frames_with_violation / len(timeline), 3),
            status_counts=status_counts, timeline=timeline,
            worst_frame=worst, worst_frame_image=worst_image,
            processing_s=round(time.perf_counter() - started, 2), video_stored=False)
    finally:
        try:
            os.remove(temp_path)                     # the clip never outlives the request
        except OSError:
            pass
