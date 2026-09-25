"""
Live webcam monitoring in the browser.

    GET /live/stream?camera=0&fps=10&required=helmet,vest    -> MJPEG video stream
    GET /live/status                                         -> latest verdicts as JSON
    GET /live                                                -> the live page

HOW THE STREAM WORKS
    The response is `multipart/x-mixed-replace`: one JPEG after another on a single
    HTTP connection. A plain <img src="/live/stream"> renders it natively, so the
    page needs no video library, no WebRTC and no extra dependency. This is the
    oldest and simplest way to put a camera in a browser, which is exactly why it
    suits a college project.

WHAT IS AND IS NOT HERE
    Each frame gets a SINGLE-FRAME verdict (same rule as edge/detect.py). The
    15-frame temporal filter needs a tracker following one person across
    consecutive frames; that lives in edge/run_edge.py and is not duplicated here.
    So the live panel is for showing detection working, and run_edge is the full
    pipeline that confirms violations.

PRIVACY
    The camera is opened by this process, frames are annotated in memory and sent
    only to the browser on this machine (127.0.0.1). Nothing is written to disk and
    nothing leaves the laptop. The camera is released as soon as the browser closes
    the connection.
"""
from __future__ import annotations

import platform
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from edge.detect import COMPLIANT, build_workers, draw, status_for
from edge.recorder import EventRecorder
from server import event_store
from server.audit import WRITE_LOCK, append_audit
from server.config import FRONTEND_DIR
from server.predict_service import ModelNotAvailable
from server.routes.predict import parse_required
from shared.decision import GO, REVIEW, STOP, decide_frame, decide_person, system_fault
from shared.events import Debouncer, event_type_for, signature_of
from shared.privacy import blur_faces, save_masked_snapshot

router = APIRouter(tags=["live"])

BOUNDARY = "frame"
MAX_FPS = 30.0


class LiveState:
    """The newest frame's summary, shared between the streaming thread and /live/status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict = {"running": False, "people": 0, "violations": 0,
                            "statuses": [], "frames": 0, "fps": 0.0, "camera": None,
                            "required": [], "updated": None,
                            # the operational decision for the newest frame
                            "decision": GO, "reason": "Camera not started.",
                            "rule": "", "evidence": [], "dropped": 0, "notice": "",
                            "latency_ms": 0.0, "clips": 0, "snapshots": 0}

    def update(self, **fields) -> None:
        with self._lock:
            self._data.update(fields)

    def snapshot(self) -> Dict:
        with self._lock:
            return dict(self._data)


def get_state(request: Request) -> LiveState:
    state = getattr(request.app.state, "live_state", None)
    if state is None:
        state = LiveState()
        request.app.state.live_state = state
    return state


def default_camera(index: int):
    """Open the webcam. CAP_DSHOW avoids a slow start-up on Windows."""
    import cv2

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


@router.get("/live", include_in_schema=False)
def live_page():
    return FileResponse(FRONTEND_DIR / "live.html")


@router.get("/live/status", summary="Latest live-monitoring verdicts")
def live_status(request: Request) -> Dict:
    return get_state(request).snapshot()


@router.get("/live/stream", summary="Annotated webcam stream (MJPEG)")
def live_stream(request: Request, camera: int = 0, fps: float = 10.0,
                conf: Optional[float] = None, required: str = "helmet,vest") -> StreamingResponse:
    import cv2

    required_items = parse_required(required)
    if conf is not None and not (0.0 < conf < 1.0):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="conf must be between 0 and 1.")
    fps = max(1.0, min(float(fps), MAX_FPS))
    min_gap = 1.0 / fps

    service = request.app.state.predict_service
    factory = getattr(request.app.state, "camera_factory", default_camera)
    capture = factory(camera)
    if not capture.isOpened():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(f"Could not open camera {camera}. Close any other app using it "
                    f"(including 'python -m edge.detect --source 0') and try again."))

    settings = getattr(request.app.state, "settings", None)
    threshold = getattr(settings, "ppe_confidence_threshold", 0.50)
    margin = getattr(settings, "review_margin", 0.15)
    fail_safe = getattr(settings, "fail_safe_decision", REVIEW)
    store_snapshots = bool(getattr(settings, "store_snapshots", False))
    snapshot_dir = getattr(settings, "evidence_dir", Path("data/snapshots/masked"))
    source_label = f"camera:{camera}"

    # Event recording: a rolling buffer in memory, a clip only when an event fires.
    recorder = None
    if getattr(settings, "record_events", False):
        recorder = EventRecorder(
            out_dir=Path(getattr(settings, "recordings_dir", "data/recordings")) / f"camera_{camera}",
            fps=fps,
            seconds_before=float(getattr(settings, "clip_seconds_before", 5.0)),
            seconds_after=float(getattr(settings, "clip_seconds_after", 5.0)),
            max_bytes=int(getattr(settings, "max_clip_bytes", 25 * 1024 * 1024)),
            # every frame is masked before it reaches the disk
            mask=lambda frame, boxes: blur_faces(frame.copy(), boxes)[0])

    analytics = getattr(request.app.state, "decision_log", None)
    debouncer = getattr(request.app.state, "debouncer", None) or Debouncer(
        frames=getattr(settings, "violation_confirmation_frames", 3))
    merge_window = float(getattr(settings, "event_merge_window_s", 30.0))
    # the DATABASE session factory - deliberately not called `factory`, which is the camera
    sessions = getattr(request.app.state, "session_factory", None)
    state = get_state(request)
    state.update(running=True, camera=camera, required=required_items, frames=0, fps=0.0,
                 decision=GO, reason="Waiting for the first frame.", rule="", dropped=0,
                 clips=0, snapshots=0, notice="")

    def frames():
        nonlocal recorder
        counted, started = 0, time.perf_counter()
        dropped, clips, snapshots = 0, 0, 0
        last_event_at, event_gap_s = 0.0, 10.0
        latency_ms = 0.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                loop_started = time.perf_counter()

                try:
                    detections, latency_ms = service.detect(frame, conf=conf)
                except ModelNotAvailable as exc:
                    # FAIL-SAFE: a system that cannot see never reports that a worker is
                    # protected. The page shows REVIEW (or STOP) with the reason.
                    fault = system_fault(f"The detection model is not available ({exc}).",
                                         fail_safe=fail_safe, source=f"camera:{camera}",
                                         required=required_items)
                    state.update(running=True, decision=fault.decision, reason=fault.reason,
                                 rule=fault.rule, statuses=[], people=0, violations=0,
                                 updated=time.time())
                    break
                except Exception as exc:        # any detector failure is a fault, not a pass
                    fault = system_fault(f"The detector failed on this frame ({exc}).",
                                         fail_safe=fail_safe, source=f"camera:{camera}",
                                         required=required_items)
                    state.update(running=True, decision=fault.decision, reason=fault.reason,
                                 rule=fault.rule, updated=time.time())
                    dropped += 1
                    continue

                workers, unassigned = build_workers(detections)
                verdicts = [status_for(worker, required_items) for worker in workers]
                statuses: List[str] = [verdict for verdict, _ in verdicts]

                # GO / REVIEW / STOP for this frame (shared/decision.py)
                calls = [decide_person({name: det.conf for name, det in worker.ppe.items()},
                                       required_items, threshold, margin,
                                       person_id=worker.display_id, source=source_label)
                         for worker in workers]
                frame_call = decide_frame(calls, source=source_label)

                # Phase 12: log violations, throttled to one line per person+status per
                # 30 s - at 10 FPS an unthrottled log would grow by 600 lines a minute.
                log = getattr(request.app.state, "violation_log", None)
                if log is not None:
                    for worker, (verdict, missing) in zip(workers, verdicts):
                        log.log_person(f"camera:{camera}", worker.display_id, verdict,
                                       missing, worker.conf, required_items)
                counted += 1
                elapsed = max(1e-6, time.perf_counter() - started)
                state.update(running=True, people=len(workers),
                             violations=sum(1 for s in statuses if s != COMPLIANT),
                             statuses=statuses, frames=counted,
                             fps=round(counted / elapsed, 1), updated=time.time(),
                             decision=frame_call.decision, reason=frame_call.reason,
                             rule=frame_call.rule, dropped=dropped,
                             latency_ms=round(float(latency_ms or 0.0), 1),
                             evidence=[e.as_dict() for e in frame_call.evidence],
                             clips=clips, snapshots=snapshots)

                if analytics is not None:
                    analytics.record(source_label, frame_call.decision, people=len(workers),
                                     violations=sum(1 for s in statuses if s != COMPLIANT),
                                     missing=sorted({item for call in calls
                                                     for item in call.missing}),
                                     required=required_items)

                # ---- evidence, only when something actually happened -------------
                person_boxes = [tuple(worker.box) for worker in workers]
                if recorder is not None:
                    finished = recorder.add_frame(frame)
                    if finished is not None and finished.ok:
                        clips += 1              # a clip completed on this frame
                        if sessions is not None and finished.event_id:
                            try:
                                session = sessions()
                                try:
                                    event_store.attach_evidence(
                                        session, finished.event_id,
                                        recording_path=str(finished.path))
                                    session.commit()
                                finally:
                                    session.close()
                            except Exception:
                                pass
                # ---- structured events: debounced, then deduplicated --------------
                # One bad frame is not an event (debounce), and a continuing violation is
                # one event that gets longer, not a hundred rows (the store merges).
                event_id = None
                created_now = False
                if sessions is not None:
                    for index, (worker, call) in enumerate(zip(workers, calls), start=1):
                        if call.decision == GO:
                            debouncer.reset(source_label, worker.display_id)
                            continue
                        signature = signature_of(event_type_for(call.decision),
                                                 call.missing, call.uncertain)
                        if not debouncer.confirm(source_label, worker.display_id, signature):
                            continue
                        try:
                            with WRITE_LOCK:
                                session = sessions()
                                try:
                                    event, created = event_store.record_event(
                                        session, camera_id=source_label,
                                        person_id=worker.display_id, decision=call.decision,
                                        required=required_items,
                                        detected={n: d.conf for n, d in worker.ppe.items()},
                                        missing=call.missing, uncertain=call.uncertain,
                                        threshold=threshold, rule_id=call.rule,
                                        confidence=worker.conf, person_index=index,
                                        merge_window_s=merge_window)
                                    event_id = event.event_id
                                    created_now = created_now or created
                                    if created:
                                        append_audit(session, actor="system",
                                                     action="SAFETY_EVENT",
                                                     event_id=event.event_id,
                                                     details={"source": source_label,
                                                              "decision": call.decision,
                                                              "rule": call.rule,
                                                              "person": event.person_id,
                                                              "missing": list(call.missing)})
                                    session.commit()
                                finally:
                                    session.close()
                        except Exception:
                            pass            # a storage hiccup must not stop the camera

                if created_now and event_id and time.time() - last_event_at > event_gap_s:
                    created_now = False          # evidence is captured once per new event
                    last_event_at = time.time()
                    # With several offenders in one frame the evidence attaches to the last
                    # event created - one frame, one picture, and the picture shows them all.
                    snapshot_path = None
                    if store_snapshots:
                        keep = [tuple(det.box) for det in detections
                                if det.cls in ("helmet", "vest")]
                        saved, _status = save_masked_snapshot(
                            frame, person_boxes, snapshot_dir, name_prefix=event_id,
                            keep_boxes=keep)
                        if saved is not None:
                            snapshots += 1
                            snapshot_path = str(saved)
                    if recorder is not None:
                        recorder.trigger(event_id, person_boxes)
                    if snapshot_path and sessions is not None:
                        try:
                            session = sessions()
                            try:
                                event_store.attach_evidence(session, event_id,
                                                            snapshot_path=snapshot_path)
                                session.commit()
                            finally:
                                session.close()
                        except Exception:
                            pass

                drawn = draw(frame, workers, unassigned, required_items)
                ok_enc, buffer = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok_enc:
                    continue
                yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buffer.tobytes() + b"\r\n")

                spare = min_gap - (time.perf_counter() - loop_started)
                if spare > 0:
                    time.sleep(spare)           # keep the laptop cool, keep the browser smooth
        finally:
            if recorder is not None:
                result = recorder.flush()       # a clip in progress is not abandoned
                if result is not None and result.ok:
                    clips += 1
            capture.release()                   # the camera is freed the moment the tab closes
            # The decision, rule and reason describe the LAST FRAME the system judged.
            # Stopping the camera does not change what that frame showed, so none of them
            # are rewritten here - an earlier version replaced the reason with "Camera
            # stopped.", which left the rule saying R-MISSING-PPE next to a reason that
            # mentioned no PPE at all. The fact that the stream ended is its own field.
            state.update(running=False, statuses=[], people=0, violations=0,
                         clips=clips, snapshots=snapshots, notice="Camera stopped.")

    return StreamingResponse(frames(),
                             media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")
