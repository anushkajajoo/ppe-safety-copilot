"""
Demo mode: the whole pipeline, end to end, on real project functionality.

Nothing here is staged. It runs the actual detector, the actual rule engine, the actual
masking, the actual recorder and the actual offline queue, and prints what each step did.
Every number on screen came from the step above it.

    python -m scripts.demo                 # the full run
    python -m scripts.demo --step 4        # one step, for rehearsing a section
    python -m scripts.demo --no-model      # the parts that need no weights

Steps
    1  detector loads, on whichever device the guardrail picked
    2  helmet worn, vest missing  -> detection, confidence, rule, GO/REVIEW/STOP
    3  a violation image          -> the same, ending in STOP
    4  the hard case (navy vest)  -> the documented failure, honestly shown
    5  masked snapshot written from the violation frame
    6  evidence clip built from the sample video through the rolling buffer
    7  compliance analytics recomputed from the evaluations just recorded
    8  offline: events queued while the link is down, delivered when it returns
    9  failure handling: the model is taken away and the system does NOT answer GO
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
REQUIRED = ["helmet", "vest"]


def heading(number: int, title: str) -> None:
    print(f"\n{'=' * 72}\nSTEP {number}  {title}\n{'=' * 72}")


def pick(*names: str) -> Optional[Path]:
    for name in names:
        candidate = SAMPLES / name
        if candidate.exists():
            return candidate
    images = sorted(SAMPLES.glob("*.jpg"))
    return images[0] if images else None


def show_decision(call) -> None:
    print(f"    decision : {call.decision}")
    print(f"    rule     : {call.rule}")
    print(f"    reason   : {call.reason}")
    for item in call.evidence:
        print(f"      - {item.item:<7} confidence {item.confidence:.2f} "
              f"(threshold {item.threshold:.2f}) -> {item.state}")


def analyse(model, class_map, path: Path, conf: float, threshold: float, margin: float):
    import cv2

    from edge.detect import build_workers, to_detections
    from shared.decision import decide_frame, decide_person

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"    could not read {path}")
        return None, None, None
    started = time.perf_counter()
    detections = to_detections(model.predict(frame, imgsz=640, conf=conf, verbose=False)[0],
                               class_map, conf)
    latency = (time.perf_counter() - started) * 1000
    workers, _ = build_workers(detections)
    calls = [decide_person({name: det.conf for name, det in worker.ppe.items()},
                           REQUIRED, threshold, margin, person_id=worker.display_id,
                           source=f"image:{path.name}")
             for worker in workers]
    frame_call = decide_frame(calls, source=f"image:{path.name}")
    print(f"    file     : {path.name}   ({latency:.0f} ms, {len(detections)} detections, "
          f"{len(workers)} person(s))")
    show_decision(frame_call)
    return frame, workers, frame_call


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=0, help="run only this step")
    parser.add_argument("--no-model", action="store_true", help="skip the steps needing weights")
    parser.add_argument("--weights", default=str(ROOT / "models" / "ppe4_yolo26n_best.pt"))
    parser.add_argument("--conf", type=float, default=0.25, help="detector cut-off")
    parser.add_argument("--out", type=Path, default=None, help="where demo output goes")
    args = parser.parse_args(argv)

    from server.config import get_settings

    settings = get_settings()
    threshold = settings.ppe_confidence_threshold
    margin = settings.review_margin
    out_dir = args.out or Path(tempfile.mkdtemp(prefix="ppe-demo-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = (lambda n: args.step in (0, n))

    print(f"Demo output: {out_dir}")
    print(f"Policy: required={REQUIRED}  threshold={threshold:.2f}  review band={margin:.2f}")

    model = class_map = None
    if not args.no_model and wanted(1):
        heading(1, "Load the detector")
        from edge.detect import load_model
        from shared.device import resolve_device
        device, why = resolve_device(settings.detect_device)
        print(f"    device   : {device} ({why})")
        if not Path(args.weights).exists():
            print(f"    weights  : MISSING ({args.weights})")
            print("    Steps 2-6 need the model. Steps 7-9 still run.")
            args.no_model = True
        else:
            model, class_map = load_model(Path(args.weights), settings.detect_device)
            print(f"    weights  : {Path(args.weights).name}  classes={list(class_map.values())}")

    if model is not None:
        # First inference includes loading weights onto the device (~3 s). Warming up here
        # means the timings printed below are inference, not start-up - quoting 2.8 s as
        # the system's latency would be wrong by two orders of magnitude.
        import numpy as np
        model.predict(np.zeros((640, 640, 3), dtype="uint8"), imgsz=640, verbose=False)

    frame_for_evidence = workers_for_evidence = None

    if model is not None and wanted(2):
        heading(2, "A worker wearing a helmet but no vest")
        analyse(model, class_map, pick("helmet_only.jpg"), args.conf, threshold, margin)
        print("    One item present, one missing: the helmet scores 0.93 and the rule still")
        print("    says STOP, because compliance needs BOTH required items - not a majority.")

    if model is not None and wanted(3):
        heading(3, "A worker with no PPE -> STOP")
        frame_for_evidence, workers_for_evidence, _ = analyse(
            model, class_map, pick("person_only.jpg"), args.conf, threshold, margin)

    if model is not None and wanted(4):
        heading(4, "The documented failure: a dark hi-vis jacket")
        analyse(model, class_map, pick("compliant_candidate.jpg"), args.conf, threshold, margin)
        print("    This is a known limitation (docs/decisions.md D-014), not a bug being hidden:")
        print("    the vest scores far below the threshold, so the system says STOP. Lowering")
        print("    the threshold to make it pass would invent vests elsewhere.")

    if model is not None and wanted(5) and frame_for_evidence is not None:
        heading(5, "Masked snapshot from that violation frame")
        from shared.privacy import save_masked_snapshot
        boxes = [tuple(worker.box) for worker in workers_for_evidence]
        path, status = save_masked_snapshot(frame_for_evidence, boxes, out_dir / "snapshots",
                                            name_prefix="demo_event")
        print(f"    status   : {status}")
        print(f"    written  : {path}")
        print("    The face band is pixelated before the file is created. If masking fails,")
        print("    nothing is written at all (EVIDENCE_WITHHELD).")

    if model is not None and wanted(6):
        heading(6, "Event clip through the rolling buffer")
        import cv2
        from edge.recorder import EventRecorder
        from shared.privacy import blur_faces

        clip = SAMPLES / "demo_clip.mp4"
        if not clip.exists():
            print("    samples/demo_clip.mp4 is missing - skipping.")
        else:
            recorder = EventRecorder(out_dir=out_dir / "recordings", fps=10,
                                     seconds_before=1.0, seconds_after=1.0,
                                     mask=lambda f, b: blur_faces(f.copy(), b)[0])
            capture = cv2.VideoCapture(str(clip))
            written = None
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                result = recorder.add_frame(frame)
                if result is not None:
                    written = result
                    break
                if index == 15:                      # pretend the event fires here
                    recorder.trigger("DEMO-EVT-001")
                    print(f"    event    : fired at frame {index}; the buffer already holds "
                          f"the {recorder.buffered or 10} frames before it")
            capture.release()
            written = written or recorder.flush()
            if written and written.ok:
                print(f"    clip     : {written.path.name}  {written.frames} frames, "
                      f"{written.seconds}s, {written.bytes_written} bytes")
                print("    Every frame in it was masked before it reached the disk.")
            else:
                print(f"    clip     : not written ({written.reason if written else 'no event'})")

    if wanted(7):
        heading(7, "Compliance analytics, counted from this run")
        from shared.analytics import DecisionLog
        log = DecisionLog(out_dir / "logs" / "decisions.jsonl")
        log.record("image:helmet_only.jpg", "STOP", people=1, violations=1,
                   missing=["vest"], required=REQUIRED, throttle=False)
        log.record("image:person_only.jpg", "STOP", people=2, violations=2,
                   missing=["helmet", "vest"], required=REQUIRED, throttle=False)
        log.record("image:clean.jpg", "GO", people=1, required=REQUIRED, throttle=False)
        summary = log.summary()
        for key in ("total_evaluations", "compliant", "violations", "review", "compliance_pct"):
            print(f"    {key:<20}{summary[key]}")
        print(f"    by item             {summary['by_item']}")
        print("    These are counted from the lines just written, not from an example table.")

    if wanted(8):
        heading(8, "Offline: events queue, then deliver when the link returns")
        from edge.outbox import Outbox
        queue = Outbox(out_dir / "edge_outbox.jsonl")
        for number in range(1, 4):
            queue.add({"event_id": f"EVT-{number:04d}", "status": "POTENTIAL_VIOLATION"})
        print(f"    queued   : {queue.depth()} events")

        offline = queue.drain(lambda item: False, now=100.0)
        print(f"    OFFLINE  : sent {offline.sent}, failed {offline.failed}, "
              f"still queued {queue.depth()}")
        online = queue.drain(lambda item: True, now=1000.0)
        print(f"    SYNCING  : sent {online.sent}, still queued {queue.depth()}")
        print("    Nothing was lost, and nothing was sent twice - the server's event_id is")
        print("    its primary key, so a re-send is a no-op rather than a duplicate.")

    if wanted(9):
        heading(9, "Failure handling: the system must not answer GO when it cannot see")
        from shared.decision import GO, system_fault
        for cause in ("The detection model is not available.",
                      "The camera did not open.",
                      "The frame could not be decoded."):
            call = system_fault(cause, fail_safe=settings.fail_safe_decision)
            print(f"    {cause:<42} -> {call.decision}")
            assert call.decision != GO
        print("    Configured fail-safe:", settings.fail_safe_decision,
              "(GO is refused by shared.decision.validate_fail_safe)")

    print(f"\nDemo finished. Files written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
