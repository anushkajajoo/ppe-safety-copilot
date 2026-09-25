"""
Simple PPE detection on ONE image, a video file, or the webcam.

This is the small, explainable entry point asked for in the project brief:

    python -m edge.detect --source samples\\worker.jpg     # one image
    python -m edge.detect --source samples\\clip.mp4       # a video file
    python -m edge.detect --source 0                      # the webcam

What it does, per frame:
    1. YOLO detects person / helmet / vest / mask boxes.
    2. edge.association decides WHICH person each PPE box belongs to
       (helmet+mask must overlap the head region, vest the torso region).
    3. A simple rule compares each person's PPE against REQUIRED_PPE and gives
       COMPLIANT / MISSING_HELMET / MISSING_VEST / MISSING_HELMET_AND_VEST.
    4. Boxes, class names, confidences and the status are drawn on the frame.

Everything runs on THIS computer: the frame is read, processed and shown locally,
and nothing is uploaded anywhere. That is the privacy-preserving part of the title.

Difference from `edge.run_edge`: this file looks at ONE frame at a time. run_edge
adds tracking, zones, a temporal filter over 15 frames, the UNCERTAIN state and
event reporting. Use this one to demonstrate and explain detection; use run_edge
for the full system.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from edge.association import associate
from edge.detector import NAME_ALIASES
from edge.types import Detection, Worker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "ppe4_yolo26n_best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# What every person must wear. Change this one list to change the site's rules.
REQUIRED_PPE = ["helmet", "vest"]

COMPLIANT = "COMPLIANT"
WINDOW_TITLE = "PPE detection - press Q or Esc to quit"

# BGR colours (OpenCV order), not RGB.
GREEN, RED, AMBER, BLUE, GREY = (60, 180, 75), (40, 40, 230), (0, 170, 255), (255, 170, 0), (160, 160, 160)
PPE_COLOUR = {"helmet": BLUE, "vest": AMBER, "mask": (200, 120, 255)}


# --------------------------------------------------------------------- the rule
def status_for(worker: Worker, required: Sequence[str] = REQUIRED_PPE) -> Tuple[str, List[str]]:
    """
    Single-frame compliance for one person.

    Returns e.g. ("COMPLIANT", []) or ("MISSING_HELMET_AND_VEST", ["helmet", "vest"]).
    The order of `required` decides the order inside the status string, so the
    wording is stable and easy to test.
    """
    missing = [item for item in required if item not in worker.ppe]
    if not missing:
        return COMPLIANT, []
    return "MISSING_" + "_AND_".join(item.upper() for item in missing), missing


def should_quit(key: int, window_closed: bool = False) -> bool:
    """
    Should the live loop stop?

    OpenCV only reports key presses while ITS window has focus - pressing Q in the
    terminal does nothing - so we also stop when the window's X button was used, and
    we accept Esc as well as Q. Ctrl+C in the terminal is handled separately.
    """
    return window_closed or key in (ord("q"), ord("Q"), 27)


def build_workers(detections: List[Detection], min_ioa: float = 0.5) -> Tuple[List[Worker], List[Detection]]:
    """Split detections into people and PPE, then give each PPE box to a person."""
    workers, ppe = [], []
    for det in detections:
        if det.cls == "person":
            index = len(workers)
            workers.append(Worker(track_id=index, display_id=f"Person-{index + 1}",
                                  box=det.box, conf=det.conf))
        else:
            ppe.append(det)
    workers, unassigned = associate(workers, ppe, min_ioa=min_ioa)
    return workers, unassigned


# --------------------------------------------------------------------- drawing
def draw(frame, workers: List[Worker], unassigned: List[Detection],
         required: Sequence[str] = REQUIRED_PPE):
    """Draw PPE boxes, person boxes and the compliance status onto the frame."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(text, x, y, colour):
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        y = max(y, th + 6)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y), colour, -1)
        cv2.putText(frame, text, (x + 3, y - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # PPE that could not be linked to anybody (e.g. a helmet on a table)
    for det in unassigned:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
        label(f"{det.cls} {det.conf:.2f} (unassigned)", x1, y1, GREY)

    violations = 0
    for worker in workers:
        status, missing = status_for(worker, required)
        if status != COMPLIANT:
            violations += 1
        colour = GREEN if status == COMPLIANT else RED

        for det in worker.ppe.values():                     # the PPE this person wears
            px1, py1, px2, py2 = (int(v) for v in det.box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), PPE_COLOUR.get(det.cls, GREY), 2)
            label(f"{det.cls} {det.conf:.2f}", px1, py1, PPE_COLOUR.get(det.cls, GREY))

        x1, y1, x2, y2 = (int(v) for v in worker.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label(f"{worker.display_id} {worker.conf:.2f}  {status}", x1, y2 + 18, colour)

    header = f"people: {len(workers)}   violations: {violations}   required: {'+'.join(required)}"
    label(header, 8, 22, RED if violations else GREEN)
    return frame


def summarise(workers: List[Worker], required: Sequence[str] = REQUIRED_PPE) -> str:
    """One text block describing the frame - printed to the terminal."""
    if not workers:
        return "  no person detected"
    lines = []
    for worker in workers:
        status, _ = status_for(worker, required)
        worn = ", ".join(f"{k} {d.conf:.2f}" for k, d in sorted(worker.ppe.items())) or "none"
        lines.append(f"  {worker.display_id}  conf {worker.conf:.2f}  wearing: {worn}  ->  {status}")
    return "\n".join(lines)


# --------------------------------------------------------------------- plumbing
def source_kind(source: str) -> str:
    """'webcam' for 0/1/2..., otherwise 'image' or 'video' from the file extension."""
    if source.isdigit():
        return "webcam"
    suffix = Path(source).suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    return "unknown"


def to_detections(result, class_map, conf_threshold: float) -> List[Detection]:
    """Convert one Ultralytics result into OUR Detection objects."""
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, classes):
        name = class_map.get(int(cls_idx))
        if name is None or float(conf) < conf_threshold:
            continue
        detections.append(Detection(cls=name, conf=float(conf), box=(float(x1), float(y1), float(x2), float(y2))))
    return detections


def load_model(weights: Path, device: Optional[str]):
    """Load YOLO and map its class indices onto our four names."""
    from shared.device import resolve_device
    if not Path(weights).exists():
        print(f"\nERROR: Model not found: {weights}")
        print("HINT : train it first (see README, 'Training'), or pass --weights <file>.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: Ultralytics is not installed in this environment.")
        print("HINT : activate .venv, then  pip install -r requirements.txt")
        sys.exit(1)

    chosen, reason = resolve_device(device)
    print(f"Device  : {chosen}  ({reason})")

    model = YOLO(str(weights))
    class_map = {}
    for index, name in model.names.items():
        ours = NAME_ALIASES.get(str(name).strip().lower())
        if ours:
            class_map[int(index)] = ours
    if "person" not in class_map.values():
        print(f"\nERROR: this model has no 'person' class. Its classes are: {model.names}")
        sys.exit(1)
    return model, class_map


def main() -> int:
    ap = argparse.ArgumentParser(description="PPE detection on an image, a video or the webcam.")
    ap.add_argument("--source", default="0", help="image path, video path, or a webcam index like 0")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.35, help="ignore detections below this confidence")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto",
                    help="'auto' (GPU if it has free VRAM, else CPU), 'cpu', or '0'")
    ap.add_argument("--required", nargs="*", default=REQUIRED_PPE, choices=["helmet", "vest", "mask"],
                    help="PPE every person must wear")
    ap.add_argument("--save", default="", help="write the annotated image/video to this path")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = ap.parse_args()

    import cv2

    kind = source_kind(args.source)
    if kind == "unknown":
        print(f"\nERROR: Unsupported source: {args.source}")
        print("HINT : use an image (.jpg/.png), a video (.mp4/.avi), or a webcam index like 0.")
        return 1
    if kind != "webcam" and not Path(args.source).exists():
        print(f"\nERROR: File not found: {args.source}")
        return 1

    model, class_map = load_model(Path(args.weights), args.device)
    required = list(args.required)
    print(f"Model   : {args.weights}")
    print(f"Source  : {args.source} ({kind})")
    print(f"Required: {'+'.join(required)}\n")

    from shared.device import resolve_device
    resolved, _ = resolve_device(args.device)
    predict = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, device=resolved)

    # ---------------------------------------------------------------- one image
    if kind == "image":
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"\nERROR: Could not read the image: {args.source}")
            return 1
        result = model.predict(frame, **predict)[0]
        workers, unassigned = build_workers(to_detections(result, class_map, args.conf))
        print(summarise(workers, required))
        frame = draw(frame, workers, unassigned, required)

        out = args.save or str(ROOT / "docs" / "evaluation" / f"detect_{Path(args.source).stem}.jpg")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out, frame)
        print(f"\nAnnotated image saved to {out}")
        if not args.no_show:
            cv2.imshow("PPE detection - press any key to close", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    # ------------------------------------------------------- webcam / video file
    capture = cv2.VideoCapture(int(args.source) if kind == "webcam" else args.source)
    if not capture.isOpened():
        print("\nERROR: Could not open webcam." if kind == "webcam"
              else f"\nERROR: Could not open the video: {args.source}")
        print("HINT : close any other app using the camera, or try --source 1.")
        return 1

    writer = None
    if args.save:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if not args.no_show:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    print("Click the preview window, then press Q or Esc to stop "
          "(Ctrl+C in this terminal also works).\n")
    frames, violation_frames, started = 0, 0, time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, **predict)[0]
            workers, unassigned = build_workers(to_detections(result, class_map, args.conf))
            if any(status_for(w, required)[0] != COMPLIANT for w in workers):
                violation_frames += 1
            frame = draw(frame, workers, unassigned, required)

            frames += 1
            if frames % 30 == 0:
                fps_now = frames / max(1e-6, time.time() - started)
                print(f"frame {frames}  {fps_now:5.1f} FPS  people {len(workers)}")
            if writer is not None:
                writer.write(frame)
            if not args.no_show:
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                try:                      # the window may already be gone
                    closed = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
                except cv2.error:
                    closed = True
                if should_quit(key, closed):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopped by the user")
    finally:
        capture.release()                 # frees the camera
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)                    # Windows needs one more tick to close the window

    elapsed = max(1e-6, time.time() - started)
    print(f"\n{frames} frames in {elapsed:.1f}s = {frames / elapsed:.1f} FPS")
    print(f"frames with at least one violation: {violation_frames}")
    if args.save:
        print(f"Annotated video saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
