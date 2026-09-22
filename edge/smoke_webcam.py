"""
Day 1 smoke test: run a PRETRAINED YOLO model on the webcam (or a video file)
and measure REAL FPS and inference time on this laptop.

This is NOT the final pipeline. It only proves: camera -> OpenCV -> YOLO on GPU -> screen.
With the COCO-pretrained model only "person" is meaningful for us (COCO has no
helmet/vest/mask classes) -- that is exactly why we fine-tune.

Run:
    python -m edge.smoke_webcam                      # webcam 0, yolo26n.pt
    python -m edge.smoke_webcam --source 1           # another camera
    python -m edge.smoke_webcam --source demo.mp4    # video file
    python -m edge.smoke_webcam --weights models/ppe4_best.pt --all-classes   # after fine-tuning
Press Q in the video window to quit.
"""
import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        if not Path(source).exists():
            raise SystemExit(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source '{source}'. Close Zoom/Teams/Camera app, or try --source 1.")
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--weights", default="yolo26n.pt", help="auto-downloads on first run")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU; default = auto")
    ap.add_argument("--all-classes", action="store_true", help="show every class (default: person only)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until Q)")
    ap.add_argument("--no-show", action="store_true", help="don't open a window (benchmark only)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    half = device != "cpu"
    model = YOLO(args.weights)
    classes = None if args.all_classes else [0]  # COCO class 0 = person
    print(f"Model: {args.weights} | device: {device} | FP16: {half} | imgsz: {args.imgsz}")

    cap = open_source(args.source)
    inf_ms, loop_ms = [], []
    n = 0
    last = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("End of stream / camera read failed.")
                break

            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device,
                                half=half, classes=classes, verbose=False)[0]
            annotated = res.plot()

            now = time.perf_counter()
            loop = (now - t0) * 1000
            n += 1
            if n > 10:  # skip warm-up frames (first GPU calls are slow)
                inf_ms.append(res.speed["inference"])
                loop_ms.append(loop)
            fps = 1.0 / max(now - last, 1e-6)
            last = now

            txt = f"FPS {fps:5.1f} | infer {res.speed['inference']:5.1f} ms | loop {loop:5.1f} ms | persons {len(res.boxes)}"
            cv2.rectangle(annotated, (0, 0), (620, 32), (20, 20, 20), -1)
            cv2.putText(annotated, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not args.no_show:
                cv2.imshow("Day 1 smoke test - press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if loop_ms:
        mean_loop = statistics.mean(loop_ms)
        summary = {
            "weights": args.weights, "device": str(device), "imgsz": args.imgsz, "frames_measured": len(loop_ms),
            "inference_ms_mean": round(statistics.mean(inf_ms), 2),
            "inference_ms_p95": round(sorted(inf_ms)[int(0.95 * (len(inf_ms) - 1))], 2),
            "loop_ms_mean": round(mean_loop, 2),
            "fps_mean": round(1000.0 / mean_loop, 2),
        }
        print("\n=== MEASURED (warm-up excluded) ===")
        for k, v in summary.items():
            print(f"  {k:<20} {v}")
        out = ROOT / "docs" / "evaluation" / "day1_smoke_benchmark.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if new:
                w.writeheader()
            w.writerow(summary)
        print(f"  appended to {out.relative_to(ROOT)}")
    else:
        print("Not enough frames to measure (need > 10).")


if __name__ == "__main__":
    main()
