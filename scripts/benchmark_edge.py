"""
Measure how fast the edge pipeline really runs on THIS machine.

    python -m scripts.benchmark_edge                          # CPU, samples/demo_clip.mp4
    python -m scripts.benchmark_edge --source 0 --frames 200  # the webcam
    python -m scripts.benchmark_edge --device 0               # GPU comparison (opt-in)

It reports, per frame:
    detect_ms   YOLO inference only
    rules_ms    association + the compliance rule
    draw_ms     annotating the frame
    total_ms    everything the loop does for one frame
and the end-to-end FPS, as mean and p95 (p95 = 19 frames in 20 are faster than this).

WHY WARM-UP FRAMES ARE DISCARDED
    The first few frames pay for CUDA initialisation and memory allocation - tens of
    times the steady-state cost. Including them would understate the system by a lot,
    so --warmup frames (default 10) are measured and then thrown away. The report says
    so; hiding it would be dishonest.

Writes docs/evaluation/edge_benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from edge.detect import DEFAULT_WEIGHTS, REQUIRED_PPE, build_workers, draw, load_model, status_for, to_detections

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: List[float], fraction: float) -> float:
    """Simple nearest-rank percentile - no numpy needed, and easy to explain."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def device_label(requested: Optional[str]) -> str:
    """
    Name the device ACTUALLY used - not just whatever GPU torch can see.

    `--device cpu` must not report the GPU's name: that would put a wrong number in
    the report. Ultralytics picks the GPU when nothing is requested, so "auto"
    resolves to the GPU name only when CUDA is really available.
    """
    if requested is not None and str(requested).lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "cpu"


def report_path(explicit: Optional[str], device: str) -> Path:
    """Separate files per device, so a CPU run cannot overwrite the GPU numbers."""
    if explicit:
        return Path(explicit)
    tag = "cpu" if device == "cpu" else "gpu"
    return ROOT / "docs" / "evaluation" / f"edge_benchmark_{tag}.json"


def summarise(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": round(statistics.fmean(samples), 2), "p95": round(percentile(samples, 0.95), 2),
            "min": round(min(samples), 2), "max": round(max(samples), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(ROOT / "samples" / "demo_clip.mp4"),
                    help="video file, or a webcam index like 0")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--frames", type=int, default=120, help="frames to MEASURE (after warm-up)")
    ap.add_argument("--warmup", type=int, default=10, help="frames to discard first")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="auto",
                    help="'auto' (GPU if it has free VRAM, else CPU), 'cpu', or '0'")
    ap.add_argument("--report", default="",
                    help="where to write the JSON (default: docs/evaluation/edge_benchmark_<gpu|cpu>.json)")
    args = ap.parse_args()

    import cv2

    model, class_map = load_model(Path(args.weights), args.device)

    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        print(f"\nERROR: could not open source: {args.source}")
        return 1

    predict = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, device=resolved)

    detect_ms: List[float] = []
    rules_ms: List[float] = []
    draw_ms: List[float] = []
    total_ms: List[float] = []
    people_seen: List[int] = []
    measured = 0
    seen = 0

    print(f"Model  : {args.weights}")
    print(f"Source : {args.source}")
    print(f"Warm-up: {args.warmup} frames (discarded)   Measuring: {args.frames} frames\n")

    while measured < args.frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)       # loop a short clip
            ok, frame = capture.read()
            if not ok:
                break
        seen += 1
        loop_started = time.perf_counter()

        t0 = time.perf_counter()
        result = model.predict(frame, **predict)[0]
        detections = to_detections(result, class_map, args.conf)
        t1 = time.perf_counter()

        workers, unassigned = build_workers(detections)
        for worker in workers:
            status_for(worker, REQUIRED_PPE)
        t2 = time.perf_counter()

        draw(frame.copy(), workers, unassigned, REQUIRED_PPE)
        t3 = time.perf_counter()

        if seen > args.warmup:                            # steady state only
            detect_ms.append((t1 - t0) * 1000)
            rules_ms.append((t2 - t1) * 1000)
            draw_ms.append((t3 - t2) * 1000)
            total_ms.append((time.perf_counter() - loop_started) * 1000)
            people_seen.append(len(workers))
            measured += 1

    capture.release()
    if not total_ms:
        print("\nERROR: no frames were measured.")
        return 1

    from shared.device import resolve_device
    resolved, why = resolve_device(args.device)
    device_used = device_label(resolved)
    print(f"Device : {device_used}  ({why})\n")

    stats = {"detect_ms": summarise(detect_ms), "rules_ms": summarise(rules_ms),
             "draw_ms": summarise(draw_ms), "total_ms": summarise(total_ms)}
    fps_mean = round(1000.0 / stats["total_ms"]["mean"], 1)
    fps_p95 = round(1000.0 / stats["total_ms"]["p95"], 1)

    print(f"{'stage':<12}{'mean ms':>10}{'p95 ms':>10}{'min':>8}{'max':>8}")
    for stage in ("detect_ms", "rules_ms", "draw_ms", "total_ms"):
        s = stats[stage]
        print(f"{stage:<12}{s['mean']:>10.2f}{s['p95']:>10.2f}{s['min']:>8.2f}{s['max']:>8.2f}")
    print(f"\nend-to-end: {fps_mean} FPS mean, {fps_p95} FPS at p95")
    print(f"people per frame: {statistics.fmean(people_seen):.2f} average")
    print(f"device: {device_used}")

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "machine": {"platform": platform.platform(), "python": platform.python_version(),
                          "device": device_used,
                          "device_requested": args.device or "auto"},
              "settings": {"source": args.source, "weights": args.weights, "imgsz": args.imgsz,
                           "conf": args.conf, "frames_measured": measured, "warmup_discarded": args.warmup},
              "latency_ms": stats, "fps_mean": fps_mean, "fps_p95": fps_p95,
              "people_per_frame_mean": round(statistics.fmean(people_seen), 2)}
    out = report_path(args.report, device_used)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
