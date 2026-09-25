"""
Resource and latency measurement: memory use, per-frame latency percentiles, throughput.

The brief asks for memory use, edge latency, frame rate and load/latency measurements. The
existing benchmark reports an average frame rate, which hides the thing that actually
matters on a device: the SLOW frames. A supervisor never notices a good average; they notice
the second where the overlay froze.

WHAT IS MEASURED
    * RSS (resident memory) at three points: before the model is loaded, after it is loaded,
      and at the end - so the model's own footprint is separated from the process baseline.
    * Peak RSS during the run, sampled every frame.
    * Latency per frame, reported as p50 / p90 / p95 / p99 and max, not just the mean.
    * Sustained throughput over the whole run.

HONEST SCOPE
    One process, one camera-sized image, on this laptop. This is not a load test of a server
    farm; it is the question "does this hold up for the length of a shift on the machine it
    will actually run on?" measured properly rather than guessed.

Run:
    python -m scripts.measure_resources                       # 100 frames, auto device
    python -m scripts.measure_resources --frames 300 --device cpu
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarise(latencies_ms: List[float]) -> Dict[str, float]:
    """Percentiles are the point: an average hides the frames a person would notice."""
    return {"frames": len(latencies_ms),
            "mean_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
            "p50_ms": round(percentile(latencies_ms, 0.50), 2),
            "p90_ms": round(percentile(latencies_ms, 0.90), 2),
            "p95_ms": round(percentile(latencies_ms, 0.95), 2),
            "p99_ms": round(percentile(latencies_ms, 0.99), 2),
            "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0.0,
            "fps_sustained": round(1000.0 / statistics.fmean(latencies_ms), 1) if latencies_ms else 0.0}


def rss_mb() -> float:
    import psutil
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5, help="frames to discard first")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default=str(ROOT / "models" / "ppe4_yolo26n_best.pt"))
    parser.add_argument("--image", default=None, help="frame to reuse; default: a sample")
    parser.add_argument("--report", default=str(ROOT / "docs" / "evaluation" / "resources.json"))
    return parser


def pick_image(given) -> Path:
    if given:
        return Path(given)
    for candidate in sorted((ROOT / "samples").glob("*.jpg")):
        return candidate
    images = ROOT / "datasets" / "ppe4" / "val" / "images"
    return sorted(images.iterdir())[0]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    from edge.detect import DEFAULT_WEIGHTS, load_model, to_detections
    from shared.device import resolve_device

    baseline_mb = rss_mb()
    path = pick_image(args.image)
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read {path}")
        return 1

    model, class_map = load_model(Path(args.weights), args.device)
    resolved, reason = resolve_device(args.device)
    loaded_mb = rss_mb()

    predict = dict(imgsz=args.imgsz, conf=0.35, verbose=False, device=resolved)
    for _ in range(max(0, args.warmup)):
        model.predict(frame, **predict)

    latencies: List[float] = []
    peak_mb = loaded_mb
    started = time.perf_counter()
    for _ in range(args.frames):
        t0 = time.perf_counter()
        model.predict(frame, **predict)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        peak_mb = max(peak_mb, rss_mb())
    wall_s = time.perf_counter() - started

    stats = summarise(latencies)
    memory = {"baseline_mb": baseline_mb, "after_model_load_mb": loaded_mb,
              "model_footprint_mb": round(loaded_mb - baseline_mb, 1),
              "peak_mb": peak_mb, "end_mb": rss_mb()}

    print(f"device: {resolved} ({reason})   image: {path.name}   frames: {args.frames}\n")
    print(f"{'latency':<16}{'ms':>9}")
    for key in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"):
        print(f"{key.replace('_ms',''):<16}{stats[key]:>9.2f}")
    print(f"\nsustained        {stats['fps_sustained']:>6.1f} FPS over {wall_s:.1f} s")
    print(f"\nmemory (RSS)")
    print(f"  before the model loaded   {memory['baseline_mb']:>8.1f} MB")
    print(f"  after the model loaded    {memory['after_model_load_mb']:>8.1f} MB"
          f"   (+{memory['model_footprint_mb']} MB for the model)")
    print(f"  peak during the run       {memory['peak_mb']:>8.1f} MB")
    print("\nRead it: p95 is what a person notices. If p95 is far above p50, something "
          "stalls\nperiodically - usually another program taking the GPU.")

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "device": resolved, "device_reason": reason, "imgsz": args.imgsz,
              "image": path.name, "warmup_frames": args.warmup,
              "latency": stats, "memory_mb": memory, "wall_seconds": round(wall_s, 2)}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
