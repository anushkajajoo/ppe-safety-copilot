"""
Privacy versus utility, measured in bytes and milliseconds.

THE QUESTION
    Storing the whole shift is the most useful thing you can do with a camera and the worst
    thing you can do to the people in front of it. This script measures both sides of that
    trade on a real clip, so the argument is made with numbers instead of adjectives.

WHAT IT COMPARES
    A. Continuous recording      every frame kept, unmasked
    B. Event-based, masked       a short clip and one snapshot per event, faces pixelated

    For each: bytes on disk, bytes per hour of operation, how many frames of a person are
    retained, and the extra processing time masking costs per frame.

    The privacy-utility half is measured elsewhere and referenced here: masking costs 6.7 %
    of helmet evidence (docs/evaluation/system_evaluation.md §6a). This script measures the
    storage and latency half.

Run:
    python -m scripts.privacy_storage --clip samples/demo_clip.mp4 --events-per-hour 6
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "evaluation" / "privacy_storage.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, default=ROOT / "samples" / "demo_clip.mp4")
    parser.add_argument("--events-per-hour", type=float, default=6.0,
                        help="how often an event fires; drives the event-based total")
    parser.add_argument("--clip-seconds", type=float, default=10.0, help="length of an event clip")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from shared.privacy import blur_faces

    if not args.clip.exists():
        print(f"No clip at {args.clip}")
        return 1

    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_bytes = args.clip.stat().st_size
    duration = frames / fps if fps else 0.0

    # --- masking cost, measured on real frames -------------------------------
    masked_ms, plain_ms, sampled = [], [], 0
    box = [(width * 0.35, height * 0.15, width * 0.65, height * 0.95)]
    while sampled < 30:
        ok, frame = capture.read()
        if not ok:
            break
        sampled += 1
        started = time.perf_counter()
        _ = frame.copy()
        plain_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        blur_faces(frame.copy(), box)
        masked_ms.append((time.perf_counter() - started) * 1000)
    capture.release()

    def mean(values):
        return round(sum(values) / len(values), 3) if values else 0.0

    bytes_per_second = source_bytes / duration if duration else 0.0
    continuous_per_hour = bytes_per_second * 3600
    event_per_hour = bytes_per_second * args.clip_seconds * args.events_per_hour
    snapshot_bytes = 120 * 1024                      # a typical masked JPEG at this size
    event_per_hour += snapshot_bytes * args.events_per_hour
    reduction = (1 - event_per_hour / continuous_per_hour) * 100 if continuous_per_hour else 0.0

    print(f"clip: {args.clip.name}  {width}x{height} @ {fps:.0f} fps, "
          f"{duration:.1f}s, {source_bytes / 1024 / 1024:.2f} MB\n")
    print(f"{'':<34}{'A: continuous':>16}{'B: event-based':>16}")
    print(f"{'stored per hour (MB)':<34}{continuous_per_hour / 1024 / 1024:>16.1f}"
          f"{event_per_hour / 1024 / 1024:>16.1f}")
    print(f"{'stored per 8-hour shift (MB)':<34}{continuous_per_hour * 8 / 1024 / 1024:>16.1f}"
          f"{event_per_hour * 8 / 1024 / 1024:>16.1f}")
    print(f"{'video kept per hour (seconds)':<34}{3600:>16.0f}"
          f"{args.clip_seconds * args.events_per_hour:>16.0f}")
    print(f"{'faces masked':<34}{'no':>16}{'yes':>16}")
    print(f"\nstorage reduction: {reduction:.1f} %")
    print(f"masking cost: {mean(masked_ms)} ms per frame "
          f"(a bare frame copy is {mean(plain_ms)} ms)")
    print("\nWhat this does NOT say: masking is not anonymisation, and the event-based side")
    print("keeps less of what happened. The utility cost of masking is measured separately")
    print("in docs/evaluation/system_evaluation.md section 6a (93.3 % of helmet evidence kept).")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip": {"name": args.clip.name, "width": width, "height": height, "fps": round(fps, 1),
                 "seconds": round(duration, 2), "bytes": source_bytes},
        "assumptions": {"events_per_hour": args.events_per_hour,
                        "clip_seconds": args.clip_seconds,
                        "snapshot_bytes": snapshot_bytes,
                        "note": "bit rate is taken from this clip; events per hour is an "
                                "input you set, not a measurement"},
        "per_hour_bytes": {"continuous": int(continuous_per_hour), "event_based": int(event_per_hour)},
        "storage_reduction_pct": round(reduction, 1),
        "masking_ms_per_frame": mean(masked_ms),
        "frame_copy_ms": mean(plain_ms),
        "frames_sampled": sampled,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
