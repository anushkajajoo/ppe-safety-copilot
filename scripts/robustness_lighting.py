"""
Robustness to lighting: how much of the detection survives a darker, brighter or hazier site?

WHY IT MATTERS MORE THAN IT SOUNDS
    The evaluation set is daylight construction photography. A real site is dawn, dusk, a
    floodlit night shift, and the inside of a shed. If the model only works at noon, the
    honest place to find that out is here, not in front of an examiner.

METHOD
    Each validation image is detected once as-is (the baseline), then re-detected under a set
    of deterministic lighting transforms. Detections are paired with the baseline by box
    overlap (the same matcher as the privacy evaluation), so the number reported is "how much
    of what we could see, can we still see" - not a raw count that can go up when the model
    starts inventing boxes in the noise.

TRANSFORMS (all deterministic, so the run is reproducible)
    gamma 0.4 / 0.6   brighter, washed out - a floodlight or a white wall
    gamma 1.6 / 2.2   darker - dusk, a shaded bay, an overcast evening
    contrast 0.5      flat, hazy light - dust or fog
    noise sigma 15    a cheap sensor at high gain, which is what night footage looks like

Run:
    python -m scripts.robustness_lighting --images 30
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["person", "helmet", "vest", "mask"]


def gamma(frame, value: float):
    """value < 1 brightens, > 1 darkens. A lookup table keeps it fast and exact."""
    import numpy as np
    table = np.array([((i / 255.0) ** value) * 255 for i in range(256)], dtype="uint8")
    import cv2
    return cv2.LUT(frame, table)


def contrast(frame, factor: float):
    import cv2
    return cv2.convertScaleAbs(frame, alpha=factor, beta=128 * (1 - factor))


def noisy(frame, sigma: float, seed: int = 7):
    import numpy as np
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype("uint8")


CONDITIONS = [
    ("bright (gamma 0.4)", lambda f: gamma(f, 0.4)),
    ("slightly bright (gamma 0.6)", lambda f: gamma(f, 0.6)),
    ("as measured", lambda f: f),
    ("slightly dark (gamma 1.6)", lambda f: gamma(f, 1.6)),
    ("dark (gamma 2.2)", lambda f: gamma(f, 2.2)),
    ("flat light (contrast 0.5)", lambda f: contrast(f, 0.5)),
    ("sensor noise (sigma 15)", lambda f: noisy(f, 15)),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default=str(ROOT / "models" / "ppe4_yolo26n_best.pt"))
    parser.add_argument("--report",
                        default=str(ROOT / "docs" / "evaluation" / "robustness_lighting.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    from edge.detect import load_model, to_detections
    from scripts.privacy_utility import gather_images, match
    from shared.device import resolve_device

    images = gather_images(args.images)
    if not images:
        print("No images found (datasets/ppe4/val/images or samples/).")
        return 1

    model, class_map = load_model(Path(args.weights), args.device)
    resolved, _ = resolve_device(args.device)
    predict = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, device=resolved)

    totals = {name: {label: {"baseline": 0, "preserved": 0, "spurious": 0}
                     for label, _ in CONDITIONS} for name in CLASSES}
    used = 0
    print(f"Testing {len(images)} images under {len(CONDITIONS)} lighting conditions ...\n")
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        used += 1
        baseline = to_detections(model.predict(frame, **predict)[0], class_map, args.conf)
        for label, transform in CONDITIONS:
            altered = to_detections(model.predict(transform(frame.copy()), **predict)[0],
                                    class_map, args.conf)
            for name, counts in match(baseline, altered).items():
                totals[name][label]["baseline"] += counts["preserved"] + counts["lost"]
                totals[name][label]["preserved"] += counts["preserved"]
                totals[name][label]["spurious"] += counts["spurious"]

    # How many detections each class had in the untouched control run. A percentage of
    # nothing is not a result, so classes with no baseline print as "-" rather than 0 %.
    sample = {name: totals[name]["as measured"]["baseline"] for name in CLASSES}

    print(f"{'condition':<30}" + "".join(f"{name:>10}" for name in CLASSES))
    print(f"{'(detections in the control)':<30}"
          + "".join(f"{('n=' + str(sample[name])):>10}" for name in CLASSES))
    rows: Dict[str, Dict[str, object]] = {}
    for label, _ in CONDITIONS:
        line = f"{label:<30}"
        rows[label] = {}
        for name in CLASSES:
            entry = totals[name][label]
            if not entry["baseline"]:
                rows[label][name] = None            # not measured, not zero
                line += f"{'-':>10}"
                continue
            kept = entry["preserved"] / entry["baseline"] * 100
            rows[label][name] = round(kept, 1)
            line += f"{kept:>9.1f}%"
        print(line)

    print(f"\nimages: {used}   (percentages are of what the model found in the original image)")
    print("'as measured' is the control - it must read 100 %, or the measurement is broken.")
    thin = [name for name in CLASSES if 0 < sample[name] < 10]
    missing = [name for name in CLASSES if sample[name] == 0]
    if missing:
        print(f"NOT MEASURED: {', '.join(missing)} - the control run found none, so there is "
              f"nothing to preserve. This says nothing about that class.")
    if thin:
        print(f"TOO FEW TO CONCLUDE: {', '.join(thin)} (fewer than 10 detections). Treat those "
              f"percentages as anecdote, not measurement.")

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": {"images": used, "conf": args.conf, "imgsz": args.imgsz,
                           "device": resolved},
              "control_detections": sample, "preserved_pct": rows, "raw": totals}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
