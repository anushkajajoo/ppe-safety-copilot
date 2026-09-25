"""
Where do helmet / vest / mask boxes actually sit inside a person box?

WHY THIS EXISTS
    shared/privacy.py blurs a geometric band inside each person box instead of
    running a face detector (see that module's docstring for why). The band's top
    and bottom edges were first chosen by eye, and scripts/privacy_utility.py then
    measured the damage: helmet detections dropped to 33% after masking. That is
    the masking destroying the very evidence the snapshot exists to keep.

    Rather than guess again, this script measures the geometry from the dataset's
    OWN ground-truth labels. No model, no GPU, no inference - just arithmetic on
    the label files. For every PPE box whose centre falls inside a person box, it
    records where that PPE box sits vertically, as a fraction of the person box
    height measured down from the person box's top edge.

    The output tells us directly which band is safe: a band that starts below the
    helmet's typical lower edge cannot erase helmet evidence.

USAGE
    python -m scripts.face_band_geometry
    python -m scripts.face_band_geometry --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

NAMES = ["person", "helmet", "vest", "mask"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "datasets" / "ppe4"


def read_label(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """One YOLO label file -> [(cls, cx, cy, w, h)] in normalised coordinates."""
    rows: List[Tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


def to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def inside(point: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def percentile(values: List[float], q: float) -> float:
    """Plain linear-interpolation percentile - avoids pulling numpy in for this."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def collect(labels_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    For each PPE class, the fractions (top, bottom, centre) of its box inside the
    person box it belongs to. 0.0 = person box top edge, 1.0 = person box bottom.
    """
    out: Dict[str, Dict[str, List[float]]] = {
        name: {"top": [], "bottom": [], "centre": []} for name in NAMES[1:]
    }
    files = sorted(labels_dir.glob("*.txt"))
    matched = 0
    for file in files:
        rows = read_label(file)
        people = [to_xyxy(*row[1:]) for row in rows if row[0] == 0]
        if not people:
            continue
        for cls, cx, cy, w, h in rows:
            if cls == 0 or cls >= len(NAMES):
                continue
            px1, py1, px2, py2 = to_xyxy(cx, cy, w, h)
            # the person box this PPE box sits in; smallest one wins if several
            hosts = [p for p in people if inside((cx, cy), p)]
            if not hosts:
                continue
            host = min(hosts, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
            hx1, hy1, hx2, hy2 = host
            person_h = hy2 - hy1
            if person_h <= 0:
                continue
            bucket = out[NAMES[cls]]
            bucket["top"].append((py1 - hy1) / person_h)
            bucket["bottom"].append((py2 - hy1) / person_h)
            bucket["centre"].append((cy - hy1) / person_h)
            matched += 1
    out["_files"] = len(files)          # type: ignore[assignment]
    out["_matched"] = matched           # type: ignore[assignment]
    return out


def summarise(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "label_files": data.pop("_files"),
        "ppe_boxes_matched_to_a_person": data.pop("_matched"),
        "classes": {},
    }
    for name, buckets in data.items():
        if not buckets["top"]:
            summary["classes"][name] = {"n": 0}       # type: ignore[index]
            continue
        summary["classes"][name] = {                   # type: ignore[index]
            "n": len(buckets["top"]),
            "top_p10": round(percentile(buckets["top"], 0.10), 3),
            "top_median": round(percentile(buckets["top"], 0.50), 3),
            "bottom_median": round(percentile(buckets["bottom"], 0.50), 3),
            "bottom_p90": round(percentile(buckets["bottom"], 0.90), 3),
            "bottom_p99": round(percentile(buckets["bottom"], 0.99), 3),
            "centre_median": round(percentile(buckets["centre"], 0.50), 3),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs" / "evaluation" / "face_band_geometry.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    labels_dir = args.dataset / args.split / "labels"
    if not labels_dir.is_dir():
        print(f"No labels at {labels_dir}")
        return 1

    summary = summarise(collect(labels_dir))
    summary["split"] = args.split

    print(f"split: {args.split}   label files: {summary['label_files']}   "
          f"PPE boxes matched: {summary['ppe_boxes_matched_to_a_person']}")
    print("fractions of person-box height, measured down from the person box top\n")
    print(f"{'class':<8}{'n':>6}{'top med':>10}{'bot med':>10}{'bot p90':>10}{'bot p99':>10}")
    for name, stats in summary["classes"].items():          # type: ignore[union-attr]
        if not stats.get("n"):
            print(f"{name:<8}{0:>6}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
            continue
        print(f"{name:<8}{stats['n']:>6}{stats['top_median']:>10.3f}"
              f"{stats['bottom_median']:>10.3f}{stats['bottom_p90']:>10.3f}"
              f"{stats['bottom_p99']:>10.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
