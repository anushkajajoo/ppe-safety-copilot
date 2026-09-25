"""
Validate datasets/ppe4 before training.

Training on a broken dataset wastes an hour and produces a useless model, so
this script checks everything that can silently go wrong:

    * image count and label count per split
    * images with no label file            (missing labels)
    * label files with no image            (missing images)
    * corrupt / unreadable images
    * malformed label lines                (wrong number of fields, not numbers)
    * invalid bounding boxes               (outside 0-1, zero width or height)
    * invalid class IDs                    (not 0..3)
    * class distribution                   (per split: boxes and images)

Run (from the project root):
    python -m scripts.check_dataset
    python -m scripts.check_dataset --data datasets/ppe4 --max-report 20

Exit code 0 = dataset is usable, 1 = problems found.
A summary is written to docs/evaluation/dataset_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from PIL import Image

CLASS_NAMES = ["person", "helmet", "vest", "mask"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def check_split(split_dir: Path, problems: Dict[str, List[str]]) -> dict:
    """Check one split folder and return its statistics."""
    images_dir, labels_dir = split_dir / "images", split_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        problems["missing_folders"].append(str(split_dir))
        return {"images": 0, "labels": 0, "boxes": 0}

    images = {p.stem: p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in sorted(labels_dir.iterdir()) if p.suffix.lower() == ".txt"}

    for stem in sorted(set(images) - set(labels)):
        problems["missing_labels"].append(f"{split_dir.name}/{stem}")
    for stem in sorted(set(labels) - set(images)):
        problems["missing_images"].append(f"{split_dir.name}/{stem}")

    boxes_per_class, images_per_class = Counter(), Counter()
    total_boxes = 0
    empty_labels = 0
    sizes = []

    for stem, path in images.items():
        # --- the image itself must open and be a real picture
        try:
            with Image.open(path) as im:
                im.verify()                     # detects truncated / corrupt files
            with Image.open(path) as im:
                sizes.append(im.size)
        except Exception as exc:
            problems["corrupt_images"].append(f"{split_dir.name}/{path.name}: {type(exc).__name__}")
            continue

        label_path = labels.get(stem)
        if label_path is None:
            continue

        classes_here = set()
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not [ln for ln in lines if ln.strip()]:
            empty_labels += 1

        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            where = f"{split_dir.name}/{label_path.name}:{number}"

            if len(parts) != 5:
                problems["malformed_lines"].append(f"{where}: expected 5 fields, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                xc, yc, w, h = (float(v) for v in parts[1:])
            except ValueError:
                problems["malformed_lines"].append(f"{where}: not numeric")
                continue

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                problems["invalid_class_ids"].append(f"{where}: class {class_id}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                problems["invalid_boxes"].append(f"{where}: values outside 0-1")
                continue
            if w <= 0 or h <= 0:
                problems["invalid_boxes"].append(f"{where}: zero width or height")
                continue
            if xc - w / 2 < -1e-3 or xc + w / 2 > 1 + 1e-3 or yc - h / 2 < -1e-3 or yc + h / 2 > 1 + 1e-3:
                problems["invalid_boxes"].append(f"{where}: box goes outside the image")
                continue

            boxes_per_class[class_id] += 1
            classes_here.add(class_id)
            total_boxes += 1

        images_per_class.update(classes_here)

    widths = [s[0] for s in sizes] or [0]
    heights = [s[1] for s in sizes] or [0]
    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "empty_label_files": empty_labels,
        "boxes_per_class": {name: boxes_per_class.get(i, 0) for i, name in enumerate(CLASS_NAMES)},
        "images_per_class": {name: images_per_class.get(i, 0) for i, name in enumerate(CLASS_NAMES)},
        "image_size": {"max_width": max(widths), "max_height": max(heights)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=Path("datasets/ppe4"), type=Path, help="dataset root")
    ap.add_argument("--max-report", type=int, default=10, help="problems to print per category")
    ap.add_argument("--report", default=Path("docs/evaluation/dataset_summary.json"), type=Path)
    args = ap.parse_args()

    root: Path = args.data
    if not root.is_dir():
        print(f"\nERROR: Dataset not found at {root.resolve()}")
        print("HINT : run  python -m training.prepare_ppe4 --src <downloaded export>  first.")
        return 1

    problems: Dict[str, List[str]] = {
        "missing_folders": [], "missing_labels": [], "missing_images": [],
        "corrupt_images": [], "malformed_lines": [], "invalid_boxes": [], "invalid_class_ids": [],
    }

    print("=" * 66)
    print(f"Dataset check: {root.resolve()}")
    print("=" * 66)

    splits = {}
    for split in SPLITS:
        splits[split] = check_split(root / split, problems)

    # ---------------- summary table
    print(f"\n{'split':<7}{'images':>8}{'labels':>8}{'boxes':>8}   " +
          "".join(f"{n:>9}" for n in CLASS_NAMES))
    for split, s in splits.items():
        per_class = s.get("boxes_per_class", {})
        print(f"{split:<7}{s['images']:>8}{s.get('labels', 0):>8}{s['boxes']:>8}   " +
              "".join(f"{per_class.get(n, 0):>9}" for n in CLASS_NAMES))

    total_images = sum(s["images"] for s in splits.values())
    print(f"\ntotal images: {total_images}")

    # ---------------- warnings that are not hard errors
    warnings = []
    if not (root / "data.yaml").exists():
        warnings.append("data.yaml is missing - Ultralytics needs it to train")
    for split, s in splits.items():
        for name in CLASS_NAMES:
            if s.get("boxes_per_class", {}).get(name, 0) == 0 and s["images"] > 0:
                warnings.append(f"{split}: class '{name}' has no boxes at all")
        if s.get("empty_label_files"):
            warnings.append(f"{split}: {s['empty_label_files']} label file(s) are empty")

    # ---------------- problem report
    error_count = sum(len(v) for v in problems.values())
    if error_count:
        print("\n-- PROBLEMS " + "-" * 53)
        for kind, items in problems.items():
            if not items:
                continue
            print(f"\n{kind} ({len(items)}):")
            for item in items[: args.max_report]:
                print(f"   {item}")
            if len(items) > args.max_report:
                print(f"   ... and {len(items) - args.max_report} more")

    if warnings:
        print("\n-- WARNINGS " + "-" * 53)
        for warning in warnings:
            print(f"   {warning}")

    summary = {
        "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(root.resolve()),
        "class_names": CLASS_NAMES,
        "splits": splits,
        "problem_counts": {k: len(v) for k, v in problems.items()},
        "problems": {k: v[:50] for k, v in problems.items() if v},
        "warnings": warnings,
        "ok": error_count == 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    if error_count == 0:
        print(f"DATASET OK  ({total_images} images)   summary -> {args.report}")
        print("=" * 66)
        return 0
    print(f"DATASET HAS {error_count} PROBLEM(S)   summary -> {args.report}")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())
