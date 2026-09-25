"""
Build our SMALL 4-class PPE dataset (datasets/ppe4) from a public YOLO export.

Source: the Roboflow Universe "Construction Site Safety" dataset
(roboflow-universe-projects/construction-site-safety, YOLOv8 export, CC BY 4.0).

    source class            ->  our id   our name
    Person                  ->  0        person
    Hardhat / Helmet        ->  1        helmet
    Safety Vest / Vest      ->  2        vest
    Mask / Face-mask        ->  3        mask
    everything else (NO-Hardhat, NO-Mask, NO-Safety Vest, machinery,
    vehicle, Safety Cone, ...)                                    -> dropped

Absence of PPE is NOT a detector class here: it is DERIVED by the rule engine
from person-PPE association, which keeps the model small and the safety
decision auditable.

Two things this script is careful about
---------------------------------------
1. AUGMENTED COPIES.  Roboflow exports several augmented copies of the same
   photo, named "<original>_jpg.rf.<hash>.jpg".  Counting those as different
   images would fake the dataset size and, far worse, put the same photo in
   two different splits.  So everything is grouped by its ORIGINAL id first.
   - evaluation splits (val/test) keep exactly ONE copy per original,
   - the training split keeps at most --copies-train copies.
2. CLASS BALANCE.  'mask' and 'vest' are much rarer than 'person'.  Plain
   random sampling can lose them, so the picker takes images for the rarest
   classes first (see pick_subset).

Everything is seeded, so the same source + the same seed rebuilds exactly the
same dataset (recorded in subset_manifest.json).

Run (Windows PowerShell, from the project root):
    python -m training.prepare_ppe4 --src "C:\\Users\\Anushka\\Downloads\\css.zip"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

# Our class order - MUST match edge/detector.py and the trained model.
TARGET_NAMES = ["person", "helmet", "vest", "mask"]

# Accepted source names (normalised: lower case, spaces/underscores -> '-').
SOURCE_ALIASES = {
    "person": 0, "worker": 0,
    "helmet": 1, "hardhat": 1, "hard-hat": 1, "safety-helmet": 1,
    "vest": 2, "safety-vest": 2, "safety-vests": 2, "reflective-vest": 2,
    "mask": 3, "face-mask": 3, "face-mask-medical": 3, "medical-mask": 3,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------- small helpers
def norm(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def original_id(stem: str) -> str:
    """
    'foo_jpg.rf.9a3f...' -> 'foo'.  Roboflow gives every augmented copy of a
    photo the same prefix and a different hash, so the prefix identifies the
    ORIGINAL photo.  Any other naming scheme falls back to the whole stem.
    """
    return stem.split(".rf.")[0]


def unzip_if_needed(src: Path) -> Path:
    """If --src is a .zip, extract it next to itself and return that folder."""
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        raise SystemExit(f"--src must be a folder or a .zip file (got: {src})")
    if not src.exists():
        raise SystemExit(f"Dataset zip not found: {src}")

    out = src.with_suffix("")
    if out.is_dir() and any(out.iterdir()):
        print(f"Already extracted: {out}")
        return out
    print(f"Extracting {src.name} ...")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(out)
    return out


def find_export_root(folder: Path) -> Path:
    """Find the folder holding data.yaml (Roboflow zips are sometimes nested)."""
    if (folder / "data.yaml").exists():
        return folder
    for candidate in sorted(folder.rglob("data.yaml")):
        return candidate.parent
    raise SystemExit(f"No data.yaml found inside {folder}. Is this a YOLO export?")


def load_source_names(root: Path) -> List[str]:
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):                       # {0: 'Person', 1: ...}
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        raise SystemExit(f"data.yaml in {root} has no 'names' list.")
    return [str(n) for n in names]


def build_id_map(names: List[str]) -> Dict[int, int]:
    """source class index -> our class id (unwanted classes are simply absent)."""
    id_map = {i: SOURCE_ALIASES[norm(n)] for i, n in enumerate(names) if norm(n) in SOURCE_ALIASES}
    found = set(id_map.values())
    missing = [TARGET_NAMES[i] for i in range(len(TARGET_NAMES)) if i not in found]
    if missing:
        raise SystemExit(
            f"The source dataset is missing these classes: {missing}\n"
            f"Its classes are: {names}\n"
            f"If they are named differently, add the alias to SOURCE_ALIASES "
            f"in training/prepare_ppe4.py."
        )
    dropped = [n for i, n in enumerate(names) if i not in id_map]
    print(f"Mapped {len(id_map)} source classes onto our 4; dropped {len(dropped)}: {dropped}")
    return id_map


def parse_label(path: Path, id_map: Dict[int, int]) -> List[Tuple[int, float, float, float, float]]:
    """
    Read one YOLO label file and return our remapped boxes.

    A normal line is:  class_id  x_center  y_center  width  height   (all 0-1).
    Some exports store polygons (class + x1 y1 x2 y2 ...); those are converted
    to their bounding box, so a segmentation export never crashes the script.
    Invalid or empty boxes are dropped.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if src_id not in id_map:
            continue                                   # a class we don't use

        if len(values) == 4:
            xc, yc, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:   # polygon -> bounding box
            xs, ys = values[0::2], values[1::2]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue

        xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w <= 1e-6 or h <= 1e-6:
            continue
        boxes.append((id_map[src_id], xc, yc, w, h))
    return boxes


def collect_candidates(root: Path, id_map: Dict[int, int]) -> Dict[str, List[dict]]:
    """
    Walk the source splits and keep every image that still has >= 1 useful box.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    candidates: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        split = SPLIT_ALIASES.get(folder.name.lower())
        images_dir, labels_dir = folder / "images", folder / "labels"
        if split is None or not images_dir.is_dir():
            continue

        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            boxes = parse_label(labels_dir / f"{img.stem}.txt", id_map)
            if not boxes:
                continue                               # nothing we care about
            candidates[split].append({
                "image": img,
                "original": original_id(img.stem),
                "boxes": boxes,
                "classes": sorted({b[0] for b in boxes}),
            })
    return candidates


def regroup(candidates: Dict[str, List[dict]], seed: int, copies_train: int,
            want_val: int, want_test: int, min_train_originals: int,
            topup_eval: bool) -> Tuple[Dict[str, List[dict]], dict]:
    """
    Turn the raw source splits into clean, leak-free splits of ORIGINAL photos.

    Steps:
      1. Group every image by its original photo id.
      2. Decide one split per original.  If the same photo appears in more than
         one source split, test wins over val, and val wins over train - so a
         photo can never be trained on AND tested on.
      3. Only if `topup_eval` is on: move spare originals from the train pool
         into val/test (seeded, never below `min_train_originals`).  This is
         OFF by default, because in the Roboflow export the training images are
         pre-augmented mosaics while valid/test are clean single photos -
         moving training images into the evaluation sets would measure the
         model on mosaics instead of on realistic pictures.
      4. Emit the images: ONE copy per original for val/test (no augmented
         duplicates when measuring performance), up to `copies_train` for train.
    """
    rng = random.Random(seed)

    by_original: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for split in SPLITS:
        for item in candidates[split]:
            by_original[item["original"]][split].append(item)

    # ---- step 2: one split per original, evaluation splits win
    assigned: Dict[str, str] = {}
    conflicts = 0
    for original, per_split in by_original.items():
        present = [s for s in SPLITS if per_split.get(s)]
        if len(present) > 1:
            conflicts += 1
        assigned[original] = "test" if "test" in present else ("val" if "val" in present else "train")

    pools: Dict[str, List[str]] = {s: sorted(o for o, sp in assigned.items() if sp == s) for s in SPLITS}

    # ---- step 3: top up val/test from the train pool if they are too small
    moved = {"val": 0, "test": 0}
    rng.shuffle(pools["train"])
    spare = max(0, len(pools["train"]) - min_train_originals) if topup_eval else 0
    needs = {"test": max(0, want_test - len(pools["test"])),
             "val": max(0, want_val - len(pools["val"]))}
    while spare > 0 and (needs["test"] > 0 or needs["val"] > 0):
        for split in ("test", "val"):                  # one each, test first
            if spare <= 0 or needs[split] <= 0:
                continue
            pools[split].append(pools["train"].pop())
            moved[split] += 1
            needs[split] -= 1
            spare -= 1
    pools["train"].sort()
    pools["val"].sort()
    pools["test"].sort()

    # ---- step 4: build the image lists
    grouped: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        allowed = 1 if split in ("val", "test") else max(1, copies_train)
        for original in pools[split]:
            copies = [i for s in SPLITS for i in by_original[original].get(s, [])]
            copies.sort(key=lambda i: i["image"].name)
            rng.shuffle(copies)
            grouped[split].extend(copies[:allowed])

    info = {
        "originals_per_split": {s: len(pools[s]) for s in SPLITS},
        "originals_in_more_than_one_source_split": conflicts,
        "originals_moved_from_train": moved,
        "min_train_originals": min_train_originals,
        "topup_eval_from_train": topup_eval,
        "copies_per_original": {"train": copies_train, "val": 1, "test": 1},
    }
    return grouped, info


def pick_subset(items: List[dict], quota: int, seed: int, min_per_class: int) -> List[dict]:
    """
    Choose `quota` images, making sure rare classes are represented.

    How it works (the bit to explain in the viva):
      1. Shuffle the candidates with a fixed seed -> reproducible.
      2. Sort the four classes from RAREST to most common.
      3. For each class in that order, take images containing it until the
         class has at least `min_per_class` images (or the quota is full).
      4. Fill any remaining places with the leftover shuffled images.
    Plain random sampling would be fine for 'person' but could easily end up
    with almost no 'mask' images, which would make that class untrainable.
    """
    if quota <= 0 or not items:
        return []
    if len(items) <= quota:
        return list(items)

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    frequency = Counter()
    for item in shuffled:
        frequency.update(item["classes"])
    rarest_first = sorted(range(len(TARGET_NAMES)), key=lambda c: frequency.get(c, 0))

    chosen: List[dict] = []
    taken = set()
    per_class = Counter()

    for class_id in rarest_first:
        if len(chosen) >= quota:
            break
        for i, item in enumerate(shuffled):
            if len(chosen) >= quota or per_class[class_id] >= min_per_class:
                break
            if i in taken or class_id not in item["classes"]:
                continue
            taken.add(i)
            chosen.append(item)
            per_class.update(item["classes"])

    for i, item in enumerate(shuffled):                 # fill the rest randomly
        if len(chosen) >= quota:
            break
        if i not in taken:
            taken.add(i)
            chosen.append(item)
    return chosen


def write_item(item: dict, split_dir: Path, max_side: int, quality: int = 88) -> dict:
    """Copy one image (downscaled) and write its remapped label file."""
    src: Path = item["image"]
    out_img = split_dir / "images" / f"{src.stem}.jpg"
    out_lbl = split_dir / "labels" / f"{src.stem}.txt"

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            im = im.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
        im.save(out_img, quality=quality)

    # Labels are normalised (0-1), so resizing does not change them.
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in item["boxes"]]
    out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"stem": src.stem, "original": item["original"], "source": str(src),
            "md5": hashlib.md5(out_img.read_bytes()).hexdigest(),
            "classes": item["classes"], "boxes": len(item["boxes"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="the downloaded YOLO export: a folder or a .zip file")
    ap.add_argument("--out", default=Path("datasets/ppe4"), type=Path)
    ap.add_argument("--train", type=int, default=1000, help="training images to keep")
    ap.add_argument("--val", type=int, default=250,
                    help="validation images wanted (capped by how many unique photos exist)")
    ap.add_argument("--test", type=int, default=250,
                    help="test images wanted (capped by how many unique photos exist)")
    ap.add_argument("--topup-eval", action="store_true",
                    help="allow moving training photos into val/test if those are small "
                         "(off by default: keeps the evaluation sets clean)")
    ap.add_argument("--min-train-originals", type=int, default=400,
                    help="never shrink the training pool below this many distinct photos")
    ap.add_argument("--copies-train", type=int, default=3,
                    help="max augmented copies of the same photo in train (val/test always 1)")
    ap.add_argument("--seed", type=int, default=42, help="fixed seed = reproducible subset")
    ap.add_argument("--max-side", type=int, default=640, help="downscale longest side to this")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality of the written images")
    ap.add_argument("--min-per-class", type=int, default=0,
                    help="minimum images per class per split (0 = quota/5, automatic)")
    args = ap.parse_args()

    root = find_export_root(unzip_if_needed(args.src.expanduser()))
    print(f"Source export : {root}")

    names = load_source_names(root)
    id_map = build_id_map(names)

    print("\nReading labels ...")
    candidates = collect_candidates(root, id_map)
    for split in SPLITS:
        print(f"  {split:5s}: {len(candidates[split]):5d} usable source images")
    if not candidates["train"]:
        raise SystemExit("No usable training images found - check the --src folder.")

    print("\nGrouping augmented copies by original photo ...")
    grouped, group_info = regroup(candidates, args.seed, args.copies_train,
                                  args.val, args.test, args.min_train_originals,
                                  args.topup_eval)
    print(f"  originals per split      : {group_info['originals_per_split']}")
    print(f"  originals in 2+ source splits (fixed): "
          f"{group_info['originals_in_more_than_one_source_split']}")
    print(f"  originals moved from train : {group_info['originals_moved_from_train']}")

    quotas = {"train": args.train, "val": args.val, "test": args.test}
    out_root: Path = args.out
    manifest: Dict[str, List[dict]] = {}
    stats: Dict[str, dict] = {}

    print("\nWriting the subset ...")
    for split in SPLITS:
        quota = quotas[split]
        min_per_class = args.min_per_class or max(10, quota // 5)
        chosen = pick_subset(grouped[split], quota, args.seed, min_per_class)

        split_dir = out_root / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        records, box_counts, image_counts = [], Counter(), Counter()
        for item in chosen:
            records.append(write_item(item, split_dir, args.max_side, args.quality))
            for class_id, *_ in item["boxes"]:
                box_counts[class_id] += 1
            for class_id in item["classes"]:
                image_counts[class_id] += 1

        manifest[split] = records
        stats[split] = {
            "images": len(records),
            "originals": len({r["original"] for r in records}),
            "boxes_per_class": {TARGET_NAMES[c]: box_counts.get(c, 0) for c in range(len(TARGET_NAMES))},
            "images_per_class": {TARGET_NAMES[c]: image_counts.get(c, 0) for c in range(len(TARGET_NAMES))},
        }
        print(f"  {split:5s}: {len(records):5d} images from {stats[split]['originals']:4d} originals  "
              f"{stats[split]['boxes_per_class']}")

    # ---- leakage check: the same photo must not be in train/val AND test
    test_originals = {r["original"] for r in manifest.get("test", [])}
    leaked = sorted({r["original"] for s in ("train", "val") for r in manifest.get(s, [])
                     if r["original"] in test_originals})
    if leaked:
        print(f"\nWARNING: {len(leaked)} photos appear in both train/val and test: {leaked[:5]}")
    else:
        print("\nLeakage check: no photo (or augmented copy of it) is in both train/val and test.")

    # ---- data.yaml for Ultralytics
    # No absolute 'path:' line - Ultralytics then resolves the split folders
    # relative to this file, so the dataset works on any machine (laptop, Colab).
    (out_root / "data.yaml").write_text(
        "# Written by training/prepare_ppe4.py - do not edit by hand.\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(TARGET_NAMES)}\n"
        f"names: {TARGET_NAMES}\n",
        encoding="utf-8")

    summary = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_export": str(root),
        "source_classes": names,
        "seed": args.seed,
        "max_side": args.max_side,
        "jpeg_quality": args.quality,
        "quotas": quotas,
        "grouping": group_info,
        "splits": stats,
        "leaked_photos": leaked,
        "class_names": TARGET_NAMES,
    }
    (out_root / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDataset written to {out_root.resolve()}")
    print("  data.yaml, stats.json, subset_manifest.json")
    print("Next: python -m scripts.check_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
