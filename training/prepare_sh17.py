"""
Convert the SH17 dataset (17 classes) into OUR 4-class dataset:

    SH17 name            ->  our id  our name
    Person               ->  0       person
    Helmet               ->  1       helmet
    Safety-vest          ->  2       vest
    Face-mask-medical    ->  3       mask
    (all other 13 classes are dropped)

What this script does, step by step:
  1. Reads SH17 class names from its YAML (never trusts hard-coded indices blindly).
  2. Uses SH17's OWN train/test file lists. The official test split stays our TEST
     set and is never touched during training or tuning.
  3. Carves a VALIDATION set out of the official train list (deterministic, seed 42).
  4. Remaps label ids and drops the 13 classes we don't use.
  5. Downscales huge photos (SH17 has images up to 8192 px wide) to --max-side px.
     YOLO labels are normalised (0-1), so they stay correct after resizing.
     This makes training MUCH faster because the dataloader no longer decodes 30 MP JPEGs.
  6. Checks for data leakage: exact duplicate files and near-duplicate images
     (perceptual hash) that appear in BOTH train/val and test.
  7. Writes data.yaml for Ultralytics and stats.json with REAL counts.

Run (Windows PowerShell, from the project root):
    python -m training.prepare_sh17 --src "D:\\datasets\\sh17" --out datasets\\ppe4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # SH17 has very large images; they are trusted files

TARGET_NAMES = ["person", "helmet", "vest", "mask"]
# normalised SH17 name -> our class id
SH17_TO_OURS = {"person": 0, "helmet": 1, "safety-vest": 2, "face-mask-medical": 3}
# Official SH17 order (used only if the YAML cannot be found; a warning is printed).
SH17_FALLBACK_NAMES = ["Person", "Head", "Face", "Glasses", "Face-mask-medical", "Face-guard", "Ear",
                       "Earmuffs", "Hands", "Gloves", "Foot", "Shoes", "Safety-vest", "Tools", "Helmet",
                       "Medical-suit", "Safety-suit"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- helpers
def norm(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def load_class_names(src: Path) -> List[str]:
    for yml in sorted(src.rglob("*.yaml")) + sorted(src.rglob("*.yml")):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            print(f"Class names read from {yml}")
            return list(names)
    print("WARNING: no YAML with 'names' found; using the official SH17 class order.")
    return SH17_FALLBACK_NAMES


def build_id_map(names: List[str]) -> Dict[int, int]:
    id_map = {}
    for i, n in enumerate(names):
        if norm(n) in SH17_TO_OURS:
            id_map[i] = SH17_TO_OURS[norm(n)]
    found = {norm(names[i]) for i in id_map}
    missing = set(SH17_TO_OURS) - found
    if missing:
        raise SystemExit(f"Could not find these SH17 classes in the names list: {missing}. Names were: {names}")
    return id_map


def read_file_list(path: Path) -> List[str]:
    """SH17 lists contain paths or bare names; we keep only the file stem."""
    stems = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            stems.append(Path(raw.replace("\\", "/")).stem)
    return stems


def find_split_lists(src: Path) -> Tuple[Optional[Path], Optional[Path]]:
    train = next(iter(sorted(src.rglob("train_files.txt"))), None)
    test = next(iter(sorted(src.rglob("test_files.txt"))), None) or next(iter(sorted(src.rglob("val_files.txt"))), None)
    return train, test


def index_files(src: Path, out: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images, labels = {}, {}
    out_resolved = out.resolve()
    for p in src.rglob("*"):
        if not p.is_file() or out_resolved in p.resolve().parents:
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images.setdefault(p.stem, p)
        elif ext == ".txt" and p.parent.name.lower() in {"labels", "label", "yolo", "annotations"} :
            labels.setdefault(p.stem, p)
    return images, labels


def remap_label_lines(text: str, id_map: Dict[int, int]) -> Tuple[List[str], Counter, int]:
    """Returns (kept YOLO lines, per-class counts, number of malformed lines)."""
    kept, counts, bad = [], Counter(), 0
    for ln in text.splitlines():
        parts = ln.split()
        if not parts:
            continue
        if len(parts) != 5:
            bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            bad += 1
            continue
        if cls not in id_map:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            bad += 1
            continue
        new = id_map[cls]
        kept.append(f"{new} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        counts[new] += 1
    return kept, counts, bad


def ahash(img: Image.Image) -> int:
    """8x8 average hash: near-identical images get (almost) the same 64-bit number."""
    g = img.convert("L").resize((8, 8), Image.BILINEAR)
    px = list(g.getdata())
    avg = sum(px) / 64.0
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v >= avg else 0)
    return bits


def process_one(job: Tuple[str, str, str, str, Dict[int, int], int]) -> dict:
    stem, img_path, lbl_path, split_dir, id_map, max_side = job
    img_path, split_dir = Path(img_path), Path(split_dir)
    raw = img_path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    with Image.open(img_path) as im:
        im.draft("RGB", (max_side, max_side))  # fast JPEG downscale while decoding
        im = ImageOps.exif_transpose(im).convert("RGB")
        w0, h0 = im.size
        if max(w0, h0) > max_side:
            s = max_side / max(w0, h0)
            im = im.resize((max(1, round(w0 * s)), max(1, round(h0 * s))), Image.LANCZOS)
        im.save(split_dir / "images" / f"{stem}.jpg", "JPEG", quality=92)
        h = ahash(im)
        size = im.size
    text = Path(lbl_path).read_text(encoding="utf-8") if lbl_path else ""
    kept, counts, bad = remap_label_lines(text, id_map)
    (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"stem": stem, "md5": md5, "ahash": h, "counts": dict(counts), "bad": bad,
            "has_label_file": bool(lbl_path), "size": size}


def leakage_check(records: Dict[str, List[dict]], max_hamming: int) -> dict:
    test = records.get("test", [])
    report = {"exact_duplicates": [], "near_duplicates": []}
    test_md5 = {r["md5"]: r["stem"] for r in test}
    for split in ("train", "val"):
        for r in records.get(split, []):
            if r["md5"] in test_md5:
                report["exact_duplicates"].append({"split": split, "stem": r["stem"], "test_stem": test_md5[r["md5"]]})
    for split in ("train", "val"):
        for r in records.get(split, []):
            for t in test:
                if (r["ahash"] ^ t["ahash"]).bit_count() <= max_hamming:
                    report["near_duplicates"].append({"split": split, "stem": r["stem"], "test_stem": t["stem"]})
    report["n_exact"] = len(report["exact_duplicates"])
    report["n_near"] = len(report["near_duplicates"])
    return report


# ----------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="SH17 root folder (contains images/ and labels/)")
    ap.add_argument("--out", default=Path("datasets/ppe4"), type=Path)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--hamming", type=int, default=3, help="near-duplicate threshold (0-64)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only process N images per split")
    ap.add_argument("--random-split", action="store_true",
                    help="only if train_files.txt/test_files.txt are missing: random 70/15/15 split")
    args = ap.parse_args(argv)

    src, out = args.src.resolve(), args.out.resolve()
    if not src.exists():
        raise SystemExit(f"--src not found: {src}")

    names = load_class_names(src)
    id_map = build_id_map(names)
    print("Class mapping (SH17 id -> ours):",
          {f"{i}:{names[i]}": f"{j}:{TARGET_NAMES[j]}" for i, j in sorted(id_map.items())})

    images, labels = index_files(src, out)
    print(f"Found {len(images)} images and {len(labels)} label files under {src}")
    if not images:
        raise SystemExit("No images found. Point --src at the extracted SH17 folder.")

    train_list, test_list = find_split_lists(src)
    rng = random.Random(args.seed)
    if train_list and test_list:
        train_stems = [s for s in read_file_list(train_list) if s in images]
        test_stems = [s for s in read_file_list(test_list) if s in images]
        overlap = set(train_stems) & set(test_stems)
        if overlap:
            print(f"WARNING: {len(overlap)} images are listed in BOTH official lists; removing them from train.")
            train_stems = [s for s in train_stems if s not in overlap]
        print(f"Official lists: {train_list.name} ({len(train_stems)}), {test_list.name} ({len(test_stems)})")
    elif args.random_split:
        allst = sorted(images)
        rng.shuffle(allst)
        k = int(0.15 * len(allst))
        test_stems, train_stems = allst[:k], allst[k:]
        print("WARNING: using a random split (official lists not found).")
    else:
        raise SystemExit("train_files.txt / test_files.txt not found. Re-download SH17 or pass --random-split.")

    train_stems = sorted(set(train_stems))
    rng.shuffle(train_stems)
    n_val = max(1, int(round(args.val_frac * len(train_stems))))
    splits = {"val": sorted(train_stems[:n_val]), "train": sorted(train_stems[n_val:]), "test": sorted(set(test_stems))}
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    jobs = []
    for split, stems in splits.items():
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        for s in stems:
            lbl = labels.get(s)
            jobs.append((s, str(images[s]), str(lbl) if lbl else "", str(out / split), id_map, args.max_side))
    split_of = {s: sp for sp, stems in splits.items() for s in stems}

    print(f"Processing {len(jobs)} images with {args.workers} worker(s). This can take a while the first time...")
    records: Dict[str, List[dict]] = {k: [] for k in splits}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(process_one, jobs, chunksize=8):
            records[split_of[rec["stem"]]].append(rec)
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")

    # ---- statistics (REAL numbers computed from the files) ----
    stats = {"source": str(src), "seed": args.seed, "val_frac": args.val_frac, "max_side": args.max_side,
             "class_names": TARGET_NAMES, "splits": {}}
    for split, recs in records.items():
        c = Counter()
        for r in recs:
            c.update({int(k): v for k, v in r["counts"].items()})
        stats["splits"][split] = {
            "images": len(recs),
            "images_without_label_file": sum(1 for r in recs if not r["has_label_file"]),
            "images_with_no_target_objects": sum(1 for r in recs if not r["counts"]),
            "malformed_label_lines": sum(r["bad"] for r in recs),
            "instances": {TARGET_NAMES[i]: c.get(i, 0) for i in range(4)},
            "images_containing": {TARGET_NAMES[i]: sum(1 for r in recs if i in {int(k) for k in r["counts"]})
                                  for i in range(4)},
        }

    leak = leakage_check(records, args.hamming)
    stats["leakage"] = {"exact_duplicates_with_test": leak["n_exact"],
                        "near_duplicates_with_test": leak["n_near"], "hamming_threshold": args.hamming}

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "leakage_report.json").write_text(json.dumps(leak, indent=2))
    data_yaml = {"path": out.as_posix(), "train": "train/images", "val": "val/images", "test": "test/images",
                 "names": {i: n for i, n in enumerate(TARGET_NAMES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print("\n=== Dataset summary (real counts) ===")
    print(f"{'split':<6} {'images':>7} {'person':>8} {'helmet':>8} {'vest':>8} {'mask':>8}")
    for split in ("train", "val", "test"):
        s = stats["splits"][split]
        i = s["instances"]
        print(f"{split:<6} {s['images']:>7} {i['person']:>8} {i['helmet']:>8} {i['vest']:>8} {i['mask']:>8}")
    print(f"\nLeakage vs test: {leak['n_exact']} exact duplicates, {leak['n_near']} near-duplicates "
          f"(hamming <= {args.hamming}). Details: {out / 'leakage_report.json'}")
    if leak["n_exact"] or leak["n_near"]:
        print("  -> Open the listed pairs and look at them. Real duplicates: delete them from train/val and rerun.")
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":  # required on Windows for multiprocessing
    sys.exit(main())
