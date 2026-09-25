"""
Privacy-utility evaluation: what does masking faces COST the detector?

The project claims that storing a masked snapshot preserves the evidence while
removing the personal data. This script measures the "preserves the evidence" half,
because a claim you have not measured is just a hope.

METHOD
    For each image:
      1. run the detector -> baseline detections
      2. blur the face band of every detected PERSON (shared/privacy.py), restoring
         the detected helmet and vest boxes afterwards (the "keep" guard)
      3. run the detector again on the masked image
      4. compare detections and mean confidence, per class

HOW TO READ IT - AND WHAT IT CANNOT TELL YOU
    'preserved' is the column that matters: original detections still found in the
    masked image, paired by box overlap (IoU >= 0.5). Raw before/after counts are NOT
    a measure - an earlier version reported 129 % "retention" for vest, because
    pixelation and the restored patches can make the detector invent boxes. Those
    invented boxes are counted separately as 'spurious'. They never change a verdict:
    compliance is decided on the original frame, before anything is stored.

    A low number for `mask` is expected and honest: a face mask sits on the face, and
    the face is what we are hiding.

    Be careful with the helmet and vest numbers: --keep (the default) restores those
    boxes after blurring, so high retention is partly by construction. Run with
    --no-keep to see what the band alone does. The first run of this script, with a
    fixed band and no keep guard, is why both the band and the guard exist: helmet
    retention was 33.3 % and helmet confidence fell from 0.818 to 0.567.

    The measurement is about the DETECTOR, not a person. Pixelation keeps coarse
    shape and colour, so a supervisor can often still read an image the detector
    can no longer parse. Retention is a lower bound on human readability.

Run:
    python -m scripts.privacy_utility                       # dataset val images
    python -m scripts.privacy_utility --images 40 --conf 0.35
    python -m scripts.privacy_utility --no-keep             # band only, no keep guard
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from edge.detect import DEFAULT_WEIGHTS, load_model, to_detections
from edge.geometry import iou
from shared.privacy import blur_faces

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["person", "helmet", "vest", "mask"]


def gather_images(limit: int) -> List[Path]:
    """Prefer the validation split (clean photos); fall back to samples/."""
    val = ROOT / "datasets" / "ppe4" / "val" / "images"
    paths = sorted(val.iterdir()) if val.is_dir() else []
    paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not paths:
        paths = sorted((ROOT / "samples").glob("*.jpg"))
    return paths[:limit]


def match(before, after, iou_min: float = 0.5) -> Dict[str, Dict[str, int]]:
    """
    Pair up detections from the two passes, per class, by box overlap.

    Counting detections alone is a bad measure: masking can CREATE boxes, which is how
    an earlier run reported 129 % "retention". This pairs each original detection with
    the best overlapping detection of the same class in the masked image and reports:
        preserved - originals that survived masking  (the number that matters)
        lost      - originals with no match          (evidence destroyed)
        spurious  - masked-image boxes matching no original (artefacts of pixelation)
    Greedy, highest overlap first - good enough at these box counts and easy to explain.
    """
    out = {name: {"preserved": 0, "lost": 0, "spurious": 0} for name in CLASSES}
    for name in CLASSES:
        originals = [d for d in before if d.cls == name]
        candidates = [d for d in after if d.cls == name]
        pairs = sorted(((iou(o.box, c.box), oi, ci)
                        for oi, o in enumerate(originals)
                        for ci, c in enumerate(candidates)),
                       key=lambda item: -item[0])
        used_o, used_c = set(), set()
        for score, oi, ci in pairs:
            if score < iou_min:
                break
            if oi in used_o or ci in used_c:
                continue
            used_o.add(oi)
            used_c.add(ci)
        out[name]["preserved"] = len(used_o)
        out[name]["lost"] = len(originals) - len(used_o)
        out[name]["spurious"] = len(candidates) - len(used_c)
    return out


def tally(detections) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {name: [] for name in CLASSES}
    for det in detections:
        if det.cls in out:
            out[det.cls].append(det.conf)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--images", type=int, default=40, help="how many images to evaluate")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report", default=None,
                    help="where to write the JSON (default: docs/evaluation/privacy_utility[_band_only].json)")
    ap.add_argument("--no-keep", dest="keep", action="store_false",
                    help="do not restore helmet/vest boxes - measures the band alone")
    ap.set_defaults(keep=True)
    args = ap.parse_args()
    if args.report is None:
        name = "privacy_utility.json" if args.keep else "privacy_utility_band_only.json"
        args.report = str(ROOT / "docs" / "evaluation" / name)

    import cv2

    images = gather_images(args.images)
    if not images:
        print("\nERROR: no images found (datasets/ppe4/val/images or samples/).")
        return 1

    model, class_map = load_model(Path(args.weights), args.device)
    from shared.device import resolve_device
    resolved, _ = resolve_device(args.device)
    predict = dict(imgsz=args.imgsz, conf=args.conf, verbose=False, device=resolved)

    base_counts = {name: 0 for name in CLASSES}
    mask_counts = {name: 0 for name in CLASSES}
    base_conf: Dict[str, List[float]] = {name: [] for name in CLASSES}
    mask_conf: Dict[str, List[float]] = {name: [] for name in CLASSES}
    matched = {name: {"preserved": 0, "lost": 0, "spurious": 0} for name in CLASSES}
    masked_regions = 0
    masked_area_pct: List[float] = []
    used = 0

    guard = "on (helmet/vest boxes restored)" if args.keep else "off (band only)"
    print(f"Evaluating {len(images)} images at conf {args.conf}, keep guard {guard} ...\n")
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        used += 1

        before = to_detections(model.predict(frame, **predict)[0], class_map, args.conf)
        person_boxes = [d.box for d in before if d.cls == "person"]

        keep_boxes = [d.box for d in before if d.cls in ("helmet", "vest")] if args.keep else []
        masked_frame, blurred = blur_faces(frame.copy(), person_boxes, keep_boxes)
        masked_regions += blurred
        if frame.size:
            changed = (masked_frame != frame).any(axis=2).sum()
            masked_area_pct.append(changed / (frame.shape[0] * frame.shape[1]) * 100)
        after = to_detections(model.predict(masked_frame, **predict)[0], class_map, args.conf)

        for name, confs in tally(before).items():
            base_counts[name] += len(confs)
            base_conf[name].extend(confs)
        for name, confs in tally(after).items():
            mask_counts[name] += len(confs)
            mask_conf[name].extend(confs)
        for name, counts in match(before, after).items():
            for key, value in counts.items():
                matched[name][key] += value

    def mean(values: List[float]) -> float:
        return round(statistics.fmean(values), 3) if values else 0.0

    print(f"{'class':<9}{'before':>8}{'after':>8}{'preserved':>11}{'lost':>7}{'spurious':>10}"
          f"{'conf before':>13}{'conf after':>12}")
    rows = {}
    for name in CLASSES:
        preserved = matched[name]["preserved"]
        kept = (preserved / base_counts[name] * 100) if base_counts[name] else 0.0
        rows[name] = {"detections_before": base_counts[name], "detections_after": mask_counts[name],
                      "preserved": preserved, "lost": matched[name]["lost"],
                      "spurious": matched[name]["spurious"],
                      "preserved_pct": round(kept, 1),
                      "mean_conf_before": mean(base_conf[name]), "mean_conf_after": mean(mask_conf[name])}
        print(f"{name:<9}{base_counts[name]:>8}{mask_counts[name]:>8}{preserved:>7} {kept:>5.1f}%"
              f"{matched[name]['lost']:>7}{matched[name]['spurious']:>10}"
              f"{mean(base_conf[name]):>13.3f}{mean(mask_conf[name]):>12.3f}")

    area = mean(masked_area_pct)
    print(f"\nimages: {used}   face regions masked: {masked_regions}   "
          f"pixels changed: {area:.2f}% of frame on average")
    print("'preserved' = originals still found in the masked image (IoU >= 0.5), the number")
    print("that matters. 'spurious' = boxes masking invented; they never affect a verdict,")
    print("because compliance is decided on the original frame before anything is stored.")
    print("A low number for 'mask' is expected - a face mask lives in the masked band.")
    if args.keep:
        print("Helmet/vest retention is partly by construction here; use --no-keep for the band alone.")

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": {"images": used, "conf": args.conf, "imgsz": args.imgsz,
                           "weights": args.weights, "device": resolved,
                           "keep_ppe_boxes": args.keep},
              "face_regions_masked": masked_regions,
              "mean_masked_area_pct_of_frame": area, "per_class": rows}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
