"""
Baseline comparison: a person inspecting photos, against the copilot doing it.

HOW THIS AVOIDS MAKING NUMBERS UP
    The manual side is MEASURED, not assumed. `--manual` shows you each image, you press a
    key to say compliant or not, and the script times you. The system side then runs the real
    detector over the same images. Both halves are recorded with the number of images, the
    date and who did the manual pass, so the comparison can be repeated and challenged.

    If you have not done a manual pass, the script says so and refuses to print a comparison.
    There is no default "a human takes 30 seconds" hiding anywhere in this file.

WHAT IS COMPARED
    time per image, total time, agreement between the two, what each one missed relative to
    the other, and whether evidence exists afterwards.

    Agreement is not accuracy: neither side is ground truth here. Where they disagree, the
    honest reading is "these are the cases a supervisor should look at", which is exactly
    what the REVIEW state is for.

Run:
    python -m scripts.baseline_comparison --manual --images 20     # you label, timed
    python -m scripts.baseline_comparison                          # system pass + compare
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "evaluation" / "baseline_comparison.json"
REQUIRED = ["helmet", "vest"]


def gather(limit: int) -> List[Path]:
    val = ROOT / "datasets" / "ppe4" / "val" / "images"
    paths = [p for p in sorted(val.iterdir())
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if val.is_dir() else []
    return (paths or sorted((ROOT / "samples").glob("*.jpg")))[:limit]


def manual_pass(images: List[Path], operator: str) -> dict:
    """You look at each image and judge it. The script only holds the stopwatch."""
    import cv2

    print("\nMANUAL INSPECTION - you are the baseline.")
    print("  C = compliant    V = violation    U = cannot tell    Q = stop\n")
    verdicts: Dict[str, str] = {}
    times: Dict[str, float] = {}
    for index, path in enumerate(images, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        cv2.imshow("Manual inspection - C compliant / V violation / U unsure / Q quit", frame)
        started = time.perf_counter()
        key = cv2.waitKey(0) & 0xFF
        elapsed = time.perf_counter() - started
        if key in (ord("q"), ord("Q"), 27):
            break
        verdict = {ord("c"): "GO", ord("C"): "GO", ord("v"): "STOP", ord("V"): "STOP"}.get(
            key, "REVIEW")
        verdicts[path.name] = verdict
        times[path.name] = elapsed
        print(f"  {index:>3}. {path.name:<40} {verdict:<7} {elapsed:6.2f} s")
    cv2.destroyAllWindows()

    total = sum(times.values())
    return {"operator": operator, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "images": len(verdicts), "verdicts": verdicts,
            "seconds_per_image": round(total / len(verdicts), 2) if verdicts else None,
            "total_seconds": round(total, 2),
            "method": "operator judged each image on screen; the script timed each keypress"}


def system_pass(images: List[Path], weights: Path, conf: float, threshold: float,
                margin: float, device: str) -> dict:
    import cv2

    from edge.detect import build_workers, load_model, to_detections
    from shared.decision import decide_frame, decide_person
    from shared.device import resolve_device

    model, class_map = load_model(weights, device)
    resolved, _ = resolve_device(device)
    verdicts: Dict[str, str] = {}
    times: Dict[str, float] = {}
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        started = time.perf_counter()
        detections = to_detections(
            model.predict(frame, imgsz=640, conf=conf, verbose=False, device=resolved)[0],
            class_map, conf)
        workers, _ = build_workers(detections)
        calls = [decide_person({name: det.conf for name, det in worker.ppe.items()},
                               REQUIRED, threshold, margin) for worker in workers]
        verdicts[path.name] = decide_frame(calls).decision
        times[path.name] = time.perf_counter() - started

    total = sum(times.values())
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": resolved, "images": len(verdicts), "verdicts": verdicts,
            "seconds_per_image": round(total / len(verdicts), 3) if verdicts else None,
            "total_seconds": round(total, 2),
            "method": "the same pipeline the dashboard uses, over the same images"}


def compare(manual: dict, system: dict) -> dict:
    shared_names = sorted(set(manual["verdicts"]) & set(system["verdicts"]))
    agree = [n for n in shared_names if manual["verdicts"][n] == system["verdicts"][n]]
    manual_stop_only = [n for n in shared_names
                        if manual["verdicts"][n] == "STOP" and system["verdicts"][n] == "GO"]
    system_stop_only = [n for n in shared_names
                        if system["verdicts"][n] == "STOP" and manual["verdicts"][n] == "GO"]
    speed = None
    if manual.get("seconds_per_image") and system.get("seconds_per_image"):
        speed = round(manual["seconds_per_image"] / system["seconds_per_image"], 1)
    return {
        "images_compared": len(shared_names),
        "agreement_pct": round(len(agree) / len(shared_names) * 100, 1) if shared_names else None,
        "manual_flagged_system_did_not": manual_stop_only,
        "system_flagged_manual_did_not": system_stop_only,
        "seconds_per_image": {"manual": manual.get("seconds_per_image"),
                              "system": system.get("seconds_per_image")},
        "times_faster": speed,
        "evidence_afterwards": {"manual": "none unless the inspector wrote something down",
                                "system": "decision, reason, confidence, rule, audit row, "
                                          "and a masked snapshot when enabled"},
        "note": ("Neither side is ground truth. Disagreements are the cases worth a "
                 "supervisor's attention, which is what REVIEW exists for."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", action="store_true", help="do the timed manual pass")
    parser.add_argument("--operator", default="", help="who did the manual pass")
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default=str(ROOT / "models" / "ppe4_yolo26n_best.pt"))
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    from server.config import get_settings
    settings = get_settings()

    images = gather(args.images)
    if not images:
        print("No images found.")
        return 1

    existing = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}

    if args.manual:
        operator = args.operator or input("Your name (recorded with the measurement): ").strip()
        if not operator:
            print("A manual measurement needs a name attached to it.")
            return 1
        existing["manual"] = manual_pass(images, operator)

    print("\nSYSTEM PASS ...")
    existing["system"] = system_pass(images, Path(args.weights), args.conf,
                                     settings.ppe_confidence_threshold,
                                     settings.review_margin, args.device)
    print(f"  {existing['system']['images']} images, "
          f"{existing['system']['seconds_per_image']} s each")

    if "manual" not in existing:
        print("\nNo manual measurement on file, so there is nothing honest to compare against.")
        print("Run:  python -m scripts.baseline_comparison --manual --images 20")
        existing.pop("comparison", None)
    else:
        existing["comparison"] = compare(existing["manual"], existing["system"])
        result = existing["comparison"]
        print(f"\n{'':<26}{'manual':>12}{'system':>12}")
        print(f"{'seconds per image':<26}{result['seconds_per_image']['manual']:>12}"
              f"{result['seconds_per_image']['system']:>12}")
        print(f"{'images compared':<26}{result['images_compared']:>12}")
        print(f"{'agreement':<26}{str(result['agreement_pct']) + ' %':>12}")
        print(f"{'system flagged, human did not':<26}{len(result['system_flagged_manual_did_not']):>12}")
        print(f"{'human flagged, system did not':<26}{len(result['manual_flagged_system_did_not']):>12}")
        if result["times_faster"]:
            print(f"\nThe system was {result['times_faster']}x faster per image on this run.")
        print(result["note"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
