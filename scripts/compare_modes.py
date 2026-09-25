"""
Does the temporal filter actually help? - a measured answer.

The project claims that judging a worker over 15 frames is better than judging each
frame on its own. This script MEASURES that claim on two scenarios, using the real
edge/temporal.py and edge/compliance.py (no model and no GPU needed: the detector's
output is simulated, so the experiment is repeatable and fast).

    Scenario A - "the detector blinks"
        The worker IS wearing helmet and vest, but the detector misses the helmet in
        a fraction of frames (motion blur, a hand in the way, a bad angle). Every
        frame flagged here is a FALSE ALARM.

    Scenario B - "a real violation"
        The worker genuinely has no vest, in every frame. Flagging is CORRECT; the
        question is how quickly each mode reacts.

    baseline : one frame, one decision (no memory)          - edge/compliance.py decide_baseline()
    proposed : 15-frame window, >=10 missing = violation    - edge/compliance.py decide()

Run:
    python -m scripts.compare_modes
    python -m scripts.compare_modes --frames 300 --miss-rate 0.2 --seed 7

Writes docs/evaluation/temporal_comparison.json with the numbers for the report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from edge.compliance import COMPLIANT, UNCERTAIN, VIOLATION, ComplianceEngine
from edge.temporal import TemporalFilter
from edge.types import Detection, Worker
from edge.zones import Zone

ROOT = Path(__file__).resolve().parent.parent
WORKER_BOX = (100.0, 100.0, 200.0, 400.0)

GENERAL_ZONE = Zone(
    zone_id="general-a", zone_name="General-A", zone_type="GENERAL",
    polygon=[(0, 0), (1279, 0), (1279, 719), (0, 719)],
    required_ppe=["helmet", "vest"], authorization_required=False, priority=0)


def make_worker(present: List[str]) -> Worker:
    """A worker whose detected PPE is exactly `present`."""
    worker = Worker(track_id=1, display_id="Worker-001", box=WORKER_BOX, conf=0.9)
    for item in present:
        worker.ppe[item] = Detection(cls=item, conf=0.8, box=WORKER_BOX)
    return worker


def run_scenario(name: str, truth: List[str], miss_item: str, miss_rate: float,
                 frames: int, seed: int, window: int) -> Dict:
    """
    Feed `frames` frames to BOTH modes and count what each one reported.

    `truth`     : what the worker is really wearing
    `miss_item` : the item the detector sometimes fails to see (miss_rate of frames)
    """
    rng = random.Random(seed)
    engine = ComplianceEngine(window_frames=window, violate_at=int(window * 2 / 3), uncertain_at=window // 3)
    temporal = TemporalFilter(window_frames=window)

    counts = {"baseline": {VIOLATION: 0, UNCERTAIN: 0, COMPLIANT: 0},
              "proposed": {VIOLATION: 0, UNCERTAIN: 0, COMPLIANT: 0}}
    first_violation = {"baseline": None, "proposed": None}

    for frame in range(frames):
        seen = [item for item in truth if not (item == miss_item and rng.random() < miss_rate)]
        worker = make_worker(seen)

        base = engine.decide_baseline(worker, GENERAL_ZONE)
        history = temporal.update(worker.display_id, GENERAL_ZONE.zone_id, seen, worker.conf)
        prop = engine.decide(worker, GENERAL_ZONE, history)

        for mode, decision in (("baseline", base), ("proposed", prop)):
            counts[mode][decision.status] += 1
            if decision.status == VIOLATION and first_violation[mode] is None:
                first_violation[mode] = frame + 1          # 1-based: "flagged on frame N"

    return {"scenario": name, "truth": truth, "miss_item": miss_item, "miss_rate": miss_rate,
            "frames": frames, "window": window, "counts": counts,
            "first_violation_frame": first_violation}


def print_table(result: Dict, is_false_alarm: bool) -> None:
    label = "FALSE ALARMS" if is_false_alarm else "correct flags"
    print(f"\n{result['scenario']}")
    print(f"  worker really wears : {', '.join(result['truth']) or 'nothing'}")
    print(f"  detector misses     : {result['miss_item']} in {result['miss_rate']:.0%} of frames")
    print(f"  {'mode':<10}{'VIOLATION':>12}{'UNCERTAIN':>12}{'COMPLIANT':>12}   first flag")
    for mode in ("baseline", "proposed"):
        counts = result["counts"][mode]
        first = result["first_violation_frame"][mode]
        print(f"  {mode:<10}{counts[VIOLATION]:>12}{counts[UNCERTAIN]:>12}{counts[COMPLIANT]:>12}"
              f"   {('frame ' + str(first)) if first else 'never'}")
    print(f"  (VIOLATION here = {label})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200, help="frames per scenario")
    ap.add_argument("--miss-rate", type=float, default=0.15,
                    help="fraction of frames where the detector misses the helmet")
    ap.add_argument("--window", type=int, default=15, help="temporal window, frames")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=str(ROOT / "docs" / "evaluation" / "temporal_comparison.json"))
    args = ap.parse_args()

    print("=" * 70)
    print("  Temporal filter vs single-frame baseline")
    print("=" * 70)

    blink = run_scenario("Scenario A - detector blinks (worker IS compliant)",
                         truth=["helmet", "vest"], miss_item="helmet",
                         miss_rate=args.miss_rate, frames=args.frames,
                         seed=args.seed, window=args.window)
    print_table(blink, is_false_alarm=True)

    real = run_scenario("Scenario B - real violation (no vest at all)",
                        truth=["helmet"], miss_item="none",
                        miss_rate=0.0, frames=args.frames,
                        seed=args.seed, window=args.window)
    print_table(real, is_false_alarm=False)

    false_alarms_saved = (blink["counts"]["baseline"][VIOLATION]
                          - blink["counts"]["proposed"][VIOLATION])
    delay = ((real["first_violation_frame"]["proposed"] or 0)
             - (real["first_violation_frame"]["baseline"] or 0))

    print("\n" + "-" * 70)
    print(f"  False alarms avoided in Scenario A : {false_alarms_saved} of "
          f"{blink['counts']['baseline'][VIOLATION]} baseline flags")
    print(f"  Cost: the real violation in Scenario B is confirmed {delay} frames later "
          f"({delay / 25:.1f}s at 25 FPS)")
    print("-" * 70)

    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": {"frames": args.frames, "miss_rate": args.miss_rate,
                           "window": args.window, "seed": args.seed},
              "scenario_a_detector_blinks": blink,
              "scenario_b_real_violation": real,
              "false_alarms_avoided": false_alarms_saved,
              "confirmation_delay_frames": delay}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
