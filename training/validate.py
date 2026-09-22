"""
Evaluate a trained model on the held-out TEST split and save per-class metrics.

Run AFTER training (Day 2 / Day 6):
    python -m training.validate --weights models/ppe4_yolo26n_best.pt
Output: docs/evaluation/detector_test_metrics.json  (real numbers for the report)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "datasets" / "ppe4" / "data.yaml"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, plots=True, project=str(ROOT / "runs" / "val"), name=args.split, exist_ok=True)
    names = m.names
    per_class = {}
    for i, cls_idx in enumerate(m.box.ap_class_index):
        cls_idx = int(cls_idx)
        per_class[names[cls_idx]] = {
            "precision": round(float(m.box.p[i]), 4),
            "recall": round(float(m.box.r[i]), 4),
            "mAP50": round(float(m.box.ap50[i]), 4),
            "mAP50_95": round(float(m.box.ap[i]), 4),
        }
    out = {
        "weights": args.weights, "split": args.split, "imgsz": args.imgsz,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                    "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)},
        "per_class": per_class,
    }
    dst = ROOT / "docs" / "evaluation" / f"detector_{args.split}_metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved {dst}")


if __name__ == "__main__":
    main()
