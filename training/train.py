"""
Fine-tune a PRETRAINED Ultralytics YOLO model on our 4-class PPE dataset.

We do NOT train from scratch: yolo26n.pt already learned general visual features
on COCO; we only adapt it to person/helmet/vest/mask (transfer learning).

Settings are conservative for a 4 GB laptop GPU. If you get "CUDA out of memory":
    1st: --batch 4      2nd: --imgsz 512      3rd: train on Colab (README)

Run:
    python -m training.train                                   # defaults
    python -m training.train --epochs 3 --fraction 0.1 --name smoke   # 5-minute sanity run
    python -m training.train --resume                          # continue after a crash / shutdown
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "ppe4" / "data.yaml"))
    ap.add_argument("--model", default="yolo26n.pt", help="pretrained checkpoint (yolo11n.pt also works)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2, help="keep low on Windows")
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    ap.add_argument("--patience", type=int, default=15, help="early stop if val mAP doesn't improve")
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of train set (quick tests)")
    ap.add_argument("--name", default="ppe4_yolo26n")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        print(f"data.yaml not found: {data}\nRun training/prepare_sh17.py first.")
        return 1

    project = ROOT / "runs" / "train"
    last_ckpt = project / args.name / "weights" / "last.pt"

    try:
        if args.resume:
            if not last_ckpt.exists():
                print(f"No checkpoint to resume at {last_ckpt}")
                return 1
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
        else:
            model = YOLO(args.model)
            model.train(
                data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, patience=args.patience,
                fraction=args.fraction, project=str(project), name=args.name, exist_ok=True,
                seed=42, deterministic=True, amp=True, cache=False, plots=True,
                # Augmentations: flips are fine for PPE; keep defaults otherwise.
                fliplr=0.5,
            )
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY. Retry with --batch 4, then --imgsz 512, or use Colab.")
        return 2

    best = project / args.name / "weights" / "best.pt"
    if not best.exists():
        print("Training finished but best.pt not found.")
        return 1

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    dst = models_dir / f"{args.name}_best.pt"
    shutil.copy2(best, dst)

    info = {
        "model_file": dst.name,
        "base_model": args.model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data),
        "data_yaml_sha256_16": sha256(data),
        "args": vars(args),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "run_dir": str(project / args.name),
        "note": "Validation metrics are in run_dir/results.csv. Test metrics: run training/validate.py.",
    }
    (models_dir / f"{args.name}_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved {dst}\nSaved {models_dir / (args.name + '_info.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
