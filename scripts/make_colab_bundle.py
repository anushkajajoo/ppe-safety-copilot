"""
Pack the project code + the prepared dataset into ONE zip for Google Colab.

Training runs on Colab (free T4 GPU) so the laptop's GPU stays free.  Colab needs
two things: our scripts and the dataset.  Rather than a GitHub clone plus a Kaggle
token, we hand it a single file:

    colab_bundle.zip
      training/...          train.py, validate.py, prepare_ppe4.py
      scripts/...           check_dataset.py and friends
      datasets/ppe4/...     the 1143-image dataset

Run:
    python -m scripts.make_colab_bundle

Then upload colab_bundle.zip to Google Drive at MyDrive/ppe_project/ and run
training/colab_train.ipynb.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["training", "scripts", "datasets/ppe4"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "_incoming", ".ipynb_checkpoints"}


def main() -> int:
    dataset = ROOT / "datasets" / "ppe4" / "data.yaml"
    if not dataset.exists():
        print(f"\nERROR: Dataset not found at {dataset}")
        print("HINT : build it first (see README, 'Dataset').")
        return 1

    out = ROOT / "colab_bundle.zip"
    started = time.time()
    count = 0
    # ZIP_STORED: the images are JPEGs already, so compressing them only wastes time.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as bundle:
        for folder in INCLUDE:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_dir() or path.suffix == ".pyc":
                    continue
                if any(part in SKIP_PARTS for part in path.parts):
                    continue
                bundle.write(path, path.relative_to(ROOT).as_posix())
                count += 1

    print(f"\n{out.name}: {count} files, {out.stat().st_size / 1e6:.1f} MB "
          f"({time.time() - started:.0f}s)")
    print("Next: upload it to Google Drive -> MyDrive/ppe_project/colab_bundle.zip")
    print("      then open training/colab_train.ipynb in Colab (T4 GPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
