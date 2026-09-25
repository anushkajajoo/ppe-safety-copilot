"""
Build samples/demo_clip.mp4 from the sample images.

WHY: the video endpoint and the dashboard's video panel need something to analyse, and a
slideshow of the four demo images is reproducible, small and licence-clean - no filming, no
extra download. It is NOT real site footage and the report says so.

Run:
    python -m scripts.make_demo_clip
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZE = (640, 640)
FPS = 10.0
SECONDS_PER_IMAGE = 3
ORDER = ["compliant_candidate", "helmet_only", "vest_only", "person_only"]


def main() -> int:
    import cv2

    out = ROOT / "samples" / "demo_clip.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        print("\nERROR: OpenCV could not open an mp4 writer on this machine.")
        return 1

    frames = 0
    for name in ORDER:
        image = cv2.imread(str(ROOT / "samples" / f"{name}.jpg"))
        if image is None:
            print(f"\nERROR: missing sample image: samples/{name}.jpg")
            writer.release()
            return 1
        image = cv2.resize(image, SIZE)
        for i in range(int(FPS * SECONDS_PER_IMAGE)):
            # a small brightness wobble so consecutive frames are not identical
            writer.write(cv2.convertScaleAbs(image, alpha=1.0, beta=(i % 5) - 2))
            frames += 1
    writer.release()

    print(f"{out}: {frames} frames, {frames / FPS:.0f} s, {out.stat().st_size / 1e6:.1f} MB")
    print("Try:  python -m edge.detect --source samples\\demo_clip.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
