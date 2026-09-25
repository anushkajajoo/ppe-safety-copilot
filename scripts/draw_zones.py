"""
Redraw the zone polygons in configs/zones.yaml for YOUR camera view.

How it works:
  1. Takes one snapshot from the webcam (or loads --image).
  2. For each zone in zones.yaml (except the full-frame GENERAL zone, unless --all),
     you click the corners in order.
       Left click  = add point      U = undo last point
       ENTER       = finish zone    S = skip (keep old polygon)     Q = quit without saving
  3. Saves the new polygons + frame_size back to zones.yaml (a backup .bak is written first).

Run:  python -m scripts.draw_zones
      python -m scripts.draw_zones --image snapshot.jpg --all
"""
import argparse
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"GENERAL": (80, 185, 63), "CAUTION": (36, 165, 245), "RESTRICTED": (73, 81, 248)}


def snapshot(cam: int):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(15):          # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit("Could not read from the camera.")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default=str(ROOT / "configs" / "zones.yaml"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--all", action="store_true", help="also redraw GENERAL zones")
    args = ap.parse_args()

    path = Path(args.zones)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    img = cv2.imread(args.image) if args.image else snapshot(args.camera)
    h, w = img.shape[:2]
    old_w, old_h = cfg.get("frame_size", [w, h])
    sx, sy = w / old_w, h / old_h

    # rescale existing polygons to the new frame size first
    for z in cfg["zones"]:
        z["polygon"] = [[round(x * sx), round(y * sy)] for x, y in z["polygon"]]

    points = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])

    cv2.namedWindow("draw zones")
    cv2.setMouseCallback("draw zones", on_mouse)

    for z in cfg["zones"]:
        if z["zone_type"] == "GENERAL" and not args.all:
            z["polygon"] = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
            continue
        points.clear()
        color = COLORS.get(z["zone_type"], (255, 255, 255))
        print(f"Draw {z['zone_id']} ({z['zone_type']}): click corners, ENTER = done, U = undo, S = skip, Q = quit")
        while True:
            canvas = img.copy()
            for other in cfg["zones"]:
                if other is not z:
                    cv2.polylines(canvas, [np.array(other["polygon"], np.int32)], True, (120, 120, 120), 1)
            if points:
                cv2.polylines(canvas, [np.array(points, np.int32)], len(points) > 2, color, 2)
                for p in points:
                    cv2.circle(canvas, tuple(p), 5, color, -1)
            cv2.putText(canvas, f"{z['zone_name']}: click corners | ENTER done  U undo  S skip  Q quit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("draw zones", canvas)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10):
                if len(points) >= 3:
                    z["polygon"] = [list(p) for p in points]
                    break
                print("  need at least 3 points")
            elif k in (ord("u"), ord("U")) and points:
                points.pop()
            elif k in (ord("s"), ord("S")):
                break
            elif k in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                print("Quit without saving.")
                return
    cv2.destroyAllWindows()

    cfg["frame_size"] = [w, h]
    shutil.copy2(path, path.with_suffix(".yaml.bak"))
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    snap = ROOT / "data" / "zones_snapshot.jpg"   # data/ is gitignored (may contain people)
    snap.parent.mkdir(exist_ok=True)
    for z in cfg["zones"]:
        cv2.polylines(img, [np.array(z["polygon"], np.int32)], True, COLORS.get(z["zone_type"], (255, 255, 255)), 2)
    cv2.imwrite(str(snap), img)
    print(f"Saved {path} (backup: {path.with_suffix('.yaml.bak').name}) and {snap.relative_to(ROOT)}")
    print("NOTE: comments in zones.yaml are not preserved; the .bak file keeps the old version.")


if __name__ == "__main__":
    main()
