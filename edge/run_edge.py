"""
Live edge pipeline: camera/video -> YOLO + ByteTrack -> association -> zones ->
temporal filter -> compliance -> overlay + events.

Run (from the project root, venv active):
    python -m edge.run_edge                                     # webcam, pretrained model (person only)
    python -m edge.run_edge --weights models/ppe4_yolo26n_best.pt    # after Colab training
    python -m edge.run_edge --source demo.mp4 --mode baseline   # single-frame baseline for comparison

With the COCO-pretrained model there are no helmet/vest/mask classes, so every
worker will (correctly!) show missing PPE. That still proves tracking, zones,
temporal filtering and events work while the fine-tuned model is training.

Keys:  Q = quit   B = toggle baseline/proposed   Z = show/hide zones
Events are appended to data/edge_events.jsonl (Day 3: offline outbox -> API).
"""
import argparse
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from edge.detector import Detector
from edge.events import EventBuilder, JsonlSink, OutboxSink
from edge.outbox import Outbox
from edge.sync import HttpSender, SyncWorker
from edge.overlay import draw_hud, draw_workers, draw_zones
from edge.pipeline import EdgePipeline
from edge.zones import ZoneManager

ROOT = Path(__file__).resolve().parent.parent


def open_source(src: str):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {src}")
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default=str(ROOT / "models" / "ppe4_yolo26n_best.pt"),
                    help="trained PPE model; pass yolo26n.pt to fall back to the COCO model")
    ap.add_argument("--zones", default=str(ROOT / "configs" / "zones.yaml"))
    ap.add_argument("--policies", default=str(ROOT / "configs" / "policies.yaml"))
    ap.add_argument("--mode", default="proposed", choices=["proposed", "baseline"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto",
                    help="'auto' (GPU if it has free VRAM, else CPU), 'cpu', or '0'")
    ap.add_argument("--device-id", default="EDGE-01")
    ap.add_argument("--events-out", default=str(ROOT / "data" / "edge_events.jsonl"))
    # Intermittent connectivity: with --server, events go to a durable outbox and are
    # drained when the server answers. Without it, they are written to a plain file.
    ap.add_argument("--server", default=None,
                    help="site server base URL, e.g. http://127.0.0.1:8000. Events are queued "
                         "durably and retried, so the link may come and go.")
    ap.add_argument("--api-key", default=None, help="X-API-Key for --server")
    ap.add_argument("--outbox", default=str(ROOT / "data" / "edge_outbox.jsonl"))
    ap.add_argument("--heartbeat-every", type=float, default=30.0)
    ap.add_argument("--max-fps", type=float, default=0, help="cap FPS to reduce laptop load (0 = no cap)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    zm = ZoneManager.from_files(Path(args.zones), Path(args.policies))
    det_cfg = zm.policies["detection"]
    ev_cfg = zm.policies["events"]
    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz,
                        person_conf=float(det_cfg["person_conf"]), ppe_conf=float(det_cfg["ppe_conf"]))
    if not detector.has_ppe:
        print("NOTE: this model has no PPE classes (COCO pretrained). Everyone will show missing PPE.")

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    model_version = Path(args.weights).stem

    syncer = None
    if args.server:
        outbox = Outbox(Path(args.outbox))
        sink = OutboxSink(outbox)
        syncer = SyncWorker(outbox=outbox,
                            send=HttpSender(base_url=args.server, api_key=args.api_key or ""),
                            device_id=args.device_id, camera_id=zm.camera_id,
                            session_id=session_id, model_version=model_version,
                            rules_version=zm.rules_version, device_label=str(args.device),
                            heartbeat_every_s=float(args.heartbeat_every))
        print(f"Sync: queueing to {args.outbox} and delivering to {args.server} "
              f"({outbox.depth()} item(s) already waiting from a previous run)")
    else:
        sink = JsonlSink(Path(args.events_out))

    def make_pipeline(mode):
        b = EventBuilder(args.device_id, zm.camera_id, session_id, model_version, zm.rules_version,
                         cooldown_s=float(ev_cfg["cooldown_seconds"]), emit_uncertain=bool(ev_cfg["emit_uncertain"]))
        return EdgePipeline(zm, b, mode=mode)

    pipe = make_pipeline(args.mode)
    show_zones = True
    cap = open_source(args.source)
    print(f"Session {session_id} | mode {pipe.mode} | model {model_version} | camera {zm.camera_id}")

    n, t_last = 0, time.perf_counter()
    fps = 0.0
    min_dt = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            persons, ppe = detector(frame)
            workers, decisions, events = pipe.step(persons, ppe, w, h)
            sink.write(events)
            if syncer is not None:
                # One call per frame; it returns immediately unless a timer is due.
                syncer.step(frames=1)
            for e in events:
                print(f"EVENT {e['worker_display_id']} {e['zone_id']} {e['status']} {e['reasons']} conf={e['confidence']}")

            if not args.no_show:
                if show_zones:
                    draw_zones(frame, zm)
                draw_workers(frame, workers, decisions)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6)) if n else 0.0
                t_last = now
                draw_hud(frame, f"{pipe.mode.upper()} | FPS {fps:4.1f} | infer {detector.last_speed.get('inference', 0):4.1f} ms"
                                f" | workers {len(workers)} | events {sink.count} | Q quit  B mode  Z zones")
                cv2.imshow("Edge Vision Safety Copilot", frame)
                key = cv2.waitKey(1) & 0xFF
                try:                      # X button on the window also stops us
                    closed = cv2.getWindowProperty("Edge Vision Safety Copilot",
                                                   cv2.WND_PROP_VISIBLE) < 1
                except cv2.error:
                    closed = True
                if closed or key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("b"), ord("B")):
                    pipe = make_pipeline("baseline" if pipe.mode == "proposed" else "proposed")
                    print("Switched to", pipe.mode)
                if key in (ord("z"), ord("Z")):
                    show_zones = not show_zones
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
            if min_dt:
                sleep = min_dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        cap.release()                     # frees the camera
        cv2.destroyAllWindows()
        cv2.waitKey(1)                    # Windows needs one more tick to close the window
    if syncer is not None:
        # One last attempt on the way out, so a clean shutdown does not strand events.
        syncer.step(now=time.time() + syncer.drain_every_s + 1)
        stats = syncer.outbox.stats()
        print(f"Processed {n} frames, queued {sink.count} events. "
              f"Outbox: {stats['depth']} still waiting, {stats['dropped']} dropped.")
        if stats["depth"]:
            print("They are kept on disk and will be delivered the next time the server answers.")
    else:
        print(f"Processed {n} frames, wrote {sink.count} events to {args.events_out}")


if __name__ == "__main__":
    main()
