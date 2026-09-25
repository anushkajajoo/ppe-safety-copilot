"""
Start the whole PPE Safety Copilot in one command.

    python run.py                 # backend + dashboard, opens the browser
    python run.py --port 8001     # if 8000 is busy
    python run.py --no-browser    # do not open a browser (e.g. over remote desktop)
    python run.py --reload        # auto-restart while editing server code

It starts the FastAPI backend (which also serves the dashboard and the live page) and
opens http://127.0.0.1:8000/dashboard. Press Ctrl+C to stop.

Why one file: a demo should not need three terminals and a memorised uvicorn command.
Everything else (training, dataset preparation, the OpenCV window) stays as its own
command, because those are development tools, not part of the running system.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "ppe4_yolo26n_best.pt"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 keeps the server on this machine only (default)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--reload", action="store_true", help="restart automatically when code changes")
    return ap


def port_is_free(host: str, port: int) -> bool:
    """A friendly check, so the user gets a sentence instead of [Errno 10048]."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) != 0


def urls(host: str, port: int) -> dict:
    base = f"http://{host}:{port}"
    return {"dashboard": f"{base}/dashboard", "live": f"{base}/live",
            "docs": f"{base}/docs", "health": f"{base}/health"}


def banner(host: str, port: int, weights_found: bool) -> str:
    links = urls(host, port)
    lines = [
        "",
        "=" * 62,
        "  Edge Vision Safety Copilot - PPE compliance monitoring",
        "=" * 62,
        f"  Dashboard (image + video) : {links['dashboard']}",
        f"  Live camera               : {links['live']}",
        f"  API documentation         : {links['docs']}",
        f"  Health check              : {links['health']}",
        "",
        "  Everything runs on this machine. No frames are uploaded anywhere.",
    ]
    if not weights_found:
        lines += ["",
                  f"  WARNING: model not found at {WEIGHTS}",
                  "  The pages will load, but detection returns 503 until the model is there.",
                  "  See README, 'Training'."]
    lines += ["", "  Press Ctrl+C to stop.", "=" * 62, ""]
    return "\n".join(lines)


def open_browser_later(url: str, delay: float = 1.5) -> None:
    """Give uvicorn a moment to bind the port before the browser asks for a page."""
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def main() -> int:
    args = build_parser().parse_args()

    if not port_is_free(args.host, args.port):
        print(f"\nERROR: port {args.port} is already in use on {args.host}.")
        print(f"HINT : another server is still running. Either stop it, or use a free port:")
        print(f"       python run.py --port {args.port + 1}")
        return 1

    try:
        import uvicorn
    except ImportError:
        print("\nERROR: uvicorn is not installed in this environment.")
        print("HINT : activate .venv, then  pip install -r requirements.txt")
        return 1

    print(banner(args.host, args.port, WEIGHTS.exists()))
    if not args.no_browser:
        open_browser_later(urls(args.host, args.port)["dashboard"])

    try:
        uvicorn.run("server.main:app", host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        pass
    print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
