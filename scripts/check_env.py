"""
Day 1 environment check. Verifies Python, PyTorch, CUDA, the exact GPU, VRAM,
Ultralytics, OpenCV and the webcam, and saves the REAL results to
docs/evaluation/env_report.json (use this file in the report / Model Card).

Run:  python -m scripts.check_env
      python -m scripts.check_env --camera 1     (if the webcam is not index 0)
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "evaluation" / "env_report.json"


def line(ok: bool, label: str, value: str) -> None:
    print(f"  [{'OK' if ok else '!!'}] {label:<22} {value}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat()}
    problems = []
    print("\n=== Environment check ===")

    # Python -------------------------------------------------------------
    py = sys.version.split()[0]
    report["python"] = py
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    line(py_ok, "Python", py + ("" if py_ok else "  (use 3.10 - 3.12; 3.11 recommended)"))
    if not py_ok:
        problems.append("python version")
    report["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    line(True, "OS", report["os"])
    report["cpu"] = platform.processor() or "unknown"
    line(True, "CPU", report["cpu"])

    # PyTorch + CUDA ------------------------------------------------------
    try:
        import torch
        report["torch"] = torch.__version__
        line(True, "PyTorch", torch.__version__)
        cuda = torch.cuda.is_available()
        report["cuda_available"] = cuda
        line(cuda, "CUDA available", str(cuda) + ("" if cuda else "  -> you installed the CPU build; see README step 3"))
        if cuda:
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = props.name
            report["gpu_vram_gb"] = round(props.total_memory / 1024 ** 3, 2)
            report["cuda_runtime"] = torch.version.cuda
            report["compute_capability"] = f"{props.major}.{props.minor}"
            line(True, "GPU", props.name)
            line(True, "VRAM", f"{report['gpu_vram_gb']} GB")
            line(True, "CUDA runtime (torch)", str(torch.version.cuda))
            # Tiny real GPU operation to prove it works (not just "is_available")
            x = torch.randn(1024, 1024, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            line(True, "GPU matmul test", "passed")
            try:
                half_ok = torch.randn(8, 8, device="cuda", dtype=torch.float16).sum().item() is not None
                report["fp16_ok"] = half_ok
            except Exception:
                report["fp16_ok"] = False
        else:
            problems.append("cuda")
    except ImportError:
        line(False, "PyTorch", "NOT INSTALLED")
        problems.append("torch")

    # nvidia-smi (driver version) ------------------------------------------
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            report["nvidia_smi"] = out
            line(True, "nvidia-smi", out)
        except Exception as e:  # pragma: no cover
            line(False, "nvidia-smi", str(e))
    else:
        line(False, "nvidia-smi", "not found on PATH (driver missing?)")

    # Ultralytics / OpenCV -------------------------------------------------
    try:
        import ultralytics
        report["ultralytics"] = ultralytics.__version__
        line(True, "Ultralytics", ultralytics.__version__)
    except ImportError:
        line(False, "Ultralytics", "NOT INSTALLED")
        problems.append("ultralytics")

    try:
        import cv2
        report["opencv"] = cv2.__version__
        line(True, "OpenCV", cv2.__version__)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            report["webcam"] = f"index {args.camera}: {w}x{h}"
            line(True, "Webcam", report["webcam"])
        else:
            report["webcam"] = f"index {args.camera}: not readable"
            line(False, "Webcam", report["webcam"] + " (close other apps using the camera, or try --camera 1)")
            problems.append("webcam")
    except ImportError:
        line(False, "OpenCV", "NOT INSTALLED")
        problems.append("opencv")

    # Backend libs ----------------------------------------------------------
    for mod in ("fastapi", "sqlalchemy", "pydantic", "psutil"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "installed")
            line(True, mod, report[mod])
        except ImportError:
            line(False, mod, "NOT INSTALLED")
            problems.append(mod)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["problems"] = problems
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT.relative_to(ROOT)}")
    print("RESULT:", "ALL GOOD" if not problems else f"FIX: {', '.join(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
