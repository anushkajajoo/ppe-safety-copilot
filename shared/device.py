"""
Choose the inference device, with a guardrail for a small laptop GPU.

THE PROBLEM
    This laptop's RTX 3050 has 4 GB of VRAM, shared with Windows' own display work
    and whatever else is open. Blindly grabbing the GPU can fail with a CUDA
    out-of-memory error in the middle of a demo - the worst possible moment.

THE RULE
    "auto" (the default) means: use the GPU only if it exists AND has enough free
    memory right now; otherwise fall back to the CPU. The nano model at batch 1
    needs roughly 1.5 GB, so that is the threshold. Everything still works on the
    CPU at about 21 FPS (docs/evaluation/system_evaluation.md), so the fallback is
    a slowdown, never a failure.

    An explicit "cpu" or "0" is always obeyed - the automatic choice is a default,
    not a policy you cannot override.

Every caller also gets a REASON string, so the terminal says why the CPU was picked
instead of leaving you guessing.
"""
from __future__ import annotations

from typing import Optional, Tuple

MIN_FREE_MB = 1500          # a nano YOLO at batch 1, plus room for CUDA's own overhead
AUTO_VALUES = {"", "auto", "none", None}


def resolve_device(preference: Optional[str] = "auto", min_free_mb: int = MIN_FREE_MB,
                   torch_module=None) -> Tuple[str, str]:
    """
    Return (device, reason).

    device : "cpu" or a GPU index like "0", ready to hand to Ultralytics
    reason : a short sentence for the console / logs
    """
    wanted = (preference or "auto").strip().lower() if isinstance(preference, str) else "auto"

    if wanted == "cpu":
        return "cpu", "CPU requested"
    if wanted not in AUTO_VALUES:
        return preference.strip(), f"device {preference.strip()} requested"

    torch = torch_module
    if torch is None:
        try:
            import torch as torch                      # noqa: F401  (imported lazily on purpose)
        except ImportError:
            return "cpu", "PyTorch is not installed"

    try:
        if not torch.cuda.is_available():
            return "cpu", "no CUDA GPU available"
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb, total_mb = free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)
        name = torch.cuda.get_device_name(0)
    except Exception as exc:                            # a driver hiccup must not crash the app
        return "cpu", f"GPU check failed ({type(exc).__name__}), using the CPU"

    if free_mb < min_free_mb:
        return "cpu", (f"{name} has only {free_mb} MB free of {total_mb} MB "
                       f"(need {min_free_mb} MB) - using the CPU instead")
    return "0", f"{name}, {free_mb} MB free of {total_mb} MB"
