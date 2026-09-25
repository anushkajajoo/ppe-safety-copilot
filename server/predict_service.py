"""
Loads the trained YOLO model ONCE and turns an image into our Detection objects.

Why a small service class instead of calling YOLO inside the route?
  * the model is loaded lazily, on the first request, so `uvicorn` starts instantly
    and the app still starts on a machine where the weights are missing;
  * the route can be unit-tested by swapping this object for a fake - no GPU, no
    model file, no Ultralytics needed in the test run;
  * there is exactly one place that knows how to map the model's class indices onto
    our four names, shared with the edge pipeline (edge.detector.NAME_ALIASES).

Privacy: the image lives in memory for the length of the request. Nothing is written
to disk here and nothing leaves the machine.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

from edge.detect import DEFAULT_WEIGHTS, to_detections
from edge.detector import NAME_ALIASES
from shared.device import resolve_device
from edge.types import Detection


class ModelNotAvailable(RuntimeError):
    """Raised with a plain-English message when the model cannot be used."""


class PredictService:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.35,
                 imgsz: int = 640, device: Optional[str] = None):
        self.weights = Path(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._class_map = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the weights on first use. Raises ModelNotAvailable with a clear message."""
        if self._model is not None:
            return
        if not self.weights.exists():
            raise ModelNotAvailable(
                f"Model not found: {self.weights}. Train it first (see README, 'Training') "
                f"or set PPE_WEIGHTS to an existing .pt file.")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelNotAvailable("Ultralytics is not installed. Run: pip install -r requirements.txt")

        # Pick the device now, with the VRAM guardrail, and say which one out loud.
        self.device, reason = resolve_device(self.device)
        print(f"[predict] device: {self.device}  ({reason})")

        model = YOLO(str(self.weights))
        class_map = {}
        for index, name in model.names.items():
            ours = NAME_ALIASES.get(str(name).strip().lower())
            if ours:
                class_map[int(index)] = ours
        if "person" not in class_map.values():
            raise ModelNotAvailable(f"This model has no 'person' class. Its classes are: {model.names}")
        self._model, self._class_map = model, class_map

    def detect(self, frame, conf: Optional[float] = None) -> Tuple[List[Detection], float]:
        """Run the model on one BGR frame. Returns (detections, milliseconds)."""
        self.load()
        threshold = self.conf if conf is None else conf
        options = dict(imgsz=self.imgsz, conf=threshold, verbose=False)
        if self.device is not None:
            options["device"] = self.device

        started = time.perf_counter()
        result = self._model.predict(frame, **options)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return to_detections(result, self._class_map, threshold), elapsed_ms
