"""
Detector + tracker wrapper around Ultralytics.

Input : a BGR frame (numpy array from OpenCV)
Output: persons (with track ids) and PPE detections, using OUR class names.

`model.track(..., persist=True, tracker="bytetrack.yaml")` runs YOLO and then ByteTrack.
ByteTrack links boxes across frames using a motion model (Kalman filter) + box overlap
(IoU). It also uses LOW-confidence boxes in a second matching round, which helps keep
IDs through brief occlusions. It uses NO appearance/face features.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from edge.types import Detection

# Model class name (lower-case) -> our name. Works for our fine-tuned model,
# raw SH17 names, and the COCO-pretrained model (which only has "person").
NAME_ALIASES = {
    "person": "person",
    "helmet": "helmet", "hardhat": "helmet", "hard-hat": "helmet",
    "vest": "vest", "safety-vest": "vest", "safety vest": "vest",
    "mask": "mask", "face-mask-medical": "mask", "face-mask": "mask",
}


class Detector:
    def __init__(self, weights: str, device=None, imgsz: int = 640, person_conf: float = 0.4,
                 ppe_conf: float = 0.3, tracker: str = "bytetrack.yaml"):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.half = self.device != "cpu"
        self.imgsz = imgsz
        self.person_conf, self.ppe_conf = person_conf, ppe_conf
        self.tracker = tracker
        self.class_map: Dict[int, str] = {}
        for idx, name in self.model.names.items():
            ours = NAME_ALIASES.get(str(name).strip().lower())
            if ours:
                self.class_map[int(idx)] = ours
        if "person" not in self.class_map.values():
            raise ValueError(f"Model has no 'person' class. Names: {self.model.names}")
        self.has_ppe = any(v != "person" for v in self.class_map.values())
        self.last_speed: Dict[str, float] = {}

    def __call__(self, frame) -> Tuple[List[Detection], List[Detection]]:
        low = min(self.person_conf, self.ppe_conf)
        res = self.model.track(frame, persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=low,
                               device=self.device, half=self.half, classes=list(self.class_map), verbose=False)[0]
        self.last_speed = dict(res.speed)
        persons, ppe = [], []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return persons, ppe
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids: Optional[list] = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else None
        for i in range(len(clss)):
            name = self.class_map.get(int(clss[i]))
            conf = float(confs[i])
            box = tuple(float(v) for v in xyxy[i])
            if name == "person":
                if conf < self.person_conf or ids is None:
                    continue   # untracked person boxes (first frames) are skipped
                persons.append(Detection("person", conf, box, track_id=int(ids[i])))
            elif name and conf >= self.ppe_conf:
                ppe.append(Detection(name, conf, box))
        return persons, ppe
