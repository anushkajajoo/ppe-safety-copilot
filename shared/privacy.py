"""
Event snapshot masking - blur faces before any image is stored.

WHY THIS EXISTS
    The system normally stores no pictures at all. But a supervisor reviewing a
    flagged violation often needs to see it, and a stored picture of a worker is
    personal data. Masking is the compromise: keep the evidence, remove the face.

WHY WE BLUR A REGION AND DO NOT DETECT FACES
    Adding a face detector to protect privacy is self-defeating - it is one more
    model trained to find faces, and it fails exactly when the worker is turned
    away or partly hidden, which is when the privacy risk is highest. Instead we
    blur a geometric region derived from the person box. That needs no extra model
    and cannot "miss" a face that the person detector already found.

WHERE THE BAND GOES - MEASURED, NOT GUESSED (see docs/decisions.md D-024a)
    The first version used a fixed band, 0.08h - 0.34h of the person box height.
    scripts/privacy_utility.py then measured what that did to the evidence: helmet
    detections fell to 33 % and helmet confidence from 0.82 to 0.57. The masking was
    destroying the very thing the snapshot exists to show.

    scripts/face_band_geometry.py explains why, from the dataset's own labels: the
    head does NOT sit at a fixed fraction of the person box, because person boxes
    come in two shapes. In a full-body box (height/width >= 2.4) the helmet ends at
    about 0.15 of the box height; in a head-and-shoulders crop (height/width <= 1.2)
    it ends at about 0.60. A single fraction cannot serve both, so the band is placed
    by interpolating on the box's aspect ratio between those two measured anchors.

    A second guard makes the failure structurally impossible rather than merely
    unlikely: any PPE box passed as `keep_boxes` is restored after pixelation. The
    face gets masked; the helmet and vest that the violation is about do not.
    A face mask is deliberately NOT restored - it sits on the face, and identity
    wins over that one class. The compliance verdict is still in the log either way.

FAIL-SAFE
    If masking fails for any reason, NO image is written and the event is marked
    EVIDENCE_WITHHELD. A missing snapshot is an inconvenience; an unmasked one is
    a privacy breach. The safe direction is obvious, so the code takes it.

WHAT WE DO NOT CLAIM
    Pixelation of a band is not anonymisation. Build, gait, clothing and context
    can still identify someone. This reduces the risk; it does not remove it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

from shared.safe_names import safe_filename

Box = Tuple[float, float, float, float]

# Measured anchors (docs/evaluation/face_band_geometry.json): for person boxes of a
# given shape, the fraction of the box height at which the helmet's lower edge - and
# so the top of the face - typically sits. Measured per aspect-ratio bucket over the
# ppe4 train and val labels, then linearly interpolated between the three points.
#   (height/width, head_end_fraction)
HEAD_END_BY_ASPECT = ((1.0, 0.60), (1.5, 0.35), (2.5, 0.15))

# The helmet box covers roughly the upper half of the head, so the rest of the face
# reaches to about 1.9x that fraction. Clamped so the band is never a sliver.
BAND_DEPTH = 1.9
BAND_MIN = 0.12          # never thinner than this fraction of the box height
FACE_PAD_X = 0.04        # a little wider than the box, faces lean out of it

PIXEL_BLOCKS = 25        # pixelation block count; smaller = blockier = safer

FACE_BLUR_OK = "FACE_BLUR_OK"
EVIDENCE_WITHHELD = "EVIDENCE_WITHHELD"
NO_EVIDENCE = "NO_EVIDENCE"


def band_fractions(person_h: float, person_w: float) -> Tuple[float, float]:
    """
    (top, bottom) of the face band as fractions of the person-box height.

    Interpolates between the two measured anchors on the box's aspect ratio, so a
    distant full-body worker and a close-up head crop both get a band over the face
    instead of over the helmet or the chest.
    """
    aspect = person_h / person_w if person_w > 0 else HEAD_END_BY_ASPECT[-1][0]
    top = _interpolate(aspect, HEAD_END_BY_ASPECT)
    bottom = min(1.0, max(top * BAND_DEPTH, top + BAND_MIN))
    return top, bottom


def _interpolate(x: float, points: Sequence[Tuple[float, float]]) -> float:
    """Piecewise-linear lookup, flat outside the measured range."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def face_band(box: Box, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    The rectangle to blur for one person box, clipped to the frame.

    Returns None when the band would be empty (a person box at the very edge, or a
    box so small that the band rounds away to nothing).
    """
    x1, y1, x2, y2 = box
    person_h = y2 - y1
    person_w = x2 - x1
    if person_h <= 0 or person_w <= 0:
        return None

    top_f, bottom_f = band_fractions(person_h, person_w)
    pad = FACE_PAD_X * person_w
    left = int(max(0, round(x1 - pad)))
    right = int(min(width, round(x2 + pad)))
    top = int(max(0, round(y1 + top_f * person_h)))
    bottom = int(min(height, round(y1 + bottom_f * person_h)))

    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _clip(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (int(max(0, round(x1))), int(max(0, round(y1))),
            int(min(width, round(x2))), int(min(height, round(y2))))


def blur_faces(frame, boxes: Sequence[Box], keep_boxes: Sequence[Box] = (),
               strength: int = PIXEL_BLOCKS) -> Tuple[object, int]:
    """
    Blur the face band of every person box, then restore every `keep_boxes` region.

    Returns (frame, regions_blurred). The frame is modified in place, so pass a copy
    if the original is still needed. `keep_boxes` are the PPE boxes whose evidence
    must survive - normally the detected helmet and vest boxes.
    """
    import cv2

    height, width = frame.shape[:2]

    # Copy the keep regions BEFORE anything is blurred, so they can be put back.
    saved = []
    for keep in keep_boxes:
        kx1, ky1, kx2, ky2 = _clip(keep, width, height)
        if kx2 - kx1 < 1 or ky2 - ky1 < 1:
            continue
        saved.append((kx1, ky1, kx2, ky2, frame[ky1:ky2, kx1:kx2].copy()))

    blurred = 0
    for box in boxes:
        band = face_band(box, width, height)
        if band is None:
            continue
        left, top, right, bottom = band
        region = frame[top:bottom, left:right]
        if region.size == 0:
            continue
        # Pixelate: shrink then blow back up. Irreversible, unlike a light blur.
        small_w = max(1, (right - left) // max(1, strength))
        small_h = max(1, (bottom - top) // max(1, strength))
        small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        frame[top:bottom, left:right] = cv2.resize(small, (right - left, bottom - top),
                                                   interpolation=cv2.INTER_NEAREST)
        blurred += 1

    for kx1, ky1, kx2, ky2, patch in saved:
        frame[ky1:ky2, kx1:kx2] = patch
    return frame, blurred


def save_masked_snapshot(frame, boxes: Sequence[Box], out_dir: Path,
                         name_prefix: str = "event",
                         keep_boxes: Sequence[Box] = ()) -> Tuple[Optional[Path], str]:
    """
    Mask the frame and write it. Returns (path or None, privacy_status).

    Never writes an unmasked frame: if anything goes wrong the file is not created
    and the status is EVIDENCE_WITHHELD.
    """
    import cv2

    try:
        masked, blurred = blur_faces(frame.copy(), boxes, keep_boxes)
        if boxes and blurred == 0:
            return None, EVIDENCE_WITHHELD          # people present but nothing masked
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
        # the prefix can carry an event id or an uploaded file name, so it is cleaned:
        # a name is a name, never a path out of the evidence folder
        path = out_dir / f"{safe_filename(name_prefix, fallback='event')}_{stamp}.jpg"
        ok, buffer = cv2.imencode(".jpg", masked, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None, EVIDENCE_WITHHELD
        path.write_bytes(buffer.tobytes())
        return path, FACE_BLUR_OK
    except Exception:
        return None, EVIDENCE_WITHHELD               # fail closed, always
