"""
Event-based recording: keep the last few seconds in memory, write a clip only when
something happens.

WHY NOT JUST RECORD EVERYTHING
    Continuous recording of a work area is the surveillance system this project exists to
    avoid, and it is also the one that fills a disk. A rolling buffer inverts the default: the
    system holds a few seconds of video in RAM at all times, and that memory is overwritten
    forever unless an event fires. Nothing reaches disk on a quiet shift.

HOW A CLIP IS BUILT
        ... buffer holds the last N seconds ...
        event fires at T
        clip = [T - seconds_before  ...  T + seconds_after]
    The "before" half is why the buffer exists: by the time the rules have confirmed a
    violation, the moment that caused it is already in the past. A recorder that starts at T
    records the aftermath.

PRIVACY IS NOT OPTIONAL HERE
    Every frame passes through the same masking used for snapshots (shared/privacy.py) before
    it is written. The buffer holds raw frames in memory - it has to, that is what a camera
    produces - but nothing unmasked is ever written to disk. If masking fails for a frame, the
    whole clip is abandoned rather than written partly masked.

BOUNDED BY DESIGN
    The buffer is a deque with a hard length, so memory cannot grow. The clip has a byte cap,
    so a stuck event cannot fill the disk. Both are configuration, not constants in the code.

TESTABLE WITHOUT A CODEC
    `writer_factory` is a seam: tests pass a fake writer and assert which frames were written,
    in what order, with no video codec, no temporary files and no OpenCV build differences.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Deque, List, Optional, Sequence, Tuple

from shared.safe_names import safe_filename

DEFAULT_FPS = 10.0
DEFAULT_BEFORE_S = 5.0
DEFAULT_AFTER_S = 5.0


class FakeWriterUnavailable(RuntimeError):
    """The video writer could not be opened - no codec, no permission, no disk."""


def default_writer_factory(path: Path, fps: float, size: Tuple[int, int]):
    """Real OpenCV writer. mp4v is the codec that ships with opencv-python everywhere."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise FakeWriterUnavailable(f"could not open a video writer for {path}")
    return writer


@dataclass
class ClipResult:
    path: Optional[Path]
    frames: int
    seconds: float
    reason: str
    event_id: Optional[str] = None
    bytes_written: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None


@dataclass
class EventRecorder:
    """
    A rolling buffer plus a one-clip-at-a-time recorder.

    Call `add_frame` for every frame. Call `trigger` when an event fires; the recorder then
    collects `seconds_after` more frames and writes the clip on the frame that completes it
    (or on `flush`, if the stream ends first).
    """
    out_dir: Path
    fps: float = DEFAULT_FPS
    seconds_before: float = DEFAULT_BEFORE_S
    seconds_after: float = DEFAULT_AFTER_S
    mask: Optional[Callable[[object], object]] = None
    writer_factory: Callable = default_writer_factory
    max_bytes: int = 25 * 1024 * 1024

    _buffer: Deque = field(default_factory=deque, init=False)
    _pending: Optional[dict] = field(default=None, init=False)
    clips_written: int = field(default=0, init=False)
    clips_failed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=max(1, int(round(self.fps * self.seconds_before))))

    # ------------------------------------------------------------------ input
    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def recording(self) -> bool:
        return self._pending is not None

    def add_frame(self, frame) -> Optional[ClipResult]:
        """
        Feed one frame. Returns a ClipResult on the frame that completes a clip, else None.

        A copy is taken: the caller usually reuses its frame buffer, and a deque of aliases
        would quietly become a deque of the same picture.
        """
        copied = frame.copy() if hasattr(frame, "copy") else frame
        if self._pending is not None:
            self._pending["after"].append(copied)
            if len(self._pending["after"]) >= self._pending["needed"]:
                return self._write()
            return None
        self._buffer.append(copied)
        return None

    def trigger(self, event_id: str, boxes: Sequence = ()) -> bool:
        """
        Start a clip around the current moment. Returns False if one is already in progress -
        a burst of events must not produce a burst of overlapping clips.
        """
        if self._pending is not None:
            return False
        self._pending = {"event_id": event_id,
                         "before": list(self._buffer),
                         "after": [],
                         "needed": max(1, int(round(self.fps * self.seconds_after))),
                         "boxes": list(boxes)}
        self._buffer.clear()
        return True

    def flush(self) -> Optional[ClipResult]:
        """Write whatever a pending clip has so far. Used when a stream ends mid-clip."""
        if self._pending is None:
            return None
        return self._write()

    # ----------------------------------------------------------------- output
    def _write(self) -> ClipResult:
        pending, self._pending = self._pending, None
        frames = pending["before"] + pending["after"]
        event_id = pending["event_id"]
        if not frames:
            self.clips_failed += 1
            return ClipResult(None, 0, 0.0, "no frames buffered", event_id)

        try:
            masked = self._mask_all(frames, pending["boxes"])
        except Exception as exc:                       # masking failed -> write nothing
            self.clips_failed += 1
            return ClipResult(None, 0, 0.0, f"masking failed ({exc}); clip discarded", event_id)

        try:
            height, width = masked[0].shape[:2]
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            # same rule as snapshots: an event id names a file, it does not choose a folder
            path = self.out_dir / f"{safe_filename(event_id, fallback='event')}_{stamp}.mp4"
            writer = self.writer_factory(path, self.fps, (width, height))
            for frame in masked:
                writer.write(frame)
            release = getattr(writer, "release", None)
            if callable(release):
                release()
        except Exception as exc:
            self.clips_failed += 1
            return ClipResult(None, 0, 0.0, f"could not write the clip ({exc})", event_id)

        size = path.stat().st_size if path.exists() else 0
        if size > self.max_bytes:
            try:
                path.unlink()
            except OSError:
                pass
            self.clips_failed += 1
            return ClipResult(None, len(masked), 0.0,
                              f"clip exceeded the {self.max_bytes} byte cap and was removed",
                              event_id)

        self.clips_written += 1
        return ClipResult(path, len(masked), round(len(masked) / max(self.fps, 1e-6), 2),
                          "clip written", event_id, size)

    def _mask_all(self, frames: List, boxes: Sequence) -> List:
        if self.mask is None:
            return frames
        return [self.mask(frame, boxes) for frame in frames]
