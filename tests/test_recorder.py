"""
Tests for edge/recorder.py - the rolling buffer and event clips.

No codec is used: `writer_factory` is a seam, so these tests assert exactly which frames a
clip would contain without depending on which video backends OpenCV was built with.
"""
import numpy as np
import pytest

from edge.recorder import ClipResult, EventRecorder

FPS = 10.0


def frame(value):
    """A tiny frame whose pixel value identifies it, so order is checkable."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


class FakeWriter:
    """Stands in for cv2.VideoWriter and records what it was asked to write."""
    instances = []

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.frames = []
        self.released = False
        FakeWriter.instances.append(self)

    def write(self, frame):
        self.frames.append(int(frame[0, 0, 0]))

    def release(self):
        self.released = True
        self.path.write_bytes(b"fake-mp4-data")      # so the size check has something to read


@pytest.fixture()
def recorder(tmp_path):
    FakeWriter.instances = []
    return EventRecorder(out_dir=tmp_path, fps=FPS, seconds_before=0.5, seconds_after=0.3,
                         writer_factory=FakeWriter)


# ------------------------------------------------------------------- the buffer
def test_the_buffer_holds_only_the_configured_seconds(recorder):
    for value in range(20):
        recorder.add_frame(frame(value))
    assert recorder.buffered == 5           # 0.5 s at 10 FPS, not 20 frames


def test_nothing_is_written_while_nothing_happens(recorder, tmp_path):
    for value in range(50):
        assert recorder.add_frame(frame(value)) is None
    assert list(tmp_path.iterdir()) == []
    assert recorder.clips_written == 0


def test_the_buffer_stores_copies_not_references(recorder):
    """A caller that reuses one frame buffer must not end up with a clip of one picture."""
    reused = frame(1)
    recorder.add_frame(reused)
    reused[:] = 99
    recorder.add_frame(frame(2))
    recorder.trigger("evt-1")
    for _ in range(3):
        result = recorder.add_frame(frame(3))
    assert FakeWriter.instances[0].frames[0] == 1, "the first frame was overwritten in place"


# -------------------------------------------------------------------- the clip
def test_a_clip_contains_the_seconds_before_and_after_the_event(recorder):
    for value in range(1, 6):               # 5 frames fill the 0.5 s buffer
        recorder.add_frame(frame(value))
    assert recorder.trigger("evt-1") is True

    result = None
    for value in (101, 102, 103):           # 0.3 s after = 3 frames
        result = recorder.add_frame(frame(value))

    assert isinstance(result, ClipResult) and result.ok
    written = FakeWriter.instances[0]
    assert written.frames == [1, 2, 3, 4, 5, 101, 102, 103]
    assert written.released is True
    assert result.frames == 8 and result.event_id == "evt-1"


def test_the_event_id_is_in_the_file_name(recorder):
    recorder.add_frame(frame(1))
    recorder.trigger("EVT-000123")
    for _ in range(3):
        result = recorder.add_frame(frame(2))
    assert "EVT-000123" in result.path.name and result.path.suffix == ".mp4"


def test_a_second_trigger_during_a_clip_is_ignored(recorder):
    """A burst of events must not produce a burst of overlapping clips."""
    recorder.add_frame(frame(1))
    assert recorder.trigger("evt-1") is True
    assert recorder.trigger("evt-2") is False
    assert recorder.recording is True


def test_the_recorder_is_ready_again_after_a_clip(recorder):
    recorder.add_frame(frame(1))
    recorder.trigger("evt-1")
    for _ in range(3):
        recorder.add_frame(frame(2))
    assert recorder.recording is False
    assert recorder.trigger("evt-2") is True


def test_a_stream_that_ends_mid_clip_still_writes_what_it_has(recorder):
    recorder.add_frame(frame(7))
    recorder.trigger("evt-1")
    recorder.add_frame(frame(8))
    result = recorder.flush()
    assert result.ok and FakeWriter.instances[0].frames == [7, 8]


def test_flushing_with_no_clip_does_nothing(recorder):
    assert recorder.flush() is None


# ------------------------------------------------------------------- privacy
def test_every_frame_is_masked_before_it_is_written(recorder):
    """The buffer holds raw frames; the disk never does."""
    seen = []

    def mask(frame, boxes):
        seen.append(int(frame[0, 0, 0]))
        return np.full_like(frame, 200)

    recorder.mask = mask
    recorder.add_frame(frame(1))
    recorder.trigger("evt-1")
    for _ in range(3):
        recorder.add_frame(frame(2))

    assert seen == [1, 2, 2, 2]                       # masking saw every frame
    assert FakeWriter.instances[0].frames == [200, 200, 200, 200]   # only masked reached disk


def test_a_masking_failure_discards_the_whole_clip(recorder, tmp_path):
    """Partly masked evidence is worse than none, so nothing is written."""
    def broken_mask(frame, boxes):
        raise ValueError("masking blew up")

    recorder.mask = broken_mask
    recorder.add_frame(frame(1))
    recorder.trigger("evt-1")
    for _ in range(3):
        result = recorder.add_frame(frame(2))

    assert not result.ok and "masking failed" in result.reason
    assert list(tmp_path.glob("*.mp4")) == []
    assert recorder.clips_failed == 1


# ------------------------------------------------------------------- failures
def test_a_writer_that_cannot_open_is_reported_not_raised(recorder):
    def refuses(path, fps, size):
        raise RuntimeError("no codec available")

    recorder.writer_factory = refuses
    recorder.add_frame(frame(1))
    recorder.trigger("evt-1")
    for _ in range(3):
        result = recorder.add_frame(frame(2))
    assert not result.ok and "could not write" in result.reason
    assert recorder.clips_failed == 1


def test_an_oversized_clip_is_deleted_rather_than_kept(tmp_path):
    FakeWriter.instances = []
    recorder = EventRecorder(out_dir=tmp_path, fps=FPS, seconds_before=0.2, seconds_after=0.1,
                             writer_factory=FakeWriter, max_bytes=5)
    recorder.add_frame(frame(1))
    recorder.trigger("evt-1")
    result = recorder.add_frame(frame(2))
    assert not result.ok and "cap" in result.reason
    assert list(tmp_path.glob("*.mp4")) == []


def test_a_clip_with_no_frames_at_all_is_reported(tmp_path):
    recorder = EventRecorder(out_dir=tmp_path, fps=FPS, seconds_before=0.1, seconds_after=0.1,
                             writer_factory=FakeWriter)
    recorder.trigger("evt-1")
    recorder._pending["after"] = []                   # simulate a stream that produced nothing
    result = recorder.flush()
    assert not result.ok and "no frames" in result.reason
