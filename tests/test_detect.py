"""
Tests for edge/detect.py - the simple image / video / webcam entry point.

No YOLO and no camera are involved: detections are handed in directly, which is
exactly why the detector is kept separate from the rules.
"""
import numpy as np

from edge import detect
from edge.types import Detection, Worker


def person(x1, y1, x2, y2, conf=0.9):
    return Detection(cls="person", conf=conf, box=(x1, y1, x2, y2))


def item(cls, x1, y1, x2, y2, conf=0.8):
    return Detection(cls=cls, conf=conf, box=(x1, y1, x2, y2))


# a person box 100x300 at (100, 100): head region ~y 70-205, torso ~y 160-325
FULL_PPE = [person(100, 100, 200, 400), item("helmet", 120, 90, 180, 140),
            item("vest", 110, 200, 190, 300)]


# ------------------------------------------------------------------ the rule
def test_person_with_helmet_and_vest_is_compliant():
    workers, unassigned = detect.build_workers(FULL_PPE)
    assert len(workers) == 1 and unassigned == []
    assert detect.status_for(workers[0]) == ("COMPLIANT", [])


def test_missing_vest():
    workers, _ = detect.build_workers([FULL_PPE[0], FULL_PPE[1]])
    assert detect.status_for(workers[0]) == ("MISSING_VEST", ["vest"])


def test_missing_helmet():
    workers, _ = detect.build_workers([FULL_PPE[0], FULL_PPE[2]])
    assert detect.status_for(workers[0]) == ("MISSING_HELMET", ["helmet"])


def test_missing_both_items_are_named_in_order():
    workers, _ = detect.build_workers([FULL_PPE[0]])
    status, missing = detect.status_for(workers[0])
    assert status == "MISSING_HELMET_AND_VEST"
    assert missing == ["helmet", "vest"]


def test_required_list_is_configurable():
    """A mask is not required by default, so requiring it changes the verdict."""
    workers, _ = detect.build_workers(FULL_PPE)
    assert detect.status_for(workers[0], ["helmet", "vest", "mask"])[0] == "MISSING_MASK"
    assert detect.status_for(workers[0], ["helmet"])[0] == "COMPLIANT"


# ------------------------------------------------------------- association
def test_ppe_far_from_the_person_is_unassigned():
    """A helmet lying on a table must not make anybody compliant."""
    workers, unassigned = detect.build_workers([person(100, 100, 200, 400),
                                                item("helmet", 500, 500, 560, 540)])
    assert [d.cls for d in unassigned] == ["helmet"]
    assert detect.status_for(workers[0])[0] == "MISSING_HELMET_AND_VEST"


def test_one_helmet_cannot_cover_two_people():
    detections = [person(100, 100, 200, 400), person(210, 100, 310, 400),
                  item("helmet", 120, 90, 180, 140), item("vest", 110, 200, 190, 300),
                  item("vest", 220, 200, 300, 300)]
    workers, _ = detect.build_workers(detections)
    statuses = [detect.status_for(w)[0] for w in workers]
    assert statuses == ["COMPLIANT", "MISSING_HELMET"]


def test_workers_are_numbered_for_display():
    workers, _ = detect.build_workers([person(0, 0, 50, 150), person(60, 0, 110, 150)])
    assert [w.display_id for w in workers] == ["Person-1", "Person-2"]


# ------------------------------------------------------------------ plumbing
def test_source_kind_recognises_webcam_image_and_video():
    assert detect.source_kind("0") == "webcam"
    assert detect.source_kind("2") == "webcam"
    assert detect.source_kind("samples/a.JPG") == "image"
    assert detect.source_kind("samples/clip.mp4") == "video"
    assert detect.source_kind("notes.txt") == "unknown"


class _FakeBoxes:
    """Mimics the parts of an Ultralytics result that to_detections() reads."""
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    class _Arr:
        def __init__(self, values):
            self._values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    @property
    def xyxy(self):
        return self._Arr([r[0] for r in self._rows])

    @property
    def conf(self):
        return self._Arr([r[1] for r in self._rows])

    @property
    def cls(self):
        return self._Arr([r[2] for r in self._rows])


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)


def test_to_detections_maps_classes_and_drops_low_confidence():
    rows = [((10, 10, 50, 200), 0.91, 0),     # person
            ((15, 5, 45, 30), 0.80, 1),       # helmet
            ((15, 60, 45, 120), 0.10, 2)]     # vest, below the threshold
    class_map = {0: "person", 1: "helmet", 2: "vest"}
    detections = detect.to_detections(_FakeResult(rows), class_map, conf_threshold=0.35)
    assert [d.cls for d in detections] == ["person", "helmet"]
    assert detections[0].box == (10.0, 10.0, 50.0, 200.0)


def test_to_detections_ignores_classes_we_do_not_use():
    rows = [((0, 0, 10, 10), 0.99, 7)]        # e.g. 'machinery'
    assert detect.to_detections(_FakeResult(rows), {0: "person"}, 0.3) == []


def test_to_detections_handles_an_empty_frame():
    assert detect.to_detections(_FakeResult([]), {0: "person"}, 0.3) == []


# ------------------------------------------------------------------- drawing
def test_draw_marks_the_frame_without_crashing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    workers, unassigned = detect.build_workers(FULL_PPE)
    out = detect.draw(frame.copy(), workers, unassigned)
    assert out.shape == frame.shape
    assert out.any(), "nothing was drawn on the frame"


def test_summarise_lists_every_person_and_their_status():
    workers, _ = detect.build_workers(FULL_PPE + [person(300, 100, 400, 400)])
    text = detect.summarise(workers)
    assert "Person-1" in text and "COMPLIANT" in text
    assert "Person-2" in text and "MISSING_HELMET_AND_VEST" in text


def test_summarise_says_when_nobody_is_there():
    assert "no person detected" in detect.summarise([])


# ------------------------------------------------------------- stopping the loop
def test_q_esc_and_a_closed_window_all_stop_the_loop():
    for key in (ord("q"), ord("Q"), 27):          # 27 = Esc
        assert detect.should_quit(key) is True
    assert detect.should_quit(255, window_closed=True) is True   # X button


def test_other_keys_do_not_stop_the_loop():
    for key in (255, ord("a"), ord("b"), ord("z"), 13):          # 255 = no key pressed
        assert detect.should_quit(key) is False


# -------------------------------------------------- device policy (D-022)
def test_the_cli_picks_the_device_automatically():
    """auto = GPU only when it has free VRAM, else CPU (shared/device.py, D-022a)."""
    import inspect
    source = inspect.getsource(detect.main)
    assert 'default="auto"' in source, "edge/detect.py should default --device to auto"
