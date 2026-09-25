"""
Tests for shared/privacy.py - event snapshot masking.

These are the tests that make the privacy claim checkable rather than rhetorical:
the face band really is blurred, the helmet area really is not, and a failure
produces no file at all.
"""
import numpy as np

from shared.privacy import (EVIDENCE_WITHHELD, FACE_BLUR_OK, band_fractions, blur_faces,
                            face_band, save_masked_snapshot)

PERSON = (100.0, 100.0, 200.0, 400.0)          # 100 x 300 person box: a full body
HEAD_CROP = (300.0, 100.0, 500.0, 300.0)       # 200 x 200 box: a head-and-shoulders crop


def noisy_frame(height=480, width=640):
    """Random pixels: blurring changes them, so a change is easy to detect."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


# ------------------------------------------------------------------ geometry
def test_the_band_sits_below_the_helmet_line():
    """A full-body box: the helmet ends near 0.15h, so the band must start there."""
    left, top, right, bottom = face_band(PERSON, 640, 480)
    assert 140 <= top <= 150          # 0.15 of the 300 px box, below the helmet
    assert 180 <= bottom <= 195       # ends around the chin, not down the chest
    assert left < 100 and right > 200  # slightly wider than the person box


def test_the_band_follows_the_shape_of_the_person_box():
    """
    The bug this guards: a fixed fraction put the band on the helmet in full-body
    boxes. A head-and-shoulders crop has its face much lower down the box.
    """
    full_top, _ = band_fractions(300.0, 100.0)     # aspect 3.0
    crop_top, _ = band_fractions(200.0, 200.0)     # aspect 1.0
    assert full_top < crop_top
    assert full_top <= 0.20 and crop_top >= 0.45


def test_a_head_crop_is_masked_low_in_the_box():
    _, top, _, bottom = face_band(HEAD_CROP, 640, 480)
    assert top >= 200                  # well below the top of a 100..300 box
    assert bottom > top


def test_the_band_is_clipped_to_the_frame():
    left, top, right, bottom = face_band((-50.0, -20.0, 60.0, 300.0), 640, 480)
    assert left >= 0 and top >= 0 and right <= 640 and bottom <= 480


def test_a_degenerate_box_has_no_band():
    assert face_band((10.0, 10.0, 10.0, 10.0), 640, 480) is None
    assert face_band((10.0, 10.0, 12.0, 12.0), 640, 480) is None


# ------------------------------------------------------------------ blurring
def test_the_face_area_is_changed():
    frame = noisy_frame()
    before = frame.copy()
    masked, count = blur_faces(frame, [PERSON])
    assert count == 1

    left, top, right, bottom = face_band(PERSON, 640, 480)
    assert not np.array_equal(before[top:bottom, left:right], masked[top:bottom, left:right])


def test_the_helmet_area_is_left_alone():
    """Blurring the helmet would destroy the evidence the snapshot exists for."""
    frame = noisy_frame()
    before = frame.copy()
    masked, _ = blur_faces(frame, [PERSON])

    _, top, _, _ = face_band(PERSON, 640, 480)
    helmet_strip = (slice(100, top - 2), slice(100, 200))     # above the face band
    assert np.array_equal(before[helmet_strip], masked[helmet_strip])


def test_the_rest_of_the_frame_is_untouched():
    frame = noisy_frame()
    before = frame.copy()
    masked, _ = blur_faces(frame, [PERSON])
    assert np.array_equal(before[400:480, 0:100], masked[400:480, 0:100])


def test_every_person_is_masked():
    frame = noisy_frame()
    _, count = blur_faces(frame, [PERSON, (300.0, 50.0, 380.0, 260.0)])
    assert count == 2


def test_no_people_means_nothing_to_blur():
    frame = noisy_frame()
    before = frame.copy()
    masked, count = blur_faces(frame, [])
    assert count == 0 and np.array_equal(before, masked)


# ------------------------------------------------------------------- saving
def test_a_masked_snapshot_is_written(tmp_path):
    path, status = save_masked_snapshot(noisy_frame(), [PERSON], tmp_path)
    assert status == FACE_BLUR_OK
    assert path is not None and path.exists() and path.stat().st_size > 0
    assert path.suffix == ".jpg"


def test_nothing_is_written_when_masking_cannot_be_done(tmp_path):
    """People present but no band could be built -> withhold the evidence."""
    tiny = (10.0, 10.0, 11.0, 11.0)
    path, status = save_masked_snapshot(noisy_frame(), [tiny], tmp_path)
    assert status == EVIDENCE_WITHHELD
    assert path is None
    assert list(tmp_path.glob("*.jpg")) == []


def test_a_broken_frame_never_writes_a_file(tmp_path):
    path, status = save_masked_snapshot("not a frame", [PERSON], tmp_path)
    assert status == EVIDENCE_WITHHELD and path is None
    assert list(tmp_path.glob("*.jpg")) == []


def test_the_saved_image_really_is_the_masked_one(tmp_path):
    import cv2
    frame = noisy_frame()
    before = frame.copy()
    path, status = save_masked_snapshot(frame, [PERSON], tmp_path)
    assert status == FACE_BLUR_OK

    saved = cv2.imread(str(path))
    left, top, right, bottom = face_band(PERSON, 640, 480)
    # the saved face area must not match the original (JPEG noise aside, pixelation is obvious)
    difference = np.abs(saved[top:bottom, left:right].astype(int)
                        - before[top:bottom, left:right].astype(int)).mean()
    assert difference > 10, "the stored image does not look masked"


# ------------------------------------------------- the evaluation script's pure parts
def test_the_evaluation_script_tallies_per_class():
    from edge.types import Detection
    from scripts.privacy_utility import CLASSES, tally

    detections = [Detection(cls="person", conf=0.9, box=(0, 0, 1, 1)),
                  Detection(cls="person", conf=0.7, box=(0, 0, 1, 1)),
                  Detection(cls="helmet", conf=0.8, box=(0, 0, 1, 1))]
    counts = tally(detections)
    assert set(counts) == set(CLASSES)
    assert len(counts["person"]) == 2 and counts["helmet"] == [0.8] and counts["vest"] == []


# ------------------------------------------------------- the keep guard
def test_kept_ppe_boxes_survive_the_blur():
    """
    Masking must never erase the evidence. A helmet or vest box passed as keep_boxes
    is restored pixel-for-pixel after the face band is pixelated.
    """
    frame = noisy_frame()
    before = frame.copy()
    helmet = (110.0, 140.0, 190.0, 175.0)          # deliberately inside the face band
    masked, count = blur_faces(frame, [PERSON], [helmet])
    assert count == 1
    assert np.array_equal(before[140:175, 110:190], masked[140:175, 110:190])


def test_the_face_is_still_masked_around_a_kept_box():
    frame = noisy_frame()
    before = frame.copy()
    helmet = (110.0, 140.0, 130.0, 150.0)          # small box, most of the band remains
    masked, _ = blur_faces(frame, [PERSON], [helmet])
    left, top, right, bottom = face_band(PERSON, 640, 480)
    assert not np.array_equal(before[top:bottom, 140:right], masked[top:bottom, 140:right])


def test_a_kept_box_outside_the_frame_is_ignored():
    frame = noisy_frame()
    masked, count = blur_faces(frame, [PERSON], [(-90.0, -90.0, -10.0, -10.0)])
    assert count == 1                               # no crash, band still blurred


# ------------------------------------ the evaluation script's matching logic
def _det(cls, box, conf=0.9):
    from edge.types import Detection
    return Detection(cls=cls, conf=conf, box=box)


def test_matching_counts_preserved_lost_and_spurious():
    """
    Counting detections alone once reported 129 % retention, because masking can
    invent boxes. Matching by overlap is what makes the number mean something.
    """
    from scripts.privacy_utility import match

    before = [_det("helmet", (0, 0, 10, 10)), _det("helmet", (100, 100, 110, 110))]
    after = [_det("helmet", (1, 1, 11, 11)),            # same helmet, shifted a little
             _det("helmet", (500, 500, 510, 510))]      # invented by the pixelation
    result = match(before, after)
    assert result["helmet"] == {"preserved": 1, "lost": 1, "spurious": 1}


def test_matching_never_pairs_one_box_twice():
    from scripts.privacy_utility import match

    before = [_det("vest", (0, 0, 10, 10))]
    after = [_det("vest", (0, 0, 10, 10)), _det("vest", (1, 1, 11, 11))]
    result = match(before, after)
    assert result["vest"]["preserved"] == 1 and result["vest"]["spurious"] == 1


def test_matching_ignores_other_classes():
    from scripts.privacy_utility import match

    result = match([_det("helmet", (0, 0, 10, 10))], [_det("vest", (0, 0, 10, 10))])
    assert result["helmet"]["lost"] == 1 and result["vest"]["spurious"] == 1
