"""Tests for scripts/check_dataset.py (the dataset validator)."""
from pathlib import Path

from PIL import Image

from scripts import check_dataset as checker


def _empty_problems():
    return {k: [] for k in ["missing_folders", "missing_labels", "missing_images",
                            "corrupt_images", "malformed_lines", "invalid_boxes",
                            "invalid_class_ids"]}


def _make_split(split_dir: Path, labels_by_stem: dict, size=(320, 240)):
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    for stem, lines in labels_by_stem.items():
        Image.new("RGB", size, (100, 120, 140)).save(split_dir / "images" / f"{stem}.jpg")
        (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_clean_split_has_no_problems(tmp_path):
    _make_split(tmp_path / "train", {
        "a": ["0 0.5 0.5 0.4 0.8", "1 0.5 0.2 0.1 0.1"],
        "b": ["0 0.3 0.5 0.2 0.6", "2 0.3 0.6 0.15 0.2", "3 0.3 0.4 0.05 0.05"],
    })
    problems = _empty_problems()
    stats = checker.check_split(tmp_path / "train", problems)

    assert sum(len(v) for v in problems.values()) == 0
    assert stats["images"] == 2 and stats["labels"] == 2 and stats["boxes"] == 5
    assert stats["boxes_per_class"] == {"person": 2, "helmet": 1, "vest": 1, "mask": 1}
    assert stats["images_per_class"]["person"] == 2


def test_missing_label_and_missing_image_are_reported(tmp_path):
    split = tmp_path / "train"
    _make_split(split, {"a": ["0 0.5 0.5 0.4 0.8"]})
    Image.new("RGB", (64, 64)).save(split / "images" / "no_label.jpg")   # image alone
    (split / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    problems = _empty_problems()
    checker.check_split(split, problems)
    assert problems["missing_labels"] == ["train/no_label"]
    assert problems["missing_images"] == ["train/ghost"]


def test_corrupt_image_is_reported(tmp_path):
    split = tmp_path / "train"
    _make_split(split, {"a": ["0 0.5 0.5 0.4 0.8"]})
    (split / "images" / "broken.jpg").write_bytes(b"this is not an image")
    (split / "labels" / "broken.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    problems = _empty_problems()
    checker.check_split(split, problems)
    assert len(problems["corrupt_images"]) == 1
    assert "broken.jpg" in problems["corrupt_images"][0]


def test_bad_label_lines_are_caught(tmp_path):
    split = tmp_path / "train"
    _make_split(split, {"a": [
        "0 0.5 0.5 0.4 0.8",        # good
        "9 0.5 0.5 0.2 0.2",        # class id out of range
        "0 0.5 0.5",                # too few fields
        "0 x y 0.2 0.2",            # not numbers
        "0 1.7 0.5 0.2 0.2",        # outside 0-1
        "0 0.5 0.5 0.0 0.2",        # zero width
        "0 0.95 0.5 0.4 0.2",       # box sticks out of the image
    ]})
    problems = _empty_problems()
    stats = checker.check_split(split, problems)

    assert stats["boxes"] == 1                       # only the good line counted
    assert len(problems["invalid_class_ids"]) == 1
    assert len(problems["malformed_lines"]) == 2
    assert len(problems["invalid_boxes"]) == 3


def test_missing_folder_is_reported(tmp_path):
    problems = _empty_problems()
    stats = checker.check_split(tmp_path / "nope", problems)
    assert problems["missing_folders"] and stats["images"] == 0
