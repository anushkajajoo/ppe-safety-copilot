"""Tests for training/prepare_ppe4.py (the small-dataset builder)."""
from pathlib import Path

import pytest
from PIL import Image

from training import prepare_ppe4 as prep

SOURCE_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "NO-Safety Vest", "Person",
                "Safety Cone", "Safety Vest", "machinery"]


def test_build_id_map_maps_aliases_and_drops_the_rest():
    id_map = prep.build_id_map(SOURCE_NAMES)
    assert id_map[SOURCE_NAMES.index("Person")] == 0
    assert id_map[SOURCE_NAMES.index("Hardhat")] == 1        # hardhat -> helmet
    assert id_map[SOURCE_NAMES.index("Safety Vest")] == 2
    assert id_map[SOURCE_NAMES.index("Mask")] == 3
    # NO-* and other classes must NOT be mapped
    for dropped in ["NO-Hardhat", "NO-Safety Vest", "Safety Cone", "machinery"]:
        assert SOURCE_NAMES.index(dropped) not in id_map


def test_build_id_map_fails_loudly_when_a_class_is_absent():
    with pytest.raises(SystemExit):
        prep.build_id_map(["Person", "Hardhat"])             # no vest, no mask


def test_parse_label_reads_normal_lines_and_remaps(tmp_path):
    id_map = prep.build_id_map(SOURCE_NAMES)
    label = tmp_path / "a.txt"
    label.write_text(
        f"{SOURCE_NAMES.index('Person')} 0.5 0.5 0.4 0.8\n"
        f"{SOURCE_NAMES.index('Hardhat')} 0.5 0.2 0.1 0.1\n"
        f"{SOURCE_NAMES.index('machinery')} 0.1 0.1 0.2 0.2\n",   # dropped
        encoding="utf-8")
    boxes = prep.parse_label(label, id_map)
    assert [b[0] for b in boxes] == [0, 1]


def test_parse_label_ignores_broken_lines(tmp_path):
    id_map = prep.build_id_map(SOURCE_NAMES)
    label = tmp_path / "b.txt"
    label.write_text(
        f"{SOURCE_NAMES.index('Person')} 0.5 0.5\n"            # too few fields
        f"{SOURCE_NAMES.index('Person')} a b c d\n"            # not numbers
        f"{SOURCE_NAMES.index('Person')} 0.5 0.5 0.0 0.3\n",   # zero width
        encoding="utf-8")
    assert prep.parse_label(label, id_map) == []


def test_parse_label_converts_a_polygon_to_a_box(tmp_path):
    id_map = prep.build_id_map(SOURCE_NAMES)
    label = tmp_path / "c.txt"
    label.write_text(f"{SOURCE_NAMES.index('Person')} 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n",
                     encoding="utf-8")
    boxes = prep.parse_label(label, id_map)
    assert len(boxes) == 1
    class_id, xc, yc, w, h = boxes[0]
    assert class_id == 0
    assert abs(xc - 0.4) < 1e-6 and abs(yc - 0.5) < 1e-6
    assert abs(w - 0.4) < 1e-6 and abs(h - 0.6) < 1e-6


def _items(counts):
    """Build fake candidates: counts = {class_id: how many images contain it}."""
    items, index = [], 0
    for class_id, how_many in counts.items():
        for _ in range(how_many):
            items.append({"image": Path(f"{index}.jpg"), "boxes": [(class_id, .5, .5, .2, .2)],
                          "classes": [class_id]})
            index += 1
    return items


def test_pick_subset_keeps_rare_classes():
    # 200 person images, only 5 mask images: random sampling could lose the masks
    items = _items({0: 200, 3: 5})
    chosen = prep.pick_subset(items, quota=20, seed=42, min_per_class=4)
    picked_classes = [c for item in chosen for c in item["classes"]]
    assert len(chosen) == 20
    assert picked_classes.count(3) >= 4          # masks survived


def test_pick_subset_is_reproducible():
    items = _items({0: 50, 1: 30, 2: 20, 3: 10})
    first = prep.pick_subset(items, quota=25, seed=42, min_per_class=5)
    again = prep.pick_subset(items, quota=25, seed=42, min_per_class=5)
    other = prep.pick_subset(items, quota=25, seed=7, min_per_class=5)
    assert [i["image"] for i in first] == [i["image"] for i in again]
    assert [i["image"] for i in first] != [i["image"] for i in other]


def test_pick_subset_returns_everything_when_quota_is_larger():
    items = _items({0: 5})
    assert len(prep.pick_subset(items, quota=100, seed=42, min_per_class=2)) == 5


def test_write_item_downscales_and_keeps_labels(tmp_path):
    source = tmp_path / "big.jpg"
    Image.new("RGB", (2000, 1000), (10, 20, 30)).save(source)
    split_dir = tmp_path / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labels").mkdir(parents=True)

    item = {"image": source, "original": "big", "boxes": [(0, 0.5, 0.5, 0.4, 0.8)], "classes": [0]}
    record = prep.write_item(item, split_dir, max_side=640)

    with Image.open(split_dir / "images" / "big.jpg") as im:
        assert max(im.size) == 640                      # downscaled
    written = (split_dir / "labels" / "big.txt").read_text(encoding="utf-8").split()
    assert written[0] == "0"
    assert abs(float(written[3]) - 0.4) < 1e-6          # normalised label unchanged
    assert record["boxes"] == 1


# --------------------------------------------------------------- grouping tests
def test_original_id_strips_the_roboflow_hash():
    assert prep.original_id("site12_jpg.rf.9a3f7c21") == "site12_jpg"
    assert prep.original_id("plain_name") == "plain_name"


def _copy(original, split_hint, class_id=0, n=1):
    """Fake candidate images belonging to one original photo."""
    return [{"image": Path(f"{original}_{split_hint}_{i}.jpg"), "original": original,
             "boxes": [(class_id, .5, .5, .2, .2)], "classes": [class_id]} for i in range(n)]


def test_regroup_keeps_one_copy_for_val_and_test_and_several_for_train():
    candidates = {
        "train": _copy("photoA", "t", n=5) + _copy("photoB", "t", n=5),
        "val": _copy("photoV", "v", n=3),      # source augmented val, should be trimmed
        "test": _copy("photoT", "s", n=2),
    }
    grouped, info = prep.regroup(candidates, seed=42, copies_train=2,
                                 want_val=0, want_test=0, min_train_originals=0,
                                 topup_eval=False)
    assert len(grouped["train"]) == 4          # 2 originals x 2 copies
    assert len(grouped["val"]) == 1            # one copy only
    assert len(grouped["test"]) == 1
    assert info["originals_per_split"] == {"train": 2, "val": 1, "test": 1}


def test_regroup_removes_a_photo_from_train_if_it_is_also_in_test():
    candidates = {
        "train": _copy("shared", "t", n=4) + _copy("trainonly", "t", n=2),
        "val": [],
        "test": _copy("shared", "s", n=1),
    }
    grouped, info = prep.regroup(candidates, seed=42, copies_train=2,
                                 want_val=0, want_test=0, min_train_originals=0,
                                 topup_eval=False)
    train_originals = {i["original"] for i in grouped["train"]}
    assert train_originals == {"trainonly"}            # 'shared' went to test only
    assert {i["original"] for i in grouped["test"]} == {"shared"}
    assert info["originals_in_more_than_one_source_split"] == 1


def test_regroup_does_not_touch_the_evaluation_splits_by_default():
    candidates = {"train": sum([_copy(f"p{i}", "t", n=2) for i in range(20)], []),
                  "val": _copy("v1", "v"), "test": _copy("s1", "s")}
    grouped, info = prep.regroup(candidates, seed=42, copies_train=1,
                                 want_val=10, want_test=10, min_train_originals=5,
                                 topup_eval=False)
    assert info["originals_moved_from_train"] == {"val": 0, "test": 0}
    assert len(grouped["val"]) == 1 and len(grouped["test"]) == 1


def test_regroup_tops_up_evaluation_splits_when_asked():
    candidates = {"train": sum([_copy(f"p{i}", "t", n=2) for i in range(20)], []),
                  "val": _copy("v1", "v"), "test": _copy("s1", "s")}
    grouped, info = prep.regroup(candidates, seed=42, copies_train=1,
                                 want_val=5, want_test=5, min_train_originals=12,
                                 topup_eval=True)
    moved = info["originals_moved_from_train"]
    assert moved["val"] + moved["test"] == 8           # 20 - 12 spare originals
    assert info["originals_per_split"]["train"] == 12
