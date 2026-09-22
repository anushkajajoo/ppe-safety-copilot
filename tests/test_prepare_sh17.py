"""
Tests the dataset converter on a TINY fake SH17 folder (no download needed).
Checks: class remapping, dropped classes, official test split kept, val carved from
train, resizing, data.yaml, stats, and leakage detection.
"""
import json

import yaml
from PIL import Image

from training.prepare_sh17 import SH17_FALLBACK_NAMES, build_id_map, main, remap_label_lines


def test_id_map_uses_names_not_positions():
    m = build_id_map(SH17_FALLBACK_NAMES)
    assert m == {0: 0, 14: 1, 12: 2, 4: 3}  # Person, Helmet, Safety-vest, Face-mask-medical


def test_remap_drops_other_classes_and_bad_lines():
    m = build_id_map(SH17_FALLBACK_NAMES)
    text = "0 0.5 0.5 0.2 0.4\n14 0.5 0.2 0.1 0.1\n9 0.3 0.3 0.1 0.1\nbroken line\n12 1.5 0.5 0.1 0.1\n"
    kept, counts, bad = remap_label_lines(text, m)
    assert [ln.split()[0] for ln in kept] == ["0", "1"]   # gloves (9) dropped
    assert counts == {0: 1, 1: 1}
    assert bad == 2                                        # malformed + out-of-range


def _fake_sh17(root, n_train=10, n_test=4, dup_of_test=True):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "sh17.yaml").write_text(yaml.safe_dump({"names": {i: n for i, n in enumerate(SH17_FALLBACK_NAMES)}}))
    train, test = [], []
    for i in range(n_train + n_test):
        stem = f"pexels-photo-{i}"
        color = (i * 20 % 255, 80, 160)
        img = Image.new("RGB", (3000, 2000), color)
        for x in range(0, 3000, 300 + i * 17):   # make each image visually different
            for y in range(0, 2000, 7):
                img.putpixel((x, y), (255, 255, 255))
        img.save(root / "images" / f"{stem}.jpeg")
        (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.3 0.6\n14 0.5 0.25 0.1 0.08\n4 0.5 0.3 0.05 0.04\n9 0.2 0.2 0.1 0.1\n")
        (train if i < n_train else test).append(f"C:\\data\\images\\{stem}.jpeg")
    if dup_of_test:  # an exact copy of a test image hidden in train = leakage
        (root / "images" / "pexels-photo-dup.jpeg").write_bytes((root / "images" / "pexels-photo-10.jpeg").read_bytes())
        (root / "labels" / "pexels-photo-dup.txt").write_text("0 0.5 0.5 0.3 0.6\n")
        train.append("pexels-photo-dup.jpeg")
    (root / "train_files.txt").write_text("\n".join(train))
    (root / "test_files.txt").write_text("\n".join(test))


def test_end_to_end_conversion(tmp_path):
    src, out = tmp_path / "sh17", tmp_path / "ppe4"
    _fake_sh17(src)
    assert main(["--src", str(src), "--out", str(out), "--val-frac", "0.2", "--workers", "1", "--max-side", "640"]) == 0

    stats = json.loads((out / "stats.json").read_text())
    s = stats["splits"]
    assert s["test"]["images"] == 4                       # official test list kept as-is
    assert s["train"]["images"] + s["val"]["images"] == 11  # 10 + 1 duplicate
    assert s["val"]["images"] == 2
    assert s["train"]["instances"]["vest"] == 0            # no vest in fake labels
    assert s["test"]["instances"] == {"person": 4, "helmet": 4, "vest": 0, "mask": 4}

    # labels only contain our 4 ids
    for lbl in (out / "train" / "labels").glob("*.txt"):
        for ln in lbl.read_text().splitlines():
            assert ln.split()[0] in {"0", "1", "2", "3"}

    # resized
    any_img = next((out / "test" / "images").glob("*.jpg"))
    assert max(Image.open(any_img).size) == 640

    # data.yaml usable by Ultralytics
    d = yaml.safe_load((out / "data.yaml").read_text())
    assert d["names"] == {0: "person", 1: "helmet", 2: "vest", 3: "mask"}

    # the planted duplicate was detected
    assert stats["leakage"]["exact_duplicates_with_test"] == 1
