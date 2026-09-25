"""Tests for shared/violation_log.py - the local JSONL violation record."""
import json

from shared.violation_log import ViolationLog

PEOPLE = [
    {"id": "Person-1", "status": "MISSING_VEST", "missing": ["vest"], "conf": 0.82},
    {"id": "Person-2", "status": "COMPLIANT", "missing": [], "conf": 0.91},
    {"id": "Person-3", "status": "MISSING_HELMET_AND_VEST", "missing": ["helmet", "vest"], "conf": 0.55},
]


def test_only_non_compliant_people_are_logged(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl")
    assert log.log_people("image:a.jpg", PEOPLE, ["helmet", "vest"], now=100.0) == 2
    statuses = [row["status"] for row in log.recent()]
    assert "COMPLIANT" not in statuses


def test_a_logged_line_holds_the_fields_the_brief_asks_for(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl")
    log.log_person("camera:0", "Person-1", "MISSING_HELMET", ["helmet"], 0.77, ["helmet", "vest"], now=1.0)
    row = json.loads((tmp_path / "v.jsonl").read_text().strip())
    assert row["source"] == "camera:0" and row["person_id"] == "Person-1"
    assert row["status"] == "MISSING_HELMET" and row["missing"] == ["helmet"]
    assert row["confidence"] == 0.77 and row["time"].endswith("+00:00")


def test_no_image_or_frame_is_ever_written(tmp_path):
    """The privacy claim: the log holds metadata only."""
    log = ViolationLog(tmp_path / "v.jsonl")
    log.log_people("image:a.jpg", PEOPLE, ["helmet", "vest"], now=1.0)
    text = (tmp_path / "v.jsonl").read_text()
    for forbidden in ["image", "frame", "jpeg", "base64", "box"]:
        assert forbidden not in text.replace("image:a.jpg", "")


def test_the_same_person_and_status_is_throttled(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl", cooldown_s=30)
    assert log.log_person("camera:0", "Person-1", "MISSING_VEST", ["vest"], 0.8, ["vest"], now=0.0)
    assert not log.log_person("camera:0", "Person-1", "MISSING_VEST", ["vest"], 0.8, ["vest"], now=10.0)
    assert log.log_person("camera:0", "Person-1", "MISSING_VEST", ["vest"], 0.8, ["vest"], now=31.0)


def test_a_different_status_is_not_throttled(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl", cooldown_s=30)
    assert log.log_person("camera:0", "Person-1", "MISSING_VEST", ["vest"], 0.8, ["vest"], now=0.0)
    assert log.log_person("camera:0", "Person-1", "MISSING_HELMET", ["helmet"], 0.8, ["helmet"], now=1.0)


def test_video_writes_one_summary_line_only_when_something_was_wrong(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl")
    assert log.log_video("video:a.mp4", 24, 24, 1.0, 2, {"MISSING_VEST": 24}, ["helmet", "vest"])
    assert not log.log_video("video:ok.mp4", 10, 0, 0.0, 1, {"COMPLIANT": 10}, ["helmet", "vest"])
    rows = log.recent()
    assert len(rows) == 1 and rows[0]["kind"] == "video" and rows[0]["frames_analysed"] == 24


def test_recent_returns_newest_first_and_respects_the_limit(tmp_path):
    log = ViolationLog(tmp_path / "v.jsonl", cooldown_s=0)
    for i in range(5):
        log.log_person("image:a.jpg", f"Person-{i}", "MISSING_VEST", ["vest"], 0.5, ["vest"], now=float(i))
    rows = log.recent(limit=3)
    assert len(rows) == 3
    assert [row["person_id"] for row in rows] == ["Person-4", "Person-3", "Person-2"]


def test_a_missing_file_is_not_an_error(tmp_path):
    assert ViolationLog(tmp_path / "nothing.jsonl").recent() == []


def test_a_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"time": "x", "status": "MISSING_VEST"}\nnot json at all\n', encoding="utf-8")
    rows = ViolationLog(path).recent()
    assert len(rows) == 1 and rows[0]["status"] == "MISSING_VEST"
