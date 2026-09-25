"""
Tests for scripts/retention.py - the job that makes `evidence_retention_days` real.

A setting that does nothing is worse than no setting, so these tests are the difference
between a retention policy and a retention paragraph.
"""
import time

from scripts.retention import expired, sweep

DAY = 86400


def snapshot(folder, name, age_days, content=b"jpegdata"):
    path = folder / name
    path.write_bytes(content)
    stamp = time.time() - age_days * DAY
    import os
    os.utime(path, (stamp, stamp))
    return path


def test_old_snapshots_go_and_recent_ones_stay(tmp_path):
    snapshot(tmp_path, "old.jpg", 40)
    snapshot(tmp_path, "recent.jpg", 3)
    result = sweep(tmp_path, older_than_days=30)
    assert result.deleted == ["old.jpg"] and result.kept == 1
    assert not (tmp_path / "old.jpg").exists()
    assert (tmp_path / "recent.jpg").exists()


def test_a_dry_run_deletes_nothing(tmp_path):
    snapshot(tmp_path, "old.jpg", 90)
    result = sweep(tmp_path, older_than_days=30, dry_run=True)
    assert result.deleted == ["old.jpg"]
    assert (tmp_path / "old.jpg").exists(), "a dry run must not touch the disk"


def test_only_images_are_touched(tmp_path):
    """The log and the database are the record; only the picture expires."""
    snapshot(tmp_path, "old.jpg", 90)
    snapshot(tmp_path, "violations.jsonl", 90)
    snapshot(tmp_path, "notes.txt", 90)
    result = sweep(tmp_path, older_than_days=30)
    assert result.deleted == ["old.jpg"]
    assert (tmp_path / "violations.jsonl").exists() and (tmp_path / "notes.txt").exists()


def test_subfolders_are_not_walked(tmp_path):
    nested = tmp_path / "keep"
    nested.mkdir()
    snapshot(nested, "deep.jpg", 90)
    assert sweep(tmp_path, older_than_days=30).deleted == []
    assert (nested / "deep.jpg").exists()


def test_a_missing_folder_is_not_an_error(tmp_path):
    result = sweep(tmp_path / "nothing-here", older_than_days=30)
    assert result.deleted == [] and result.kept == 0


def test_the_freed_space_is_reported(tmp_path):
    snapshot(tmp_path, "old.jpg", 90, content=b"x" * (2 * 1024 * 1024))
    result = sweep(tmp_path, older_than_days=30)
    assert result.freed_mb == 2.0


def test_the_boundary_is_the_period_itself(tmp_path):
    now = time.time()
    young = snapshot(tmp_path, "just_inside.jpg", 29.9)
    old = snapshot(tmp_path, "just_outside.jpg", 30.1)
    assert not expired(young, 30, now) and expired(old, 30, now)


def test_nothing_is_deleted_when_the_period_is_zero_days_in_the_future(tmp_path):
    """A negative or zero period must not become 'delete everything ever'."""
    snapshot(tmp_path, "today.jpg", 0)
    result = sweep(tmp_path, older_than_days=1)
    assert result.deleted == [] and result.kept == 1
