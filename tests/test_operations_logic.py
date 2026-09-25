"""
Pure-logic tests for the operations layer: the tool allow-list, the approval rules and the
compliance analytics. No web framework, no database - these run anywhere Python does.
"""
import json

import pytest

from shared.analytics import DecisionLog
from shared.approvals import (APPROVED, KINDS, PENDING, REJECTED, NotApprovable,
                              WRITABLE_KEYS, check_changes, check_kind)
from shared.tools import ALLOWED, DENIED, TOOLS, ToolDenied, ToolLog, check_tool

# ============================================================ the tool allow-list
def test_only_listed_tools_exist():
    assert set(TOOLS) == {"get_camera_status", "get_latest_detection", "get_compliance_status",
                          "get_system_health", "list_recent_events", "create_event",
                          "save_snapshot", "save_recording", "acknowledge_alert",
                          "generate_report"}


def test_there_is_no_shell_filesystem_or_database_tool():
    """Their absence is the design. If one appears, this test is the alarm."""
    for forbidden in ("run_command", "shell", "exec", "eval", "read_file", "write_file",
                      "delete_file", "query_database", "drop_table", "http_request",
                      "send_email", "python"):
        assert forbidden not in TOOLS


def test_an_unknown_tool_is_denied():
    with pytest.raises(ToolDenied):
        check_tool("run_command")
    with pytest.raises(ToolDenied):
        check_tool("get_camera_status ")          # a trailing space is not the same tool


def test_a_writing_tool_needs_authority():
    with pytest.raises(ToolDenied):
        check_tool("save_snapshot", can_write=False)
    assert check_tool("save_snapshot", can_write=True).writes is True
    assert check_tool("get_camera_status", can_write=False).name == "get_camera_status"


def test_the_tool_log_records_denials_as_well_as_calls(tmp_path):
    log = ToolLog(tmp_path / "tools.jsonl")
    log.record("get_camera_status", ALLOWED, "executed", "ana")
    log.record("run_command", DENIED, "not on the tool allow-list", "ana")
    assert len(log.recent()) == 2
    denials = log.denials()
    assert len(denials) == 1 and denials[0]["tool"] == "run_command"


def test_a_broken_tool_log_never_breaks_the_request(tmp_path):
    """Logging is evidence, not a dependency: it must not be able to fail a call."""
    log = ToolLog(tmp_path / "nope" / "x" / "tools.jsonl")
    log.path = tmp_path                      # a directory, so writing must fail
    assert log.record("get_camera_status", ALLOWED, "executed").outcome == ALLOWED


# ============================================================== approvals (pure)
def test_every_approval_kind_states_what_it_changes_and_why():
    for kind, entry in KINDS.items():
        assert entry["what"] and entry["why"]
        assert isinstance(entry["keys"], list)


def test_a_change_can_only_touch_the_keys_its_kind_declares():
    assert check_changes("change_threshold", {"ppe_confidence_threshold": 0.6})
    with pytest.raises(NotApprovable):
        check_changes("change_threshold", {"store_snapshots": True})
    with pytest.raises(NotApprovable):
        check_changes("change_privacy", {"database_url": "sqlite:///evil.db"})
    with pytest.raises(NotApprovable):
        check_kind("run_anything")


def test_no_kind_can_write_a_setting_outside_the_global_list():
    for entry in KINDS.values():
        for key in entry["keys"]:
            assert key in WRITABLE_KEYS


def test_dangerous_settings_are_not_writable_at_all():
    """A change request must never be able to repoint the database or the model."""
    for never in ("database_url", "weights", "edge_api_key", "users_file", "session_secret",
                  "evidence_dir", "storage_root"):
        assert never not in WRITABLE_KEYS


# ============================================================== analytics (pure)
def test_compliance_is_counted_from_real_rows(tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.record("image:a.jpg", "GO", people=1, throttle=False)
    log.record("image:b.jpg", "GO", people=2, throttle=False)
    log.record("image:c.jpg", "STOP", people=1, violations=1, missing=["helmet"], throttle=False)
    log.record("image:d.jpg", "REVIEW", people=1, missing=[], throttle=False)

    summary = log.summary()
    assert summary["total_evaluations"] == 4
    assert summary["compliant"] == 2 and summary["violations"] == 1 and summary["review"] == 1
    assert summary["compliance_pct"] == 50.0
    assert summary["by_item"]["helmet"] == 1 and summary["by_item"]["vest"] == 0
    assert summary["people_seen"] == 5


def test_with_nothing_evaluated_the_rate_is_unknown_not_zero(tmp_path):
    """0 % says everyone failed. None says nobody has been checked. They are different."""
    assert DecisionLog(tmp_path / "empty.jsonl").summary()["compliance_pct"] is None


def test_a_live_camera_does_not_flood_the_analytics(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", min_gap_s=5.0)
    written = sum(log.record("camera:0", "GO", now=100 + tick * 0.1) for tick in range(50))
    assert written == 1


def test_a_change_of_decision_is_always_written(tmp_path):
    """The moment compliance turns into a violation must never be the throttled line."""
    log = DecisionLog(tmp_path / "d.jsonl", min_gap_s=60.0)
    assert log.record("camera:0", "GO", now=100) is True
    assert log.record("camera:0", "GO", now=101) is False
    assert log.record("camera:0", "STOP", now=101.5, violations=1, missing=["helmet"]) is True


def test_a_torn_analytics_line_is_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"at": "x", "decision": "GO"}) + "\n{ cut off",
                    encoding="utf-8")
    assert DecisionLog(path).summary()["total_evaluations"] == 1


