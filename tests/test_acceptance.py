"""
Tests for scripts/check_acceptance.py - the go/no-go gate.

The gate's own failure modes are what matter here: a missing report must never read as a
pass, and a criterion whose measurement is absent must not quietly disappear.
"""
import json

from scripts.check_acceptance import (FAIL, PASS, SKIPPED, check_one, dig, load_criteria,
                                      verdict)

CRITERION = {"id": "demo", "what": "w", "why": "y", "report": "r.json",
             "path": ["overall", "mAP50"], "operator": ">=", "threshold": 0.7}


def write(tmp_path, data, name="r.json"):
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_a_value_above_the_threshold_passes(tmp_path):
    write(tmp_path, {"overall": {"mAP50": 0.8036}})
    result = check_one(CRITERION, tmp_path)
    assert result["status"] == PASS and result["value"] == 0.8036


def test_a_value_below_the_threshold_fails(tmp_path):
    write(tmp_path, {"overall": {"mAP50": 0.51}})
    assert check_one(CRITERION, tmp_path)["status"] == FAIL


def test_a_missing_report_is_skipped_never_passed(tmp_path):
    """The failure this guards: an empty evaluation folder reading as a clean bill of health."""
    result = check_one(CRITERION, tmp_path)
    assert result["status"] == SKIPPED and result["value"] is None
    assert "not produced" in result["detail"]


def test_a_missing_value_inside_a_present_report_is_skipped(tmp_path):
    write(tmp_path, {"overall": {"something_else": 1}})
    result = check_one(CRITERION, tmp_path)
    assert result["status"] == SKIPPED and "no value at" in result["detail"]


def test_a_corrupt_report_is_skipped_not_crashed(tmp_path):
    (tmp_path / "r.json").write_text("{ not json", encoding="utf-8")
    assert check_one(CRITERION, tmp_path)["status"] == SKIPPED


def test_every_operator_works(tmp_path):
    write(tmp_path, {"overall": {"mAP50": 0.5}})
    for operator, expected in ((">=", FAIL), (">", FAIL), ("<=", PASS), ("<", PASS), ("==", FAIL)):
        criterion = dict(CRITERION, operator=operator)
        assert check_one(criterion, tmp_path)["status"] == expected, operator


def test_an_unknown_operator_is_refused(tmp_path):
    write(tmp_path, {"overall": {"mAP50": 0.9}})
    result = check_one(dict(CRITERION, operator="~="), tmp_path)
    assert result["status"] == SKIPPED and "unknown operator" in result["detail"]


def test_a_nested_path_with_spaces_is_found(tmp_path):
    """Lighting conditions are keyed by their human label - dig must handle that."""
    write(tmp_path, {"preserved_pct": {"slightly dark (gamma 1.6)": {"person": 71.0}}})
    criterion = dict(CRITERION, path=["preserved_pct", "slightly dark (gamma 1.6)", "person"],
                     threshold=60.0)
    assert check_one(criterion, tmp_path)["status"] == PASS


def test_dig_returns_none_rather_than_raising():
    assert dig({"a": {"b": 1}}, ["a", "b"]) == 1
    assert dig({"a": {"b": 1}}, ["a", "zzz"]) is None
    assert dig({"a": 1}, ["a", "b"]) is None


def test_the_verdict_is_the_worst_of_the_results():
    assert verdict([{"status": PASS}, {"status": PASS}]) == "GO"
    assert verdict([{"status": PASS}, {"status": SKIPPED}]) == "INCOMPLETE"
    assert verdict([{"status": SKIPPED}, {"status": FAIL}]) == "NO-GO"


def test_the_real_criteria_file_is_valid_and_complete():
    """Every claim in the README should have a criterion behind it."""
    criteria = load_criteria(__import__("pathlib").Path("configs/acceptance.yaml"))
    assert len(criteria) >= 10
    ids = {c["id"] for c in criteria}
    for expected in ("detector-map", "detector-recall", "false-alarms",
                     "privacy-helmet-evidence", "latency-p95", "memory"):
        assert expected in ids
    for criterion in criteria:
        assert criterion["operator"] in {">=", ">", "<=", "<", "=="}
        assert criterion.get("why"), f"{criterion['id']} has no stated reason"
        assert isinstance(criterion.get("path"), list) and criterion["path"]
