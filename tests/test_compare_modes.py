"""
Tests for scripts/compare_modes.py - the temporal-filter experiment.

These assert the BEHAVIOUR the report claims: a flickering detector produces false
alarms frame-by-frame but not through the temporal filter, and a real violation is
still caught, just later.
"""
from edge.compliance import COMPLIANT, UNCERTAIN, VIOLATION
from scripts import compare_modes


def blink(**kw):
    defaults = dict(name="blink", truth=["helmet", "vest"], miss_item="helmet",
                    miss_rate=0.15, frames=200, seed=42, window=15)
    defaults.update(kw)
    return compare_modes.run_scenario(**defaults)


def real_violation(**kw):
    defaults = dict(name="real", truth=["helmet"], miss_item="none",
                    miss_rate=0.0, frames=200, seed=42, window=15)
    defaults.update(kw)
    return compare_modes.run_scenario(**defaults)


def test_a_flickering_detector_makes_the_baseline_cry_wolf():
    result = blink()
    assert result["counts"]["baseline"][VIOLATION] > 0        # false alarms
    assert result["counts"]["proposed"][VIOLATION] == 0       # none survive the window
    assert result["first_violation_frame"]["proposed"] is None


def test_a_real_violation_is_still_caught_by_both_modes():
    result = real_violation()
    assert result["counts"]["baseline"][VIOLATION] > 0
    assert result["counts"]["proposed"][VIOLATION] > 0


def test_the_temporal_filter_costs_about_one_window_of_delay():
    result = real_violation(window=15)
    assert result["first_violation_frame"]["baseline"] == 1
    assert result["first_violation_frame"]["proposed"] == 15   # one full window of evidence


def test_a_shorter_window_confirms_sooner():
    assert real_violation(window=9)["first_violation_frame"]["proposed"] == 9


def test_before_the_window_fills_the_answer_is_uncertain_not_compliant():
    """Never claim 'compliant' on thin evidence - that is the dangerous direction."""
    result = real_violation(frames=20, window=15)
    assert result["counts"]["proposed"][UNCERTAIN] == 14       # frames 1..14
    assert result["counts"]["proposed"][COMPLIANT] == 0


def test_the_experiment_is_reproducible():
    assert blink(seed=7) == blink(seed=7)
    assert blink(seed=7) != blink(seed=8)


def test_a_worker_wearing_everything_is_never_flagged():
    result = compare_modes.run_scenario(name="clean", truth=["helmet", "vest"],
                                        miss_item="none", miss_rate=0.0,
                                        frames=50, seed=1, window=15)
    assert result["counts"]["baseline"][VIOLATION] == 0
    assert result["counts"]["proposed"][VIOLATION] == 0
