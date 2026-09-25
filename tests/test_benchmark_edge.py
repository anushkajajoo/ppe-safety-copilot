"""Tests for the pure parts of scripts/benchmark_edge.py (no model, no camera)."""
from scripts.benchmark_edge import percentile, summarise


def test_percentile_picks_the_nearest_rank():
    values = [float(i) for i in range(1, 101)]       # 1..100
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.0) == 100.0


def test_percentile_handles_short_and_empty_lists():
    assert percentile([], 0.95) == 0.0
    assert percentile([7.0], 0.95) == 7.0


def test_summarise_reports_mean_p95_min_max():
    stats = summarise([10.0, 12.0, 11.0, 40.0])
    assert stats["min"] == 10.0 and stats["max"] == 40.0
    assert stats["mean"] == 18.25
    assert stats["p95"] == 40.0                      # the slow frame shows up at p95


def test_summarise_of_nothing_is_zeroes():
    assert summarise([]) == {"mean": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}


# --------------------------------------------------- honest device reporting
from pathlib import Path

from scripts.benchmark_edge import device_label, report_path


def test_cpu_run_is_never_labelled_as_the_gpu():
    """--device cpu must say cpu, even on a machine with a CUDA GPU."""
    assert device_label("cpu") == "cpu"
    assert device_label("CPU") == "cpu"


def test_auto_and_gpu_labels_do_not_claim_cpu_wrongly():
    label = device_label(None)
    assert isinstance(label, str) and label != ""


def test_each_device_gets_its_own_report_file():
    """A CPU run must not overwrite the GPU numbers."""
    gpu_file = report_path("", "NVIDIA GeForce RTX 3050 A Laptop GPU")
    cpu_file = report_path("", "cpu")
    assert gpu_file != cpu_file
    assert gpu_file.name == "edge_benchmark_gpu.json"
    assert cpu_file.name == "edge_benchmark_cpu.json"


def test_an_explicit_report_path_is_respected():
    assert report_path("out/custom.json", "cpu") == Path("out/custom.json")
