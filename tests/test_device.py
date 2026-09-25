"""
Tests for shared/device.py - the GPU guardrail.

A fake torch module stands in for the real one, so every branch can be tested on any
machine, with or without a GPU.
"""
from shared.device import MIN_FREE_MB, resolve_device


class FakeCuda:
    def __init__(self, available=True, free_mb=3000, total_mb=4096, name="Fake GPU", raises=None):
        self._available = available
        self._free = free_mb * 1024 * 1024
        self._total = total_mb * 1024 * 1024
        self._name = name
        self._raises = raises

    def is_available(self):
        if self._raises:
            raise self._raises
        return self._available

    def mem_get_info(self):
        return self._free, self._total

    def get_device_name(self, index=0):
        return self._name


class FakeTorch:
    def __init__(self, **kw):
        self.cuda = FakeCuda(**kw)


def test_auto_uses_the_gpu_when_there_is_room():
    device, reason = resolve_device("auto", torch_module=FakeTorch(free_mb=3000))
    assert device == "0"
    assert "3000 MB free" in reason


def test_auto_falls_back_to_cpu_when_vram_is_tight():
    """A 4 GB laptop GPU with a game or browser open must not be grabbed."""
    device, reason = resolve_device("auto", torch_module=FakeTorch(free_mb=400))
    assert device == "cpu"
    assert "only 400 MB free" in reason and "using the CPU" in reason


def test_the_threshold_is_the_boundary():
    assert resolve_device("auto", torch_module=FakeTorch(free_mb=MIN_FREE_MB))[0] == "0"
    assert resolve_device("auto", torch_module=FakeTorch(free_mb=MIN_FREE_MB - 1))[0] == "cpu"


def test_auto_uses_cpu_when_there_is_no_gpu():
    device, reason = resolve_device("auto", torch_module=FakeTorch(available=False))
    assert device == "cpu" and "no CUDA GPU" in reason


def test_a_driver_error_never_crashes_the_app():
    device, reason = resolve_device("auto", torch_module=FakeTorch(raises=RuntimeError("driver")))
    assert device == "cpu" and "GPU check failed" in reason


def test_an_explicit_choice_is_always_obeyed():
    assert resolve_device("cpu", torch_module=FakeTorch(free_mb=8000)) == ("cpu", "CPU requested")
    device, reason = resolve_device("0", torch_module=FakeTorch(free_mb=100))
    assert device == "0" and "requested" in reason         # user overrides the guardrail


def test_empty_and_none_mean_auto():
    for value in ("", None, "AUTO", " auto "):
        assert resolve_device(value, torch_module=FakeTorch(free_mb=3000))[0] == "0"


def test_a_custom_threshold_is_respected():
    assert resolve_device("auto", min_free_mb=2000, torch_module=FakeTorch(free_mb=1800))[0] == "cpu"
    assert resolve_device("auto", min_free_mb=1000, torch_module=FakeTorch(free_mb=1800))[0] == "0"
