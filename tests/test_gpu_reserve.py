import io
import sys
import types

import pytest

from octantvector.generation import config, embed
from octantvector.generation.errors import GenerationError

GB = 2**30


def fake_torch(monkeypatch, total_gb=16, initialized=False):
    calls = []
    cuda = types.SimpleNamespace(
        current_device=lambda: 0,
        is_initialized=lambda: initialized,
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=total_gb * GB),
        set_per_process_memory_fraction=lambda f, d: calls.append((f, d)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


def test_reserve_caps_torch_at_total_minus_reserve(monkeypatch):
    calls = fake_torch(monkeypatch)
    log = io.StringIO()
    embed._reserve_gpu(4, log)
    assert calls == [(12 / 16, 0)]
    assert "at most 12.0 of 16.0 GB" in log.getvalue()


def test_zero_reserve_leaves_torch_uncapped(monkeypatch):
    calls = fake_torch(monkeypatch)
    embed._reserve_gpu(0, io.StringIO())
    assert calls == []  # CUDA not started yet: nothing to undo, and CUDA is not started for it


def test_zero_reserve_lifts_an_earlier_cap_in_the_same_process(monkeypatch):
    calls = fake_torch(monkeypatch, initialized=True)
    embed._reserve_gpu(8, io.StringIO())
    embed._reserve_gpu(0, io.StringIO())
    assert calls == [(0.5, 0), (1.0, 0)]


def test_reserve_as_large_as_the_gpu_is_an_error(monkeypatch):
    fake_torch(monkeypatch, total_gb=4)
    with pytest.raises(GenerationError, match="gpu_reserve_gb"):
        embed._reserve_gpu(4, io.StringIO())


def test_setting_defaults_to_4_and_reads_env(monkeypatch):
    monkeypatch.delenv("BV_GPU_RESERVE_GB", raising=False)
    with config.given({}):
        assert config.load().gpu_reserve_gb == 4
    monkeypatch.setenv("BV_CONFIG", "/nonexistent.toml")
    monkeypatch.setenv("BV_GPU_RESERVE_GB", "2.5")
    assert config.load().gpu_reserve_gb == 2.5
