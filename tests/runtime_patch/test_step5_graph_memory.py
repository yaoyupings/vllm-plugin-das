"""The exact profile must retain native startup isolation and fallback."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def runtime():
    path = Path(__file__).parents[2] / "vllm_hcu/v1/step5_graph_memory.py"
    spec = importlib.util.spec_from_file_location("step5_graph_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_EXACT_GRAPH_MEMORY", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=["Step5ForCausalLM"]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1,
            decode_context_parallel_size=1, prefill_context_parallel_size=1),
    )


@pytest.fixture
def native(monkeypatch):
    class Manager:
        def __init__(self, limit=2, samples=None):
            self._max_full_descs_to_capture = limit
            self._capture_mem_samples = samples
            self.seen = []
        def capture(self, factory, progress_bar_desc="Capturing"):
            descs = list(range(6))
            limit = self._max_full_descs_to_capture
            if limit is not None:
                descs = descs[:limit]
            for d in descs:
                factory(d)
                self.seen.append(d)
                if self._capture_mem_samples is not None:
                    self._capture_mem_samples.append([100, 60, 3, 4, 2, 1][d])
            return progress_bar_desc
    def extrapolate(samples, count):
        if not samples:
            return 0
        return samples[0] + (count - 1) * max(samples[1], 1)
    module = SimpleNamespace(CudaGraphManager=Manager,
        _extrapolate_full_graph_memory=extrapolate)
    for name in ["vllm", "vllm.v1", "vllm.v1.worker",
                 "vllm.v1.worker.gpu", "vllm.logger"]:
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules["vllm.v1.worker.gpu"].cudagraph_utils = module
    sys.modules["vllm.logger"].init_logger = lambda name: SimpleNamespace(info=lambda *a: None)
    return module


@pytest.mark.parametrize("field,value", [
    ("flag", "0"), ("runner", "0"), ("arch", "Other"),
    ("pipeline_parallel_size", 2), ("decode_context_parallel_size", 2),
    ("prefill_context_parallel_size", 2),
])
def test_guard(runtime, config, monkeypatch, field, value):
    if field == "flag":
        monkeypatch.setenv("VLLM_STEP5_EXACT_GRAPH_MEMORY", value)
    elif field == "runner":
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
    elif field == "arch":
        config.model_config.architectures = [value]
    else:
        setattr(config.parallel_config, field, value)
    assert not runtime.enabled(config)


def test_exact_profile_captures_all_and_cancels_extrapolation(runtime, config, native):
    original = native.CudaGraphManager.capture
    samples = []
    manager = native.CudaGraphManager(samples=samples)
    with runtime.exact_graph_memory(config):
        assert manager.capture(lambda d: None, "profile") == "profile"
        assert manager.seen == list(range(6))
        assert native._extrapolate_full_graph_memory(samples, 6) == sum(samples)
        measured = 500  # Shared pool also includes PIECEWISE and MTP.
        assert measured - sum(samples) + native._extrapolate_full_graph_memory(samples, 6) == measured
    assert manager._max_full_descs_to_capture == 2
    assert not runtime._exact_profile.get()
    # Installing twice keeps one wrapper and preserves native fallback.
    wrapped = native.CudaGraphManager.capture
    runtime.install(native)
    assert native.CudaGraphManager.capture is wrapped
    assert wrapped.__wrapped__ is original
    manager = native.CudaGraphManager(samples=[])
    manager.capture(lambda d: None)
    assert manager.seen == [0, 1]
    assert native._extrapolate_full_graph_memory([100, 60], 6) == 400


def test_real_capture_and_partial_samples_keep_native_behavior(runtime, config, native):
    with runtime.exact_graph_memory(config):
        manager = native.CudaGraphManager(limit=None, samples=None)
        manager.capture(lambda d: None)
        assert manager.seen == list(range(6))
        assert manager._capture_mem_samples is None
        # If native capture filters descriptors, never lower an incomplete estimate.
        assert native._extrapolate_full_graph_memory([100, 60], 6) == 400
        assert native._extrapolate_full_graph_memory([], 6) == 0


def test_exception_restores_context_and_capture_limit(runtime, config, native):
    manager = native.CudaGraphManager(samples=[])
    with pytest.raises(RuntimeError):
        with runtime.exact_graph_memory(config):
            manager.capture(lambda d: (_ for _ in ()).throw(RuntimeError("capture failed")))
    assert manager._max_full_descs_to_capture == 2
    assert not runtime._exact_profile.get()
    fallback = native.CudaGraphManager(samples=[])
    fallback.capture(lambda d: None)
    assert fallback.seen == [0, 1]


def test_other_thread_retains_two_sample_profile(runtime, config, native):
    def capture():
        manager = native.CudaGraphManager(samples=[])
        manager.capture(lambda d: None)
        return manager.seen
    with runtime.exact_graph_memory(config), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(capture).result() == [0, 1]
        assert capture() == list(range(6))


def test_disabled_context_does_not_install(runtime, config, native, monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_EXACT_GRAPH_MEMORY", "0")
    with runtime.exact_graph_memory(config):
        assert not runtime._exact_profile.get()
    assert not hasattr(native, "_step5_exact_graph_memory_installed")


def test_signed_full_deltas_preserve_whole_pool_measurement(runtime, config, native):
    # Live ROCm capture can free a large eager workspace between descriptors.
    # Signed per-FULL deltas must cancel without lowering the measured total.
    samples = [-1113587712, -1130364928, -1113587712, -1109393408, -1109393408]
    measured = 2952790016
    with runtime.exact_graph_memory(config):
        estimate = native._extrapolate_full_graph_memory(samples, len(samples))
        assert measured - sum(samples) + estimate == measured


def test_margin_adds_headroom_without_changing_native_pool_accounting(runtime, config, native, monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES", str(512*1024*1024))
    samples = [-10, -20, -30]
    measured = 2952790016
    with runtime.exact_graph_memory(config):
        assert measured - sum(samples) + native._extrapolate_full_graph_memory(samples, 3) == measured + 512*1024*1024
        assert native._extrapolate_full_graph_memory([100, 60], 6) == 400 + 512*1024*1024
    assert native._extrapolate_full_graph_memory([100, 60], 6) == 400


@pytest.mark.parametrize("raw", ["-1", "512MB", "nan"])
def test_margin_rejects_invalid_bytes_before_profiling(runtime, config, native, monkeypatch, raw):
    monkeypatch.setenv("VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES", raw)
    with pytest.raises(ValueError):
        with runtime.exact_graph_memory(config):
            pytest.fail("Invalid margin must fail before profiling")
    assert not runtime._exact_profile.get()
    assert not hasattr(native, "_step5_exact_graph_memory_installed")
