# SPDX-License-Identifier: Apache-2.0
"""Measure all Step5 FULL graphs instead of extrapolating two samples."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import threading

_exact_profile = ContextVar("step5_exact_graph_memory_profile", default=False)
_install_lock = threading.RLock()


def enabled(config) -> bool:
    if os.getenv("VLLM_STEP5_EXACT_GRAPH_MEMORY", "0") != "1":
        return False
    from vllm_hcu.step5_dcp2_config import step5_context_parallel_supported
    parallel = config.parallel_config
    architectures = getattr(config.model_config, "architectures", ()) or ()
    return (
        any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
            for a in architectures)
        and parallel.pipeline_parallel_size == 1
        and step5_context_parallel_supported(config)
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0") == "1"
    )


def memory_margin_bytes() -> int:
    raw = os.getenv("VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("Step5 Graph memory margin must be integer bytes") from exc
    if value < 0:
        raise ValueError("Step5 Graph memory margin must be nonnegative")
    return value


def install(module) -> None:
    # The native profiler calls these globals after constructing a temporary
    # manager/pool. ContextVar scopes the change to this startup call; neither
    # its cleanup nor the real capture/replay lifecycle is replaced.
    with _install_lock:
        if getattr(module, "_step5_exact_graph_memory_installed", False):
            return
        capture = module.CudaGraphManager.capture
        extrapolate = module._extrapolate_full_graph_memory

        @wraps(capture)
        def capture_all(self, *args, **kwargs):
            limit = self._max_full_descs_to_capture
            samples = self._capture_mem_samples
            if not (_exact_profile.get() and limit is not None
                    and samples is not None):
                return capture(self, *args, **kwargs)
            self._max_full_descs_to_capture = None
            try:
                return capture(self, *args, **kwargs)
            finally:
                self._max_full_descs_to_capture = limit

        @wraps(extrapolate)
        def measured_full(samples, total_graphs):
            if _exact_profile.get() and len(samples) == total_graphs:
                from vllm.logger import init_logger
                margin = memory_margin_bytes()
                init_logger("vllm.hcu.step5_graph_memory").info(
                    "Step5 all-FULL Graph profile: samples=%d; "
                    "signed deltas=%s; sum=%d bytes; extra headroom=%d bytes",
                    total_graphs, samples, sum(samples), margin,
                )
                # All FULL descriptors were captured. Replacing the sampled
                # contribution with this sum leaves the native measured delta
                # (including PIECEWISE and MTP) intact, without extrapolation.
                return sum(samples) + margin
            result = extrapolate(samples, total_graphs)
            return result + memory_margin_bytes() if _exact_profile.get() else result

        module.CudaGraphManager.capture = capture_all
        module._extrapolate_full_graph_memory = measured_full
        module._step5_exact_graph_memory_installed = True


@contextmanager
def exact_graph_memory(config):
    if not enabled(config):
        yield
        return
    from vllm.v1.worker.gpu import cudagraph_utils
    # Reject invalid margins before any model profiling/capture.
    memory_margin_bytes()
    install(cudagraph_utils)
    token = _exact_profile.set(True)
    try:
        yield
    finally:
        _exact_profile.reset(token)
