# SPDX-License-Identifier: Apache-2.0
"""Explicit LL capture for Step5 auto; HT remains outside CUDA graphs."""
from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from contextvars import ContextVar

_CAPTURE_POLICY = ContextVar("step5_auto_decode_capture_policy", default=None)


def step5_auto_decode_graph_enabled(config, *, allow_draft=True):
    if (os.getenv("VLLM_STEP5_AUTO_DECODE_GRAPH", "0") != "1"
            or os.getenv("VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS", "0") != "1"
            or os.getenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE", "0") != "1"
            or config is None):
        return False
    from vllm_hcu.patch.config import get_hcu_config
    if not get_hcu_config(config).deepep_auto:
        return False
    model = config.model_config
    architectures = getattr(model, "architectures", None) or getattr(
        getattr(model, "hf_config", None), "architectures", ()) or ()
    accepted = {"Step5ForCausalLM", "Step5ForConditionalGeneration"}
    if allow_draft:
        accepted.add("Step5MTP")
    p = config.parallel_config
    return bool(
        accepted.intersection(architectures)
        and (p.tensor_parallel_size, p.data_parallel_size) == (4, 2)
        and p.pipeline_parallel_size == 1
        and getattr(p, "decode_context_parallel_size", 1) == 1
        and getattr(p, "prefill_context_parallel_size", 1) == 1
        and getattr(p, "enable_expert_parallel", False)
        and not getattr(p, "enable_dbo", False)
        and not getattr(model, "enforce_eager", False)
        and getattr(config.compilation_config.cudagraph_mode, "name", "")
        == "FULL_DECODE_ONLY"
    )


def validate_decode_capture_descriptors(manager):
    if getattr(manager.cudagraph_mode, "name", "") != "FULL_DECODE_ONLY":
        raise RuntimeError("Step5 auto can capture only FULL_DECODE_ONLY graphs")
    query = int(manager.decode_query_len)
    if query not in (1, 4) or manager.varlen_decode:
        raise RuntimeError("Step5 auto capture requires fixed MTP3/decode queries")
    maximum = int(manager.vllm_config.scheduler_config.max_num_seqs) * query
    for mode, descriptors in manager._capture_descs.items():
        if getattr(mode, "name", "") != "FULL":
            raise RuntimeError("Step5 auto must never capture a mixed/HT graph")
        for desc in descriptors:
            if (desc.num_reqs is None or desc.num_reqs <= 0
                    or desc.uniform_token_count != query
                    or desc.max_query_len not in (None, query)
                    or desc.num_tokens != desc.num_reqs * query
                    or not 0 < desc.num_tokens <= maximum):
                raise RuntimeError(f"Invalid Step5 auto Decode descriptor: {desc}")
    return maximum


@contextmanager
def step5_auto_decode_capture_scope(manager):
    maximum = validate_decode_capture_descriptors(manager)
    token = _CAPTURE_POLICY.set(maximum)
    try:
        yield
    finally:
        _CAPTURE_POLICY.reset(token)


def step5_auto_capture_uses_ll(config, num_tokens):
    maximum = _CAPTURE_POLICY.get()
    if maximum is None:
        return None
    if not step5_auto_decode_graph_enabled(config):
        raise RuntimeError("Step5 auto Decode capture entered an unsupported config")
    if num_tokens is None or not 0 < int(num_tokens) <= maximum:
        raise RuntimeError("Step5 auto Decode capture exceeded its explicit token bound")
    # Every captured descriptor was verified before warmup. No host collective
    # belongs inside capture; runtime FULL dispatch already uses DP-wide voting.
    return True


def install_step5_auto_capture():
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    original = CudaGraphManager.capture
    if getattr(original, "_step5_auto_decode_capture", False):
        return

    @functools.wraps(original)
    def capture(self, *args, **kwargs):
        if not step5_auto_decode_graph_enabled(self.vllm_config):
            return original(self, *args, **kwargs)
        with step5_auto_decode_capture_scope(self):
            return original(self, *args, **kwargs)

    capture._step5_auto_decode_capture = True
    CudaGraphManager.capture = capture


def step5_auto_workspace_bound(config):
    """Bound HT assignment padding and static LL scratch before graph locking."""
    if not step5_auto_decode_graph_enabled(config, allow_draft=False):
        return None
    model = config.model_config
    text = getattr(model, "hf_text_config", None)
    if text is None:
        hf = model.hf_config
        text = getattr(hf, "text_config", hf)
    hidden = int(text.hidden_size)
    intermediate = int(text.moe_intermediate_size)
    experts = int(text.moe_num_experts)
    topk = int(text.moe_top_k)
    budget = int(config.scheduler_config.max_num_batched_tokens)
    sequences = int(config.scheduler_config.max_num_seqs)
    spec = getattr(config, "speculative_config", None)
    speculative = int(getattr(spec, "num_speculative_tokens", 0))
    p = config.parallel_config
    if (min(hidden, intermediate, experts, topk, budget, sequences) <= 0
            or topk > experts or speculative != 3
            or getattr(p, "enable_eplb", False)):
        raise RuntimeError("Unsupported Step5 auto workspace geometry")
    tp, dp = p.tensor_parallel_size, p.data_parallel_size
    dispatchers = tp * dp
    local_experts = (experts + dispatchers - 1) // dispatchers
    sender_rows = (budget + tp - 1) // tp
    received_rows = sender_rows * dispatchers
    assignments = received_rows * topk
    # Each local expert adds at most ALIGNMENT-1 padding rows. This covers
    # every routing distribution, including all assignments landing here.
    alignment = 256
    aligned_rows = ((assignments + local_experts * (alignment - 1)
                     + alignment - 1) // alignment) * alignment
    ht_bytes = 2 * aligned_rows * (
        max(hidden, intermediate) + max(hidden, 2 * intermediate))
    ll_rows = local_experts * sequences * (1 + speculative) * dispatchers
    ll_bytes = 2 * ll_rows * (max(hidden, intermediate) + 2 * intermediate)
    # BF16 HT uses bounded received-token chunks and top-k scratch.
    bf16_rows = min(received_rows, 4096) * topk
    bf16_bytes = 2 * bf16_rows * (max(hidden, 2 * intermediate) + intermediate)
    bound = (max(ht_bytes, ll_bytes, bf16_bytes) + 255) // 256 * 256
    return dict(bytes=bound, ht_bytes=ht_bytes, ll_bytes=ll_bytes,
                bf16_ht_bytes=bf16_bytes, sender_rows=sender_rows,
                received_rows=received_rows, assignments=assignments,
                aligned_rows=aligned_rows, local_experts=local_experts,
                dispatchers=dispatchers, hidden=hidden, intermediate=intermediate)


def reserve_step5_auto_workspace(config, device):
    bound = step5_auto_workspace_bound(config)
    if bound is None:
        return None
    import torch
    from vllm.v1.worker.workspace import current_workspace_manager
    from vllm.logger import init_logger
    manager = current_workspace_manager()
    # Called inside memory profiling, before CUDA Graph capture. The manager
    # retains this storage; later HT/LL calls take views without moving it.
    manager.get_simultaneous(((bound["bytes"],), torch.uint8))
    init_logger("vllm.hcu.step5_auto_graph").info_once(
        "Step5 auto reserved %.2f GiB scratch before graph locking; "
        "HT bound=%d routed assignments, %d aligned rows.",
        bound["bytes"] / 2**30, bound["assignments"], bound["aligned_rows"])
    return bound
