# SPDX-License-Identifier: Apache-2.0
"""Bound Step5 LL input without changing the scheduler or full-batch gate."""
from __future__ import annotations

import os


def configured_chunk_tokens(moe_config, vllm_config) -> int:
    raw = os.getenv("VLLM_STEP5_LL_MOE_CHUNK_TOKENS", "0")
    if raw == "0":
        return 0
    if raw not in ("128", "256", "512"):
        raise ValueError("Step5 LL MoE chunk tokens must be 0, 128, 256 or 512")
    cfg = vllm_config or getattr(moe_config, "_hcu_vllm_config", None)
    if cfg is None:
        return 0
    p = cfg.parallel_config
    architectures = getattr(cfg.model_config, "architectures", ()) or ()
    mp = moe_config.moe_parallel_config
    if not (
        any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
            for a in architectures)
        and (p.tensor_parallel_size, p.data_parallel_size) == (4, 2)
        and p.pipeline_parallel_size == 1
        and getattr(p, "decode_context_parallel_size", 1) == 1
        and getattr(p, "prefill_context_parallel_size", 1) == 1
        and p.all2all_backend == "deepep_low_latency"
        and mp.is_sequence_parallel and mp.sp_size == 4 and mp.ep_size == 8
        and os.getenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE", "0") == "1"
        and os.getenv("VLLM_STEP5_SP_LL_WORKSPACE", "0") == "1"
    ):
        return 0
    if getattr(p, "enable_dbo", False):
        raise ValueError("Step5 LL input chunks require DBO disabled")
    if (os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0") != "1"
        or os.getenv("VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP", "0") == "1"):
        raise ValueError("Step5 LL input chunks require shared-expert overlap disabled")
    return int(raw)


def uniform_chunk_count(local_rows: int, counts, sp_size: int, chunk: int) -> int:
    """All EP ranks use the existing CPU scheduler counts, including idle DP."""
    if chunk <= 0 or sp_size <= 0 or local_rows < 0:
        raise ValueError("Invalid Step5 LL chunk geometry")
    maximum = local_rows
    if counts is not None:
        if counts.device.type != "cpu":
            raise ValueError("Step5 LL chunks require CPU scheduler counts")
        values = counts.tolist()
        if any(int(n) < 0 for n in values):
            raise ValueError("Step5 LL scheduler counts cannot be negative")
        maximum = max(((int(n) + sp_size - 1) // sp_size for n in values), default=0)
        if local_rows > maximum:
            raise ValueError("Step5 LL input exceeds scheduler token counts")
    return max(1, (maximum + chunk - 1) // chunk)


def apply_input_chunks(apply, *, hidden_states, router_logits,
                       shared_experts_input, chunk: int, num_chunks: int):
    """Keep full gate logits; copy each output before reusable expert buffers turn."""
    import torch
    rows = hidden_states.shape[0]
    if rows > chunk * num_chunks:
        raise ValueError("Step5 LL chunk plan does not cover local input")
    if router_logits is None or router_logits.shape[0] != rows:
        raise ValueError("Step5 LL chunks require full-batch router logits")
    if shared_experts_input is not None and shared_experts_input.shape[0] != rows:
        raise ValueError("Step5 LL shared input must use the same local rows")
    fused_result = None
    shared_result = None
    has_shared = None
    for i in range(num_chunks):
        start = i * chunk
        live = max(0, min(chunk, rows - start))

        def padded(t):
            if t is None:
                return None
            out = t.new_zeros((chunk, *t.shape[1:]))
            if live:
                out[:live].copy_(t[start:start + live])
            return out

        shared, fused = apply(hidden_states=padded(hidden_states),
                              router_logits=padded(router_logits),
                              shared_experts_input=padded(shared_experts_input))
        if fused.shape[0] != chunk:
            raise ValueError("Step5 LL expert output has unexpected rows")
        if fused_result is None:
            fused_result = fused.new_empty((rows, *fused.shape[1:]))
            has_shared = shared is not None
            if has_shared:
                shared_result = shared.new_empty((rows, *shared.shape[1:]))
        if (shared is not None) != has_shared:
            raise ValueError("Step5 LL shared expert output changed across chunks")
        if live:
            fused_result[start:start + live].copy_(fused[:live])
            if shared is not None:
                shared_result[start:start + live].copy_(shared[:live])
    return shared_result, fused_result
