# SPDX-License-Identifier: Apache-2.0
# Bound Step5 low-latency dispatch storage to the sequence-parallel input.
from __future__ import annotations
import copy
import os

def sequence_parallel_dispatch_bound(moe_config, vllm_config) -> int | None:
    if os.getenv("VLLM_STEP5_SP_LL_WORKSPACE", "0") != "1":
        return None
    if vllm_config is None:
        vllm_config = getattr(moe_config, "_hcu_vllm_config", None)
    if vllm_config is None:
        return None
    from vllm_hcu.step5_dcp2_config import step5_context_parallel_supported
    parallel = vllm_config.parallel_config
    hf = getattr(vllm_config.model_config, "hf_config", None)
    architectures = getattr(
        vllm_config.model_config, "architectures",
        getattr(hf, "architectures", ()),
    ) or ()
    moe_parallel = moe_config.moe_parallel_config
    if not (
        any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
            for a in architectures)
        and (parallel.tensor_parallel_size, parallel.data_parallel_size)
        in ((2, 4), (4, 2))
        and parallel.pipeline_parallel_size == 1
        and step5_context_parallel_supported(vllm_config)
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and parallel.all2all_backend == "deepep_low_latency"
        and os.getenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE", "0") == "1"
        and moe_parallel.is_sequence_parallel
        and moe_parallel.sp_size == parallel.tensor_parallel_size
        and moe_parallel.ep_size == 8
    ):
        return None
    # Account for target graph and speculative request padding before SP slicing.
    scheduler = vllm_config.scheduler_config
    captures = getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", ()) or ()
    spec = vllm_config.speculative_config
    speculative_tokens = getattr(spec, "num_speculative_tokens", 0) if spec else 0
    full_tokens = max(
        int(moe_config.max_num_tokens),
        int(scheduler.max_num_batched_tokens),
        max(captures, default=0),
        int(scheduler.max_num_seqs) * (1 + int(speculative_tokens)),
    )
    if full_tokens <= 0:
        raise ValueError("Step5 dispatch requires a positive token capacity")
    local_tokens = (full_tokens + moe_parallel.sp_size - 1) // moe_parallel.sp_size
    if os.getenv("VLLM_STEP5_LL_MOE_CHUNK_TOKENS", "0") != "0":
        from vllm_hcu.model_executor.layers.fused_moe.step5_ll_chunk import configured_chunk_tokens
        chunk = configured_chunk_tokens(moe_config, vllm_config)
        if chunk:
            local_tokens = min(local_tokens, chunk)
    # Keep the original alignment unless the qualified gfx938 candidate is
    # explicitly selected. HCU DeepEP supports 192-row dispatch: its receive
    # storage need not be padded to 256 merely for the old 128-row bound.
    raw_alignment = os.getenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT", "128")
    if raw_alignment not in ("64", "128"):
        raise ValueError("Step5 LL dispatch alignment must be 64 or 128")
    alignment = int(raw_alignment)
    if alignment == 64:
        import torch
        arch = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":")[0]
        if arch != "gfx938":
            raise ValueError("Step5 64-token LL dispatch alignment requires gfx938")
    return ((local_tokens + alignment - 1) // alignment) * alignment

def bound_sequence_parallel_workspace(moe_config, vllm_config):
    bound = sequence_parallel_dispatch_bound(moe_config, vllm_config)
    if bound is None:
        return moe_config
    # Keep the runner/quantization layer's config intact. Use the same private
    # bound for native DeepEP handle, prepare/finalize and expert construction.
    bounded = copy.copy(moe_config)
    bounded.max_num_tokens = bound
    from vllm.logger import init_logger
    init_logger("vllm.hcu.step5_sp_workspace").info_once(
        "Step5 SP LL workspace: full token bound=%s; per-dispatcher bound=%s",
        moe_config.max_num_tokens, bound,
    )
    return bounded

def step5_auto_ll_dispatch_bound(moe_config, vllm_config) -> int | None:
    """Bound only Auto LL by the decoded input after sequence-parallel slicing."""
    if os.getenv("VLLM_STEP5_AUTO_SP_LL_WORKSPACE", "0") != "1":
        return None
    from vllm_hcu.step5_auto_graph import step5_auto_decode_graph_enabled
    if not step5_auto_decode_graph_enabled(vllm_config, allow_draft=False):
        return None
    mp = moe_config.moe_parallel_config
    if not (mp.is_sequence_parallel and mp.sp_size == 4 and mp.ep_size == 8):
        return None
    speculative = getattr(vllm_config, "speculative_config", None)
    if int(getattr(speculative, "num_speculative_tokens", 0)) != 3:
        raise ValueError("Step5 Auto SP LL capacity requires MTP3")
    sequences = int(vllm_config.scheduler_config.max_num_seqs)
    if sequences <= 0:
        raise ValueError("Step5 Auto SP LL capacity requires positive max_num_seqs")
    # FULL Decode descriptors use <= max_num_seqs queries, at most four
    # tokens per request. A mixed/Prefill batch is globally dispatched via HT.
    # The same private capacity is passed to the handle, LL prepare/finalize,
    # and then the experts factory; the HT token budget is never narrowed.
    full_tokens = sequences * 4
    local_tokens = (full_tokens + mp.sp_size - 1) // mp.sp_size
    alignment = 64
    bound = ((local_tokens + alignment - 1) // alignment) * alignment
    from vllm.logger import init_logger
    init_logger("vllm.hcu.step5_sp_workspace").info_once(
        "Step5 Auto SP LL workspace: full Decode bound=%d; "
        "per-dispatcher bound=%d; HT budget unchanged",
        full_tokens, bound)
    return bound
