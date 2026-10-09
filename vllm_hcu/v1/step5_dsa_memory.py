# SPDX-License-Identifier: Apache-2.0
"""Budget Step5's physical-block-indexed DSA side pools before KV allocation."""
from __future__ import annotations
import os

def plan_dsa_cache_budget(available: int, pool_per_block: int,
                          summary_per_block: int, fixed: int) -> tuple[int, int, int]:
    if min(available, summary_per_block, fixed) < 0 or pool_per_block <= 0:
        raise ValueError("Invalid Step5 DSA memory budget")
    blocks = max(0, available - fixed) // (pool_per_block + summary_per_block)
    cache = blocks * pool_per_block
    reserve = blocks * summary_per_block + fixed
    return cache, reserve, blocks

def reserve_step5_dsa_memory(worker, available: int) -> int:
    if os.getenv("VLLM_STEP5_DSA_MEMORY_BUDGET", "0") != "1":
        return available
    config = worker.vllm_config
    from vllm_hcu.step5_dcp2_config import step5_context_parallel_supported
    parallel = config.parallel_config
    architectures = getattr(config.model_config, "architectures", ()) or ()
    if not (any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
                for a in architectures)
            and parallel.pipeline_parallel_size == 1
            and step5_context_parallel_supported(config)
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1):
        return available
    import torch
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_groups, _pool_bytes_per_block,
    )
    specs = worker.get_kv_cache_spec()
    indexers = []
    seen_indexers = set()
    for module in worker.model_runner.model.modules():
        indexer = getattr(module, "indexer", None)
        if (indexer is not None and id(indexer) not in seen_indexers
                and indexer.layer_name in specs):
            indexers.append(indexer)
            seen_indexers.add(id(indexer))
    if not indexers:
        return available
    groups = get_kv_cache_groups(config, dict(specs))
    pool_per_block = _pool_bytes_per_block(groups)
    states = {}
    for indexer in indexers:
        states.setdefault(id(indexer.state), []).append(indexer)
    summary_per_block = 0
    fixed = 0
    details = []
    for members in states.values():
        state = members[0].state
        geometry = state.geometry
        block_sizes = {specs[i.layer_name].block_size for i in members}
        if len(block_sizes) != 1:
            raise ValueError("A Step5 DSA state requires a uniform block size")
        # DCP2 summaries are replicated for a global logical page128 even
        # though each physical BF16 page stores only 64 striped tokens.
        state_cp = getattr(state, "context_parallel_size", 1)
        if state_cp != parallel.decode_context_parallel_size:
            raise ValueError("Step5 DSA state and KV budget context-parallel sizes differ")
        block_size = block_sizes.pop() * state_cp
        if block_size % geometry.region_size:
            raise ValueError("A Step5 DSA region must fit inside its KV page")
        dtype_size = torch.empty((), dtype=state.dtype).element_size()
        # Match ensure_allocated: one FP8 proxy vector per region, per layer.
        summary = (state.num_layers * (block_size // geometry.region_size)
                   * geometry.proxy_dim)
        pending = (2 * state.num_layers * (state.max_pending + 1)
                   * geometry.region_size * geometry.proxy_dim * dtype_size)
        score = 0
        graph_vectors = 0
        if state.graph_decode:
            regions = (state.max_model_len + geometry.region_size - 1) // geometry.region_size
            score = state.num_kv_groups * state.max_num_reqs * regions * 4
            graph_vectors = state.max_num_reqs * (5 * 8 + 3 * 4 + 1)
        # Cached RoPE is shared only for identical indexer geometry.
        # Reserve construction peak: cos/sin are FP32 before conversion.
        rope_keys = {(float(i.rope_theta), repr(i.rope_scaling), i.max_position)
                     for i in members}
        rope = sum(max_position * (geometry.rope_span * (4 + dtype_size) + 4)
                   for _, _, max_position in rope_keys)
        summary_per_block += summary
        fixed += pending + score + graph_vectors + rope
        details.append(dict(layers=state.num_layers, summary_per_block=summary,
                            pending=pending, score=score, graph_vectors=graph_vectors,
                            rope_construction_peak=rope))
    swa_workspace = 0
    if os.getenv("VLLM_STEP5_SWA_PREFILL_CROP", "0") == "1":
        from vllm.v1.kv_cache_interface import SlidingWindowSpec
        scheduler = config.scheduler_config
        for spec in specs.values():
            if isinstance(spec, SlidingWindowSpec):
                # The native library keeps a262K-token allocation floor.
                # Cropping caps the sum of real retained tokens across requests.
                total_tokens = max(
                    262144,
                    int(scheduler.max_num_batched_tokens)
                    + int(scheduler.max_num_seqs)
                    * (spec.sliding_window - 1 + spec.block_size - 1),
                )
                dtype_size = torch.empty((), dtype=spec.dtype).element_size()
                value_width = spec.head_size_v or spec.head_size
                bytes_ = total_tokens * spec.num_kv_heads * (
                    spec.head_size + value_width) * dtype_size
                swa_workspace = max(swa_workspace, bytes_)
        fixed += swa_workspace
    cache, reserve, blocks = plan_dsa_cache_budget(
        available, pool_per_block, summary_per_block, fixed,
    )
    from vllm.logger import init_logger
    init_logger("vllm.hcu.step5_dsa_memory").info(
        "Step5 DSA memory budget: profiled=%d; KV=%d; side-pool reserve=%d; "
        "blocks=%d; KV bytes/block=%d; summary bytes/block=%d; fixed=%d; states=%s",
        available, cache, reserve, blocks, pool_per_block, summary_per_block, fixed, details,
    )
    worker._step5_dsa_memory_budget = dict(
        available=available, cache=cache, reserve=reserve, blocks=blocks,
        pool_per_block=pool_per_block, summary_per_block=summary_per_block,
        fixed=fixed, swa_workspace=swa_workspace, states=details,
    )
    worker.available_kv_cache_memory_bytes = cache
    return cache
