# SPDX-License-Identifier: Apache-2.0
"""Size Step5 HT NVL storage from the actual DeepEP dispatch/combine configs."""
import os

def choose_nvl_bytes(default: int, dispatch_hint: int, combine_hint: int) -> int:
    if min(default, dispatch_hint, combine_hint) <= 0:
        raise ValueError("DeepEP buffer capacities must be positive")
    alignment = 64 * 1024 * 1024
    needed = max(dispatch_hint, combine_hint)
    rounded = (needed + alignment - 1) // alignment * alignment
    # Preserve the existing size when no reduction is possible.
    return min(default, rounded) if needed <= default else default

def step5_ht_nvl_buffer_bytes(manager, default: int) -> int:
    if os.getenv("VLLM_STEP5_HT_RIGHTSIZE_BUFFER", "0") != "1":
        return default
    from vllm.config import get_current_vllm_config_or_none
    config = get_current_vllm_config_or_none()
    if config is None:
        return default
    parallel = config.parallel_config
    architectures = getattr(config.model_config, "architectures", ()) or ()
    if not (any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP")
                for a in architectures)
            and parallel.pipeline_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and not parallel.enable_dbo and manager.world_size == 8):
        return default
    import torch,deep_ep
    if config.model_config.dtype != torch.bfloat16:
        return default
    hidden = int(config.model_config.hf_text_config.hidden_size)
    # BF16 is the largest dispatched element type in this Step5 configuration.
    hidden_bytes = hidden * 2
    dispatch = deep_ep.Buffer.get_dispatch_config(manager.world_size)
    combine = deep_ep.Buffer.get_combine_config(manager.world_size)
    dispatch_hint = dispatch.get_nvl_buffer_size_hint(hidden_bytes, manager.world_size)
    combine_hint = combine.get_nvl_buffer_size_hint(hidden_bytes, manager.world_size)
    size = choose_nvl_bytes(default, dispatch_hint, combine_hint)
    manager._step5_ht_buffer_budget = dict(
        default=default, selected=size, dispatch_hint=dispatch_hint,
        combine_hint=combine_hint, hidden_bytes=hidden_bytes,
        ep_size=manager.world_size, actual_buffer_sms=deep_ep.Buffer.num_sms)
    from vllm.logger import init_logger
    init_logger("vllm.hcu.step5_ht_buffer").info(
        "Step5 HT NVL buffer: default=%d; selected=%d; dispatch hint=%d; "
        "combine hint=%d; hidden bytes=%d; EP=%d; actual buffer SMs=%d",
        default, size, dispatch_hint, combine_hint, hidden_bytes,
        manager.world_size, deep_ep.Buffer.num_sms)
    return size
