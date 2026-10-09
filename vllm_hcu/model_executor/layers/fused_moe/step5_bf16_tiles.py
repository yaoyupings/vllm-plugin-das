# SPDX-License-Identifier: Apache-2.0
"""Measured BF16 tiles for small Step5 TP2 DeepEP LL batches."""
from __future__ import annotations
import functools
import os
import torch
from vllm.logger import init_logger
logger = init_logger("vllm.hcu.step5_bf16_tiles")
_step5_model_config = None


def configure_step5_bf16_tiles(vllm_config):
    """Keep the model's construction-time config for graph capture and replay."""
    global _step5_model_config
    _step5_model_config = vllm_config
    if os.environ.get("VLLM_STEP5_BF16_TILE_TUNING", "0") == "1":
        from vllm.model_executor.layers.fused_moe.experts import fused_batched_moe
        install_step5_bf16_tiles(fused_batched_moe)
        logger.info("Step5 BF16 tile hook configured: text_model_type=%s",
                    vllm_config.model_config.hf_text_config.model_type)



@functools.lru_cache(maxsize=None)
def _is_gfx938(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").startswith("gfx938")


def small_batch_tile_config(config, context, dp_size):
    """Select tiles using existing host scheduler counts, never GPU counts."""
    if context is None:
        return config
    cache_key = "_step5_bf16_global_tokens"
    if hasattr(context, cache_key):
        total = getattr(context, cache_key)
    else:
        metadata = getattr(context, "dp_metadata", None)
        counts = getattr(metadata, "num_tokens_across_dp_cpu", None)
        total = None
        if counts is not None and counts.device.type == "cpu":
            total = sum(counts.tolist())
        if total is None or total <= 0:
            descriptor = getattr(context, "batch_descriptor", None)
            if descriptor is not None:
                total = int(descriptor.num_tokens) * dp_size
        setattr(context, cache_key, total)
    if total is not None and 0 < total <= 512:
        return dict(config, BLOCK_SIZE_M=32, BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=64, num_warps=4, num_stages=3)
    return config


def install_step5_bf16_tiles(target):
    original = target.invoke_moe_batched_triton_kernel
    if getattr(original, "_step5_bf16_tiles", False):
        return
    reported = False
    rejected = set()

    def report_rejection(reason):
        if reason not in rejected:
            logger.info("Step5 BF16 tile guard: %s", reason)
            rejected.add(reason)

    @functools.wraps(original)
    def invoke(*args, **kwargs):
        nonlocal reported
        if os.environ.get("VLLM_STEP5_BF16_TILE_TUNING", "0") != "1":
            return original(*args, **kwargs)
        A = args[0] if args else kwargs["A"]
        B = args[1] if len(args) > 1 else kwargs["B"]
        if (A.ndim != 3 or A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16
            or (B.shape[1], B.shape[2]) not in ((3072, 4096), (4096, 1536))
            or not _is_gfx938(A.device)):
            return original(*args, **kwargs)
        from vllm.config import get_current_vllm_config_or_none
        from vllm.forward_context import get_forward_context
        cfg = _step5_model_config or get_current_vllm_config_or_none()
        if cfg is None:
            report_rejection("missing model config")
            return original(*args, **kwargs)
        from vllm_hcu.step5_dcp2_config import step5_context_parallel_supported
        p = cfg.parallel_config
        if (cfg.model_config.hf_text_config.model_type != "step5"
            or (p.tensor_parallel_size, p.data_parallel_size) not in ((2, 4), (4, 2))
            or p.pipeline_parallel_size != 1 or not step5_context_parallel_supported(cfg)
            or p.all2all_backend != "deepep_low_latency"):
            report_rejection(f"config type={cfg.model_config.hf_text_config.model_type}, TP={p.tensor_parallel_size}, DP={p.data_parallel_size}, backend={p.all2all_backend}")
            return original(*args, **kwargs)
        config = args[11] if len(args) > 11 else kwargs["config"]
        context = get_forward_context()
        tuned = small_batch_tile_config(config, context, p.data_parallel_size)
        if (os.environ.get("VLLM_STEP5_BF16_COMPACT_GRID", "0") == "1"
            and A.shape[0] == 44
            and not any(args[8:11] if len(args) > 10 else (
                kwargs.get("use_fp8_w8a8", False),
                kwargs.get("use_int8_w8a16", False),
                kwargs.get("use_int4_w4a16", False),
            ))):
            from .step5_bf16_batched import invoke_step5_bf16_experts
            C = args[2] if len(args) > 2 else kwargs["C"]
            counts = args[3] if len(args) > 3 else kwargs["expert_num_tokens"]
            if not reported:
                logger.info(
                    "Step5 BF16 compact expert grid enabled: shape=%s, config=%s",
                    tuple(A.shape), tuned,
                )
                reported = True
            # Counts remain on the GPU. This also works during PIECEWISE
            # compilation, where batch_descriptor is None.
            return invoke_step5_bf16_experts(A, B, C, counts, tuned)
        if tuned is config:
            return original(*args, **kwargs)
        if not reported:
            logger.info("Step5 BF16 small-batch tiles enabled: M32 N64 K64, "
                        "warps4 stages3, using host token counts")
            reported = True
        if len(args) > 11:
            positional = list(args)
            positional[11] = tuned
            return original(*positional, **kwargs)
        return original(*args, **dict(kwargs, config=tuned))

    invoke._step5_bf16_tiles = True
    target.invoke_moe_batched_triton_kernel = invoke
