# SPDX-License-Identifier: Apache-2.0
"""Reuse the existing MoE arena for native Step5 SWA paged conversion."""
from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.hcu.step5_swa_workspace")

# Retain the vendor's allocation floor; small native buffers have a documented
# framework failure. This optimization changes storage ownership, not that ABI.
_NATIVE_TOKEN_FLOOR = 262144


def eligible(query, key_cache, value_cache, cu_query, max_query, window):
    rows = cu_query.numel() - 1
    uniform_mtp = 1 < max_query <= 4 and max_query * rows == query.shape[0]
    return (
        rows > 0 and max_query > 1 and not uniform_mtp
        and window[0] >= 0 and window[1] == 0
        and query.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
        and query.shape[-1] == key_cache.shape[-1] == value_cache.shape[-1] == 192
        and key_cache.ndim == value_cache.ndim == 4
        and key_cache.shape[1] == value_cache.shape[1] == 64
    )


def native_swa_varlen_with_workspace(
    query, key_cache, value_cache, output, cu_query, seq_lens, table,
    max_query, max_key, softmax_scale, window,
):
    from flash_attn.flash_attn_interface import flash_attn_cuda
    from vllm.v1.worker.workspace import current_workspace_manager

    if not eligible(query, key_cache, value_cache, cu_query, max_query, window):
        raise ValueError("Unsupported Step5 native SWA workspace geometry")
    # Step5WindowMetadataBuilder has already removed the common invisible
    # prefix. Use its static launch bound; never sum device lengths on CPU.
    rows = cu_query.numel() - 1
    tokens = max(_NATIVE_TOKEN_FLOOR, rows * max_key)
    heads = key_cache.shape[2]
    logger.info_once(
        "Step5 SWA native conversion uses shared workspace: "
        "tokens=%d; heads=%d; width=192", tokens, heads,
    )
    k, v, cu_key = current_workspace_manager().get_simultaneous(
        ((tokens, heads, 192), query.dtype),
        ((tokens, heads, 192), query.dtype),
        ((rows + 1,), cu_query.dtype),
    )
    flash_attn_cuda.pagedkv_to_contiguouskv(
        k, v, key_cache, value_cache, table, seq_lens, cu_key, max_key, 2,
    )
    result = flash_attn_cuda.varlen_fwd(
        query, k, v, output, cu_query, cu_key,
        None, None, None, None, max_query, max_key, 0.0, softmax_scale,
        False, True, window[0], window[1], 0.0, False,
        None, None, None, None, None,
    )
    return result[0]
