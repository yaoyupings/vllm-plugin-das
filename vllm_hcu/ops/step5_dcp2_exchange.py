# SPDX-License-Identifier: Apache-2.0
"""Head exchange and BF16 striped-cache write for Step5 TP2/DCP2.

The QKV weights remain TP sharded. Pack *normalized, rotated* local Q/K and
local V, all-gather the packed rows along dim=1 in TP rank order, then unpack
global heads and write owned KV slots in one launch. A DCP2 group must match
the TP2 group and its rank order. This module does not enable model DCP2:
summary indexing, selected regions, prefill and FP32 attention-state exchange
must also be integrated before removing the launcher's DCP1 restriction.

Cache tensors are [physical_pages*64, 4, 192] or [physical_pages, 64, 4,
192]. Both affine interleaved K/V (sparse) and per-page K/V sections (SWA)
are supported through explicit page/token/head strides. Use the model runner's slot
mapping unchanged: CP_SIZE=2 and CP_INTERLEAVE=1 maps a logical 128-token page
to two local 64-token pages and masks the other rank's tokens with -1.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_local_qkv_kernel(
    q, k, v, packed,
    Q_ROW, K_ROW, V_ROW,
    Q_WIDTH: tl.constexpr, KV_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = Q_WIDTH + 2 * KV_WIDTH
    qv = tl.load(q + row * Q_ROW + d, mask=d < Q_WIDTH, other=0)
    kd = d - Q_WIDTH
    kv = tl.load(k + row * K_ROW + kd,
                 mask=(kd >= 0) & (kd < KV_WIDTH), other=0)
    vd = d - Q_WIDTH - KV_WIDTH
    vv = tl.load(v + row * V_ROW + vd,
                 mask=(vd >= 0) & (vd < KV_WIDTH), other=0)
    x = tl.where(d < Q_WIDTH, qv, tl.where(d < Q_WIDTH + KV_WIDTH, kv, vv))
    tl.store(packed + row * width + d, x, mask=d < width)


@triton.jit
def _unpack_qkv_cache_kernel(
    gathered, slots, query, key, value, key_cache, value_cache,
    GATHER_ROW, CACHE_SLOTS, K_CACHE_ROW, K_CACHE_HEAD, V_CACHE_ROW, V_CACHE_HEAD,
    K_CACHE_PAGE, V_CACHE_PAGE,
    Q_WIDTH: tl.constexpr, KV_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = Q_WIDTH + 2 * KV_WIDTH
    q_live = d < 2 * Q_WIDTH
    kd = d - 2 * Q_WIDTH
    k_live = (kd >= 0) & (kd < 2 * KV_WIDTH)
    vd = d - 2 * Q_WIDTH - 2 * KV_WIDTH
    v_live = (vd >= 0) & (vd < 2 * KV_WIDTH)

    # All-gather is rank-major Q0/K0/V0/Q1/K1/V1. Outputs are
    # component-major Q0/Q1, K0/K1, V0/V1, preserving provider group order.
    rank = tl.where(q_live, d // Q_WIDTH,
                    tl.where(k_live, kd // KV_WIDTH, vd // KV_WIDTH))
    local_d = tl.where(q_live, d % Q_WIDTH,
                       tl.where(k_live, Q_WIDTH + kd % KV_WIDTH,
                                Q_WIDTH + KV_WIDTH + vd % KV_WIDTH))
    x = tl.load(gathered + row * GATHER_ROW + rank * width + local_d,
                mask=d < 2 * width, other=0)
    tl.store(query + row * (2 * Q_WIDTH) + d, x, mask=q_live)
    tl.store(key + row * (2 * KV_WIDTH) + kd, x, mask=k_live)
    tl.store(value + row * (2 * KV_WIDTH) + vd, x, mask=v_live)

    slot = tl.load(slots + row)
    owned = (slot >= 0) & (slot < CACHE_SLOTS)
    tl.store(key_cache + (slot // 64) * K_CACHE_PAGE + (slot % 64) * K_CACHE_ROW
             + (kd // 192) * K_CACHE_HEAD + kd % 192,
             x, mask=owned & k_live)
    tl.store(value_cache + (slot // 64) * V_CACHE_PAGE + (slot % 64) * V_CACHE_ROW
             + (vd // 192) * V_CACHE_HEAD + vd % 192,
             x, mask=owned & v_live)


def _head_tensor(tensor: torch.Tensor, heads: int, tokens: int) -> bool:
    return (
        tensor.shape == (tokens, heads, 192)
        and tensor.dtype == torch.bfloat16
        and tensor.stride(-1) == 1
        and tensor.stride(-2) == 192
        and tensor.stride(0) >= heads * 192
    )



def _cache_tensor(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim in (3, 4) and tensor.shape[-2:] == (4, 192)
        and tensor.dtype == torch.bfloat16
        and tensor.stride(-1) == 1 and tensor.stride(-2) >= 192
        and tensor.stride(-3) >= 4 * tensor.stride(-2)
        and (tensor.ndim == 3 or (
            tensor.shape[1] == 64 and tensor.stride(0) >= 64 * tensor.stride(1)
        ))
    )


def _cache_geometry(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    slots = tensor.shape[0] * (64 if tensor.ndim == 4 else 1)
    page = tensor.stride(0) if tensor.ndim == 4 else 64 * tensor.stride(0)
    return slots, tensor.stride(-3), tensor.stride(-2), page


def pack_local_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Pack local TP2 heads; caller all-gathers the result with dim=1."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Q/K/V must have rank three")
    tokens, q_heads, _ = query.shape
    if tokens < 1 or q_heads < 1 or q_heads % 2:
        raise ValueError("requires nonempty TP2 queries with two local KV groups")
    if not (
        _head_tensor(query, q_heads, tokens)
        and _head_tensor(key, 2, tokens)
        and _head_tensor(value, 2, tokens)
    ):
        raise ValueError("requires BF16 HD192 Q/K/V with contiguous head rows")
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        raise ValueError("Q/K/V must share a GPU")
    q_width, kv_width = q_heads * 192, 2 * 192
    packed = torch.empty(
        (tokens, q_width + 2 * kv_width), dtype=query.dtype, device=query.device
    )
    _pack_local_qkv_kernel[(tokens, triton.cdiv(packed.shape[1], 256))](
        query, key, value, packed,
        query.stride(0), key.stride(0), value.stride(0),
        Q_WIDTH=q_width, KV_WIDTH=kv_width, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    return packed


def unpack_qkv_and_write_cache(
    gathered: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    local_query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack rank-ordered heads and write BF16 KV at already-striped slots.

    -1 slots leave the cache untouched while Q/K/V outputs are still populated
    (both ranks need all current-query heads). Positive out-of-range slots are
    also masked in the kernel. FP8 cache conversion requires separate scale
    validation and is deliberately unsupported here.
    """
    if local_query_heads < 1 or local_query_heads % 2:
        raise ValueError("requires two local KV groups")
    tokens = gathered.shape[0] if gathered.ndim == 2 else 0
    width = (local_query_heads + 4) * 192
    if (
        tokens < 1 or gathered.ndim != 2
        or gathered.shape[1] != 2 * width
        or gathered.dtype != torch.bfloat16 or gathered.stride(-1) != 1
        or slots.ndim != 1 or slots.shape[0] < tokens
        or slots.dtype not in (torch.int32, torch.int64) or not slots.is_contiguous()
    ):
        raise ValueError("invalid rank-major gathered QKV or slot mapping")
    if (
        not _cache_tensor(key_cache) or not _cache_tensor(value_cache)
        or value_cache.shape != key_cache.shape
        or _cache_geometry(key_cache)[0] < 64
        or _cache_geometry(key_cache)[0] % 64
    ):
        raise ValueError("requires local64 BF16 caches with four KV heads and contiguous features")
    if gathered.device.type != "cuda" or any(
        x.device != gathered.device for x in [slots, key_cache, value_cache]
    ):
        raise ValueError("all exchange/cache tensors must share a GPU")
    query = torch.empty(
        (tokens, 2 * local_query_heads, 192),
        dtype=torch.bfloat16, device=gathered.device,
    )
    key = torch.empty((tokens, 4, 192), dtype=query.dtype, device=query.device)
    value = torch.empty_like(key)
    cache_slots, kr, kh, kp = _cache_geometry(key_cache)
    _, vr, vh, vp = _cache_geometry(value_cache)
    _unpack_qkv_cache_kernel[(tokens, triton.cdiv(2 * width, 256))](
        gathered, slots, query, key, value, key_cache, value_cache,
        gathered.stride(0), cache_slots, kr, kh, vr, vh, kp, vp,
        Q_WIDTH=local_query_heads * 192, KV_WIDTH=2 * 192, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    return query, key, value


@triton.jit
def _decode_words_to_regions_kernel(
    words, counts, regions,
    ROWS: tl.constexpr, TOPK: tl.constexpr,
    REGION_SIZE: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // TOPK, offset % TOPK
    count = tl.load(counts + row, mask=row < ROWS, other=0)
    live = (row < ROWS) & (col < count)
    word = tl.load(words + offset, mask=live, other=0)
    # The source decoder stores global token offset in low32 bits and the
    # physical cache region in high32 bits. Physical IDs cannot be inverted
    # to logical IDs for shuffled/shared block tables.
    region = ((word & 0xFFFFFFFF) // REGION_SIZE).to(tl.int32)
    tl.store(regions + offset, tl.where(live, region, -1), mask=row < ROWS)


def decode_words_to_regions(
    words: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Retain selected logical region8 IDs from the existing Decode word.

    Counts, order and forced current region are preserved. Zero padding outside
    counts is masked, since a zero word can also be the valid first region.
    Gather these group-major logical rows along dim=0 before DCP2 attention.
    """
    if (
        words.ndim != 2 or min(words.shape) < 1
        or words.dtype != torch.int64 or not words.is_contiguous()
        or counts.shape != (words.shape[0],)
        or counts.dtype != torch.int32 or not counts.is_contiguous()
    ):
        raise ValueError("requires contiguous group-major int64 decode words and int32 counts")
    if words.device.type != "cuda" or counts.device != words.device:
        raise ValueError("decode words and counts must share a GPU")
    regions = torch.empty(words.shape, dtype=torch.int32, device=words.device)
    _decode_words_to_regions_kernel[(triton.cdiv(words.numel(), 256),)](
        words, counts, regions, ROWS=words.shape[0], TOPK=words.shape[1],
        REGION_SIZE=8, BLOCK=256, num_warps=4, num_stages=1,
    )
    return regions
