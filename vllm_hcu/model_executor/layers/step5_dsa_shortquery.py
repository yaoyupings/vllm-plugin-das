# SPDX-License-Identifier: Apache-2.0
"""Short-query Step5 DSA attention, adapted from the vendored HD192 decode kernel.

The reference release remains unchanged. Prefill metadata already packs the visible
token count per region, so splitting its slots preserves each query's causal horizon.
"""
import os
import torch
from vllm.logger import init_logger
from vllm.triton_utils import triton, tl
from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
    NEGATIVE_INFINITY, REGION_ID_MASK, REGION_VALID_SHIFT,
    _fit_block_regions, merge_split_states,
)

bf16_tile_logger = init_logger("vllm.step5_bf16_dsa")



def is_short_query_batch(num_tokens, max_query_len=0, *, verification=False,
                         capturing=False):
    """Keep ragged MTP verification on the device-only attention path.

    Total token count alone cannot distinguish an ordinary short prefill from
    2--4 queries per request. The builder already knows the maximum query
    length on the host, so this decision needs no tensor readback.
    """
    return 0 < num_tokens <= 64 or (
        64 < num_tokens <= 128
        and (verification or capturing or 0 < max_query_len <= 4)
    )


@triton.jit
def _sparse_attn_short_prefill_hd192_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    stride_out_split,
    stride_out_token,
    stride_out_head,
    stride_lse_split,
    stride_lse_token,
    softmax_scale,
    v_scale,
    q_scale,
    num_reqs,
    topk: tl.constexpr,
    heads_per_group: tl.constexpr,
    region_size: tl.constexpr,
    slots_per_split: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DK: tl.constexpr,
    QK_FP8: tl.constexpr,
    PV_FP8: tl.constexpr,
    P_SCALE: tl.constexpr,
    STORE_LSE: tl.constexpr,
) -> None:
    """HD192 split attention over the selector\'s causal int32 prefill regions."""
    request = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    row = group * num_reqs + request
    count = tl.load(counts_ptr + row).to(tl.int32)

    heads = tl.arange(0, BLOCK_H)
    dk = tl.arange(0, DK)
    head_ok = heads < heads_per_group

    qh = (group * heads_per_group + heads) * stride_q_head
    qr0 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (0 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr1 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (1 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    qr2 = tl.load(
        q_ptr + request * stride_q_token + qh[:, None] + (2 * DK + dk)[None, :],
        mask=head_ok[:, None],
        other=0.0,
    )
    if QK_FP8:
        # 3 * BLOCK_H * DK elements quantized once, against the whole selected cache
        # that now never gets widened.
        q0 = (qr0.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q1 = (qr1.to(tl.float32) * q_scale).to(tl.float8e4nv)
        q2 = (qr2.to(tl.float32) * q_scale).to(tl.float8e4nv)
    else:
        q0, q1, q2 = qr0, qr1, qr2

    running_max = tl.full((BLOCK_H,), NEGATIVE_INFINITY, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_H, DK), dtype=tl.float32)

    offsets_in_region = tl.arange(0, region_size)
    window_begin = split * slots_per_split
    window_end = window_begin + slots_per_split
    kg = group * stride_k_group
    vg = group * stride_v_group

    for base in tl.range(window_begin, window_end, BLOCK_R):
        slots = base + tl.arange(0, BLOCK_R)
        # ``slots < window_end`` is not redundant: ``slots_per_split`` is a ceiling
        # division, so it is a multiple of BLOCK_R only by luck. The production
        # width 513 over 16 splits gives 33 slots per split against BLOCK_R=16, and
        # the last tile of every split would otherwise reach 15 slots into the next
        # split's window. Both splits then accumulate those regions and the merge
        # double-counts them.
        slot_ok = (slots < count) & (slots < topk) & (slots < window_end)
        meta = tl.load(
            packed_ptr + row * stride_packed_row + slots, mask=slot_ok, other=-1
        )
        physical = meta & REGION_ID_MASK
        valid = (meta >> REGION_VALID_SHIFT) & 0xFF
        live = slot_ok & (valid > 0)

        safe_physical = tl.where(live, physical, 0)
        tokens = tl.reshape(
            safe_physical[:, None] * region_size + offsets_in_region[None, :],
            (BLOCK_R * region_size,),
        )
        token_ok = tl.reshape(
            live[:, None] & (offsets_in_region[None, :] < valid[:, None]),
            (BLOCK_R * region_size,),
        )

        # Load K already transposed as ``[DK, tokens]``. On the fp8 path the tile goes
        # into the dot as fp8; otherwise it is cast to the query's compute dtype, and an
        # fp8 cache is dequantized by that cast. Either way the descale itself rides on
        # softmax_scale (which the caller has already multiplied by k_scale, and divided
        # by q_scale) and on v_scale at the end.
        kb = k_ptr + tokens[None, :] * stride_k_token + kg
        k0 = tl.load(kb + (0 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k1 = tl.load(kb + (1 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        k2 = tl.load(kb + (2 * DK + dk)[:, None], mask=token_ok[None, :], other=0.0)
        if not QK_FP8:
            k0 = k0.to(qr0.dtype)
            k1 = k1.to(qr0.dtype)
            k2 = k2.to(qr0.dtype)

        s = tl.dot(q0, k0, out_dtype=tl.float32)
        s = tl.dot(q1, k1, acc=s, out_dtype=tl.float32)
        s = tl.dot(q2, k2, acc=s, out_dtype=tl.float32)
        s = s * softmax_scale
        s = tl.where(token_ok[None, :] & head_ok[:, None], s, NEGATIVE_INFINITY)

        tile_max = tl.maximum(running_max, tl.max(s, axis=1))
        empty = tile_max == NEGATIVE_INFINITY
        shift = tl.where(empty, 0.0, tile_max)
        rescale = tl.where(empty, 1.0, tl.exp2(running_max - shift))

        probabilities = tl.exp2(s - shift[:, None])
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)

        vb = v_ptr + tokens[:, None] * stride_v_token + vg
        v0 = tl.load(vb + (0 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v1 = tl.load(vb + (1 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        v2 = tl.load(vb + (2 * DK + dk)[None, :], mask=token_ok[:, None], other=0.0)
        if PV_FP8:
            pb = (probabilities * P_SCALE).to(tl.float8e4nv)
        else:
            pb = probabilities.to(qr0.dtype)
            v0 = v0.to(qr0.dtype)
            v1 = v1.to(qr0.dtype)
            v2 = v2.to(qr0.dtype)
        acc0 = acc0 * rescale[:, None] + tl.dot(pb, v0, out_dtype=tl.float32)
        acc1 = acc1 * rescale[:, None] + tl.dot(pb, v1, out_dtype=tl.float32)
        acc2 = acc2 * rescale[:, None] + tl.dot(pb, v2, out_dtype=tl.float32)
        running_max = tile_max

    selected = running_sum > 0
    # One reciprocal per head instead of a divide per output element, as in the hd192
    # prefill kernel. P_SCALE rides out here too, so neither it nor q_scale reaches the
    # output, and both -- like v_scale -- are constant per split and so commute with the
    # LSE-weighted merge.
    if PV_FP8:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum * P_SCALE), 0.0) * v_scale
    else:
        inv = tl.where(selected, tl.div_rn(1.0, running_sum), 0.0) * v_scale
    output0 = acc0 * inv[:, None]
    output1 = acc1 * inv[:, None]
    output2 = acc2 * inv[:, None]
    oh = (group * heads_per_group + heads) * stride_out_head
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (0 * DK + dk)[None, :],
        output0.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (1 * DK + dk)[None, :],
        output1.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    tl.store(
        out_ptr
        + split * stride_out_split
        + request * stride_out_token
        + oh[:, None]
        + (2 * DK + dk)[None, :],
        output2.to(out_ptr.dtype.element_ty),
        mask=head_ok[:, None],
    )
    if STORE_LSE:
        tl.store(
            lse_ptr
            + split * stride_lse_split
            + request * stride_lse_token
            + group * heads_per_group
            + heads,
            # running_max/running_sum are base-2; the stored lse is natural log, as the
            # general kernel's is, so convert rather than changing the contract.
            tl.where(
                selected,
                (running_max + tl.log2(running_sum)) * 0.6931471805599453,
                NEGATIVE_INFINITY,
            ),
            mask=head_ok,
        )



def short_query_num_splits(num_queries: int, num_kv_groups: int, topk: int) -> int:
    """BW1101 split policy for the stage-1 HD192 verification kernel."""
    rows = num_queries * num_kv_groups
    if rows <= 8:
        splits = 32
    elif rows <= 32:
        splits = 16
    elif rows <= 128:
        splits = 4
    elif rows <= 256:
        splits = 2
    else:
        splits = 1
    # Each split must still own at least two BLOCK_R=8 region tiles.
    return max(1, min(splits, topk // 16))

def sparse_attention_short_prefill(query, key_cache, value_cache, packed_regions,
                                  region_counts, *, num_kv_groups, region_size=8,
                                  softmax_scale=None, num_splits=1,
                                  k_scale=1.0, v_scale=1.0,
                                  output=None, return_lse=True):
    """Return merged short-query attention without quantizing Q or probabilities."""
    num_queries, num_heads, head_dim = query.shape
    if head_dim != 192 or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("short prefill requires BF16/FP16 queries with head_dim=192")
    if packed_regions.dtype != torch.int32:
        raise ValueError("short prefill requires causal int32 region metadata")
    if num_heads % num_kv_groups or packed_regions.shape[0] != num_queries * num_kv_groups:
        raise ValueError("query heads or metadata rows do not match KV groups")
    if num_splits < 1:
        raise ValueError("num_splits must be positive")
    heads_per_group = num_heads // num_kv_groups
    topk = int(packed_regions.shape[1])
    slots_per_split = -(-topk // num_splits)
    block_r = _fit_block_regions(
        min(8, triton.next_power_of_2(slots_per_split)), region_size=region_size,
        block_d=64, element_size=query.element_size())
    # Spawned workers may import this module before runtime env propagation.
    tile_mode = os.environ.get("VLLM_STEP5_BF16_DSA_TUNED_TILES", "0")
    if tile_mode not in ("0", "1"):
        raise ValueError("VLLM_STEP5_BF16_DSA_TUNED_TILES must be 0 or 1")
    # Qualified against per-head packed BF16 KV on BW1101. Keep all other dtypes,
    # layouts, split counts and shapes on their existing tile policy.
    if (
        tile_mode == "1"
        and num_queries in (16, 32, 64)
        and query.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
        and num_heads == 32 and num_kv_groups == 2 and region_size == 8
        and topk == 513
        and num_splits == short_query_num_splits(num_queries, num_kv_groups, topk)
        and key_cache.stride() == value_cache.stride() == (768, 384, 1)
    ):
        block_r = 4 if num_queries == 32 else 2
        bf16_tile_logger.info_once(
            "Step5 BF16 sparse Decode tiles enabled: M16/64 BR2, M32 BR4",
            scope="process",
        )
    # BW1101 loses occupancy when this HD192 kernel pipelines two K/V tiles:
    # BLOCK_R=8 already consumes enough LDS that the second stage serializes
    # waves instead of hiding memory latency. One stage is also faster for the
    # smaller split windows used by M=1/4/16. At the production top-513 shape
    # it is 1.06x faster at M=32, 1.80x at M=64, and 3.03x at M=128, with
    # bit-identical output and LSE.
    stages = 1
    expected_output = (num_queries, num_heads, head_dim)
    if output is not None and (
        output.shape != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError(
            f"short prefill output must be {expected_output} on {query.device} "
            f"with {query.dtype}"
        )
    direct = output is not None and num_splits == 1
    out = (
        output.unsqueeze(0)
        if direct
        else torch.empty(
            (num_splits, num_queries, num_heads, head_dim),
            device=query.device, dtype=query.dtype,
        )
    )
    needs_partial_lse = return_lse or num_splits > 1
    lse = (
        torch.empty((num_splits, num_queries, num_heads),
                    device=query.device, dtype=torch.float32)
        if needs_partial_lse
        else torch.empty((1,), device=query.device, dtype=torch.float32)
    )
    scale = (softmax_scale if softmax_scale is not None else head_dim ** -0.5) * k_scale
    _sparse_attn_short_prefill_hd192_kernel[(num_queries, num_kv_groups, num_splits)](
        query, key_cache, value_cache, packed_regions, region_counts,
        region_counts, out, lse, query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        packed_regions.stride(0), out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1), scale * 1.4426950408889634,
        v_scale, 1.0, num_queries, topk=topk,
        heads_per_group=heads_per_group, region_size=region_size,
        slots_per_split=slots_per_split,
        BLOCK_H=max(16, triton.next_power_of_2(heads_per_group)),
        BLOCK_R=block_r, DK=64, QK_FP8=False, PV_FP8=False, P_SCALE=256.0,
        STORE_LSE=needs_partial_lse, num_warps=2, num_stages=stages)
    if num_splits == 1:
        return out[0], lse[0] if return_lse else None
    return merge_split_states(
        out, lse, out=output, write_lse=return_lse,
    )


@triton.jit
def _prefill_request_indices_kernel(starts, output, TOKENS: tl.constexpr,
                                    REQUESTS: tl.constexpr,
                                    DECODE: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first = tl.load(starts)
    absolute = first + token
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), REQUESTS, tl.int32)
    # Upper-bound search handles empty requests at the beginning, middle,
    # and end without materializing a repeat-interleave result.
    while tl.sum((lo < hi).to(tl.int32), axis=0) > 0:
        mid = (lo + hi) // 2
        searching = lo < hi
        end = tl.load(starts + mid + 1, mask=searching, other=2147483647)
        advance = searching & (absolute >= end)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(searching & ~advance, mid, hi)
    # FULL batches include an empty final request row. Use that in-range
    # row for trailing padded tokens, whose actual query ownership and KV
    # slot masks remain empty. Consumers never receive an invalid table row.
    request = DECODE + tl.minimum(lo, REQUESTS - 1)
    tl.store(output + token, request, mask=token < TOKENS)


def prefill_request_indices_device(query_start_loc, num_decodes,
                                   num_prefills, num_prefill_tokens):
    """Map query rows on device, including the fixed FULL-graph padding bound.

    repeat_interleave(output_size=...) requires the bound to equal the sum
    of repeats. FULL padding makes it larger, which can trigger a GPU VMFault.
    Trailing padding uses the final (empty in FULL metadata) request row;
    attention still determines live queries from the unchanged boundaries.
    """
    if min(num_decodes, num_prefills, num_prefill_tokens) < 0:
        raise ValueError("Request-map dimensions must be nonnegative")
    if query_start_loc.ndim != 1 or query_start_loc.dtype != torch.int32:
        raise ValueError("Request-map boundaries must be one-dimensional int32")
    stop = num_decodes + num_prefills
    if query_start_loc.numel() < stop + 1:
        raise ValueError("Request-map boundaries do not cover the batch")
    if num_prefills == 0 and num_prefill_tokens:
        raise ValueError("Cannot map tokens without prefill requests")
    output = torch.empty(num_prefill_tokens, device=query_start_loc.device,
                         dtype=torch.int32)
    if num_prefill_tokens == 0:
        return output
    starts = query_start_loc[num_decodes:stop + 1].contiguous()
    if query_start_loc.device.type == "cpu":
        tokens = torch.arange(num_prefill_tokens, dtype=torch.int32) + starts[0]
        return torch.searchsorted(starts[1:], tokens, right=True,
                                  out_int32=True).clamp_max(num_prefills - 1) + num_decodes
    _prefill_request_indices_kernel[(triton.cdiv(num_prefill_tokens, 128),)](
        starts, output, TOKENS=num_prefill_tokens, REQUESTS=num_prefills,
        DECODE=num_decodes, BLOCK=128, num_warps=4, num_stages=1)
    return output
