# SPDX-License-Identifier: Apache-2.0
"""Step5 TP2/DCP2 Attention execution bridge.

TP weights remain sharded: exchange normalized/rotated local QKV and local
provider-group selections, evaluate the striped local cache, then combine
FP32 softmax states and scatter the original TP query heads. This module
does not by itself enable DCP2 in model/config/cache constructors.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .step5_dcp2 import merge, partial
from .step5_dcp2_exchange import pack_local_qkv, unpack_qkv_and_write_cache
from .step5_dcp2_fused_exchange import fused_exchange


@triton.jit
def _query_metadata_kernel(starts, lengths, positions, requests, qpos,
                           TOKENS: tl.constexpr, REQUESTS: tl.constexpr,
                           BLOCK: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    begin = tl.load(starts + rows, mask=rows < REQUESTS, other=0)
    end = tl.load(starts + rows + 1, mask=rows < REQUESTS, other=0)
    length = tl.load(lengths + rows, mask=rows < REQUESTS, other=0)
    live = (rows < REQUESTS) & (token >= begin) & (token < end) & (length > 0)
    request = tl.min(tl.where(live, rows, REQUESTS), axis=0)
    position = tl.load(positions + token).to(tl.int32)
    valid = (request < REQUESTS) & (position >= 0)
    tl.store(requests + token, tl.where(valid, request, -1))
    tl.store(qpos + token, tl.where(valid, position, -1))


def query_metadata(positions, starts, lengths):
    if (positions.ndim != 1 or positions.numel() < 1
        or positions.dtype not in (torch.int32, torch.int64)
        or starts.ndim != 1 or lengths.ndim != 1
        or starts.numel() != lengths.numel() + 1 or lengths.numel() < 1
        or any(x.dtype != torch.int32 for x in (starts, lengths))
        or any(not x.is_contiguous() for x in (positions, starts, lengths))):
        raise ValueError("DCP2 requires contiguous global query boundaries/positions")
    if positions.device.type != "cuda" or any(
        x.device != positions.device for x in (starts, lengths)
    ):
        raise ValueError("DCP2 query metadata must share a GPU")
    requests = torch.empty_like(positions, dtype=torch.int32)
    qpos = torch.empty_like(requests)
    _query_metadata_kernel[(positions.numel(),)](
        starts, lengths, positions, requests, qpos,
        TOKENS=positions.numel(), REQUESTS=lengths.numel(),
        BLOCK=triton.next_power_of_2(lengths.numel()), num_warps=4,
    )
    return requests, qpos


@triton.jit
def _window_regions_kernel(qpos, requests, regions, counts,
                           TOKENS: tl.constexpr, TOPK: tl.constexpr,
                           WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    query = row % TOKENS
    position = tl.load(qpos + query)
    request = tl.load(requests + query)
    first = tl.maximum(0, position - WINDOW + 1) // 8
    last = position // 8
    count = tl.where((request >= 0) & (position >= 0), last-first+1, 0)
    slot = tl.arange(0, BLOCK)
    tl.store(regions + row*TOPK + slot, tl.where(slot < count, first+slot, -1),
             mask=slot < TOPK)
    tl.store(counts + row, count)


def window_regions(qpos, requests, *, window=512):
    if window != 512:
        raise ValueError("Step5 DCP2 window bridge currently requires window512")
    if qpos.dtype != torch.int32 or requests.dtype != torch.int32:
        raise ValueError("window metadata must be int32")
    if qpos.ndim != 1 or requests.shape != qpos.shape or qpos.numel() < 1:
        raise ValueError("invalid window query metadata")
    if (not qpos.is_contiguous() or not requests.is_contiguous()
        or qpos.device.type != "cuda" or qpos.device != requests.device):
        raise ValueError("window metadata must be contiguous on the same GPU")
    tokens, topk = qpos.numel(), (window+6)//8+1
    regions = torch.empty((4*tokens, topk), device=qpos.device, dtype=torch.int32)
    counts = torch.empty(4*tokens, device=qpos.device, dtype=torch.int32)
    _window_regions_kernel[(4*tokens,)](
        qpos, requests, regions, counts, TOKENS=tokens, TOPK=topk,
        WINDOW=window, BLOCK=triton.next_power_of_2(topk), num_warps=4,
    )
    return regions, counts


@triton.jit
def _assemble_plan_kernel(words, dcounts, logical, pcounts, payload,
                          TOKENS: tl.constexpr, DECODE: tl.constexpr,
                          PREFILL: tl.constexpr, TOPK: tl.constexpr,
                          BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, col = offset // (TOPK+1), offset % (TOPK+1)
    group, query = row // TOKENS, row % TOKENS
    live = row < 2*TOKENS
    if DECODE > 0:
        drow = group*DECODE + query
        dc = tl.load(dcounts+drow, mask=live & (query < DECODE), other=0)
        word = tl.load(words+drow*TOPK+col,
                       mask=live & (query < DECODE) & (col < TOPK) & (col < dc),
                       other=0)
        dregion = ((word & 0xFFFFFFFF)//8).to(tl.int32)
    else:
        dc, dregion = tl.full((BLOCK,), 0, tl.int32), tl.full((BLOCK,), -1, tl.int32)
    if PREFILL > 0:
        pq = query-DECODE
        prow = group*PREFILL + pq
        plive = live & (pq >= 0) & (pq < PREFILL)
        pc = tl.load(pcounts+prow, mask=plive, other=0)
        pregion = tl.load(logical+prow*TOPK+col,
                         mask=plive & (col < TOPK) & (col < pc), other=-1)
    else:
        pc, pregion = tl.full((BLOCK,), 0, tl.int32), tl.full((BLOCK,), -1, tl.int32)
    count = tl.where(query < DECODE, dc, pc)
    region = tl.where(query < DECODE, dregion, pregion)
    value = tl.where(col == TOPK, count, tl.where(col < count, region, -1))
    tl.store(payload+offset, value, mask=live)


@triton.jit
def _unpack_plan_kernel(payload, regions, counts, ROWS: tl.constexpr,
                        TOPK: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    row, col = offset//TOPK, offset%TOPK
    x = tl.load(payload+row*(TOPK+1)+col, mask=row < ROWS, other=-1)
    tl.store(regions+offset, x, mask=row < ROWS)
    count = tl.load(payload+row*(TOPK+1)+TOPK, mask=(row < ROWS) & (col == 0),
                    other=0)
    tl.store(counts+row, count, mask=(row < ROWS) & (col == 0))


def gather_sparse_regions(plan, md, tokens, group):
    """One collective for both provider-group region IDs and counts."""
    if plan is None or plan.md is not md or md.block_size != 128:
        raise RuntimeError("DCP2 requires this step's logical128 sparse plan")
    nd, np = md.num_decode_tokens, md.num_actual_tokens-md.num_decode_tokens
    if min(nd, np) < 0 or nd+np > tokens:
        raise ValueError("sparse plan token split exceeds queries")
    if (nd > 0 and plan.decode is None) or (np > 0 and plan.prefill is None):
        raise RuntimeError("DCP2 sparse plan is missing a query half")
    dwords, dcounts = plan.decode if nd else (None, None)
    logical = plan.prefill[2] if np else None
    pcounts = plan.prefill[1] if np else None
    if np and logical is None:
        raise RuntimeError("DCP2 requires retained logical prefill region IDs")
    first = dwords if nd else logical
    if first is None:
        raise RuntimeError("cannot build an empty DCP2 sparse plan")
    topk = first.shape[1]
    for reg, count, n, dtype in (
        (dwords, dcounts, nd, torch.int64), (logical, pcounts, np, torch.int32)
    ):
        if not n:
            continue
        if (reg.shape != (2*n, topk) or count.shape != (2*n,)
            or reg.dtype != dtype or count.dtype != torch.int32
            or not reg.is_contiguous() or not count.is_contiguous()
            or reg.device != first.device or count.device != first.device):
            raise ValueError("DCP2 selection rows must have two local provider groups")
    if first.device.type != "cuda":
        raise ValueError("DCP2 selection must be on a GPU")
    # Dummy pointers for compile-time absent halves are never read.
    dummy = first
    payload = torch.empty((2*tokens, topk+1), device=first.device, dtype=torch.int32)
    _assemble_plan_kernel[(triton.cdiv(payload.numel(), 256),)](
        dwords if nd else dummy, dcounts if nd else dummy,
        logical if np else dummy, pcounts if np else dummy, payload,
        TOKENS=tokens, DECODE=nd, PREFILL=np, TOPK=topk, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    gathered = group.all_gather(payload, dim=0)
    regions = torch.empty((4*tokens, topk), device=first.device, dtype=torch.int32)
    counts = torch.empty(4*tokens, device=first.device, dtype=torch.int32)
    _unpack_plan_kernel[(triton.cdiv(regions.numel(), 256),)](
        gathered, regions, counts, ROWS=4*tokens, TOPK=topk, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    return regions, counts


@triton.jit
def _weight_rank_state_kernel(out, lses, weighted, global_lse,
                              OUT_TOKEN, OUT_HEAD, OUT_DIM,
                              TOKENS: tl.constexpr, HEADS: tl.constexpr,
                              RANK: tl.constexpr, BLOCK_D: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    a = tl.load(lses+token*HEADS+head)
    b = tl.load(lses+(TOKENS+token)*HEADS+head)
    a = tl.where((a != a) | (a == float("inf")), -float("inf"), a)
    b = tl.where((b != b) | (b == float("inf")), -float("inf"), b)
    peak = tl.maximum(a, b)
    empty = peak == -float("inf")
    shift = tl.where(empty, 0., peak)
    wa, wb = tl.exp(a-shift), tl.exp(b-shift)
    total = wa+wb
    factor = tl.where(RANK == 0, wa, wb)/tl.maximum(total, 1.e-20)
    dims = tl.arange(0, BLOCK_D)
    x = tl.load(out+token*OUT_TOKEN+head*OUT_HEAD+dims*OUT_DIM,
                mask=dims < 192, other=0.)
    # Head-major output is already in ReduceScatter's rank-chunk layout.
    tl.store(weighted+(head*TOKENS+token)*192+dims,
             tl.where(factor > 0., x*factor, 0.), mask=dims < 192)
    tl.store(global_lse+token*HEADS+head,
             tl.where(empty, -float("inf"), shift+tl.log(total)))


@triton.jit
def _finish_local_heads_kernel(scattered, output, TOKENS: tl.constexpr,
                               HEADS: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    dim = offset%192
    head = (offset//192)%HEADS
    token = offset//(192*HEADS)
    x = tl.load(scattered+(head*TOKENS+token)*192+dim,
                mask=token < TOKENS, other=0.)
    tl.store(output+offset, x, mask=token < TOKENS)


def combine_rank_states(output, lse, group, *, return_lse=False):
    if group.world_size != 2 or group.rank_in_group not in (0, 1):
        raise ValueError("Step5 DCP2 requires an ordered two-rank group")
    if (output.ndim != 3 or output.shape[1:] != (64, 192)
        or output.shape[0] < 1 or output.dtype != torch.float32
        or lse.shape != output.shape[:2] or lse.dtype != torch.float32
        or not lse.is_contiguous() or output.stride(-1) != 1
        or output.device.type != "cuda" or output.device != lse.device):
        raise ValueError("DCP2 combine requires FP32 states [queries,64,192]")
    tokens = output.shape[0]
    all_lse = group.all_gather(lse, dim=0)
    weighted = torch.empty((64, tokens, 192), device=output.device, dtype=torch.float32)
    global_lse = torch.empty_like(lse)
    _weight_rank_state_kernel[(tokens, 64)](
        output, all_lse, weighted, global_lse, *output.stride(),
        TOKENS=tokens, HEADS=64, RANK=group.rank_in_group, BLOCK_D=256,
        num_warps=4, num_stages=1,
    )
    scattered = group.reduce_scatter(weighted, dim=0)
    if scattered.shape != (32, tokens, 192) or scattered.dtype != torch.float32:
        raise RuntimeError("DCP2 ReduceScatter changed state dtype or head geometry")
    result = torch.empty((tokens, 32, 192), device=output.device, dtype=torch.bfloat16)
    _finish_local_heads_kernel[(triton.cdiv(result.numel(), 256),)](
        scattered, result, TOKENS=tokens, HEADS=32, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    if return_lse:
        start = group.rank_in_group*32
        return result, global_lse[:, start:start+32].contiguous()
    return result


def dcp2_attention(query, key, value, key_cache, value_cache, slots,
                   positions, starts, lengths, table, group, *,
                   window=0, plan=None, metadata=None, splits=1,
                   return_lse=False, fused_exchange_enabled=False, norm_data=None):
    """Execute gathered heads/selections and return the original 32 TP heads."""
    if group.world_size != 2 or group.rank_in_group not in (0, 1):
        raise ValueError("Step5 DCP2 requires an ordered two-rank group")
    if query.ndim != 3 or query.shape[1:] != (32, 192):
        raise ValueError("Step5 DCP2 bridge requires local32Q/2KV heads")
    tokens = query.shape[0]
    if positions.shape != (tokens,) or slots.shape != (tokens,):
        raise ValueError("positions/slots must describe every padded query")
    if window not in (0, 512):
        raise ValueError("only Step5 sparse or sliding512 attention is supported")
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("DCP2 reader requires affine sparse-style cache views")
    eligible = (fused_exchange_enabled and metadata is not None
                and 1 <= metadata.max_query_len <= 4 and tokens <= 64)
    if norm_data is not None and not eligible:
        raise ValueError("Raw QKV norm exchange requires the eligible fused short-query path")
    if eligible:
        full_q, regions, counts, requests, qpos = fused_exchange(
            query, key, value, key_cache, value_cache, slots,
            positions, starts, lengths, group, window=window,
            plan=plan, metadata=metadata, norm_data=norm_data,
        )
    else:
        requests, qpos = query_metadata(positions, starts, lengths)
        packed = pack_local_qkv(query, key, value)
        gathered = group.all_gather(packed, dim=1)
        full_q, _, _ = unpack_qkv_and_write_cache(
            gathered, slots, key_cache, value_cache, local_query_heads=32,
        )
        if window:
            regions, counts = window_regions(qpos, requests, window=window)
        else:
            regions, counts = gather_sparse_regions(plan, metadata, tokens, group)
    states, lses, _ = partial(
        full_q, key_cache, value_cache, regions, counts, qpos, requests, table,
        rank=group.rank_in_group, window=window, splits=splits,
        global_region_size=8,
    )
    local_out, local_lse = merge(states, lses, dtype=torch.float32)
    return combine_rank_states(local_out, local_lse, group, return_lse=return_lse)
