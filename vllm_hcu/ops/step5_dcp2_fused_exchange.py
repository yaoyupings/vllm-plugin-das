# SPDX-License-Identifier: Apache-2.0
"""Single-collective Step5 DCP2 QKV/selection exchange.

BF16 values travel as exact pairs of 16-bit words inside int32 payloads.
Logical sparse regions and counts share the same rank-major payload.
The runtime enables this path only for short queries with an explicit flag.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .step5_dcp2_exchange import _head_tensor, _cache_tensor, _cache_geometry
from .step5_dcp2_norm_exchange import pack_norm_exchange


@triton.jit
def _pack_exchange(q, k, v, words, dcounts, logical, pcounts,
                   positions, starts, lengths, payload, requests, qpos,
                   Q_ROW, K_ROW, V_ROW,
                   TOKENS: tl.constexpr, REQUESTS: tl.constexpr,
                   DECODE: tl.constexpr, PREFILL: tl.constexpr,
                   TOPK: tl.constexpr, WINDOW: tl.constexpr,
                   REQ_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    d = block * BLOCK + tl.arange(0, BLOCK)
    # 32 local query heads and two local K/V heads, HD192.
    qwidth: tl.constexpr = 6144
    kvwidth: tl.constexpr = 384
    qwords: tl.constexpr = 3456
    pwidth: tl.constexpr = qwords + (0 if WINDOW else 2 * (TOPK + 1))
    bf = 2*d
    qa = tl.load(q + row*Q_ROW + bf, mask=bf < qwidth, other=0)
    qb = tl.load(q + row*Q_ROW + bf+1, mask=bf < qwidth, other=0)
    kd = bf-qwidth
    ka = tl.load(k + row*K_ROW + kd, mask=(kd >= 0) & (kd < kvwidth), other=0)
    kb = tl.load(k + row*K_ROW + kd+1, mask=(kd >= 0) & (kd < kvwidth), other=0)
    vd = kd-kvwidth
    va = tl.load(v + row*V_ROW + vd, mask=(vd >= 0) & (vd < kvwidth), other=0)
    vb = tl.load(v + row*V_ROW + vd+1, mask=(vd >= 0) & (vd < kvwidth), other=0)
    lo = tl.where(bf < qwidth, qa, tl.where(kd < kvwidth, ka, va))
    hi = tl.where(bf < qwidth, qb, tl.where(kd < kvwidth, kb, vb))
    packed = lo.to(tl.uint16, bitcast=True).to(tl.uint32) | (
        hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
    tl.store(payload + row*pwidth + d, packed.to(tl.int32, bitcast=True), mask=d < qwords)
    if not WINDOW:
        pd = d-qwords
        group, col = pd//(TOPK+1), pd%(TOPK+1)
        live = (pd >= 0) & (group < 2)
        if DECODE:
            dr = group*DECODE + row
            dc = tl.load(dcounts+dr, mask=live & (row < DECODE), other=0)
            word = tl.load(words+dr*TOPK+col,
                           mask=live & (row < DECODE) & (col < TOPK) & (col < dc), other=0)
            region_d = ((word & 0xFFFFFFFF)//8).to(tl.int32)
        else:
            dc = tl.full((BLOCK,), 0, tl.int32)
            region_d = tl.full((BLOCK,), -1, tl.int32)
        if PREFILL:
            pr = group*PREFILL+row-DECODE
            plive = live & (row >= DECODE) & (row < DECODE+PREFILL)
            pc = tl.load(pcounts+pr, mask=plive, other=0)
            region_p = tl.load(logical+pr*TOPK+col,
                               mask=plive & (col < TOPK) & (col < pc), other=-1)
        else:
            pc = tl.full((BLOCK,), 0, tl.int32)
            region_p = tl.full((BLOCK,), -1, tl.int32)
        count = tl.where(row < DECODE, dc, pc)
        region = tl.where(row < DECODE, region_d, region_p)
        result = tl.where(col == TOPK, count, tl.where(col < count, region, -1))
        tl.store(payload+row*pwidth+d, result, mask=live)
    if block == 0:
        rr = tl.arange(0, REQ_BLOCK)
        begin = tl.load(starts+rr, mask=rr < REQUESTS, other=0)
        end = tl.load(starts+rr+1, mask=rr < REQUESTS, other=0)
        length = tl.load(lengths+rr, mask=rr < REQUESTS, other=0)
        active = (rr < REQUESTS) & (row >= begin) & (row < end) & (length > 0)
        req = tl.min(tl.where(active, rr, REQUESTS), 0)
        pos = tl.load(positions+row).to(tl.int32)
        valid = (req < REQUESTS) & (pos >= 0)
        tl.store(requests+row, tl.where(valid, req, -1))
        tl.store(qpos+row, tl.where(valid, pos, -1))


@triton.jit
def _unpack_exchange(gathered, slots, query, key_cache, value_cache,
                     requests, qpos, regions, counts,
                     CACHE_SLOTS, K_ROW, K_HEAD, K_PAGE, V_ROW, V_HEAD, V_PAGE,
                     TOKENS: tl.constexpr, TOPK: tl.constexpr,
                     WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    qwidth: tl.constexpr = 6144
    kvwidth: tl.constexpr = 384
    width: tl.constexpr = 6912
    qwords: tl.constexpr = 3456
    pwidth: tl.constexpr = qwords + (0 if WINDOW else 2*(TOPK+1))
    qlive = d < 2*qwidth
    kd = d-2*qwidth
    klive = (kd >= 0) & (kd < 2*kvwidth)
    vd = kd-2*kvwidth
    vlive = (vd >= 0) & (vd < 2*kvwidth)
    rank = tl.where(qlive, d//qwidth, tl.where(klive, kd//kvwidth, vd//kvwidth))
    local_d = tl.where(qlive, d%qwidth,
                       tl.where(klive, qwidth+kd%kvwidth, qwidth+kvwidth+vd%kvwidth))
    word = tl.load(gathered+row*(2*pwidth)+rank*pwidth+local_d//2,
                   mask=d < 2*width, other=0).to(tl.uint32, bitcast=True)
    value = ((word >> ((local_d%2)*16)) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(query+row*(2*qwidth)+d, value, mask=qlive)
    slot = tl.load(slots+row)
    owned = (slot >= 0) & (slot < CACHE_SLOTS)
    tl.store(key_cache+(slot//64)*K_PAGE+(slot%64)*K_ROW+(kd//192)*K_HEAD+kd%192,
             value, mask=owned & klive)
    tl.store(value_cache+(slot//64)*V_PAGE+(slot%64)*V_ROW+(vd//192)*V_HEAD+vd%192,
             value, mask=owned & vlive)
    pd = d-2*width
    group, col = pd//(TOPK+1), pd%(TOPK+1)
    live = (pd >= 0) & (group < 4)
    if WINDOW:
        position = tl.load(qpos+row)
        request = tl.load(requests+row)
        first = tl.maximum(0, position-WINDOW+1)//8
        last = position//8
        count = tl.where((request >= 0) & (position >= 0), last-first+1, 0)
        region = tl.where(col < count, first+col, -1)
        result = tl.where(col == TOPK, count, region)
    else:
        result = tl.load(gathered+row*(2*pwidth)+(group//2)*pwidth+qwords
                         +(group%2)*(TOPK+1)+col, mask=live, other=-1)
    outrow = group*TOKENS+row
    tl.store(regions+outrow*TOPK+col, result, mask=live & (col < TOPK))
    tl.store(counts+outrow, result, mask=live & (col == TOPK))


def fused_exchange(query, key, value, kc, vc, slots, positions,
                   starts, lengths, group, *, window=0, plan=None, metadata=None, norm_data=None):
    tokens = query.shape[0]
    if (group.world_size != 2 or group.rank_in_group not in (0, 1)
        or tokens < 1 or window not in (0, 512)
        or not _head_tensor(query, 32, tokens)
        or not _head_tensor(key, 2, tokens) or not _head_tensor(value, 2, tokens)
        or not _cache_tensor(kc) or not _cache_tensor(vc) or kc.shape != vc.shape
        or slots.shape != (tokens,) or positions.shape != (tokens,)
        or slots.dtype not in (torch.int32, torch.int64)
        or positions.dtype not in (torch.int32, torch.int64)
        or starts.ndim != 1 or lengths.ndim != 1 or lengths.numel() < 1
        or starts.numel() != lengths.numel()+1
        or starts.dtype != torch.int32 or lengths.dtype != torch.int32
        or any(not x.is_contiguous() for x in (slots, positions, starts, lengths))
        or query.device.type != 'cuda'
        or any(x.device != query.device for x in (key,value,kc,vc,slots,positions,starts,lengths))):
        raise ValueError('fused exchange requires TP2/DCP2 BF16 HD192 and device metadata')
    cache_slots, kr, kh, kp = _cache_geometry(kc)
    _, vr, vh, vp = _cache_geometry(vc)
    if cache_slots < 64 or cache_slots%64:
        raise ValueError('fused exchange requires local64 cache pages')
    nd = np = 0
    words = dcounts = logical = pcounts = query
    if window:
        topk = (window+6)//8+1
    else:
        if plan is None or plan.md is not metadata or metadata.block_size != 128:
            raise RuntimeError('fused exchange requires current global128 sparse plan')
        nd = metadata.num_decode_tokens
        np = metadata.num_actual_tokens-nd
        if min(nd,np) < 0 or nd+np > tokens or nd+np < 1:
            raise ValueError('invalid sparse plan query split')
        if (nd and plan.decode is None) or (np and plan.prefill is None):
            raise RuntimeError('missing sparse plan half')
        words,dcounts = plan.decode if nd else (None,None)
        logical = plan.prefill[2] if np else None
        pcounts = plan.prefill[1] if np else None
        first = words if nd else logical
        if first is None or first.ndim != 2 or first.shape[1] < 1:
            raise ValueError('missing logical sparse regions')
        topk = first.shape[1]
        for reg,count,n,dtype in ((words,dcounts,nd,torch.int64),(logical,pcounts,np,torch.int32)):
            if not n:
                continue
            if (reg is None or count is None or reg.shape != (2*n,topk)
                or count.shape != (2*n,) or reg.dtype != dtype or count.dtype != torch.int32
                or not reg.is_contiguous() or not count.is_contiguous()
                or reg.device != query.device or count.device != query.device):
                raise ValueError('invalid sparse region/count layout')
        words = first if words is None else words
        dcounts = first if dcounts is None else dcounts
        logical = first if logical is None else logical
        pcounts = first if pcounts is None else pcounts
    pwidth = 3456+(0 if window else 2*(topk+1))
    payload = torch.empty((tokens,pwidth),device=query.device,dtype=torch.int32)
    requests = torch.empty(tokens,device=query.device,dtype=torch.int32)
    qpos = torch.empty_like(requests)
    if norm_data is None:
        _pack_exchange[(tokens,triton.cdiv(pwidth,256))](
            query,key,value,words,dcounts,logical,pcounts,positions,starts,lengths,payload,requests,qpos,
            query.stride(0),key.stride(0),value.stride(0),TOKENS=tokens,REQUESTS=lengths.numel(),
            DECODE=nd,PREFILL=np,TOPK=topk,WINDOW=window,
            REQ_BLOCK=triton.next_power_of_2(lengths.numel()),BLOCK=256,num_warps=4,num_stages=1)
    else:
        packed,qw,kw,rope_cache,pairs,epsilon = norm_data
        if (metadata is None or not 1 <= metadata.max_query_len <= 4 or tokens > 64
            or packed.shape != (tokens,6912) or packed.dtype != torch.bfloat16
            or packed.stride(1) != 1 or packed.stride(0) < 6912
            or qw.shape != (192,) or kw.shape != (192,)
            or qw.dtype != torch.float32 or kw.dtype != torch.float32
            or not qw.is_contiguous() or not kw.is_contiguous()
            or pairs not in (32,96) or not epsilon > 0
            or rope_cache.ndim != 2 or rope_cache.shape[1] != 2*pairs
            or rope_cache.dtype != torch.bfloat16 or rope_cache.stride(1) != 1
            or any(x.device != query.device for x in (packed,qw,kw,rope_cache))):
            raise ValueError("Step5 norm exchange requires raw BF16 TP2 QKV, FP32 weights and short-query metadata")
        pack_norm_exchange[(tokens,9+(0 if window else triton.cdiv(2*(topk+1),256)))](
            packed,qw,kw,rope_cache,words,dcounts,logical,pcounts,
            positions,starts,lengths,payload,requests,qpos,
            packed.stride(0),rope_cache.stride(0),
            TOKENS=tokens,REQUESTS=lengths.numel(),DECODE=nd,PREFILL=np,
            TOPK=topk,WINDOW=window,ROTARY_PAIRS=pairs,EPS=epsilon,
            CAST_NORM=False,REQ_BLOCK=triton.next_power_of_2(lengths.numel()),
            num_warps=4,num_stages=1)
    gathered = group.all_gather(payload,dim=1)
    full_q = torch.empty((tokens,64,192),device=query.device,dtype=torch.bfloat16)
    regions = torch.empty((4*tokens,topk),device=query.device,dtype=torch.int32)
    counts = torch.empty(4*tokens,device=query.device,dtype=torch.int32)
    _unpack_exchange[(tokens,triton.cdiv(13824+4*(topk+1),256))](
        gathered,slots,full_q,kc,vc,requests,qpos,regions,counts,
        cache_slots,kr,kh,kp,vr,vh,vp,TOKENS=tokens,TOPK=topk,WINDOW=window,BLOCK=256,
        num_warps=4,num_stages=1)
    return full_q,regions,counts,requests,qpos
