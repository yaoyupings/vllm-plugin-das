# SPDX-License-Identifier: Apache-2.0
"""Step5 short-query QK norm/RoPE fused into the DCP2 payload prologue."""
import torch
import triton
import triton.language as tl

@triton.jit
def pack_norm_exchange(qkv, qw, kw, cache, words, dcounts, logical, pcounts,
                       positions, starts, lengths, payload, requests, qpos,
                       X_ROW, CACHE_ROW,
                       TOKENS: tl.constexpr, REQUESTS: tl.constexpr,
                       DECODE: tl.constexpr, PREFILL: tl.constexpr,
                       TOPK: tl.constexpr, WINDOW: tl.constexpr,
                       ROTARY_PAIRS: tl.constexpr, EPS: tl.constexpr,
                       CAST_NORM: tl.constexpr, REQ_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    BLOCK: tl.constexpr = 256
    qwords: tl.constexpr = 3456
    pwidth: tl.constexpr = qwords+(0 if WINDOW else 2*(TOPK+1))
    d = qwords+(block-9)*BLOCK+tl.arange(0,BLOCK)
    if block < 9:
        head = block*4+tl.arange(0,4)
        dim = tl.arange(0,256)
        xx = tl.load(qkv+row*X_ROW+head[:,None]*192+dim[None,:],
                     mask=(head[:,None]<36)&(dim[None,:]<192),other=0).to(tl.float32)
        variance = tl.div_rn(tl.sum(xx*xx,1),192.)
        inv = tl.rsqrt(variance+EPS)
        wq = tl.load(qw+dim,mask=dim<192,other=0).to(tl.float32)
        wk = tl.load(kw+dim,mask=dim<192,other=0).to(tl.float32)
        weight = tl.where(head[:,None]<32,wq[None,:],wk[None,:])
        norm = tl.where(head[:,None]<34,xx*inv[:,None]*(1.+weight),xx)
        if CAST_NORM:
            norm = norm.to(tl.bfloat16).to(tl.float32)
        partner_dim = tl.where(dim<ROTARY_PAIRS,dim+ROTARY_PAIRS,
                      tl.where(dim<2*ROTARY_PAIRS,dim-ROTARY_PAIRS,dim))
        indices = tl.broadcast_to(partner_dim[None,:],(4,256))
        partner = tl.gather(norm,indices,1)
        position = tl.load(positions+row)
        cache_position = tl.maximum(position,0)
        cos = tl.load(cache+cache_position*CACHE_ROW+dim%ROTARY_PAIRS).to(tl.float32)
        sin = tl.load(cache+cache_position*CACHE_ROW+ROTARY_PAIRS+dim%ROTARY_PAIRS).to(tl.float32)
        cos = tl.where(position>=0,cos,1.)
        sin = tl.where(position>=0,sin,0.)
        rotated = norm*cos[None,:]+partner*tl.where(dim<ROTARY_PAIRS,-sin,sin)[None,:]
        result = tl.where((head[:,None]<34)&(dim[None,:]<2*ROTARY_PAIRS),
                          rotated,norm).to(tl.bfloat16)
        pair = tl.arange(0,128)
        lo = tl.gather(result,tl.broadcast_to((2*pair)[None,:],(4,128)),1)
        hi = tl.gather(result,tl.broadcast_to((2*pair+1)[None,:],(4,128)),1)
        packed = lo.to(tl.uint16,bitcast=True).to(tl.uint32)|(
                 hi.to(tl.uint16,bitcast=True).to(tl.uint32)<<16)
        tl.store(payload+row*pwidth+head[:,None]*96+pair[None,:],
                 packed.to(tl.int32,bitcast=True),
                 mask=(head[:,None]<36)&(pair[None,:]<96))
    if not WINDOW and block >= 9:
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
