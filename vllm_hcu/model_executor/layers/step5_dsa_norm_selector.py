# SPDX-License-Identifier: Apache-2.0
"""Joint causal region packing and Step5 native-carry QK norm/RoPE.

The two CTA ranges are independent. Keeping the original four-head reduction
layout preserves BF16 output bits while removing one launch. Native Top512 and
sparse FlashAttention remain separate stages.
"""
from __future__ import annotations

import torch
from lightop import op
from vllm.triton_utils import triton, tl
from .step5_dsa_lightop_selector import _prepare_scores, supports_native_selector

@triton.jit
def _pack_norm(IDS,P,T,R,O,C,LO,X,QW,KW,ROPE,QO,KO,
               QROWS:tl.constexpr,PS:tl.constexpr,TS:tl.constexpr,RS:tl.constexpr,
               OUTS:tl.constexpr,REGION:tl.constexpr,RPP:tl.constexpr,
               LOGICAL:tl.constexpr,XS:tl.constexpr,CS:tl.constexpr,ROTARY:tl.constexpr,EPS:tl.constexpr,
               NQ:tl.constexpr,NK:tl.constexpr):
    pid=tl.program_id(0)
    if pid<NK*QROWS:
        row=pid
        q=tl.load(P+(row%QROWS)*PS).to(tl.int32)
        req=tl.load(R+row*RS).to(tl.int32)
        current=tl.maximum(q//REGION,0)
        history=tl.minimum(current,512)
        slot=tl.arange(0,512)
        ids=tl.load(IDS+row*512+slot)
        ids=tl.sort(tl.where(ids>=0,ids,2147483647),descending=False)
        live=(slot<history)&(ids!=2147483647)
        safe=tl.where(live,ids,0)
        page=tl.load(T+req*TS+safe//RPP,mask=live,other=-1)
        physical=page*RPP+safe%RPP
        valid=tl.minimum(tl.maximum(q+1-safe*REGION,0),REGION)
        words=tl.where(live&(page>=0),physical|(valid<<24),-1)
        # The causal slot belongs only to the scalar store below. Otherwise
        # another lane could overwrite it with history padding.
        tl.store(O+row*OUTS+slot,words,mask=slot!=history)
        tl.store(O+row*OUTS+512,-1,mask=history!=512)
        if LOGICAL:
            tl.store(LO+row*OUTS+slot,tl.where(live,ids,-1),mask=slot!=history)
            tl.store(LO+row*OUTS+512,-1,mask=history!=512)
        page=tl.load(T+req*TS+current//RPP)
        physical=page*RPP+current%RPP
        valid=tl.minimum(tl.maximum(q+1-current*REGION,0),REGION)
        tl.store(O+row*OUTS+history,tl.where(page>=0,physical|(valid<<24),-1))
        if LOGICAL:
            tl.store(LO+row*OUTS+history,tl.where(page>=0,current,-1))
        tl.store(C+row,history+1)
    
    else:
        if ROTARY==96:
            rows=(pid-NK*QROWS)*4+tl.arange(0,4)
            token=rows//(NQ+NK);head=rows%(NQ+NK);is_q=head<NQ
            lane=tl.arange(0,128)
            half=tl.arange(0,2)
            dims=lane[:,None]+96*half[None,:]
            mask=(token[:,None,None]<QROWS)&(lane[None,:,None]<96)
            x=tl.load(X+token[:,None,None]*XS+head[:,None,None]*192+dims[None,:,:],mask=mask,other=0).to(tl.float32)
            ss=tl.div_rn(tl.sum(tl.sum(x*x,2),1),192.0)
            inv=tl.rsqrt(ss+EPS)
            wq=tl.load(QW+dims,mask=lane[:,None]<96,other=0).to(tl.float32)
            wk=tl.load(KW+dims,mask=lane[:,None]<96,other=0).to(tl.float32)
            w=tl.where(is_q[:,None,None],wq[None,:,:],wk[None,:,:])
            norm=x*inv[:,None,None]*(1.0+w)
            real,imag=tl.split(norm)
            positions=tl.load(P+token,mask=token<QROWS,other=0)
            rm=(token[:,None]<QROWS)&(lane[None,:]<96)
            c=tl.load(ROPE+positions[:,None]*CS+lane[None,:],mask=rm,other=0).to(tl.float32)
            s=tl.load(ROPE+positions[:,None]*CS+96+lane[None,:],mask=rm,other=0).to(tl.float32)
            yr=(real*c-imag*s).to(tl.bfloat16);yi=(real*s+imag*c).to(tl.bfloat16)
            qbase=token[:,None]*(NQ*192)+head[:,None]*192+lane[None,:]
            kbase=token[:,None]*(NK*192)+(head[:,None]-NQ)*192+lane[None,:]
            tl.store(QO+qbase,yr,mask=rm&is_q[:,None]);tl.store(QO+qbase+96,yi,mask=rm&is_q[:,None])
            tl.store(KO+kbase,yr,mask=rm&~is_q[:,None]);tl.store(KO+kbase+96,yi,mask=rm&~is_q[:,None])
        else:
            rows=(pid-NK*QROWS)*4+tl.arange(0,4)
            token=rows//(NQ+NK);head=rows%(NQ+NK);is_q=head<NQ
            group=tl.arange(0,4)
            lane=tl.arange(0,32)
            half=tl.arange(0,2)
            dims=group[:,None,None]*64+lane[None,:,None]+half[None,None,:]*32
            mask=(token[:,None,None,None]<QROWS)&(group[None,:,None,None]<3)
            x=tl.load(X+token[:,None,None,None]*XS+head[:,None,None,None]*192+dims[None,:,:,:],mask=mask,other=0).to(tl.float32)
            ss=tl.div_rn(tl.sum(tl.sum(tl.sum(x*x,3),2),1),192.0)
            inv=tl.rsqrt(ss+EPS)
            wq=tl.load(QW+dims,mask=group[:,None,None]<3,other=0).to(tl.float32)
            wk=tl.load(KW+dims,mask=group[:,None,None]<3,other=0).to(tl.float32)
            w=tl.where(is_q[:,None,None,None],wq[None,:,:,:],wk[None,:,:,:])
            norm=x*inv[:,None,None,None]*(1.0+w)
            real,imag=tl.split(norm)
            positions=tl.load(P+token,mask=token<QROWS,other=0)
            c=tl.load(ROPE+positions[:,None]*CS+lane[None,:],mask=token[:,None]<QROWS,other=0).to(tl.float32)
            s=tl.load(ROPE+positions[:,None]*CS+32+lane[None,:],mask=token[:,None]<QROWS,other=0).to(tl.float32)
            rot_real=real*c[:,None,:]-imag*s[:,None,:]
            rot_imag=imag*c[:,None,:]+real*s[:,None,:]
            yr=tl.where(group[None,:,None]==0,rot_real,real).to(tl.bfloat16)
            yi=tl.where(group[None,:,None]==0,rot_imag,imag).to(tl.bfloat16)
            rm=(token[:,None,None]<QROWS)&(group[None,:,None]<3)
            qbase=token[:,None,None]*(NQ*192)+head[:,None,None]*192+group[None,:,None]*64+lane[None,None,:]
            kbase=token[:,None,None]*(NK*192)+(head[:,None,None]-NQ)*192+group[None,:,None]*64+lane[None,None,:]
            tl.store(QO+qbase,yr,mask=rm&is_q[:,None,None])
            tl.store(QO+qbase+32,yi,mask=rm&is_q[:,None,None])
            tl.store(KO+kbase,yr,mask=rm&~is_q[:,None,None])
            tl.store(KO+kbase+32,yi,mask=rm&~is_q[:,None,None])

_IDENTITIES: dict[torch.device, torch.Tensor] = {}


def supports_main_norm_selector(logits, positions, requests, main_norm):
    qkv, qw, kw, cache, rotary_pairs, epsilon = main_norm
    m = positions.numel()
    nk = {3456: 1, 6912: 2}.get(qkv.shape[1], 0) if qkv.ndim == 2 else 0
    return (
        supports_native_selector(
            logits, positions, requests, topk=512, num_kv_groups=nk
        )
        and 1 <= m <= 64 and positions.stride(0) == 1
        and nk in (1, 2) and qkv.ndim == 2 and qkv.shape == (m, nk * 3456)
        and qkv.dtype == torch.bfloat16 and qkv.stride(1) == 1
        and qw.shape == kw.shape == (192,)
        and qw.dtype == kw.dtype == torch.float32
        and qw.stride(0) == kw.stride(0) == 1
        and rotary_pairs in (32, 96)
        and cache.ndim == 2 and cache.shape[1] == 2 * rotary_pairs
        and cache.stride(1) == 1 and cache.dtype == torch.bfloat16
        and all(t.device == logits.device for t in (qkv, qw, kw, cache))
    )


def prefill_meta_with_main_norm(
    logits, positions, table, requests, main_norm, *, regions_per_page,
):
    if not supports_main_norm_selector(logits, positions, requests, main_norm):
        raise ValueError("Unsupported joint Step5 causal pack/QK norm shape")
    if table.dtype != torch.int32 or table.device != logits.device or table.stride(1) != 1:
        raise ValueError("Step5 causal pack requires an int32 block table")
    qkv, qw, kw, cache, rotary_pairs, epsilon = main_norm
    rows, width = logits.shape
    m = positions.numel()
    identity = _IDENTITIES.get(logits.device)
    if identity is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 joint selector requires an uncaptured warmup")
        identity = torch.zeros((1, 1), device=logits.device, dtype=torch.int32)
        _IDENTITIES[logits.device] = identity
    lengths = torch.empty(rows, device=logits.device, dtype=torch.int32)
    ids = torch.empty((rows, 512), device=logits.device, dtype=torch.int32)
    normalized = torch.empty_like(logits)
    packed = torch.empty((rows, 513), device=logits.device, dtype=torch.int32)
    counts = torch.empty(rows, device=logits.device, dtype=torch.int32)
    nk = qkv.shape[1] // 3456
    nq = nk * 16
    q = qkv.new_empty((m, nq * 192))
    k = qkv.new_empty((m, nk * 192))
    _prepare_scores[(rows, triton.cdiv(width, 1024))](
        logits, normalized, positions, lengths,
        rows, m, positions.stride(0), width, 8, 1024, num_warps=4,
    )
    op.topk_transform_512(
        normalized, lengths, identity.expand(rows, 1), ids,
        triton.next_power_of_2(max(1, width)), None,
    )
    _pack_norm[(rows + triton.cdiv(m * (nq + nk), 4),)](
        ids, positions, table, requests, packed, counts, packed,
        qkv, qw, kw, cache, q, k,
        m, positions.stride(0), table.stride(0), requests.stride(0),
        packed.stride(0), 8, regions_per_page, False,
        qkv.stride(0), cache.stride(0), rotary_pairs, epsilon, nq, nk,
        num_warps=4, enable_fp_fusion=True,
    )
    return packed, counts, q, k
