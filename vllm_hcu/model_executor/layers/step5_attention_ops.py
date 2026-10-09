# SPDX-License-Identifier: Apache-2.0
"""Fused Step5 QK RMSNorm/RoPE for the native Inductor FP32 carry path."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op

@triton.jit
def _qknorm_rope_full(packed,qw,kw,cache,pos,qo,ko,stride_x,stride_c,tokens,
                NQ:tl.constexpr,NK:tl.constexpr,ROWS:tl.constexpr,EPS:tl.constexpr):
 rows=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
 token=rows//(NQ+NK);head=rows%(NQ+NK);is_q=head<NQ
 lane=tl.arange(0,128)
 half=tl.arange(0,2)
 dims=lane[:,None]+96*half[None,:]
 mask=(token[:,None,None]<tokens)&(lane[None,:,None]<96)
 x=tl.load(packed+token[:,None,None]*stride_x+head[:,None,None]*192+dims[None,:,:],mask=mask,other=0).to(tl.float32)
 ss=tl.div_rn(tl.sum(tl.sum(x*x,2),1),192.0)
 inv=tl.rsqrt(ss+EPS)
 wq=tl.load(qw+dims,mask=lane[:,None]<96,other=0).to(tl.float32)
 wk=tl.load(kw+dims,mask=lane[:,None]<96,other=0).to(tl.float32)
 w=tl.where(is_q[:,None,None],wq[None,:,:],wk[None,:,:])
 norm=x*inv[:,None,None]*(1.0+w)
 real,imag=tl.split(norm)
 positions=tl.load(pos+token,mask=token<tokens,other=0)
 rm=(token[:,None]<tokens)&(lane[None,:]<96)
 c=tl.load(cache+positions[:,None]*stride_c+lane[None,:],mask=rm,other=0).to(tl.float32)
 s=tl.load(cache+positions[:,None]*stride_c+96+lane[None,:],mask=rm,other=0).to(tl.float32)
 yr=(real*c-imag*s).to(tl.bfloat16);yi=(real*s+imag*c).to(tl.bfloat16)
 qbase=token[:,None]*(NQ*192)+head[:,None]*192+lane[None,:]
 kbase=token[:,None]*(NK*192)+(head[:,None]-NQ)*192+lane[None,:]
 tl.store(qo+qbase,yr,mask=rm&is_q[:,None]);tl.store(qo+qbase+96,yi,mask=rm&is_q[:,None])
 tl.store(ko+kbase,yr,mask=rm&~is_q[:,None]);tl.store(ko+kbase+96,yi,mask=rm&~is_q[:,None])


@triton.jit
def _qknorm_rope_partial(packed,qw,kw,cache,pos,qo,ko,stride_x,stride_c,tokens,
             NQ:tl.constexpr,NK:tl.constexpr,ROWS:tl.constexpr,EPS:tl.constexpr):
 rows=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
 token=rows//(NQ+NK);head=rows%(NQ+NK);is_q=head<NQ
 group=tl.arange(0,4)
 lane=tl.arange(0,32)
 half=tl.arange(0,2)
 dims=group[:,None,None]*64+lane[None,:,None]+half[None,None,:]*32
 mask=(token[:,None,None,None]<tokens)&(group[None,:,None,None]<3)
 x=tl.load(packed+token[:,None,None,None]*stride_x+head[:,None,None,None]*192+dims[None,:,:,:],mask=mask,other=0).to(tl.float32)
 ss=tl.div_rn(tl.sum(tl.sum(tl.sum(x*x,3),2),1),192.0)
 inv=tl.rsqrt(ss+EPS)
 wq=tl.load(qw+dims,mask=group[:,None,None]<3,other=0).to(tl.float32)
 wk=tl.load(kw+dims,mask=group[:,None,None]<3,other=0).to(tl.float32)
 w=tl.where(is_q[:,None,None,None],wq[None,:,:,:],wk[None,:,:,:])
 norm=x*inv[:,None,None,None]*(1.0+w)
 real,imag=tl.split(norm)
 positions=tl.load(pos+token,mask=token<tokens,other=0)
 c=tl.load(cache+positions[:,None]*stride_c+lane[None,:],mask=token[:,None]<tokens,other=0).to(tl.float32)
 s=tl.load(cache+positions[:,None]*stride_c+32+lane[None,:],mask=token[:,None]<tokens,other=0).to(tl.float32)
 rot_real=real*c[:,None,:]-imag*s[:,None,:]
 rot_imag=imag*c[:,None,:]+real*s[:,None,:]
 yr=tl.where(group[None,:,None]==0,rot_real,real).to(tl.bfloat16)
 yi=tl.where(group[None,:,None]==0,rot_imag,imag).to(tl.bfloat16)
 rm=(token[:,None,None]<tokens)&(group[None,:,None]<3)
 qbase=token[:,None,None]*(NQ*192)+head[:,None,None]*192+group[None,:,None]*64+lane[None,None,:]
 kbase=token[:,None,None]*(NK*192)+(head[:,None,None]-NQ)*192+group[None,:,None]*64+lane[None,None,:]
 tl.store(qo+qbase,yr,mask=rm&is_q[:,None,None])
 tl.store(qo+qbase+32,yi,mask=rm&is_q[:,None,None])
 tl.store(ko+kbase,yr,mask=rm&~is_q[:,None,None])
 tl.store(ko+kbase+32,yi,mask=rm&~is_q[:,None,None])


def qknorm_rope_impl(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, positions: torch.Tensor,
    num_q_heads: int, num_kv_heads: int, rotary_pairs: int, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = (num_q_heads + 2 * num_kv_heads) * 192
    if qkv.ndim != 2 or qkv.shape[1] != width or qkv.stride(1) != 1:
        raise ValueError("Step5 fused norm requires TP-local packed QKV with head_dim=192")
    if qkv.dtype != torch.bfloat16 or q_weight.dtype != torch.float32 or k_weight.dtype != torch.float32:
        raise ValueError("Step5 fused norm requires BF16 activations and FP32 norm weights")
    if rotary_pairs not in (32, 96):
        raise ValueError("Step5 fused norm requires 64- or 192-dimensional NeoX rotation")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Step5 fused norm requires positive head counts")
    if q_weight.shape != (192,) or k_weight.shape != (192,) or q_weight.stride(0) != 1 or k_weight.stride(0) != 1:
        raise ValueError("Step5 fused norm requires contiguous 192-element norm weights")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 2 * rotary_pairs or cos_sin_cache.dtype != qkv.dtype or cos_sin_cache.stride(1) != 1:
        raise ValueError("Step5 fused norm requires an activation-typed cosine/sine cache")
    if positions.ndim != 1 or positions.shape[0] != qkv.shape[0] or positions.stride(0) != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Step5 fused norm requires one contiguous integer position per token")
    if not qkv.is_cuda or any(t.device != qkv.device for t in (q_weight,k_weight,cos_sin_cache,positions)):
        raise ValueError("Step5 fused norm requires all tensors on the same GPU")
    tokens = qkv.shape[0]
    q = qkv.new_empty((tokens, num_q_heads * 192))
    k = qkv.new_empty((tokens, num_kv_heads * 192))
    if tokens:
        rows = 4 if tokens <= 64 else (16 if tokens <= 1024 else (32 if rotary_pairs == 96 else 64))
        kernel = _qknorm_rope_full if rotary_pairs == 96 else _qknorm_rope_partial
        # The compiled native reference keeps normalized values in FP32 until
        # the final BF16 store. An intermediate BF16 cast changes its arithmetic.
        kernel[(triton.cdiv(tokens * (num_q_heads + num_kv_heads),rows),)](
            qkv,q_weight,k_weight,cos_sin_cache,positions,q,k,
            qkv.stride(0),cos_sin_cache.stride(0),tokens,
            NQ=num_q_heads,NK=num_kv_heads,ROWS=rows,EPS=epsilon,
            num_warps=4,enable_fp_fusion=True,
        )
    return q,k


def _fake(qkv: torch.Tensor,q_weight: torch.Tensor,k_weight: torch.Tensor,
          cos_sin_cache: torch.Tensor,positions: torch.Tensor,
          num_q_heads: int,num_kv_heads: int,rotary_pairs: int,
          epsilon: float) -> tuple[torch.Tensor,torch.Tensor]:
    return (qkv.new_empty((qkv.shape[0],num_q_heads*192)),
            qkv.new_empty((qkv.shape[0],num_kv_heads*192)))


direct_register_custom_op(
    op_name="step5_qknorm_rope",op_func=qknorm_rope_impl,
    mutates_args=[],fake_impl=_fake,
)


def fused_qknorm_rope(qkv: torch.Tensor,q_weight: torch.Tensor,
                     k_weight: torch.Tensor,cos_sin_cache: torch.Tensor,
                     positions: torch.Tensor,num_q_heads: int,
                     num_kv_heads: int,rotary_pairs: int,
                     epsilon: float) -> tuple[torch.Tensor,torch.Tensor]:
    return torch.ops.vllm.step5_qknorm_rope(
        qkv,q_weight,k_weight,cos_sin_cache,positions,
        num_q_heads,num_kv_heads,rotary_pairs,epsilon,
    )
