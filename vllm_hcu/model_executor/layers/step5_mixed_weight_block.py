"""Mixed-precision projection weights in one byte-addressed allocation.

The FP8 QKV bytes, BF16 residual bytes and FP32 per-channel scales are copied
without numerical conversion. Views retain native KN/NK layouts. Runtime/model
integration is not implied by this standalone experimental load-time helper.
"""
from dataclasses import dataclass
import torch

def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment

@dataclass(frozen=True)
class WeightBlockPlan:
    input_size: int
    qkv_width: int
    residual_width: int
    alignment: int
    residual_byte_offset: int
    scale_byte_offset: int
    total_bytes: int

    @classmethod
    def create(cls, input_size: int, qkv_width: int, residual_width: int,
               alignment: int = 256):
        if any(type(v) is not int or v <= 0 for v in (input_size,qkv_width,residual_width)):
            raise ValueError("Positive integer projection dimensions required")
        if type(alignment) is not int or alignment < 4 or alignment & (alignment-1):
            raise ValueError("Alignment must be a power of two, at least four")
        residual_offset=_align(input_size*qkv_width,alignment)
        scale_offset=_align(residual_offset+2*input_size*residual_width,alignment)
        total=_align(scale_offset+4*qkv_width,alignment)
        return cls(input_size,qkv_width,residual_width,alignment,
                   residual_offset,scale_offset,total)

@dataclass(frozen=True)
class MixedProjectionWeightBlock:
    storage: torch.Tensor
    qkv_kn: torch.Tensor
    residual_nk: torch.Tensor
    qkv_scale: torch.Tensor
    plan: WeightBlockPlan

@torch.no_grad()
def pack_mixed_projection_weights(qkv_kn: torch.Tensor,
                                  residual_nk: torch.Tensor,
                                  qkv_scale: torch.Tensor,
                                  alignment: int = 256) -> MixedProjectionWeightBlock:
    if qkv_kn.ndim != 2 or qkv_kn.dtype != torch.float8_e4m3fn:
        raise ValueError("QKV must be FP8 FN [K,N]")
    if residual_nk.ndim != 2 or residual_nk.dtype != torch.bfloat16:
        raise ValueError("Residual indexer/gate must be BF16 [R,K]")
    if qkv_scale.dtype != torch.float32 or qkv_scale.ndim != 2:
        raise ValueError("QKV channel scales must be FP32 [N,1]")
    k,n=qkv_kn.shape
    r=residual_nk.shape[0]
    plan=WeightBlockPlan.create(k,n,r,alignment)
    if residual_nk.shape[1] != k or tuple(qkv_scale.shape) != (n,1):
        raise ValueError("Projection or channel scale geometry mismatch")
    if qkv_kn.stride() != (1,k) or not residual_nk.is_contiguous() or not qkv_scale.is_contiguous():
        raise ValueError("Native KN QKV and contiguous NK residual/scales required")
    if qkv_kn.device != residual_nk.device or qkv_kn.device != qkv_scale.device:
        raise ValueError("Packed weights and scales must be on the same device")
    if qkv_kn.requires_grad or residual_nk.requires_grad or qkv_scale.requires_grad:
        raise ValueError("Inference-only weights and scales required")
    raw=torch.zeros(plan.total_bytes,device=qkv_kn.device,dtype=torch.uint8)
    qbytes=k*n
    rbytes=2*k*r
    raw[:qbytes].copy_(qkv_kn.t().view(torch.uint8).reshape(-1))
    raw[plan.residual_byte_offset:plan.residual_byte_offset+rbytes].copy_(
        residual_nk.view(torch.uint8).reshape(-1))
    raw[plan.scale_byte_offset:plan.scale_byte_offset+4*n].copy_(
        qkv_scale.view(torch.uint8).reshape(-1))
    qview=raw[:qbytes].view(torch.float8_e4m3fn).view(n,k).t()
    rview=raw[plan.residual_byte_offset:plan.residual_byte_offset+rbytes].view(
        torch.bfloat16).view(r,k)
    sview=raw[plan.scale_byte_offset:plan.scale_byte_offset+4*n].view(
        torch.float32).view(n,1)
    return MixedProjectionWeightBlock(raw,qview,rview,sview,plan)
