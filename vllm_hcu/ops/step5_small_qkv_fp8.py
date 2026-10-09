# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 small-batch QKV activation quantization and FP8 GEMM."""
import torch,triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger

logger = init_logger("vllm.hcu.step5.small_qkv_fp8")
import triton.language as tl

@triton.jit
def _token_amax(x, rows, M:tl.constexpr,K:tl.constexpr,stride_m:tl.constexpr,BA:tl.constexpr):
    kk=tl.arange(0,BA)
    amax=tl.full(rows.shape,0,tl.float32)
    for b in range(triton.cdiv(K,BA)):
        k=b*BA+kk
        values=tl.load(x+rows[:,None]*stride_m+k[None,:],
                       (rows<M)[:,None]&(k<K)[None,:],0).to(tl.float32)
        amax=tl.maximum(amax,tl.max(tl.abs(values),1))
    return amax

@triton.jit
def _quant_debug(x,q,scale,M:tl.constexpr,K:tl.constexpr,stride_m:tl.constexpr,
                 BM:tl.constexpr,BA:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    amax=_token_amax(x,rows,M,K,stride_m,BA)
    s=tl.div_rn(amax,448.)
    inv=tl.where(amax>0,tl.div_rn(448.,amax),0.)
    tl.store(scale+rows,s,rows<M)
    for b in range(triton.cdiv(K,BK)):
        k=b*BK+tl.arange(0,BK)
        a=tl.load(x+rows[:,None]*stride_m+k[None,:],
                  (rows<M)[:,None]&(k<K)[None,:],0).to(tl.float32)
        aq=(a*inv[:,None]).to(tl.float8e4nv)
        tl.store(q+rows[:,None]*K+k[None,:],aq,
                 (rows<M)[:,None]&(k<K)[None,:])

@triton.jit
def _fused_qkv(x,w,ws,out,bias,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
               stride_x:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
               BA:tl.constexpr,BIAS:tl.constexpr,STAGES:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    amax=_token_amax(x,rows,M,K,stride_x,BA)
    scale=tl.div_rn(amax,448.)
    inv=tl.where(amax>0,tl.div_rn(448.,amax),0.)
    acc=tl.zeros((BM,BN),tl.float32)
    for ki in tl.range(0,triton.cdiv(K,BK),num_stages=STAGES):
        k=ki*BK+rk
        a=tl.load(x+rows[:,None]*stride_x+k[None,:],
                  (rows<M)[:,None]&(k<K)[None,:],0).to(tl.float32)
        aq=(a*inv[:,None]).to(tl.float8e4nv)
        b=tl.load(w+k[:,None]+cols[None,:]*K,
                  (k<K)[:,None]&(cols<N)[None,:],0.)
        acc=tl.dot(aq,b,acc)
    bscale=tl.load(ws+cols,cols<N,0.)
    result=(acc*bscale[None,:])*scale[:,None]
    if BIAS:
        result+=tl.load(bias+cols,cols<N,0.)[None,:]
    tl.store(out+rows[:,None]*N+cols[None,:],result.to(out.dtype.element_ty),
             (rows<M)[:,None]&(cols<N)[None,:])

def quant_debug(x, bm=16,ba=512):
    m,k=x.shape
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    s=torch.empty((m,1),device=x.device,dtype=torch.float32)
    _quant_debug[(triton.cdiv(m,bm),)](x,q,s,M=m,K=k,stride_m=x.stride(0),
        BM=bm,BA=ba,BK=128,num_warps=4)
    return q,s

def fused_qkv(x,w,ws,bias=None,*,bm=16,bn=64,bk=128,ba=512,warps=4,stages=2,out=None):
    m,k=x.shape;n=w.shape[1]
    if out is None:out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    _fused_qkv[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
        x,w,ws,out,bias,M=m,N=n,K=k,stride_x=x.stride(0),
        BM=bm,BN=bn,BK=bk,BA=ba,BIAS=bias is not None,STAGES=stages,num_warps=warps)
    return out

_SMALL_M_SHAPES = (1, 4, 8, 16, 32, 64)

def small_qkv_shape_supported(tokens):
    return tokens in _SMALL_M_SHAPES

def _validate_runtime_layout(x, weight, weight_scale):
    m, k = x.shape
    if (x.device.type != "cuda" or m < 0 or k != 4096 or weight.shape != (4096, 3456)
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn
            or weight_scale.dtype != torch.float32 or weight_scale.shape != (3456, 1)
            or not x.is_contiguous() or weight.stride() != (1, 4096)
            or not weight_scale.is_contiguous()
            or x.device != weight.device or x.device != weight_scale.device):
        raise ValueError("Step5 small QKV requires qualified TP4 FP8 layouts and token shapes")


def _native_qkv(x, weight, weight_scale):
    # These exact operators are registered during QKV finalization.  The
    # fallback retains the existing LightOp activation and channel FP8 GEMM.
    quantized, scale = torch.ops.vllm.lightop_per_token_quant_fp8(x)
    return torch.ops.vllm.hcu_channel_fp8_target_triton_scaled_mm(
        quantized, weight, scale, weight_scale, torch.bfloat16, None,
    )


def _run_small_qkv(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    _validate_runtime_layout(x, weight, weight_scale)
    m = x.shape[0]
    if not small_qkv_shape_supported(m):
        if m == 0:
            return torch.empty((0, weight.shape[1]), device=x.device, dtype=torch.bfloat16)
        return _native_qkv(x, weight, weight_scale)
    logger.info_once("Step5 TP4 small QKV runtime fused path: M=%d", m, scope="process")
    return fused_qkv(x, weight, weight_scale, bm=16, bn=128,
                     bk=128 if m == 64 else 256, ba=512, warps=4, stages=2)

def _fake_small_qkv(x, weight, weight_scale):
    return torch.empty((x.shape[0], weight.shape[1]), device=x.device,
                       dtype=torch.bfloat16)

_registered = False
def ensure_registered():
    global _registered
    if not _registered:
        direct_register_custom_op(op_name="step5_tp4_small_qkv_fp8",
            op_func=_run_small_qkv, mutates_args=[], fake_impl=_fake_small_qkv)
        _registered = True
