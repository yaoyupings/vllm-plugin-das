# SPDX-License-Identifier: Apache-2.0
"""Step5 Gemma norm, explicit BF16 narrowing and E4M3 token quantization.

Explicit layouts preserve qualified per-row FP32 reduction orders. The auto
profile chooses from tensor shape metadata; model adoption requires precision
and performance qualification.
The operator retains a BF16 hidden tensor for the indexer and gate, and leaves
the FP32 residual and norm weight untouched. The model candidate is opt-in.
"""

import math

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from vllm.utils.torch_utils import direct_register_custom_op

# rows/CTA, norm columns, columns/thread, row warps, total warps.
# Each native profile defines a per-row reduction order on gfx938 wave64.
_LAYOUTS = (
    (1, 2048, 4, 1, 8),
    (1, 4096, 8, 1, 8),
    (2, 2048, 8, 2, 8),
    (4, 2048, 8, 2, 8),
    (1, 2048, 8, 1, 4),
)

@g.jit
def fused_norm_quant(X,W,H,Q,S,M:gl.constexpr,EPS:gl.constexpr,ROWS:gl.constexpr,
                     RBLOCK:gl.constexpr,SPT:gl.constexpr,WROW:gl.constexpr,
                     WARPS:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,SPT],[1,64],[WROW,WARPS//WROW],[1,0])
    row=gl.program_id(0)*ROWS+gl.arange(0,ROWS,layout=gl.SliceLayout(1,layout))
    r=gl.arange(0,RBLOCK,layout=gl.SliceLayout(0,layout))
    acc=gl.full((ROWS,RBLOCK),0.,gl.float32,layout)
    for offset in range(0,4096,RBLOCK):
        x=gl.load(X+row[:,None]*4096+offset+r[None,:],row[:,None]<M,other=0.).to(gl.float32)
        acc=gl.where(row[:,None]<M,acc+x*x,acc)
    variance=gl.sum(acc,1)/4096.
    i=gl.arange(0,4096,layout=gl.SliceLayout(0,layout))
    x=gl.load(X+row[:,None]*4096+i[None,:],row[:,None]<M,other=0.).to(gl.float32)
    w=gl.load(W+i)
    n=(x*gl.rsqrt(variance[:,None]+EPS))*(w[None,:]+1.)
    h=n.to(gl.bfloat16)
    a=h.to(gl.float32)
    maximum=gl.max(gl.abs(a),1)
    scale=gl.div_rn(maximum,448.)
    inverse=gl.where(maximum>0.,gl.div_rn(1.,scale),0.)
    q=gl.minimum(gl.maximum(a*inverse[:,None],-448.),448.).to(gl.float8e4nv)
    gl.store(H+row[:,None]*4096+i[None,:],h,row[:,None]<M)
    gl.store(Q+row[:,None]*4096+i[None,:],q,row[:,None]<M)
    gl.store(S+row,scale,row<M)


def _validate(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, layout: int
) -> None:
    if layout < -1 or layout >= len(_LAYOUTS):
        raise ValueError("Unqualified Step5 norm/quant layout")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Step5 norm epsilon must be finite and positive")
    if (
        x.ndim != 2 or x.shape[1] != 4096 or x.dtype not in (torch.float32, torch.bfloat16)
        or not x.is_contiguous() or not x.is_cuda
    ):
        raise ValueError("Step5 norm/quant requires contiguous FP32/BF16 GPU [M,4096]")
    if (
        weight.shape != (4096,) or weight.dtype != torch.float32
        or not weight.is_contiguous() or weight.device != x.device
    ):
        raise ValueError("Step5 norm/quant requires matching FP32 [4096] weight")


def _fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, layout: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(x, weight, epsilon, layout)
    return (
        torch.empty_like(x, dtype=torch.bfloat16),
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


def _auto_layout(num_tokens: int, dtype: torch.dtype) -> int:
    # BF16 draft and FP32 residual norms select different native reductions.
    # Profiles 2/3 share profile 4's per-row order. BF16 M32/128/256 use
    # profile 0 instead; reusing the FP32 table changes near-tie BF16 rounding.
    if dtype == torch.bfloat16:
        return 4 if num_tokens in (4, 8, 16, 64) else 0
    return 4 if num_tokens == 4 or 16 < num_tokens <= 256 else 0


def _impl(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, layout: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(x, weight, epsilon, layout)
    arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "")
    if arch.split(":")[0] != "gfx938":
        raise ValueError("Step5 explicit-layout norm/quant is qualified only on gfx938")
    h, q, scale = _fake(x, weight, epsilon, layout)
    if x.shape[0]:
        if layout == -1:
            # Shape metadata only; no device count or tensor value is read.
            layout = _auto_layout(x.shape[0], x.dtype)
        rows, rblock, spt, wrow, warps = _LAYOUTS[layout]
        fused_norm_quant[(triton.cdiv(x.shape[0], rows),)](
            x, weight, h, q, scale, x.shape[0], epsilon, rows, rblock, spt, wrow, warps,
            num_warps=warps, num_stages=1, enable_fp_fusion=True,
        )
    return h, q, scale


direct_register_custom_op(
    op_name="step5_gemma_norm_fp8_quant",
    op_func=_impl,
    mutates_args=[],
    fake_impl=_fake,
)


def step5_gemma_norm_fp8_quant(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, layout: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 hidden, E4M3FN hidden and FP32 [M,1] scales.

    Layout -1 chooses a profile from host shape and dtype metadata.
    BF16 draft inputs are promoted inside the kernel; norm weights remain FP32.
    Layout IDs 0/1/2/3 match respectively rows1/R2048, rows1/R4096,
    rows2/R2048 and rows4/R2048 native norm profiles. Layout4 retains
    the per-row order of profiles2/3 with one row and four warps per CTA,
    increasing the grid for small Decode batches. Zero rows have scale0.
    CPU execution and casting FP32 norm weights are deliberately unsupported.
    """
    return torch.ops.vllm.step5_gemma_norm_fp8_quant(
        x, weight, epsilon, layout
    )
