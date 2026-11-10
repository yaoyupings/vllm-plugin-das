# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny bf16 GEMM for the lm_head at decode batch sizes.

For M <= 64 the projection is a pure weight stream (N x K bf16). hipBLASLt picks a
256x256 macro tile there and reaches ~0.93 TB/s on gfx938; this kernel tiles only over
N with a full-M block and reaches 1.4-1.7 TB/s (Step5 lm_head shard 32224 x 4096:
285 -> 153 us at M=16, 190 us at M=64).

Disable with STEP5_SKINNY_LM_HEAD=0.

skinny_top_pairs() fuses the vocab-parallel argmax of the draft (get_top_tokens) into the
same kernel, so the logits are never written (Step5 MTP draft, M=32: 186 -> 161 us);
disable with STEP5_SKINNY_TOP_TOKENS=0.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.platforms import envs as henvs

_ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("STEP5_SKINNY_LM_HEAD", "1") != "0"
)
# Fused lm_head + argmax for the vocab-parallel draft argmax (get_top_tokens).
TOP_PAIRS = _ENABLED and os.environ.get("STEP5_SKINNY_TOP_TOKENS", "1") != "0"
MAX_M = 64
_BK = 128


@triton.jit
def _skinny_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pid * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0
        )
        wt = tl.load(
            w_ptr + rn[None, :] * K + rk[:, None], mask=nmask[None, :], other=0.0
        )
        acc += tl.dot(x, wt)
    tl.store(
        o_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.bfloat16),
        mask=mmask[:, None] & nmask[None, :],
    )


@triton.jit
def _skinny_argmax_kernel(
    x_ptr,
    w_ptr,
    pv_ptr,
    pi_ptr,
    M,
    N,
    NV,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # _skinny_kernel's tiling (so the bf16 logits are bit-identical), then the tile's
    # per-row max and first argmax over columns < NV as partials [M, num_programs(0)]
    pid = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pid * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0
        )
        wt = tl.load(
            w_ptr + rn[None, :] * K + rk[:, None], mask=nmask[None, :], other=0.0
        )
        acc += tl.dot(x, wt)
    v = acc.to(tl.bfloat16).to(tl.float32)
    v = tl.where((rn < NV)[None, :], v, float("-inf"))
    mx = tl.max(v, axis=1)
    am = tl.min(tl.where(v == mx[:, None], rn[None, :], 2147483647), axis=1)
    nb = tl.num_programs(0)
    tl.store(pv_ptr + rm * nb + pid, mx, mask=mmask)
    tl.store(pi_ptr + rm * nb + pid, am, mask=mmask)


@triton.jit
def _argmax_pairs_kernel(
    pv_ptr,
    pi_ptr,
    out_ptr,
    NB,
    off,
    rank,
    TP: tl.constexpr,
    BLK: tl.constexpr,
):
    # row m: (max, first argmax + off) over the partials into out[m, rank], zeros in the
    # other ranks' slots, so an all-reduce sum of out gathers every rank's pair
    m = tl.program_id(0)
    b = tl.arange(0, BLK)
    bm = b < NB
    v = tl.load(pv_ptr + m * NB + b, mask=bm, other=float("-inf"))
    i = tl.load(pi_ptr + m * NB + b, mask=bm, other=2147483647)
    mx = tl.max(v, axis=0)
    am = tl.min(tl.where(v == mx, i, 2147483647), axis=0)
    r = tl.arange(0, TP)
    o = out_ptr + m * TP * 2 + r * 2
    tl.store(o, tl.where(r == rank, mx, 0.0))
    tl.store(o + 1, tl.where(r == rank, (am + off).to(tl.float32), 0.0))


def _tiles(M):
    if M <= 16:
        return 16, 64, 2
    if M <= 32:
        return 32, 64, 2
    return 64, 128, 4


def usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and x.shape[-1] == weight.shape[1]
        and weight.shape[1] % _BK == 0
    )


def skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [..., K] @ weight[N, K].T in bf16 (fp32 accumulate); requires usable()."""
    N, K = weight.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BM, BN, nw = _tiles(M)
    _skinny_kernel[(triton.cdiv(N, BN), triton.cdiv(M, BM))](
        x2, weight, out, M, N, K, BM=BM, BN=BN, BK=_BK, num_warps=nw, num_stages=1
    )
    return out.reshape(*x.shape[:-1], N)


def skinny_top_pairs(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_valid: int,
    vocab_start: int,
    rank: int,
    tp: int,
) -> torch.Tensor:
    """Per-row (max bf16 logit, global argmax) of x @ weight[:n_valid].T as fp32 [M, tp, 2]
    with only slot `rank` filled (the rest 0); requires usable(). Same values and
    lowest-index tie-break as logits.max(-1) on skinny_linear's output."""
    N, K = weight.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    BM, BN, nw = _tiles(M)
    nb = triton.cdiv(N, BN)
    pv = torch.empty((M, nb), dtype=torch.float32, device=x.device)
    pi = torch.empty((M, nb), dtype=torch.int32, device=x.device)
    out = torch.empty((M, tp, 2), dtype=torch.float32, device=x.device)
    _skinny_argmax_kernel[(nb, triton.cdiv(M, BM))](
        x2,
        weight,
        pv,
        pi,
        M,
        N,
        n_valid,
        K,
        BM=BM,
        BN=BN,
        BK=_BK,
        num_warps=nw,
        num_stages=1,
    )
    _argmax_pairs_kernel[(M,)](
        pv, pi, out, nb, vocab_start, rank, TP=tp, BLK=triton.next_power_of_2(nb)
    )
    return out
