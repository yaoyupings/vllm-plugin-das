# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse-indexer Q/K normalization and RoPE in one launch."""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton


def short_index_query_rounding_enabled(num_tokens, proxy_dim, dtype):
    """Only fold the indexer's deployed E4M3 rounding into short BF16 norms."""
    return (
        os.environ.get("VLLM_STEP5_INDEXER_ROUND_IN_NORM", "0") == "1"
        and 0 < num_tokens <= 128
        and proxy_dim == 256
        and dtype == torch.bfloat16
    )


@triton.jit
def _fused_indexer_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_q_token,
    stride_k_token,
    stride_q_out_token,
    stride_k_out_token,
    stride_cos,
    eps,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    ROUND_Q_E4M3: tl.constexpr,
    ROUND_NORM_WEIGHTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Run Q RMSNorm and K LayerNorm/RoPE on a joint head grid.

    Programs ``[0, NUM_Q_HEADS)`` process Q heads and the last program
    processes the single K head. The arithmetic and bf16 rounding boundaries
    intentionally match ``step4_dsa_kernels.indexer_norm_rope``.
    """
    token = tl.program_id(0)
    lane = tl.program_id(1)
    is_k = lane == NUM_Q_HEADS
    q_head = lane

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    q_values = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + dims,
        mask=dim_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    k_values = tl.load(
        k_ptr + token * stride_k_token + dims,
        mask=dim_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(is_k, k_values, q_values)

    sum_sq = tl.sum(values * values)
    raw_mean = tl.sum(values) / head_dim
    mean = tl.where(is_k, raw_mean, 0.0)
    variance = tl.where(
        is_k,
        sum_sq / head_dim - raw_mean * raw_mean,
        sum_sq / head_dim,
    )
    rstd = tl.rsqrt(variance + eps)

    q_scale = tl.load(q_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    k_scale = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    k_bias = tl.load(k_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    if ROUND_NORM_WEIGHTS:
        # Preserve the original parameter.to(BF16) boundary before the affine.
        q_scale = q_scale.to(tl.bfloat16).to(tl.float32)
        k_scale = k_scale.to(tl.bfloat16).to(tl.float32)
        k_bias = k_bias.to(tl.bfloat16).to(tl.float32)
    scale = tl.where(is_k, k_scale, q_scale + 1.0)
    bias = tl.where(is_k, k_bias, 0.0)
    normalized = (values - mean) * rstd * scale + bias

    # Preserve the two deployed casts: normalized/RoPE output first rounds
    # to BF16, then to E4M3 and back. K remains unrounded for CSA compression.
    if ROUND_Q_E4M3:
        unrotated_q = (
            normalized.to(q_out_ptr.dtype.element_ty)
            .to(tl.float8e4nv)
            .to(q_out_ptr.dtype.element_ty)
        )
    else:
        unrotated_q = normalized.to(q_out_ptr.dtype.element_ty)
    unrotated = dim_ok & (dims >= 2 * rotary_dim)
    tl.store(
        q_out_ptr + token * stride_q_out_token + q_head * head_dim + dims,
        unrotated_q,
        mask=unrotated & ~is_k,
    )
    tl.store(
        k_out_ptr + token * stride_k_out_token + dims,
        normalized.to(k_out_ptr.dtype.element_ty),
        mask=unrotated & is_k,
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    q_real = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + half,
        mask=half_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    q_imag = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + rotary_dim + half,
        mask=half_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    k_real = tl.load(
        k_ptr + token * stride_k_token + half,
        mask=half_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    k_imag = tl.load(
        k_ptr + token * stride_k_token + rotary_dim + half,
        mask=half_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    real = tl.where(is_k, k_real, q_real)
    imag = tl.where(is_k, k_imag, q_imag)

    q_scale_real = tl.load(q_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    q_scale_imag = tl.load(
        q_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    k_scale_real = tl.load(k_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    k_scale_imag = tl.load(
        k_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    k_bias_real = tl.load(k_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    k_bias_imag = tl.load(
        k_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    if ROUND_NORM_WEIGHTS:
        q_scale_real = q_scale_real.to(tl.bfloat16).to(tl.float32)
        q_scale_imag = q_scale_imag.to(tl.bfloat16).to(tl.float32)
        k_scale_real = k_scale_real.to(tl.bfloat16).to(tl.float32)
        k_scale_imag = k_scale_imag.to(tl.bfloat16).to(tl.float32)
        k_bias_real = k_bias_real.to(tl.bfloat16).to(tl.float32)
        k_bias_imag = k_bias_imag.to(tl.bfloat16).to(tl.float32)
    real = (
        (real - mean)
        * rstd
        * tl.where(is_k, k_scale_real, q_scale_real + 1.0)
        + tl.where(is_k, k_bias_real, 0.0)
    ).to(q_out_ptr.dtype.element_ty).to(tl.float32)
    imag = (
        (imag - mean)
        * rstd
        * tl.where(is_k, k_scale_imag, q_scale_imag + 1.0)
        + tl.where(is_k, k_bias_imag, 0.0)
    ).to(q_out_ptr.dtype.element_ty).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(
        cos_ptr + position * stride_cos + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + position * stride_cos + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    rotated_real = real * cos - imag * sin
    rotated_imag = real * sin + imag * cos

    if ROUND_Q_E4M3:
        q_rotated_real = (
            rotated_real.to(q_out_ptr.dtype.element_ty)
            .to(tl.float8e4nv)
            .to(q_out_ptr.dtype.element_ty)
        )
        q_rotated_imag = (
            rotated_imag.to(q_out_ptr.dtype.element_ty)
            .to(tl.float8e4nv)
            .to(q_out_ptr.dtype.element_ty)
        )
    else:
        q_rotated_real = rotated_real
        q_rotated_imag = rotated_imag
    q_base = q_out_ptr + token * stride_q_out_token + q_head * head_dim
    k_base = k_out_ptr + token * stride_k_out_token
    tl.store(q_base + half, q_rotated_real, mask=half_ok & ~is_k)
    tl.store(q_base + rotary_dim + half, q_rotated_imag, mask=half_ok & ~is_k)
    tl.store(k_base + half, rotated_real, mask=half_ok & is_k)
    tl.store(k_base + rotary_dim + half, rotated_imag, mask=half_ok & is_k)


def fused_indexer_qk_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
    round_q_e4m3: bool = False,
    round_norm_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if round_norm_weights and (
        index_q.dtype != torch.bfloat16
        or index_k.dtype != torch.bfloat16
        or index_k.device != index_q.device
        or head_dim != 256
        or any(
            w.dtype != torch.float32
            or w.shape != (head_dim,)
            or not w.is_contiguous()
            or w.device != index_q.device
            for w in (q_norm_weight, k_norm_weight, k_norm_bias)
        )
    ):
        raise ValueError(
            "Fused indexer norm weight casts require contiguous FP32 weights "
            "and matching BF16 HD256 inputs"
        )
    if round_q_e4m3 and index_q.dtype != torch.bfloat16:
        raise ValueError("Fused indexer E4M3 rounding requires BF16 queries")
    if index_k.shape[1] != head_dim:
        raise ValueError(
            "Step5 fused indexer norm expects exactly one K head; "
            f"got shape {tuple(index_k.shape)} and head_dim={head_dim}"
        )
    q_out = torch.empty_like(index_q)
    k_out = torch.empty_like(index_k)
    _fused_indexer_qk_norm_rope_kernel[(index_q.shape[0], num_q_heads + 1)](
        index_q,
        index_k,
        q_out,
        k_out,
        q_norm_weight,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        positions,
        index_q.stride(0),
        index_k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos.stride(0),
        eps,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        NUM_Q_HEADS=num_q_heads,
        ROUND_Q_E4M3=round_q_e4m3,
        ROUND_NORM_WEIGHTS=round_norm_weights,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
        # One wave is enough for the 256-wide reduction and avoids the
        # multi-wave synchronization cost that made the old joint launch
        # slower at 64 verification tokens on BW1101.
        num_warps=1,
        num_stages=1,
    )
    return q_out, k_out
