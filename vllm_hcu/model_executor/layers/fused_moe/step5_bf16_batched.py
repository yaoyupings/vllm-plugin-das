# SPDX-License-Identifier: Apache-2.0
"""Device-counted BF16 expert tiles without a grid over padded capacity."""
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    expert_triton_kernel,
)


@triton.jit
def _step5_bf16_experts_kernel(
    A, B, C, counts, K: tl.constexpr, N: tl.constexpr,
    stride_ae: tl.constexpr, stride_am: tl.constexpr,
    stride_ak: tl.constexpr, stride_be: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_ce: tl.constexpr, stride_cm: tl.constexpr,
    stride_cn: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_tokens = tl.load(counts + expert)
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    cta_n_start = pid_n * BLOCK_N
    cta_n_size = min(BLOCK_N, N - cta_n_start)
    # Dispatch counts are device tensors and can change on each graph replay.
    # Reuse the upstream tile calculation exactly, visiting only live M tiles.
    for m_start in range(0, num_tokens, BLOCK_M):
        expert_triton_kernel(
            A + expert * stride_ae + m_start * stride_am,
            B + expert * stride_be + cta_n_start * stride_bn,
            C + expert * stride_ce + m_start * stride_cm
              + cta_n_start * stride_cn,
            expert, tl.bfloat16, min(BLOCK_M, num_tokens - m_start),
            cta_n_size, K, None, None, None,
            stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn, 0, 0, 0, 0, 0, 0,
            offs_bn, 0, 0, False, False, False,
            BLOCK_M, BLOCK_N, BLOCK_K, False,
        )


def invoke_step5_bf16_experts(A, B, C, expert_num_tokens, config):
    """Compute live rows; preserve strides, padding, and native BF16 arithmetic."""
    # The native invoke uses Triton's defaults (4 warps, 3 stages), rather
    # than forwarding the config's num_warps/num_stages. Preserve that launch.
    _step5_bf16_experts_kernel[
        (expert_num_tokens.numel(), triton.cdiv(B.shape[1], config["BLOCK_SIZE_N"]))
    ](
        A, B, C, expert_num_tokens, A.shape[2], C.shape[2],
        *A.stride(), B.stride(0), B.stride(2), B.stride(1), *C.stride(),
        BLOCK_M=config["BLOCK_SIZE_M"], BLOCK_N=config["BLOCK_SIZE_N"],
        BLOCK_K=config["BLOCK_SIZE_K"], num_warps=4, num_stages=3,
    )
