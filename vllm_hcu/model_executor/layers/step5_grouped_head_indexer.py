# SPDX-License-Identifier: Apache-2.0
"""Score all provider groups and index heads in one GEMM for short Step5 MTP queries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_head_score(
    q, summary, table, requests, positions, weights, out,
    stride_q_token, stride_q_group, stride_q_head, stride_summary, stride_table,
    stride_weight, stride_weight_group, stride_out,
    TOKENS: tl.constexpr, GROUPS: tl.constexpr, WIDTH: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
    REGIONS_PER_PAGE: tl.constexpr, REGION_SIZE: tl.constexpr,
    QPR: tl.constexpr, BQ: tl.constexpr, BQH: tl.constexpr, BK: tl.constexpr,
):
    token_base = tl.program_id(0) * QPR
    key = tl.program_id(1) * BK + tl.arange(0, BK)
    group_query = tl.arange(0, BQ)
    query = group_query % QPR
    group = group_query // QPR
    query_ok = group_query < GROUPS * QPR
    position = tl.load(positions + token_base + query, query_ok, 0)
    visible = position // REGION_SIZE
    # FULL graphs reserve the maximum context width. Skip score CTAs whose
    # whole region tile is causally invisible, without any host count read.
    if tl.program_id(1) * BK >= tl.max(visible, 0):
        return
    key_ok = (key < tl.max(visible, 0)) & (key < WIDTH)
    request = tl.load(requests + token_base).to(tl.int64)
    page = tl.load(table + request * stride_table + key // REGIONS_PER_PAGE,
                   key_ok, 0).to(tl.int64)
    slot = page * REGIONS_PER_PAGE + key % REGIONS_PER_PAGE
    dim = tl.arange(0, DIM)
    k = tl.load(summary + slot[None, :] * stride_summary + dim[:, None],
                key_ok[None, :], 0.0).to(q.dtype.element_ty)
    row = tl.arange(0, BQH)
    token = token_base + row % QPR
    q_group = row // (QPR * HEADS)
    head = row // QPR % HEADS
    a = tl.load(q + token[:, None] * stride_q_token
                + q_group[:, None] * stride_q_group
                + head[:, None] * stride_q_head + dim[None, :],
                (row < QPR * HEADS * GROUPS)[:, None], 0)
    scores = tl.dot(a, k, out_dtype=tl.float32)
    acc = tl.zeros((BQ, BK), tl.float32)
    # Preserve native head-by-head FP32 accumulation, including FMA rounding.
    for h in tl.static_range(HEADS):
        index = group[:, None] * HEADS * QPR + h * QPR + query[:, None] + tl.zeros((1, BK), tl.int32)
        score = tl.gather(scores, index, 0)
        w = tl.load(weights + (token_base + query) * stride_weight
                    + group * stride_weight_group + h,
                    query_ok, 0).to(tl.float32)
        acc += tl.maximum(score, 0.0) * w[:, None]
    tl.store(out + (group[:, None] * TOKENS + token_base + query[:, None]) * stride_out + key[None, :],
             acc, (query_ok)[:, None] & (key < WIDTH)[None, :]
             & (key[None, :] < visible[:, None]))


def grouped_head_indexer_logits(
    quantized_q, weights, summary, block_table, request_indices, positions, out,
    *, regions_per_page, queries_per_request, region_size=8, block_k=128,
    num_warps=4, num_stages=2,
):
    tokens, groups, heads, dim = quantized_q.shape
    if groups not in (1, 2, 4) or heads not in (4, 8) or dim != 256:
        raise ValueError("Grouped-head Step5 scoring requires G=1/2/4,H=4/8,D=256")
    if queries_per_request not in (1, 2, 3, 4) or tokens % queries_per_request:
        raise ValueError("Expected complete request groups of 1--4 queries")
    _grouped_head_score[(tokens // queries_per_request,
                         triton.cdiv(out.shape[1], block_k))](
        quantized_q, summary, block_table, request_indices, positions, weights, out,
        quantized_q.stride(0), quantized_q.stride(1), quantized_q.stride(2), summary.stride(0),
        block_table.stride(0), weights.stride(0), weights.stride(1), out.stride(0),
        TOKENS=tokens, GROUPS=groups, WIDTH=out.shape[1], HEADS=heads, DIM=dim,
        REGIONS_PER_PAGE=regions_per_page, REGION_SIZE=region_size,
        QPR=queries_per_request, BQ=triton.next_power_of_2(queries_per_request * groups),
        BQH=max(16, triton.next_power_of_2(queries_per_request * heads * groups)), BK=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
