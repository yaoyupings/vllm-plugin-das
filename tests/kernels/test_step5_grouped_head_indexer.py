# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA/HIP unavailable")


@pytest.mark.parametrize("qpr", [1, 2, 3, 4])
@pytest.mark.parametrize("summary_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("groups,heads,width", [(1, 8, 127), (2, 4, 8192),
                                              (4, 4, 128)])
def test_grouped_heads_match_native_and_graph_replay(monkeypatch, qpr, groups,
                                                    heads, width, summary_dtype):
    monkeypatch.setenv("VLLM_STEP4_DSA_FP8_MATH", "off")
    monkeypatch.setenv("VLLM_STEP5_GROUPED_HEAD_INDEXER", "0")
    from vllm_hcu.model_executor.layers.step4_dsa import (
        short_prefill_indexer_logits,
    )
    requests_count = 4
    tokens = requests_count * qpr
    cols = (width + 7) // 8
    torch.manual_seed(107)
    q = torch.randn(tokens, groups, heads, 256,
                    device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, groups, heads, device="cuda", dtype=torch.float32)
    summary = torch.randn(requests_count * cols * 8, 256,
                          device="cuda", dtype=torch.bfloat16).to(summary_dtype)
    table = torch.randperm(requests_count * cols, device="cuda",
                          dtype=torch.int32).view(requests_count, cols)
    requests = torch.arange(requests_count, device="cuda",
                            dtype=torch.int32).repeat_interleave(qpr)
    positions = torch.tensor([width * 8 - qpr + j
                              for j in range(qpr)] * requests_count,
                             device="cuda", dtype=torch.int64)
    # Noncontiguous output rows exercise group-major addressing and padding.
    ref = torch.empty(groups * tokens, width + 8,
                      device="cuda", dtype=torch.float32)
    actual = torch.empty_like(ref)

    def score(out):
        out.fill_(-5)
        short_prefill_indexer_logits(
            q, weights, summary, table, requests, positions, out[:, :width],
            regions_per_page=8, queries_per_request=qpr,
        )

    score(ref)
    monkeypatch.setenv("VLLM_STEP5_GROUPED_HEAD_INDEXER", "1")
    score(actual)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        score(actual)
    for end in [0, 8, min(width * 8, 65), width * 8]:
        positions.copy_(torch.tensor([max(end - qpr + j, 0)
                                      for j in range(qpr)] * requests_count,
                                     device="cuda", dtype=torch.int64))
        q.normal_()
        weights.normal_()
        monkeypatch.setenv("VLLM_STEP5_GROUPED_HEAD_INDEXER", "0")
        score(ref)
        graph.replay()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
