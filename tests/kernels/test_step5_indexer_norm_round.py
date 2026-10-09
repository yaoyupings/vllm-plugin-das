# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.step5_dsa_norm_rope import (
    short_index_query_rounding_enabled,
)


def test_round_fusion_requires_opt_in_and_short_bf16_geometry(monkeypatch):
    monkeypatch.delenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', raising=False)
    assert not short_index_query_rounding_enabled(64, 256, torch.bfloat16)
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')
    for tokens in (1, 4, 32, 64, 128):
        assert short_index_query_rounding_enabled(tokens, 256, torch.bfloat16)
    for tokens, dim, dtype in (
        (0, 256, torch.bfloat16),
        (129, 256, torch.bfloat16),
        (64, 192, torch.bfloat16),
        (64, 256, torch.float16),
        (64, 256, torch.float32),
    ):
        assert not short_index_query_rounding_enabled(tokens, dim, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP unavailable')
@pytest.mark.parametrize('qpr', [1, 2, 3, 4])
@pytest.mark.parametrize('rotary', [16, 64])
def test_projection_and_scores_match_native_casts_in_full_graph(
    monkeypatch, qpr, rotary,
):
    monkeypatch.setenv('VLLM_STEP4_DSA_FP8_MATH', 'off')
    monkeypatch.setenv('VLLM_STEP5_GROUPED_HEAD_INDEXER', '1')
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '0')
    from vllm_hcu.model_executor.layers import step4_dsa as driver

    requests_count, groups, heads, dim, width = 4, 2, 4, 256, 8192
    tokens = requests_count * qpr
    torch.manual_seed(191)
    packed = torch.randn(
        tokens, groups * heads * dim + 2 * dim + groups * heads,
        device='cuda', dtype=torch.bfloat16,
    )
    qw = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    kb = torch.randn_like(qw)
    cos = torch.randn(65536, rotary, device='cuda', dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.tensor(
        [65532 + j for j in range(qpr)] * requests_count, device='cuda',
    )
    table = torch.randperm(
        requests_count * 1024, device='cuda', dtype=torch.int32,
    ).view(requests_count, 1024)
    requests = torch.arange(
        requests_count, device='cuda', dtype=torch.int32,
    ).repeat_interleave(qpr)
    summary = torch.randn(
        1, requests_count * width, dim, device='cuda', dtype=torch.bfloat16,
    )
    scores = torch.full(
        (groups * tokens, width), -5., device='cuda', dtype=torch.float32,
    )
    indexer = object.__new__(driver.Step4Indexer)
    indexer.fuse_norm_weight_casts = False
    indexer.owner = SimpleNamespace(
        sparse_indexer_q_norm=SimpleNamespace(weight=qw),
        sparse_indexer_k_norm=SimpleNamespace(weight=kw, bias=kb),
    )
    indexer.geometry = SimpleNamespace(proxy_dim=dim)
    indexer.num_heads = groups * heads
    indexer.num_kv_groups = groups
    indexer.heads_per_group = heads
    indexer.eps = 1e-6
    indexer.slot = 0
    indexer.weight_prescale = 0.5
    indexer.rope_theta = 10000.
    indexer.rope_scaling = None
    indexer.max_position = 65536
    indexer.state = SimpleNamespace(
        rope_tables=lambda **kwargs: (cos, sin),
        summary=summary, regions_per_page=8, _verification_logits=scores,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=tokens, positions=positions, block_table=table,
    )
    schedule = SimpleNamespace(
        num_requests=requests_count, queries_per_request=qpr,
        graph_regions=width, request_indices=requests,
    )

    def project():
        return indexer._project(packed, positions, tokens, packed)

    def reference():
        q, k, z, weights = project()
        out = torch.full_like(scores, -5.)
        driver.short_prefill_indexer_logits(
            q, weights, summary[0], table, requests, positions, out,
            regions_per_page=8, queries_per_request=qpr,
        )
        return driver.round_activations_e4m3(q), k, z, out

    expected_q, expected_k, expected_z, expected_scores = reference()
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')

    def fused():
        q, k, z, weights = project()
        out, _, _ = indexer._score_verification(metadata, schedule, q, weights)
        return q, k, z, out

    def forbidden_cast(*args, **kwargs):
        raise AssertionError('scorer launched a redundant E4M3 rounding stage')

    native_cast = driver.round_activations_e4m3
    monkeypatch.setattr(driver, 'round_activations_e4m3', forbidden_cast)
    q, k, z, out = fused()
    for actual, expected in zip(
        (q, k, z, out), (expected_q, expected_k, expected_z, expected_scores),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, k, z, out = fused()
    for end in (8, 65, 65536):
        packed.normal_()
        positions.copy_(torch.tensor(
            [max(end - qpr + j, 0) for j in range(qpr)] * requests_count,
            device='cuda',
        ))
        scores.fill_(-5.)
        graph.replay()
        actual_scores = out.clone()
        monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '0')
        monkeypatch.setattr(driver, 'round_activations_e4m3', native_cast)
        expected_q, expected_k, expected_z, expected_scores = reference()
        for actual, expected in zip(
            (q, k, z, actual_scores),
            (expected_q, expected_k, expected_z, expected_scores),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')
        monkeypatch.setattr(driver, 'round_activations_e4m3', forbidden_cast)


@pytest.mark.parametrize("tokens,proxy,dtype", [
    (8, 256, torch.bfloat16), (64, 256, torch.bfloat16),
    (128, 256, torch.bfloat16), (129, 256, torch.bfloat16),
    (8, 128, torch.bfloat16), (8, 256, torch.float16),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_weight_cast_routing_preserves_norm_rounding(
    monkeypatch, tokens, proxy, dtype, enabled,
):
    from vllm_hcu.model_executor.layers import step4_dsa as driver

    calls = []
    def capture(q, k, qw, kw, kb, *args, **kwargs):
        calls.append((qw, kw, kb, kwargs.get("round_norm_weights", False)))
        return q, k

    def long_capture(q, k, z, qw, kw, kb, *args, **kwargs):
        capture(q, k, qw, kw, kb)
        return q, k, z

    monkeypatch.setattr(driver, "fused_indexer_qk_norm_rope", capture)
    monkeypatch.setattr(driver, "indexer_norm_rope", long_capture)
    monkeypatch.setenv("VLLM_STEP5_INDEXER_ROUND_IN_NORM", "0")
    heads = 8
    parameters = [torch.full((proxy,), 1.00390625) for _ in range(3)]
    owner = SimpleNamespace(
        sparse_indexer_q_norm=SimpleNamespace(weight=parameters[0]),
        sparse_indexer_k_norm=SimpleNamespace(
            weight=parameters[1], bias=parameters[2],
        ),
    )
    indexer = object.__new__(driver.Step4Indexer)
    indexer.__dict__.update(
        owner=owner, geometry=SimpleNamespace(proxy_dim=proxy),
        num_heads=heads, num_kv_groups=2, heads_per_group=4,
        eps=1e-6, weight_prescale=.5, rope_theta=10000.,
        rope_scaling=None, max_position=65536,
        fuse_norm_weight_casts=enabled,
        state=SimpleNamespace(rope_tables=lambda **kw: (
            torch.ones(2, 16, dtype=dtype), torch.zeros(2, 16, dtype=dtype),
        )),
    )
    packed = torch.randn(tokens, heads * proxy + 2 * proxy + heads, dtype=dtype)
    q, k, z, weights = indexer._project(
        packed, torch.zeros(tokens, dtype=torch.int64), tokens, packed,
    )
    qw, kw, kb, fused = calls.pop()
    assert not calls
    expected_fused = enabled and tokens <= 128 and proxy == 256 and dtype == torch.bfloat16
    assert fused == expected_fused
    for actual, original in zip((qw, kw, kb), parameters):
        assert actual.dtype == (torch.float32 if expected_fused else dtype)
        torch.testing.assert_close(actual.to(dtype), original.to(dtype), rtol=0, atol=0)
        if expected_fused:
            assert actual is original
    assert q.shape == (tokens, 2, 4, proxy)
    assert k.shape == z.shape == (tokens, 1, proxy)
    torch.testing.assert_close(
        weights, packed[:, -heads:].float().view(tokens, 2, 4) * .5,
        rtol=0, atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP unavailable")
@pytest.mark.parametrize("tokens", [8, 64])
@pytest.mark.parametrize("q_round", [False, True])
def test_weight_cast_fusion_matches_changing_parameters_in_graph(tokens, q_round):
    from vllm_hcu.model_executor.layers.step5_dsa_norm_rope import (
        fused_indexer_qk_norm_rope as norm,
    )

    torch.manual_seed(4707)
    heads, dim, rotary = 8, 256, 16
    q = torch.randn(tokens, heads * dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    parameters = [torch.randn(dim, device="cuda", dtype=torch.float32) for _ in range(3)]
    cos = torch.randn(65536, rotary, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) + 65400

    def invoke(fused):
        weights = parameters if fused else [w.to(torch.bfloat16) for w in parameters]
        return norm(q, k, *weights, cos, sin, positions,
                    head_dim=dim, num_q_heads=heads, rotary_dim=rotary,
                    round_q_e4m3=q_round, round_norm_weights=fused)

    invoke(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_k = invoke(True)
    for value in (None, 1.00390625, -1.0003, .003906251, 0.):
        q.normal_()
        k.normal_()
        for w in parameters:
            if value is None:
                w.normal_()
            else:
                w.fill_(value)
        positions.copy_(torch.randperm(65536, device="cuda")[:tokens])
        graph.replay()
        expected_q, expected_k = invoke(False)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
