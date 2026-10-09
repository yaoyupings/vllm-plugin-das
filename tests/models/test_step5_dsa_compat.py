# SPDX-License-Identifier: Apache-2.0
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.step5_dsa_compat import (
    region_topk_ids,
    region_topk_pack,
)


def test_flash_cache_spec_without_current_config_context():
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType
    from vllm_hcu.models.step5 import Step5FlashAttentionBackend

    layer = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        attn_backend=Step5FlashAttentionBackend,
        sliding_window=512,
        num_kv_heads=1,
        head_size=192,
        head_size_v=192,
        kv_cache_torch_dtype=torch.bfloat16,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, skip_page_size_padded=None)
    )
    spec = Attention.get_kv_cache_spec(layer, config)
    assert (
        spec.block_size,
        spec.sliding_window,
        spec.num_kv_heads,
        spec.head_size,
    ) == (64, 512, 1, 192)


@pytest.mark.parametrize("length", [7, 70])
def test_hcu_flash_cache_prefill_and_decode_match_torch(length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step5_dsa_compat import flash_cache_view
    from vllm_hcu.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backend import AttentionType

    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        num_heads=8,
        head_size=192,
        num_kv_heads=1,
        scale=192 ** -0.5,
        alibi_slopes=None,
        sliding_window=(511, 0),
        logits_soft_cap=0.0,
        kv_sharing_target_layer_name=None,
        vllm_flash_attn_version=get_flash_attn_version(head_size=192),
        batch_invariant_enabled=False,
        supports_quant_query_input=False,
        sinks=None,
        dcp_world_size=1,
        use_pcp=False,
    )
    scale = torch.tensor(1.0, device="cuda")
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    raw = torch.zeros(2, 64, 1, 384, dtype=torch.bfloat16, device="cuda")
    cache = flash_cache_view(raw.permute(0, 2, 1, 3))
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.manual_seed(19)
    key = torch.randn(length, 1, 192, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    queries = torch.randn(length, 8, 192, dtype=torch.bfloat16, device="cuda")
    FlashAttentionImpl.do_kv_cache_update(
        impl, layer, key, value, cache, torch.arange(length, device="cuda")
    )
    for decode in [False, True]:
        q = queries[-1:] if decode else queries
        count = q.shape[0]
        md = FlashAttentionMetadata(
            num_actual_tokens=count,
            max_query_len=count,
            query_start_loc=torch.tensor([0, count], dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            seq_lens=torch.tensor([length], dtype=torch.int32, device="cuda"),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(length - count, length, device="cuda"),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        actual = torch.empty_like(q)
        FlashAttentionImpl.forward(
            impl, layer, q, key[-count:], value[-count:], cache, md, actual
        )
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            key.transpose(0, 1)[None],
            value.transpose(0, 1)[None],
            is_causal=not decode,
            enable_gqa=True,
        )[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("heads", [1, 4])
def test_v2_cache_alias_and_sparse_attention_parity(heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAImpl
    from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
        sparse_attention_prefill,
    )

    # The runner exposes BHNC while physical LBNHC packs K/V per head.
    raw = torch.zeros((2, 8, heads, 384), dtype=torch.bfloat16, device="cuda")
    allocated = raw.permute(0, 2, 1, 3)
    cache = sparse_cache_view(allocated)
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.manual_seed(7)
    key = torch.randn((17, heads, 192), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.tensor(list(range(16)) + [-1], device="cuda")
    k_flat, v_flat = Step4DSAImpl._write_kv(cache, key, value, slots)
    torch.testing.assert_close(raw[..., :192].flatten(0, 1), key[:16], rtol=0, atol=0)
    torch.testing.assert_close(raw[..., 192:].flatten(0, 1), value[:16], rtol=0, atol=0)
    query = torch.randn((16, heads * 2, 192), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(16, device="cuda", dtype=torch.int32)
    logits = torch.zeros((heads * 16, 2), device="cuda")
    lengths = ((positions + 8) // 8).repeat(heads)
    logical = torch.empty((heads * 16, 2), dtype=torch.int32, device="cuda")
    packed, logical = region_topk_pack(
        logits,
        lengths,
        positions.repeat(heads),
        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        topk=2,
        regions_per_page=1,
        _logical_out=logical,
    )
    counts = lengths.to(torch.int32)
    out, _ = sparse_attention_prefill(
        query,
        k_flat,
        v_flat,
        packed,
        counts,
        num_kv_groups=heads,
        logical_regions=logical,
        q_positions=positions,
        request_indices=torch.zeros(heads * 16, device="cuda", dtype=torch.int32),
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None],
        key[:16].transpose(0, 1)[None],
        value[:16].transpose(0, 1)[None],
        is_causal=True,
        enable_gqa=True,
    )[0].transpose(0, 1)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)


def test_v2_cache_rejects_interleaved_layers():
    from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view

    raw = torch.zeros((2, 3, 8, 1, 384))
    with pytest.raises(ValueError, match="LBNHC"):
        sparse_cache_view(raw[:, 1].permute(0, 2, 1, 3))


def _ordered(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return ((~bits) & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("topk", [1, 3, 8])
def test_selection_and_packing_match_scalar_reference(device, topk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    values = [[-1.0, 0.0, -0.0, 2.0, 2.0, -3.0], [0.0] * 6] * 3
    lengths = [-2, 0, 2, 4, 6, 99]
    positions = [0, 11, 20, 35, 44, 49]
    requests = [0, 1, 0, 1, 0, 1]
    table = [[3, 1, -1], [8, -1, 2]]
    expected_ids, expected_pack = [], []
    for row, length, pos, req in zip(values, lengths, positions, requests):
        candidates = range(max(0, min(length, len(row))))
        chosen = sorted(sorted(candidates, key=lambda i: (-_ordered(row[i]), i))[:topk])
        packed = []
        for i in chosen:
            page = table[req][i // 2]
            physical = page * 2 + i % 2
            valid = max(0, min(pos + 1 - i * 8, 8))
            packed.append(physical | (valid << 24) if page >= 0 else -1)
        expected_ids.append(chosen + [-1] * (topk - len(chosen)))
        expected_pack.append(packed + [-1] * (topk - len(packed)))
    logits = torch.tensor(values, dtype=torch.float32, device=device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    ids = region_topk_ids(logits, lens, topk=topk)
    logical_out = torch.empty_like(ids)
    packed, returned_ids = region_topk_pack(
        logits,
        lens,
        torch.tensor(positions, device=device),
        torch.tensor(table, device=device),
        torch.tensor(requests, device=device),
        topk=topk,
        regions_per_page=2,
        _logical_out=logical_out,
    )
    assert returned_ids is logical_out
    assert ids.cpu().tolist() == expected_ids
    assert returned_ids.cpu().tolist() == expected_ids
    assert packed.cpu().tolist() == expected_pack


def test_shared_block_table_ignores_request_index():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = region_topk_pack(
        scores,
        torch.tensor([4]),
        torch.tensor([31]),
        torch.tensor([[5, 6]]),
        torch.tensor([999]),
        topk=2,
        regions_per_page=2,
    )
    assert out.tolist() == [[10 | (8 << 24), 11 | (8 << 24)]]


def test_native_radix_selector_gfx938_workaround():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels
    from vllm_hcu.model_executor.layers.step5_dsa_selector import (
        install_native_radix_selectors,
    )

    original_threshold = kernels._radix_topk_threshold
    original_ids = kernels.region_topk_ids
    original_pack = kernels.region_topk_pack
    try:
        install_native_radix_selectors()
        torch.manual_seed(41)
        rows, width, topk = 4, 2049, 512
        scores = torch.randn(rows, width, device="cuda", dtype=torch.float32)
        scores[:, ::11] = 0.0
        lengths = torch.tensor(
            [0, 1, 2048, width], device="cuda", dtype=torch.int32
        )
        positions = lengths * 8 + 3
        table = torch.arange(
            2 * 1030, device="cuda", dtype=torch.int32
        ).reshape(2, 1030)
        requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2

        expected_ids = region_topk_ids(scores, lengths, topk=topk)
        actual_ids = kernels.region_topk_ids(scores, lengths, topk=topk)
        expected_logical = torch.empty_like(expected_ids)
        actual_logical = torch.empty_like(expected_ids)
        expected_pack, expected_logical = region_topk_pack(
            scores, lengths, positions, table, requests,
            topk=topk, regions_per_page=2,
            _logical_out=expected_logical,
        )
        actual_pack, actual_logical = kernels.region_topk_pack(
            scores, lengths, positions, table, requests,
            topk=topk, regions_per_page=2,
            _logical_out=actual_logical,
        )
        torch.cuda.synchronize()

        assert torch.equal(actual_ids, expected_ids)
        assert torch.equal(actual_logical, expected_logical)
        assert torch.equal(actual_pack, expected_pack)

        # A one-column score buffer is scalarized by the DTK compiler, so the
        # wrapper must retain the exact compatibility path for this corner.
        tiny_scores = scores[:2, :1].contiguous()
        tiny_lengths = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        tiny_positions = tiny_lengths * 8 + 3
        tiny_requests = requests[:2]
        expected_tiny_ids = region_topk_ids(
            tiny_scores, tiny_lengths, topk=topk
        )
        actual_tiny_ids = kernels.region_topk_ids(
            tiny_scores, tiny_lengths, topk=topk
        )
        expected_tiny_pack = region_topk_pack(
            tiny_scores,
            tiny_lengths,
            tiny_positions,
            table,
            tiny_requests,
            topk=topk,
            regions_per_page=2,
        )
        actual_tiny_pack = kernels.region_topk_pack(
            tiny_scores,
            tiny_lengths,
            tiny_positions,
            table,
            tiny_requests,
            topk=topk,
            regions_per_page=2,
        )
        assert torch.equal(actual_tiny_ids, expected_tiny_ids)
        assert torch.equal(actual_tiny_pack, expected_tiny_pack)
    finally:
        kernels._radix_topk_threshold = original_threshold
        kernels.region_topk_ids = original_ids
        kernels.region_topk_pack = original_pack


@pytest.mark.parametrize("queries", [16, 65, 96, 128])
def test_flash_sparse_short_prefill_matches_triton(queries):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step4_dsa import _flash_attn_prefill
    from vllm_hcu.model_executor.layers.step5_dsa_shortquery import (
        sparse_attention_short_prefill,
    )

    torch.manual_seed(53)
    groups, heads, head_dim = 2, 32, 192
    region_size, topk = 8, 32
    tokens = region_size * topk
    query = torch.randn(
        queries, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        tokens, groups, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    physical = torch.arange(topk, device="cuda", dtype=torch.int32)
    packed = (
        physical | (torch.full_like(physical, region_size) << 24)
    ).repeat(groups * queries, 1).contiguous()
    counts = torch.full(
        (groups * queries,), topk, device="cuda", dtype=torch.int32
    )

    expected, expected_lse = sparse_attention_short_prefill(
        query,
        key,
        value,
        packed,
        counts,
        num_kv_groups=groups,
        region_size=region_size,
        num_splits=4,
    )
    direct = torch.empty_like(expected)
    direct_out, direct_lse = sparse_attention_short_prefill(
        query,
        key,
        value,
        packed,
        counts,
        num_kv_groups=groups,
        region_size=region_size,
        num_splits=4,
        output=direct,
        return_lse=False,
    )
    assert direct_out.data_ptr() == direct.data_ptr()
    assert direct_lse is None
    assert torch.equal(direct_out, expected)
    actual, actual_lse = _flash_attn_prefill(
        query,
        key,
        value,
        packed,
        counts,
        num_kv_groups=groups,
        region_size=region_size,
        softmax_scale=head_dim**-0.5,
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=2e-5)

def test_fused_step5_indexer_norm_and_batched_scoring_match_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step4_dsa import (
        short_prefill_indexer_logits,
    )
    from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
        indexer_logits,
        indexer_norm_rope,
        round_activations_e4m3,
    )
    from vllm_hcu.model_executor.layers.step5_dsa_norm_rope import (
        fused_indexer_qk_norm_rope,
    )

    torch.manual_seed(67)
    tokens, groups, heads, proxy, rotary = 8, 2, 4, 256, 64
    q = torch.randn(
        tokens, groups * heads * proxy, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(tokens, proxy, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(k)
    q_weight = torch.randn(proxy, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(proxy, device="cuda", dtype=torch.bfloat16)
    k_bias = torch.randn(proxy, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(256, rotary, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.arange(120, 128, device="cuda")

    expected_q, expected_k, _ = indexer_norm_rope(
        q, k, z, q_weight, k_weight, k_bias, cos, sin, positions,
        head_dim=proxy, num_q_heads=groups * heads, num_k_heads=1,
        rotary_dim=rotary, eps=1e-6, q_norm_weight_bias=1.0,
    )
    actual_q, actual_k = fused_indexer_qk_norm_rope(
        q, k, q_weight, k_weight, k_bias, cos, sin, positions,
        head_dim=proxy, num_q_heads=groups * heads,
        rotary_dim=rotary, eps=1e-6,
    )
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)

    index_q = round_activations_e4m3(
        actual_q.view(tokens, groups, heads, proxy)
    )
    weights = torch.randn(
        tokens, groups, heads, device="cuda", dtype=torch.float32
    )
    requests = torch.tensor([0] * 4 + [1] * 4, device="cuda", dtype=torch.int32)
    score_positions = torch.tensor(
        [124, 125, 126, 127] * 2, device="cuda", dtype=torch.int32
    )
    regions_per_page, width = 8, 15
    block_table = torch.tensor(
        [[1, 2], [3, 4]], device="cuda", dtype=torch.int32
    )
    summary = torch.randn(
        5 * regions_per_page, 1, proxy,
        device="cuda", dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    expected = torch.zeros(
        groups * tokens, width, device="cuda", dtype=torch.float32
    )
    logical = torch.arange(width, device="cuda")
    for request in range(2):
        slots = (
            block_table[request, logical // regions_per_page].to(torch.int64)
            * regions_per_page
            + logical % regions_per_page
        )
        indexer_logits(
            index_q[request * 4 : (request + 1) * 4],
            weights[request * 4 : (request + 1) * 4],
            summary[slots].to(torch.bfloat16),
            out=expected,
            out_row_base=request * 4,
            out_group_stride=tokens,
            quantize_keys=False,
        )
    actual = torch.zeros_like(expected)
    short_prefill_indexer_logits(
        index_q, weights, summary, block_table, requests, score_positions, actual,
        regions_per_page=regions_per_page,
    )
    assert torch.equal(actual, expected)

    grouped = torch.zeros_like(expected)
    short_prefill_indexer_logits(
        index_q, weights, summary, block_table, requests, score_positions, grouped,
        regions_per_page=regions_per_page, queries_per_request=4,
    )
    assert torch.equal(grouped, expected)


def test_fused_step5_prefill_metadata_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels
    from vllm_hcu.model_executor.layers.step5_dsa_selector import (
        install_native_radix_selectors,
        prefill_sparse_meta_fused,
    )

    original_threshold = kernels._radix_topk_threshold
    original_ids = kernels.region_topk_ids
    original_pack = kernels.region_topk_pack
    try:
        install_native_radix_selectors()
        torch.manual_seed(71)
        rows, width, topk, regions_per_page = 4, 2049, 512, 8
        scores = torch.randn(rows, width, device="cuda", dtype=torch.float32)
        scores[:, ::13] = 0.0
        positions = torch.tensor(
            [7, 4095, 8191, width * 8 - 1],
            device="cuda",
            dtype=torch.int32,
        )
        requests = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
        pages = (width + 1 + regions_per_page - 1) // regions_per_page
        block_table = (
            torch.arange(2 * pages, device="cuda", dtype=torch.int32)
            .reshape(2, pages)
            + 1
        )

        expected = kernels.prefill_sparse_meta(
            scores,
            positions,
            block_table,
            requests,
            topk=topk,
            region_size=8,
            regions_per_page=regions_per_page,
            return_logical_ids=True,
        )
        actual = prefill_sparse_meta_fused(
            scores,
            positions,
            block_table,
            requests,
            topk=topk,
            region_size=8,
            regions_per_page=regions_per_page,
            return_logical_ids=True,
        )
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(actual[2], expected[2])
        packed_only = prefill_sparse_meta_fused(
            scores,
            positions,
            block_table,
            requests,
            topk=topk,
            region_size=8,
            regions_per_page=regions_per_page,
            return_logical_ids=False,
        )
        assert torch.equal(packed_only[0], expected[0])
        assert torch.equal(packed_only[1], expected[1])
    finally:
        kernels._radix_topk_threshold = original_threshold
        kernels.region_topk_ids = original_ids
        kernels.region_topk_pack = original_pack


def test_decode_token_insert_matches_scatter_and_replays_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step4_dsa import _insert_decode_tokens

    rows, region, proxy = 17, 8, 256
    torch.manual_seed(97)
    initial_key = torch.randn(
        (rows, region, proxy), dtype=torch.bfloat16, device="cuda"
    )
    initial_z = torch.randn_like(initial_key)
    source_key = torch.randn(
        (rows, 1, proxy), dtype=torch.bfloat16, device="cuda"
    )
    source_z = torch.randn_like(source_key)
    heads = torch.arange(rows, dtype=torch.int64, device="cuda") % region

    expected_key = initial_key.clone()
    expected_z = initial_z.clone()
    scatter_index = heads[:, None, None].expand(rows, 1, proxy)
    expected_key.scatter_(1, scatter_index, source_key)
    expected_z.scatter_(1, scatter_index, source_z)

    actual_key = initial_key.clone()
    actual_z = initial_z.clone()
    _insert_decode_tokens(actual_key, actual_z, source_key, source_z, heads)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(actual_z, expected_z, rtol=0, atol=0)

    # Compile before capture, then verify that one captured launch accepts new
    # data and destination rows without recording shape- or pointer-dependent
    # state. This reproduces the target model's replay contract in isolation.
    static_key = initial_key.clone()
    static_z = initial_z.clone()
    static_source_key = source_key.clone()
    static_source_z = source_z.clone()
    static_heads = heads.clone()
    _insert_decode_tokens(
        static_key, static_z, static_source_key, static_source_z, static_heads
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _insert_decode_tokens(
            static_key, static_z, static_source_key, static_source_z, static_heads
        )

    for shift in (1, 5):
        replay_heads = (heads + shift) % region
        replay_key = source_key + shift
        replay_z = source_z - shift
        static_key.copy_(initial_key)
        static_z.copy_(initial_z)
        static_source_key.copy_(replay_key)
        static_source_z.copy_(replay_z)
        static_heads.copy_(replay_heads)
        graph.replay()
        torch.cuda.synchronize()

        expected_key = initial_key.clone()
        expected_z = initial_z.clone()
        scatter_index = replay_heads[:, None, None].expand(rows, 1, proxy)
        expected_key.scatter_(1, scatter_index, replay_key)
        expected_z.scatter_(1, scatter_index, replay_z)
        torch.testing.assert_close(static_key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(static_z, expected_z, rtol=0, atol=0)


def test_mtp_verification_schedule_matches_general_schedule():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step4_dsa import (
        Step4DSAGeometry,
        Step4DSAMetadata,
        Step4DSAState,
        Step4Indexer,
    )

    torch.manual_seed(113)
    heads = list(range(8))
    seq_lens_cpu = torch.tensor([head + 4 for head in heads], dtype=torch.int32)
    query_start_cpu = torch.arange(0, 33, 4, dtype=torch.int32)
    block_table = torch.stack(
        [torch.tensor([req + 1, req + 9], dtype=torch.int32) for req in range(8)]
    ).cuda()
    metadata = Step4DSAMetadata(
        num_decodes=0,
        num_prefills=8,
        num_decode_tokens=0,
        num_prefill_tokens=32,
        num_actual_tokens=32,
        block_table=block_table,
        slot_mapping=torch.arange(32, device="cuda", dtype=torch.long),
        seq_lens=seq_lens_cpu.cuda(),
        query_start_loc=query_start_cpu.cuda(),
        query_start_loc_cpu=query_start_cpu,
        seq_lens_cpu=seq_lens_cpu,
        positions=torch.cat(
            [torch.arange(head, head + 4) for head in heads]
        ).cuda(),
        max_seq_len=11,
        block_size=64,
        block_table_np=block_table.cpu().numpy().astype(np.int32),
    )
    initial_key = torch.randn(
        (8, 8, 256), device="cuda", dtype=torch.bfloat16
    )
    initial_z = torch.randn_like(initial_key)
    index_k = torch.randn((32, 1, 256), device="cuda", dtype=torch.bfloat16)
    index_z = torch.randn_like(index_k)

    def make_state():
        state = Step4DSAState(
            num_layers=1,
            geometry=Step4DSAGeometry(),
            max_pending=64,
            dtype=torch.bfloat16,
            max_num_reqs=8,
            max_model_len=128,
            num_speculative_tokens=3,
        )
        state.num_kv_groups = 2
        state.heads_per_group = 4
        cache = torch.zeros(
            (2, 24, 64, 2, 192), device="cuda", dtype=torch.bfloat16
        )
        state.ensure_allocated(cache, torch.device("cuda"))
        for req, head in enumerate(heads):
            if head:
                slot = int(block_table[req, 0]) * state.regions_per_page
                row = state.pending.acquire(slot)
                state.pending_key[0, row, :head] = initial_key[req, :head]
                state.pending_z[0, row, :head] = initial_z[req, :head]
        return state

    def make_indexer(state):
        obj = Step4Indexer.__new__(Step4Indexer)
        obj.state = state
        obj.geometry = state.geometry
        obj.slot = 0
        obj.num_kv_groups = 2
        return obj

    reference_state = make_state()
    graph_state = make_state()
    reference = make_indexer(reference_state)
    graph = make_indexer(graph_state)
    reference_schedule = reference_state._general_schedule(metadata)
    reference._update_summaries(reference_schedule, index_k, index_z)
    schedule = graph_state._verification_schedule(metadata, persistent=True)
    graph._update_summaries(schedule, index_k, index_z)
    torch.cuda.synchronize()

    assert schedule.verification
    assert schedule.request_indices.tolist() == [i for i in range(8) for _ in range(4)]
    assert torch.equal(reference_state.summary, graph_state.summary)
    assert torch.equal(reference_state.pending_key, graph_state.pending_key)
    assert torch.equal(reference_state.pending_z, graph_state.pending_z)

    index_q = torch.randn(
        (32, 2, 4, 256), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn((32, 2, 4), device="cuda", dtype=torch.float32)
    reference_logits, _, reference_width = reference._score(
        metadata, reference_schedule, index_q, weights
    )
    graph_logits, _, _ = graph._score(metadata, schedule, index_q, weights)
    assert reference_logits is not None
    width = max(reference_width, 1)
    columns = torch.arange(width, device="cuda").unsqueeze(0)
    score_positions = metadata.positions.repeat(graph_state.num_kv_groups)
    visible = columns < score_positions.unsqueeze(1) // graph_state.geometry.region_size
    torch.testing.assert_close(
        graph_logits[:, :width][visible],
        reference_logits[visible],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tokens,max_query,expected", [
    (0, 4, False), (64, 64, True), (65, 4, True), (96, 3, True),
    (127, 4, True), (128, 4, True), (129, 4, False),
    (96, 0, False), (96, 5, False), (96, 96, False),
])
def test_short_query_routes_ragged_mtp_without_tensor_readback(
    tokens, max_query, expected,
):
    from vllm_hcu.model_executor.layers.step5_dsa_shortquery import (
        is_short_query_batch,
    )
    assert is_short_query_batch(tokens, max_query) is expected
    assert is_short_query_batch(128, verification=True)
    assert is_short_query_batch(128, capturing=True)


@pytest.mark.parametrize("max_query,needs_logical", [(4, False), (96, True)])
def test_ragged_mtp_selection_omits_unused_tiled_metadata(
    monkeypatch, max_query, needs_logical,
):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa
    monkeypatch.setenv("VLLM_STEP5_DSA_SHORT_QUERY", "1")
    monkeypatch.setattr(dsa, "PREFILL_QK_FP8", False)
    monkeypatch.setattr(dsa, "PREFILL_PV_FP8", False)
    tokens, groups = 96, 2
    observed = []

    def select(scores, positions, table, request_indices, **kwargs):
        observed.append(kwargs["return_logical_ids"])
        packed = torch.zeros(groups * tokens, 3, dtype=torch.int32)
        counts = torch.ones(groups * tokens, dtype=torch.int32)
        if kwargs["return_logical_ids"]:
            return packed, counts, packed.clone()
        return packed, counts

    monkeypatch.setattr(dsa, "prefill_sparse_meta", select)
    monkeypatch.setattr(dsa, "prefill_sparse_meta_fused", select, raising=False)
    md = SimpleNamespace(
        num_decodes=0, num_decode_tokens=0, num_prefills=32,
        num_actual_tokens=tokens, positions=torch.arange(tokens),
        block_table=torch.zeros(32, 2, dtype=torch.int32),
        prefill_request_indices=torch.arange(32).repeat_interleave(3),
        schedule=None, max_query_len=max_query,
    )
    plan = dsa.build_plan(
        md, torch.zeros(groups * tokens, 16), None,
        topk=2, num_kv_groups=groups, regions_per_page=8,
    )
    assert observed == [needs_logical]
    assert (plan.prefill[2] is not None) is needs_logical
