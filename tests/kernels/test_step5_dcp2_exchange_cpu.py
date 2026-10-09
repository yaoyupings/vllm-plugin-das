# SPDX-License-Identifier: Apache-2.0
"""Exact head order / owned cache-slot qualification via CPU interpreter."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("requires explicit Triton CPU interpreter", allow_module_level=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source = Path(__file__).resolve().parents[2] / "vllm_hcu/ops/step5_dcp2_exchange.py"
spec = importlib.util.spec_from_file_location("step5_dcp2_exchange_cpu", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("local_heads", [4, 32])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 16, 64, 128, 384])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_rank_order_and_striped_cache(local_heads, tokens, slot_dtype):
    torch.manual_seed(106 + tokens + local_heads)
    # Independent rank data makes Q0/K0/V0/Q1/K1/V1 reorder bugs visible.
    qkv = []
    packed_ranks = []
    width = (local_heads + 4) * 192
    for rank in range(2):
        base = torch.randn(tokens, local_heads + 4, 192).bfloat16()
        q = base[:, :local_heads]
        k = base[:, local_heads:local_heads + 2]
        v = base[:, local_heads + 2:]
        # Norm/RoPE may produce contiguous q/k while V remains strided.
        q, k = q.contiguous(), k.contiguous()
        qkv.append((q, k, v))
        packed = torch.empty(tokens, width, dtype=torch.bfloat16)
        module._pack_local_qkv_kernel[(tokens, (width + 255) // 256)](
            q, k, v, packed, q.stride(0), k.stride(0), v.stride(0),
            Q_WIDTH=local_heads * 192, KV_WIDTH=384, BLOCK=256,
        )
        expected = torch.cat([x.flatten(1) for x in (q, k, v)], dim=1)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        packed_ranks.append(packed)
    gathered = torch.cat(packed_ranks, dim=1)
    # Include row padding stride as well as the normal contiguous case.
    backing = torch.zeros(tokens, 2 * width + 192, dtype=gathered.dtype)
    backing[:, :2 * width] = gathered
    gathered = backing[:, :2 * width]

    # Global physical pages are shuffled; slot calculation is an independent
    # oracle for vLLM CP_SIZE2 / INTERLEAVE1 / local block64.
    pages = (tokens + 127) // 128
    physical = torch.randperm(pages)
    slots_per_cache = pages * 64
    for rank in range(2):
        positions = torch.arange(tokens)
        slots = physical[positions // 128] * 64 + (positions % 128) // 2
        slots[positions % 2 != rank] = -1
        # Padded/invalid query rows still get QKV, never overwrite cache.
        slots[-1] = -1
        if tokens >= 3:
            slots[-3] = slots_per_cache
        slots = slots.to(slot_dtype)
        kc = torch.full((slots_per_cache, 4, 192), -123, dtype=torch.bfloat16)
        vc = torch.full_like(kc, 123)
        expected_kc, expected_vc = kc.clone(), vc.clone()
        expected_q = torch.cat([r[0] for r in qkv], dim=1)
        expected_k = torch.cat([r[1] for r in qkv], dim=1)
        expected_v = torch.cat([r[2] for r in qkv], dim=1)
        for row, slot in enumerate(slots.tolist()):
            if 0 <= slot < slots_per_cache:
                expected_kc[slot] = expected_k[row]
                expected_vc[slot] = expected_v[row]
        q, k, v = [torch.empty_like(x) for x in [expected_q, expected_k, expected_v]]
        module._unpack_qkv_cache_kernel[(tokens, (2 * width + 255) // 256)](
            gathered, slots, q, k, v, kc, vc, gathered.stride(0), slots_per_cache,
            kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
            64*kc.stride(0), 64*vc.stride(0),
            Q_WIDTH=local_heads * 192, KV_WIDTH=384, BLOCK=256,
        )
        for actual, expected in [
            (q, expected_q), (k, expected_k), (v, expected_v),
            (kc, expected_kc), (vc, expected_vc),
        ]:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [1, 2, 513])
def test_decode_word_logical_regions(topk):
    # Real Decode words contain token offsets in low32 and physical IDs in
    # high32. Physical IDs intentionally disagree with logical order.
    rows = 8
    logical = torch.arange(rows*topk, dtype=torch.int64).reshape(rows, topk) % 8193
    physical = torch.flip(logical, dims=[0, 1]) + 4096
    words = (physical << 32) | (logical * 8)
    # Region0 is valid with count1; outside counts zero words are padding.
    words[0, 0] = 0
    logical[0, 0] = 0
    counts = torch.tensor([0, 1, topk, max(0, topk-1), topk, 0, 1, topk],
                          dtype=torch.int32)
    expected = logical.to(torch.int32)
    for row in range(rows):
        expected[row, counts[row]:] = -1
        words[row, counts[row]:] = 0
    regions = torch.empty_like(expected)
    module._decode_words_to_regions_kernel[((words.numel()+255)//256,)](
        words, counts, regions, ROWS=rows, TOPK=topk, REGION_SIZE=8, BLOCK=256,
    )
    torch.testing.assert_close(regions, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["separate", "sparse_v2", "flash_v2"])
@pytest.mark.parametrize("heads", [4, 32])
def test_actual_v2_cache_strides(layout, heads):
    tokens, slots_count = 4, 128
    width = (heads + 4) * 192
    gathered = torch.randn(tokens, 2*width).bfloat16()
    if layout == "separate":
        kc = torch.full((slots_count, 4, 192), -123, dtype=torch.bfloat16)
        vc = torch.full_like(kc, 123)
    elif layout == "sparse_v2":
        # Exact sparse_cache_view flattening: [slot,head,K/V,HD].
        backing = torch.full((slots_count, 4, 2, 192), -123, dtype=torch.bfloat16)
        kc, vc = backing[:, :, 0], backing[:, :, 1]
        vc.fill_(123)
    else:
        # K/V are separated within each page; crossing page64 has a gap.
        backing = torch.full((2, 2, 64, 4, 192), -123, dtype=torch.bfloat16)
        kc, vc = backing[:, 0], backing[:, 1]
        vc.fill_(123)
    assert module._cache_tensor(kc) and module._cache_tensor(vc)
    expected_kc, expected_vc = kc.clone(), vc.clone()
    slots = torch.tensor([-1, 63, 64, 1 << 40], dtype=torch.int64)
    q = torch.empty(tokens, 2*heads, 192, dtype=torch.bfloat16)
    k = torch.empty(tokens, 4, 192, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    pieces = [gathered[:, rank*width:(rank+1)*width] for rank in range(2)]
    expected_q = torch.cat([x[:, :heads*192] for x in pieces], dim=1).view_as(q)
    expected_k = torch.cat([x[:, heads*192:(heads+2)*192] for x in pieces], dim=1).view_as(k)
    expected_v = torch.cat([x[:, (heads+2)*192:] for x in pieces], dim=1).view_as(v)
    for cache, rows in [(expected_kc, expected_k), (expected_vc, expected_v)]:
        flat = cache.view(-1, 4, 192)
        flat[63], flat[64] = rows[1], rows[2]
    cache_slots, kr, kh, kp = module._cache_geometry(kc)
    _, vr, vh, vp = module._cache_geometry(vc)
    module._unpack_qkv_cache_kernel[(tokens, (2*width+255)//256)](
        gathered, slots, q, k, v, kc, vc, gathered.stride(0), slots_count,
        kr, kh, vr, vh, kp, vp,
        Q_WIDTH=heads*192, KV_WIDTH=384, BLOCK=256,
    )
    for actual, expected in [(q,expected_q),(k,expected_k),(v,expected_v),
                             (kc,expected_kc),(vc,expected_vc)]:
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
