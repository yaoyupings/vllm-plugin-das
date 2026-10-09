# SPDX-License-Identifier: Apache-2.0
"""DCP2 bridge layout checks using the explicit CPU Triton interpreter."""
import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("requires explicit Triton CPU interpreter", allow_module_level=True)
torch.set_num_threads(1)
source = Path(__file__).resolve().parents[2] / "vllm_hcu/ops"
package = types.ModuleType("_step5_runtime_cpu_snapshot")
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
runtime = importlib.import_module(package.__name__+".step5_dcp2_runtime")


def test_ragged_query_mapping_and_padding():
    starts = torch.tensor([0, 1, 5, 5, 8], dtype=torch.int32)
    lengths = torch.tensor([65536, 129, 0, 0], dtype=torch.int32)
    positions = torch.tensor([65535, 125, 126, 127, 128, 5, 6, 7, 999, -1])
    requests = torch.empty(10, dtype=torch.int32)
    qpos = torch.empty_like(requests)
    runtime._query_metadata_kernel[(10,)](
        starts, lengths, positions, requests, qpos,
        TOKENS=10, REQUESTS=4, BLOCK=4,
    )
    assert requests.tolist() == [0, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    assert qpos.tolist() == [65535, 125, 126, 127, 128, -1, -1, -1, -1, -1]


def test_window512_logical_regions_cover_all_visible_tokens():
    qpos = torch.tensor([0, 1, 511, 512, 518, 65535, -1], dtype=torch.int32)
    requests = torch.tensor([0, 1, 2, 3, 4, 5, -1], dtype=torch.int32)
    regions = torch.empty((4*7, 65), dtype=torch.int32)
    counts = torch.empty(4*7, dtype=torch.int32)
    runtime._window_regions_kernel[(4*7,)](
        qpos, requests, regions, counts, TOKENS=7, TOPK=65, WINDOW=512, BLOCK=128,
    )
    for group in range(4):
        for query, position in enumerate(qpos.tolist()):
            count = int(counts[group*7+query])
            expected = (list(range(max(0, position-511)//8, position//8+1))
                        if position >= 0 else [])
            assert regions[group*7+query, :count].tolist() == expected
            assert (regions[group*7+query, count:] == -1).all()
            actual_tokens = {r*8+t for r in expected for t in range(8)
                             if max(0, position-511) <= r*8+t <= position}
            assert actual_tokens == set(range(max(0, position-511), position+1))


@pytest.mark.parametrize("nd,np,pad", [(3, 0, 2), (0, 8, 0), (2, 8, 2)])
def test_mixed_group_major_selection_payload(nd, np, pad):
    topk, tokens = 17, nd+np+pad
    dr = torch.arange(2*max(nd, 1)*topk, dtype=torch.int64).reshape(-1, topk)
    dc = torch.arange(2*max(nd, 1), dtype=torch.int32) % topk
    words = ((dr.flip([0, 1])+64) << 32) | (dr*8)
    preg = torch.arange(2*max(np, 1)*topk, dtype=torch.int32).reshape(-1, topk)+4096
    pc = torch.arange(2*max(np, 1), dtype=torch.int32) % topk
    local_payload = torch.empty((2*tokens, topk+1), dtype=torch.int32)
    runtime._assemble_plan_kernel[((local_payload.numel()+255)//256,)](
        words, dc, preg, pc, local_payload,
        TOKENS=tokens, DECODE=nd, PREFILL=np, TOPK=topk, BLOCK=256,
    )
    expected = torch.full_like(local_payload, -1)
    for group in range(2):
        for query in range(tokens):
            row = group*tokens+query
            count = (int(dc[group*nd+query]) if query < nd
                     else int(pc[group*np+query-nd]) if query < nd+np else 0)
            expected[row, topk] = count
            if count:
                expected[row, :count] = (dr[group*nd+query, :count] if query < nd
                                        else preg[group*np+query-nd, :count])
    torch.testing.assert_close(local_payload, expected, rtol=0, atol=0)
    # Rank1 uses distinct provider-group IDs to detect gather/group reorder bugs.
    other = local_payload.clone()
    other[:, :topk] = torch.where(other[:, :topk] >= 0, other[:, :topk]+2048, -1)
    gathered = torch.cat([local_payload, other], dim=0)
    regions = torch.empty((4*tokens, topk), dtype=torch.int32)
    counts = torch.empty(4*tokens, dtype=torch.int32)
    runtime._unpack_plan_kernel[((regions.numel()+255)//256,)](
        gathered, regions, counts, ROWS=4*tokens, TOPK=topk, BLOCK=256,
    )
    torch.testing.assert_close(regions, gathered[:, :topk], rtol=0, atol=0)
    torch.testing.assert_close(counts, gathered[:, topk], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_head192_fp32_weighting_and_local_head_slice(rank):
    torch.manual_seed(106)
    tokens = 4
    backing = torch.randn(tokens, 64, 208)
    states = backing[:, :, :192]
    lse = torch.randn(2, tokens, 64)
    lse[0, 0] = -float("inf")
    lse[1, 1] = -float("inf")
    lse[:, 2] = -float("inf")
    lse[0, 3, 0] = float("nan")
    lse[1, 3, 1] = float("inf")
    weighted = torch.empty(64, tokens, 192)
    global_lse = torch.empty(tokens, 64)
    runtime._weight_rank_state_kernel[(tokens, 64)](
        states, lse, weighted, global_lse, *states.stride(),
        TOKENS=tokens, HEADS=64, RANK=rank, BLOCK_D=256,
    )
    clean = torch.where(torch.isnan(lse) | torch.isposinf(lse), -float("inf"), lse)
    peak = clean.max(0).values
    weight = torch.exp(clean-torch.where(torch.isneginf(peak), 0., peak))
    factor = weight[rank]/weight.sum(0).clamp_min(1.e-20)
    expected = states*factor.unsqueeze(-1)
    torch.testing.assert_close(weighted, expected.permute(1, 0, 2), rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(global_lse, torch.logsumexp(clean, dim=0),
                               rtol=3e-6, atol=3e-6)
    assert not weighted.isnan().any()
    scattered = weighted[rank*32:(rank+1)*32].contiguous()
    # The CPU interpreter truncates BF16 casts; verify addressing in FP32.
    # Native qualification separately requires exact BF16 rounding.
    final = torch.empty(tokens, 32, 192, dtype=torch.float32)
    runtime._finish_local_heads_kernel[((final.numel()+255)//256,)](
        scattered, final, TOKENS=tokens, HEADS=32, BLOCK=256,
    )
    torch.testing.assert_close(final, scattered.permute(1, 0, 2),
                               rtol=0, atol=0)
