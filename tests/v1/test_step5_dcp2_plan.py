# SPDX-License-Identifier: Apache-2.0
"""Retain logical MTP selections only when the DCP2 reader needs them."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import step4_dsa as dsa


@pytest.mark.parametrize("cp,short,expected_logical", [
    (1, True, False), (2, True, True),
    (1, False, True), (2, False, True),
])
def test_short_verification_retains_dcp2_logical_ids(
    monkeypatch, cp, short, expected_logical
):
    monkeypatch.setenv("VLLM_STEP5_DSA_SHORT_QUERY", "1" if short else "0")
    monkeypatch.setattr(dsa, "_STEP5_DSA_SELECTOR", "1")
    monkeypatch.setattr(dsa, "PREFILL_QK_FP8", False)
    monkeypatch.setattr(dsa, "PREFILL_PV_FP8", False)
    tokens, groups, topk = 4, 2, 3
    packed = torch.zeros(groups*tokens, topk+1, dtype=torch.int32)
    counts = torch.full((groups*tokens,), topk, dtype=torch.int32)
    logical = torch.arange(groups*tokens*(topk+1), dtype=torch.int32).reshape(
        groups*tokens, topk+1)
    calls = []

    def selector(logits, positions, table, request_indices, **kwargs):
        calls.append(kwargs["return_logical_ids"])
        assert kwargs["regions_per_page"] == (16 if cp == 2 else 8)
        assert request_indices.tolist() == [0]*8
        return (packed, counts, logical) if kwargs["return_logical_ids"] else (packed, counts)

    monkeypatch.setattr(dsa, "prefill_sparse_meta_fused", selector, raising=False)
    md = SimpleNamespace(
        num_decodes=0, num_prefills=1, num_decode_tokens=0,
        num_actual_tokens=tokens, positions=torch.arange(65532,65536,dtype=torch.int32),
        prefill_request_indices=torch.zeros(tokens,dtype=torch.int32),
        max_query_len=4, schedule=SimpleNamespace(verification=True),
        block_table=torch.zeros(1,512,dtype=torch.int32),
    )
    plan = dsa.build_plan(
        md, torch.zeros(groups*tokens,8192), None,
        topk=topk, num_kv_groups=groups,
        regions_per_page=16 if cp == 2 else 8, context_parallel_size=cp,
    )
    assert plan.md is md
    assert calls == [expected_logical]
    assert plan.prefill[0] is packed
    assert plan.prefill[1] is counts
    assert plan.prefill[2] is (logical if expected_logical else None)


def test_unsupported_context_parallel_selection_fails():
    with pytest.raises(ValueError, match="CP1"):
        dsa.build_plan(
            SimpleNamespace(), None, None, topk=512,
            num_kv_groups=2, regions_per_page=16, context_parallel_size=4,
        )
