# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_hcu.v1.attention.backends.step5_swa_prefill_crop import (
    crop_sliding_paged_kv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA/HIP unavailable")


@pytest.mark.parametrize("left", [0, 1, 63, 64, 511])
def test_page_crop_keeps_all_visible_keys(left):
    ends = [0, 1, 63, 64, 65, 575, 576, 577, 8193, 65536]
    queries = [0, 1, 2, 4, 3, 64, 65, 128, 3, 512]
    table = torch.randperm(len(ends) * 1024, device="cuda",
                          dtype=torch.int32).view(len(ends), 1024)
    seq = torch.tensor(ends, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(queries).cumsum(0).tolist()),
                      device="cuda", dtype=torch.int32)
    seq_out, table_out, bound = crop_sliding_paged_kv(
        seq, cu, table, max(queries), max(ends), left,
    )
    assert bound == min(max(ends), max(queries) + left + 63)
    for row, (end, qlen) in enumerate(zip(ends, queries)):
        start = max(end - qlen - left, 0) // 64
        length = end - start * 64
        assert seq_out[row].item() == length
        live = (length + 63) // 64
        assert torch.equal(table_out[row, :live], table[row, start:start + live])
        assert (table_out[row, live:] == 0).all()
        # Independently check the physical position of every unmasked key.
        for qpos in range(end - qlen, end):
            lo = max(qpos - left, 0)
            assert lo >= start * 64
            for keypos in (lo, qpos):
                shifted = keypos - start * 64
                assert table_out[row, shifted // 64] == table[row, keypos // 64]
                assert shifted % 64 == keypos % 64


@pytest.mark.parametrize("ends,queries", [
    ([65536, 8193], [64, 3]),
    ([65536, 8193], [512, 8]),
    ([65536, 8193], [4, 3]),
    ([575, 576, 577, 65], [64, 65, 128, 3]),
    ([63, 64, 65, 576], [2, 4, 3, 8]),
])
def test_cropped_attention_matches_native(ends, queries):
    from flash_attn import flash_attn_varlen_func
    torch.manual_seed(107)
    cols = (max(ends) + 63) // 64
    rows = len(ends)
    table = torch.randperm(rows * cols, device="cuda", dtype=torch.int32)
    table = table.view(rows, cols)
    k = torch.randn(rows * cols, 64, 2, 192,
                    device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(sum(queries), 32, 192,
                    device="cuda", dtype=torch.bfloat16)
    seq = torch.tensor(ends, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(queries).cumsum(0).tolist()),
                      device="cuda", dtype=torch.int32)

    def attention(lengths, pages, maxk):
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, max_seqlen_q=max(queries),
            seqused_k=lengths, max_seqlen_k=maxk, block_table=pages,
            causal=True, window_size=(511, 0),
        )

    expected = attention(seq, table, max(ends))
    lengths, pages, maxk = crop_sliding_paged_kv(
        seq, cu, table, max(queries), max(ends), 511,
    )
    actual = attention(lengths, pages, maxk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
