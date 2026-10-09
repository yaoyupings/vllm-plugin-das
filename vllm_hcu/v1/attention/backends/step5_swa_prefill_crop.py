# SPDX-License-Identifier: Apache-2.0
"""Bound Step5 sliding Attention's paged KV conversion to its live window."""

import torch
import triton
import triton.language as tl


@triton.jit
def _crop_swa_table(
    seq, cuq, table, out_seq, out_table,
    OLD_COLS: tl.constexpr, NEW_COLS: tl.constexpr,
    PAGE: tl.constexpr, LEFT: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    end = tl.load(seq + row)
    qlen = tl.load(cuq + row + 1) - tl.load(cuq + row)
    first_page = tl.maximum(end - qlen - LEFT, 0) // PAGE
    length = end - first_page * PAGE
    col = tl.arange(0, BLOCK)
    old_col = first_page + col
    live = ((col < NEW_COLS) & (old_col < OLD_COLS)
            & (col < tl.cdiv(length, PAGE)))
    page = tl.load(table + row * OLD_COLS + old_col, live, 0)
    tl.store(out_table + row * NEW_COLS + col, page, col < NEW_COLS)
    tl.store(out_seq + row, length)


def crop_sliding_paged_kv(
    seqused_k, cu_seqlens_q, block_table, max_seqlen_q, max_seqlen_k,
    window_left, block_size=64,
):
    """Remove only pages masked for *every* query in each request.

    Shifting K and its right-aligned query positions by the same page-aligned
    prefix preserves native causal/window coordinates. At most PAGE-1 extra
    prefix tokens survive and remain masked by the unchanged Attention window.
    Lengths and offsets stay on device; no routing/count read synchronizes CPU.
    """
    if block_size != 64 or window_left < 0 or max_seqlen_q <= 1:
        raise ValueError("Step5 SWA crop requires multi-query batches, "
                         "a finite left window and 64-token pages")
    if (
        block_table.ndim != 2 or seqused_k.ndim != 1 or cu_seqlens_q.ndim != 1
        or block_table.shape[0] != seqused_k.shape[0]
        or cu_seqlens_q.shape[0] != seqused_k.shape[0] + 1
        or not block_table.is_contiguous() or not seqused_k.is_contiguous()
        or not cu_seqlens_q.is_contiguous()
        or max_seqlen_k <= 0 or block_table.shape[1] <= 0
    ):
        raise ValueError("Invalid Step5 SWA table geometry")
    bound = min(max_seqlen_k, max_seqlen_q + window_left + block_size - 1)
    columns = min(block_table.shape[1], triton.cdiv(bound, block_size))
    cropped = torch.empty((block_table.shape[0], columns),
                          device=block_table.device, dtype=block_table.dtype)
    lengths = torch.empty_like(seqused_k)
    _crop_swa_table[(block_table.shape[0],)](
        seqused_k, cu_seqlens_q, block_table, lengths, cropped,
        block_table.shape[1], columns, block_size, window_left,
        triton.next_power_of_2(columns),
    )
    return lengths, cropped, bound
