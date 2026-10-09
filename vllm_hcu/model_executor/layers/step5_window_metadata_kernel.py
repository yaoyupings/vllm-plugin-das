# SPDX-License-Identifier: Apache-2.0
"""Write sliding-window tail metadata directly to Graph-stable buffers."""
import triton
import triton.language as tl

@triton.jit
def _tail_metadata(seq, cuq, table, out_seq, out_table,
                   OLD_COLS: tl.constexpr, NEW_COLS: tl.constexpr,
                   PAGE: tl.constexpr, LEFT: tl.constexpr, BLOCK: tl.constexpr,
                   SEQ_STRIDE: tl.constexpr, CU_STRIDE: tl.constexpr,
                   TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
                   OUT_SEQ_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    end = tl.load(seq + row * SEQ_STRIDE)
    qlen = tl.load(cuq + (row + 1) * CU_STRIDE) - tl.load(cuq + row * CU_STRIDE)
    first_page = tl.maximum(end - qlen - LEFT, 0) // PAGE
    col = tl.arange(0, BLOCK)
    old_col = first_page + col
    # Match existing metadata's full table, including invisible trailing cells.
    valid = (col < NEW_COLS) & (old_col < OLD_COLS)
    physical = tl.load(table + row * TABLE_ROW + old_col * TABLE_COL, valid, 0)
    tl.store(out_table + row * NEW_COLS + col, physical, col < NEW_COLS)
    tl.store(out_seq + row * OUT_SEQ_STRIDE, end - first_page * PAGE)

def fused_tail_into(seq, cuq, table, out_seq, out_table, window=512, block_size=64):
    assert block_size>0 and window>0
    assert table.ndim==2 and out_table.ndim==2
    assert seq.numel()==table.shape[0]==out_seq.numel()==out_table.shape[0]
    assert cuq.numel()==seq.numel()+1
    assert seq.ndim == cuq.ndim == out_seq.ndim == 1
    # Vendor Attention reads output rows by width; input strides may vary.
    assert out_table.is_contiguous()
    if not seq.numel():
        return
    _tail_metadata[(seq.numel(),)](seq,cuq,table,out_seq,out_table,
         table.shape[1],out_table.shape[1],block_size,window-1,
         triton.next_power_of_2(out_table.shape[1]),
         seq.stride(0),cuq.stride(0),table.stride(0),table.stride(1),
         out_seq.stride(0))
