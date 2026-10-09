# SPDX-License-Identifier: Apache-2.0
"""Fused integer bookkeeping for Step5's exact Torch TopK fallback.

Float-to-rank encoding, low-ID tie breaks and ascending selected IDs are
unchanged. Torch still selects and sorts the same integer ranks.
"""
import torch
from vllm.triton_utils import triton, tl
from vllm.logger import init_logger

_selector_logger = init_logger("vllm.hcu.step5.sparse_selector")


@triton.jit
def _ordered_float_key_gfx938(scores):
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(
        scores < 0.0,
        (~bits) & 0xFFFFFFFF,
        bits | 0x80000000,
    )


@triton.jit
def _radix_topk_threshold_gfx938(
    logits_ptr,
    row,
    stride_logits_row,
    visible,
    budget,
    seq_regions,
    BLOCK_R: tl.constexpr,
):
    """Step4 radix threshold with the DTK scalar/vector workaround.

    DTK's gfx938 backend assigns ``remaining`` to an SGPR pair when the
    caller compares it with a vector after multiple region tiles. Broadcast
    both return values here so the caller receives VGPR operands. The radix
    arithmetic and tie semantics are otherwise identical to the vendored
    kernel.
    """
    # Graph buffers cover max_model_len; skip masked tiles beyond this row's
    # causal horizon. The skipped tiles contribute zero to every histogram.
    threshold = budget.to(tl.int64) * 0
    remaining = budget
    bucket_ids = tl.arange(0, 256)
    for radix_pass in tl.static_range(4):
        shift = 32 - 8 * (radix_pass + 1)
        histogram = tl.zeros((256,), dtype=tl.int32)
        for base in tl.range(0, visible, BLOCK_R):
            regions = base + tl.arange(0, BLOCK_R)
            live = regions < visible
            scores = tl.load(
                logits_ptr + row * stride_logits_row + regions,
                mask=live,
                other=0.0,
            )
            keys = _ordered_float_key_gfx938(scores)
            if radix_pass == 0:
                matches = live
            else:
                high = shift + 8
                matches = live & ((keys >> high) == (threshold >> high))
            buckets = tl.where(
                matches, ((keys >> shift) & 255).to(tl.int32), 0
            )
            counts = tl.histogram(buckets, 256)
            parked = tl.sum((~matches).to(tl.int32))
            histogram += counts - tl.where(bucket_ids == 0, parked, 0)
        total = tl.sum(histogram)
        above = total - tl.cumsum(histogram, axis=0)
        chosen = tl.min(tl.where(above < remaining, bucket_ids, 256))
        remaining -= tl.sum(tl.where(bucket_ids > chosen, histogram, 0))
        threshold |= chosen.to(tl.int64) << shift
    lanes = tl.arange(0, BLOCK_R)
    return (
        threshold + lanes.to(tl.int64) * 0,
        remaining.to(tl.int32) + lanes * 0,
    )


@triton.jit(do_not_specialize=["width", "stride_row", "stride_col"])
def _make_ranks(logits, lengths, ranks, width,
                stride_row, stride_col,
                tie_bits: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(lengths + row).to(tl.int64), 0), width)
    score = tl.load(logits + row * stride_row + ids * stride_col,
                    mask=ids < width, other=0.0)
    bits = score.to(tl.int32, bitcast=True).to(tl.uint32)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits,
                       bits ^ 0x80000000).to(tl.int64)
    keys = tl.where(ids < visible, ordered, -1)
    value = (keys << tie_bits) + (width - 1 - ids).to(tl.int64)
    tl.store(ranks + row * width + ids, value, mask=ids < width)


@triton.jit(do_not_specialize=["width"])
def _mask_selected(selected, lengths, sortable, width,
                   take: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(lengths + row).to(tl.int64), 0), width)
    ids = tl.load(selected + row * take + col, mask=col < take, other=width)
    ids = tl.where(ids < visible, ids, width)
    tl.store(sortable + row * take + col, ids, mask=col < take)


@triton.jit(do_not_specialize=["width"])
def _write_ids(sorted_ids, output, width, take: tl.constexpr,
               topk: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(sorted_ids + row * take + col, mask=col < take, other=width)
    tl.store(output + row * topk + col, tl.where(ids < width, ids, -1).to(tl.int32),
             mask=col < topk)



@triton.jit(do_not_specialize=["width"])
def _sort_write_ids(selected, lengths, output, width,
                    take: tl.constexpr, topk: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(lengths + row).to(tl.int64), 0), width)
    ids = tl.load(selected + row * take + col,
                  mask=col < take, other=width).to(tl.int32)
    ids = tl.where((col < take) & (ids < visible), ids, width).to(tl.int32)
    ordered = tl.sort(ids, descending=False)
    tl.store(output + row * topk + col,
             tl.where(ordered < width, ordered, -1).to(tl.int32),
             mask=col < topk)


def region_topk_ids_fused(logits, lengths, *, topk):
    rows, width = logits.shape
    topk = int(topk)
    if logits.dtype != torch.float32:
        raise ValueError("selector expects float32 scores")
    if logits.device != lengths.device or not lengths.is_contiguous():
        raise ValueError("selector lengths must be contiguous on the score device")
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if width == 0 or topk == 0:
        out.fill_(-1)
        return out
    tie_bits = max(1, (width - 1).bit_length())
    if tie_bits > 31:
        raise ValueError("selector width exceeds exact int64 rank capacity")
    take = min(topk, width)
    chunk_rows = max(1, min(rows, (4 * 1024 * 1024) // width))
    for start in range(0, rows, chunk_rows):
        scores = logits[start:start + chunk_rows]
        lens = lengths[start:start + chunk_rows]
        count = scores.shape[0]
        ranks = torch.empty((count, width), dtype=torch.int64, device=logits.device)
        _make_ranks[(triton.cdiv(width, 256), count)](
            scores, lens, ranks, width, scores.stride(0), scores.stride(1),
            tie_bits, BLOCK=256, num_warps=4)
        selected = torch.topk(ranks, take, dim=-1, sorted=False).indices
        if width < 2**31 and topk <= 1024:
            _sort_write_ids[(count,)](
                selected, lens, out[start:start + chunk_rows], width, take, topk,
                BLOCK=triton.next_power_of_2(topk), num_warps=4)
        else:
            sortable = torch.empty_like(selected)
            _mask_selected[(count,)](selected, lens, sortable, width, take,
                                    BLOCK=triton.next_power_of_2(take), num_warps=4)
            sorted_ids = sortable.sort(dim=-1).values
            _write_ids[(triton.cdiv(topk, 256), count)](
                sorted_ids, out[start:start + chunk_rows], width, take, topk,
                BLOCK=256, num_warps=4)
    return out


@triton.jit(do_not_specialize=["table_stride"])
def _pack_regions(logical, positions, table, requests, output,
                  topk: tl.constexpr, table_stride,
                  position_stride: tl.constexpr, request_stride: tl.constexpr,
                  region_size: tl.constexpr, regions_per_page: tl.constexpr,
                  SINGLE_REQUEST: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(logical + row * topk + col, mask=col < topk, other=-1).to(tl.int64)
    safe = tl.maximum(ids, 0)
    if SINGLE_REQUEST:
        request = 0
    else:
        request = tl.load(requests + row * request_stride).to(tl.int64)
    pages = tl.load(table + request * table_stride + safe // regions_per_page,
                    mask=col < topk, other=-1).to(tl.int64)
    physical = pages * regions_per_page + safe % regions_per_page
    position = tl.load(positions + row * position_stride).to(tl.int64)
    valid = tl.minimum(tl.maximum(position + 1 - safe * region_size, 0), region_size)
    packed = tl.where((ids >= 0) & (pages >= 0),
                      physical | (valid << 24), -1).to(tl.int32)
    tl.store(output + row * topk + col, packed, mask=col < topk)


def pack_regions_fused(logical, q_positions, block_table, request_indices, *,
                       region_size, regions_per_page):
    rows, topk = logical.shape
    output = torch.empty_like(logical)
    if rows == 0 or topk == 0:
        return output
    if block_table.stride(1) != 1:
        raise ValueError("selector block-table columns must be contiguous")
    single = request_indices is None or block_table.shape[0] == 1
    requests = q_positions if single else request_indices
    _pack_regions[(triton.cdiv(topk, 256), rows)](
        logical, q_positions, block_table, requests, output,
        topk, block_table.stride(0), q_positions.stride(0), requests.stride(0),
        region_size, regions_per_page, SINGLE_REQUEST=single,
        BLOCK=256, num_warps=4)
    return output


@triton.jit
def _prefill_sparse_meta_fused_kernel(
    logits_ptr,
    q_positions_ptr,
    block_table_ptr,
    request_ptr,
    out_ptr,
    logical_out_ptr,
    counts_ptr,
    seq_regions,
    stride_logits_row,
    stride_table_row,
    stride_out_row,
    stride_logical_row,
    topk: tl.constexpr,
    region_size: tl.constexpr,
    regions_per_page: tl.constexpr,
    WRITE_LOGICAL_IDS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    QUERY_POSITION_STRIDE: tl.constexpr,
    BROADCAST_POSITIONS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    """Select, pack, and append the query's current region in one launch."""
    row = tl.program_id(0)
    position_row = row % QUERY_ROWS if BROADCAST_POSITIONS else row
    q_position = tl.load(
        q_positions_ptr + position_row * QUERY_POSITION_STRIDE
    ).to(tl.int32)
    request = tl.load(request_ptr + row).to(tl.int32)
    current_region = tl.maximum(q_position // region_size, 0)
    visible = tl.minimum(current_region, seq_regions)
    budget = tl.minimum(topk, visible)

    slots = tl.arange(0, BLOCK_TOPK)
    tl.store(
        out_ptr + row * stride_out_row + slots,
        -1,
        mask=slots < topk + 1,
    )
    if WRITE_LOGICAL_IDS:
        tl.store(
            logical_out_ptr + row * stride_logical_row + slots,
            -1,
            mask=slots < topk + 1,
        )

    if visible > 0:
        threshold, remaining = _radix_topk_threshold_gfx938(
            logits_ptr,
            row,
            stride_logits_row,
            visible,
            budget,
            seq_regions,
            BLOCK_R=BLOCK_R,
        )
        cursor = 0
        ties_taken = 0
        for base in tl.range(0, visible, BLOCK_R):
            regions = base + tl.arange(0, BLOCK_R)
            live = regions < visible
            scores = tl.load(
                logits_ptr + row * stride_logits_row + regions,
                mask=live,
                other=0.0,
            )
            keys = _ordered_float_key_gfx938(scores)
            greater = live & (keys > threshold)
            tied = live & (keys == threshold)
            tied_rank = ties_taken + tl.cumsum(tied.to(tl.int32), axis=0) - 1
            selected = greater | (tied & (tied_rank < remaining))
            positions = cursor + tl.cumsum(selected.to(tl.int32), axis=0) - 1
            safe_regions = tl.where(selected, regions, 0)
            pages = tl.load(
                block_table_ptr
                + request * stride_table_row
                + safe_regions // regions_per_page,
                mask=selected,
                other=0,
            )
            physical = pages * regions_per_page + safe_regions % regions_per_page
            valid_tokens = tl.minimum(
                tl.maximum(q_position + 1 - safe_regions * region_size, 0),
                region_size,
            )
            packed = physical | (valid_tokens << 24)
            packed = tl.where(pages >= 0, packed, -1)
            tl.store(
                out_ptr + row * stride_out_row + positions,
                packed,
                mask=selected & (positions < topk),
            )
            if WRITE_LOGICAL_IDS:
                tl.store(
                    logical_out_ptr + row * stride_logical_row + positions,
                    regions,
                    mask=selected & (positions < topk),
                )
            cursor += tl.sum(selected.to(tl.int32))
            ties_taken += tl.sum(tied.to(tl.int32))

    history_count = tl.minimum(current_region, topk)
    page = tl.load(
        block_table_ptr
        + request * stride_table_row
        + current_region // regions_per_page
    )
    physical = page * regions_per_page + current_region % regions_per_page
    valid_tokens = tl.minimum(
        tl.maximum(q_position + 1 - current_region * region_size, 0),
        region_size,
    )
    packed = physical | (valid_tokens << 24)
    tl.store(
        out_ptr + row * stride_out_row + history_count,
        tl.where(page >= 0, packed, -1),
    )
    if WRITE_LOGICAL_IDS:
        tl.store(
            logical_out_ptr + row * stride_logical_row + history_count,
            tl.where(page >= 0, current_region, -1),
        )
    tl.store(counts_ptr + row, history_count + 1)


def prefill_sparse_meta_fused(
    logits,
    q_positions,
    block_table,
    request_indices=None,
    *,
    topk,
    region_size=8,
    regions_per_page,
    return_logical_ids=False,
    num_kv_groups=1,
):
    """Build exact selection metadata from group-major scores.

    Query positions may be supplied once per token; the kernel broadcasts them
    across num_kv_groups and converts int64 positions to int32 on device.
    Request indices remain group-major, as in the original call contract.
    """
    rows, width = logits.shape
    topk = int(topk)
    num_kv_groups = int(num_kv_groups)
    if num_kv_groups < 1 or q_positions.numel() * num_kv_groups != rows:
        raise ValueError("positions must cover each token in every KV group")
    if width <= 1:
        from .step4_dsa_kernels import prefill_sparse_meta

        return prefill_sparse_meta(
            logits,
            q_positions.repeat(num_kv_groups).to(torch.int32),
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
            return_logical_ids=return_logical_ids,
        )
    if request_indices is None:
        request_indices = torch.zeros(
            rows, device=logits.device, dtype=torch.int32
        )
    if request_indices.numel() != rows:
        raise ValueError(
            "fused prefill metadata needs one request per score row"
        )
    if block_table.stride(1) != 1:
        raise ValueError("selector block-table columns must be contiguous")

    packed = torch.empty(
        (rows, topk + 1), device=logits.device, dtype=torch.int32
    )
    logical_ids = (
        torch.empty_like(packed) if return_logical_ids else None
    )
    counts = torch.empty(rows, device=logits.device, dtype=torch.int32)
    if rows == 0:
        return (packed, counts, logical_ids) if return_logical_ids else (packed, counts)
    table_stride = 0 if block_table.shape[0] == 1 else block_table.stride(0)
    _prefill_sparse_meta_fused_kernel[(rows,)](
        logits,
        q_positions,
        block_table,
        request_indices,
        packed,
        packed if logical_ids is None else logical_ids,
        counts,
        width,
        logits.stride(0),
        table_stride,
        packed.stride(0),
        packed.stride(0) if logical_ids is None else logical_ids.stride(0),
        topk=topk,
        region_size=int(region_size),
        regions_per_page=int(regions_per_page),
        WRITE_LOGICAL_IDS=return_logical_ids,
        QUERY_ROWS=q_positions.numel(),
        QUERY_POSITION_STRIDE=q_positions.stride(0),
        BROADCAST_POSITIONS=num_kv_groups != 1,
        BLOCK_R=min(2048, triton.next_power_of_2(width)),
        BLOCK_TOPK=triton.next_power_of_2(topk + 1),
    )
    if num_kv_groups > 1:
        _selector_logger.info_once(
            "Step5 sparse selector position broadcast active: dtype=%s, "
            "stride=%s, KV groups=%d",
            q_positions.dtype, tuple(q_positions.stride()), num_kv_groups,
            scope="process",
        )
    if return_logical_ids:
        return packed, counts, logical_ids
    return packed, counts


_native_region_topk_ids = None
_native_region_topk_pack = None


def _region_topk_ids_radix(logits, lengths, *, topk, block_r=2048):
    # BLOCK_R=1 is scalarized again by DTK and hits the same register-class
    # compiler bug. It is an early-request corner with no performance value.
    if logits.shape[1] <= 1:
        from .step5_dsa_compat import region_topk_ids

        return region_topk_ids(logits, lengths, topk=topk, block_r=block_r)
    return _native_region_topk_ids(
        logits, lengths, topk=topk, block_r=block_r
    )


def _region_topk_pack_radix(
    logits,
    lengths,
    q_positions,
    block_table,
    request_indices=None,
    *,
    topk,
    region_size=8,
    regions_per_page,
    block_r=2048,
    _logical_out=None,
):
    if logits.shape[1] <= 1:
        from .step5_dsa_compat import region_topk_pack

        return region_topk_pack(
            logits,
            lengths,
            q_positions,
            block_table,
            request_indices,
            topk=topk,
            region_size=region_size,
            regions_per_page=regions_per_page,
            block_r=block_r,
            _logical_out=_logical_out,
        )
    return _native_region_topk_pack(
        logits,
        lengths,
        q_positions,
        block_table,
        request_indices,
        topk=topk,
        region_size=region_size,
        regions_per_page=regions_per_page,
        block_r=block_r,
        _logical_out=_logical_out,
    )


def install_native_radix_selectors():
    """Enable the exact Triton radix selector with the gfx938 workaround."""
    global _native_region_topk_ids, _native_region_topk_pack
    from . import step4_dsa_kernels as kernels

    if _native_region_topk_ids is None:
        _native_region_topk_ids = kernels.region_topk_ids
        _native_region_topk_pack = kernels.region_topk_pack
    kernels._radix_topk_threshold = _radix_topk_threshold_gfx938
    kernels.region_topk_ids = _region_topk_ids_radix
    kernels.region_topk_pack = _region_topk_pack_radix
