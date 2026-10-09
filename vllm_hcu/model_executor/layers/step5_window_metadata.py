# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Limit Step5 sliding attention's paged KV reads to the visible tail.

from __future__ import annotations

import copy
import os

import torch

from vllm_hcu.v1.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
)


def truncate_window_metadata(metadata, *, block_size: int, window: int):
    if (
        window <= 0
        or metadata.use_cascade
        or metadata.causal is not True
        or metadata.block_table.shape[1] == 0
    ):
        return metadata
    # Causal attention aligns Q to the right edge of K. Keep the earliest
    # query's full window, rounding the start down to the physical page.
    # Truncating this common prefix shifts Q and K equally and preserves
    # every visible Q/K pair, including partial pages and ragged requests.
    query_lens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
    first_visible = (metadata.seq_lens - query_lens - (window - 1)).clamp_min(0)
    first_page = torch.div(first_visible, block_size, rounding_mode="floor")
    # A graph's launch bound must not change with the live context length.
    max_tail = metadata.max_query_len + window - 1 + block_size - 1
    columns = min(
        metadata.block_table.shape[1], (max_tail + block_size - 1) // block_size
    )
    offsets = first_page.to(torch.int64)[:, None] + torch.arange(
        columns, device=metadata.block_table.device, dtype=torch.int64
    )[None, :]
    valid = offsets < metadata.block_table.shape[1]
    safe_offsets = offsets.clamp_max(metadata.block_table.shape[1] - 1)
    table = torch.gather(metadata.block_table, 1, safe_offsets)
    table = torch.where(valid, table, 0)
    result = copy.copy(metadata)
    result.block_table = table
    result.seq_lens = metadata.seq_lens - first_page * block_size
    result.max_seq_len = max_tail
    return result


class Step5WindowMetadataBuilder(FlashAttentionMetadataBuilder):
    # A replacement physical block table has its original logical origin.
    # Rebuild metadata so the tail offset is applied to that table as well.
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._step5_window = int(getattr(kv_cache_spec, "sliding_window", 0) or 0)
        self._step5_window_enabled = (
            self.dcp_world_size == 1
            and self.parallel_config.prefill_context_parallel_size == 1
        )
        self._step5_tail_table = None
        self._step5_tail_lens = None
        if self._step5_window_enabled and self._step5_window > 0:
            max_queries = vllm_config.scheduler_config.max_num_batched_tokens
            max_tail = max_queries + self._step5_window - 1 + self.block_size - 1
            max_columns = (max_tail + self.block_size - 1) // self.block_size
            max_requests = vllm_config.scheduler_config.max_num_seqs
            self._step5_tail_table = torch.empty(
                (max_requests, max_columns), dtype=torch.int32, device=device
            )
            self._step5_tail_lens = torch.empty(
                max_requests, dtype=torch.int32, device=device
            )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        if not self._step5_window_enabled or self._step5_window <= 0:
            return metadata
        if (
            os.getenv("VLLM_STEP5_FUSED_WINDOW_METADATA", "0") == "1"
            and not metadata.use_cascade
            and metadata.causal is True
            and metadata.block_table.shape[1] > 0
        ):
            return self._stage_fused_tail_metadata(metadata)
        tail = truncate_window_metadata(
            metadata, block_size=self.block_size, window=self._step5_window
        )
        if tail is metadata:
            return metadata
        return self._stage_tail_metadata(tail)

    def _tail_buffers(self, rows, columns):
        # The builder runs before every replay. Stage into fixed addresses so
        # recorded attention kernels see this step's lengths and physical pages.
        if rows > self._step5_tail_table.shape[0] or columns > self._step5_tail_table.shape[1]:
            raise RuntimeError("Step5 sliding-window metadata exceeds its scheduler capacity")
        # Vendor paged attention indexes rows by the table width, without a
        # row-stride parameter. A column-sliced view of one larger arena is
        # therefore unsafe. Give each width its own contiguous, persistent
        # table; graph replay keeps the same address for that query shape.
        if not hasattr(self, "_step5_tail_tables"):
            self._step5_tail_tables = {}
        if columns not in self._step5_tail_tables:
            self._step5_tail_tables[columns] = torch.empty(
                (self._step5_tail_table.shape[0], columns),
                dtype=self._step5_tail_table.dtype,
                device=self._step5_tail_table.device,
            )
        table = self._step5_tail_tables[columns][:rows]
        lengths = self._step5_tail_lens[:rows]
        return table, lengths

    def _stage_fused_tail_metadata(self, metadata):
        from .step5_window_metadata_kernel import fused_tail_into
        max_tail = metadata.max_query_len + self._step5_window - 1 + self.block_size - 1
        rows = metadata.block_table.shape[0]
        columns = min(
            metadata.block_table.shape[1],
            (max_tail + self.block_size - 1) // self.block_size,
        )
        table, lengths = self._tail_buffers(rows, columns)
        fused_tail_into(
            metadata.seq_lens, metadata.query_start_loc, metadata.block_table,
            lengths, table, self._step5_window, self.block_size,
        )
        result = copy.copy(metadata)
        result.block_table = table
        result.seq_lens = lengths
        result.max_seq_len = max_tail
        return result

    def _stage_tail_metadata(self, metadata):
        table, lengths = self._tail_buffers(*metadata.block_table.shape)
        table.copy_(metadata.block_table)
        lengths.copy_(metadata.seq_lens)
        result = copy.copy(metadata)
        result.block_table = table
        result.seq_lens = lengths
        return result
