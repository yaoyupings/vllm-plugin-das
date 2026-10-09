# SPDX-License-Identifier: Apache-2.0
"""Logical-page capacity for Step5's experimental BF16 DCP2 window cache."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from vllm.v1.kv_cache_interface import KVQuantMode, SlidingWindowSpec


@dataclass(frozen=True, kw_only=True)
class Step5DCP2SlidingWindowSpec(SlidingWindowSpec):
    """Local page64 stores one parity of a global page128 with all four KV heads.

    Keep the upstream SlidingWindowManager's recycling behavior. Its base
    constructor already multiplies its allocation block_size by DCP size;
    admission and startup sizing must use that same logical page128.
    Prefix-cache lookup is not supported for this experimental path.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.block_size != 64 or self.num_kv_heads != 4
            or self.head_size != 192 or self.head_size_v != 192
            or self.dtype != torch.bfloat16
            or self.sliding_window != 512 or self.extra_retained_tokens < 0
            or self.kv_quant_mode != KVQuantMode.NONE
            or self.tokens_per_state != 1
            or self.num_head_slots not in (None, 4)
            or self.state_content_bytes not in (None, 768)
        ):
            raise ValueError(
                "Step5 DCP2 SWA requires local64/four KV heads/BF16 HD192/window512"
            )

    @property
    def prefix_cacheable(self) -> bool:
        return False

    def is_uniform_with_collection(self, kv_cache_specs) -> bool:
        # A collection represented by this spec must be DCP-aware throughout.
        # Integration must convert every SWA layer (including MTP); the stock
        # spec's grouping method does not reject a mixed collection when first.
        return all(
            isinstance(spec, Step5DCP2SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )

    @classmethod
    def from_spec(cls, spec: SlidingWindowSpec) -> "Step5DCP2SlidingWindowSpec":
        return cls(**{
            field.name: getattr(spec, field.name)
            for field in fields(SlidingWindowSpec) if field.init
        })

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        if max_in_flight_tokens < 0 or max_model_len < 1:
            raise ValueError("Invalid Step5 window-cache admission bounds")
        live_tokens = min(
            self.sliding_window - 1 + self.extra_retained_tokens + max_in_flight_tokens,
            max_model_len,
        )
        logical_page_size = self.block_size * 2
        # The window may begin inside a logical page, requiring one extra page.
        return (live_tokens + logical_page_size - 1) // logical_page_size + 1

    def max_memory_usage_bytes(self, vllm_config) -> int:
        parallel = vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 2
            or parallel.prefill_context_parallel_size != 1
            or vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError("Step5 DCP2 SWA requires DCP2/PCP1 and no prefix cache")
        return self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        ) * self.page_size_bytes
