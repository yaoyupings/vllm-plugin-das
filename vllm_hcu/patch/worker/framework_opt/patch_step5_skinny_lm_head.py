# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType

import torch

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "worker.framework_opt.step5_skinny_lm_head"
TARGETS = (
    "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",
    "vllm.model_executor.layers.logits_processor.LogitsProcessor.get_top_tokens",
)
_MARKER = "_vllm_hcu_step5_skinny_lm_head_applied"

SIGNATURES = (
    (
        "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
    (
        "vllm.model_executor.layers.logits_processor.LogitsProcessor.get_top_tokens",
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os
    from vllm_hcu.platforms import envs as henvs

    cls = getattr(target, "LogitsProcessor", None)
    original = require_callable(cls, "_apply_head", TARGETS[0])
    original_top = require_callable(cls, "get_top_tokens", TARGETS[1])

    @functools.wraps(original)
    def apply_head(self, lm_head, hidden_states, embedding_bias=None):
        # Only processors created by the Step5 models opt into this kernel.
        if (
            henvs.optional_custom_op_enabled(
                os.environ.get("STEP5_SKINNY_LM_HEAD", "1") != "0"
            )
            and getattr(self, "_step5_skinny_lm_head", False)
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
        ):
            from vllm_hcu.model_executor.layers import skinny_lm_head

            if (
                embedding_bias is None
                and type(lm_head.quant_method)
                in (target.UnquantizedEmbeddingMethod, target.UnquantizedLinearMethod)
                and skinny_lm_head.usable(hidden_states, lm_head.weight)
            ):
                target.logger.info_once(
                    "lm_head: using the skinny Triton GEMM for M <= %d",
                    skinny_lm_head.MAX_M,
                )
                return skinny_lm_head.skinny_linear(hidden_states, lm_head.weight)
        return original(self, lm_head, hidden_states, embedding_bias)

    @functools.wraps(original_top)
    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        # Fused lm_head + local argmax for the MTP draft: the logits are never written,
        # only one (value, index) pair per rank reaches the reduction. Same tokens as
        # the logits argmax; see skinny_lm_head.skinny_top_pairs.
        from vllm_hcu.model_executor.layers import skinny_lm_head

        if (
            skinny_lm_head.TOP_PAIRS
            and getattr(self, "_step5_skinny_lm_head", False)
        ):
            tp_size = lm_head.tp_size
            if (
                tp_size > 1
                and self.soft_cap is None
                and self.scale == 1.0
                and embedding_bias is None
                and (
                    self.head_dtype is None
                    or self.head_dtype == hidden_states.dtype
                )
                and type(lm_head.quant_method)
                in (
                    target.UnquantizedEmbeddingMethod,
                    target.UnquantizedLinearMethod,
                )
                and skinny_lm_head.usable(hidden_states, lm_head.weight)
            ):
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                    tensor_model_parallel_all_reduce,
                )

                # The (value, index) pairs are gathered by an all-reduce of a
                # [batch, tp, 2] buffer that is zero outside this rank's slot (exact
                # in fp32; the custom all-reduce is faster than the NCCL all-gather).
                n_valid = (
                    lm_head.weight.shape[0]
                    - lm_head.shard_indices.num_org_vocab_padding
                )
                pairs = skinny_lm_head.skinny_top_pairs(
                    hidden_states,
                    lm_head.weight,
                    n_valid,
                    lm_head.shard_indices.org_vocab_start_index,
                    get_tensor_model_parallel_rank(),
                    tp_size,
                )
                gathered = tensor_model_parallel_all_reduce(pairs)
                max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
                top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
                return top_tokens.squeeze(-1).to(torch.int64)
        return original_top(self, lm_head, hidden_states, embedding_bias)

    cls._apply_head = apply_head
    cls.get_top_tokens = get_top_tokens
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
