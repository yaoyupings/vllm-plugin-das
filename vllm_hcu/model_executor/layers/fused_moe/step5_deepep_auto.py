# SPDX-License-Identifier: Apache-2.0
"""Step5 HT/LL experts with a single qualified gfx938 FP8 weight layout."""
from __future__ import annotations

import os
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm_hcu.model_executor.layers.fused_moe.step5_ht_workspace import (
    Step5HTChunkedTritonExperts,
)
from vllm.model_executor.utils import replace_parameter
from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
    DeepEPAutoDeepGemmExperts,
)


def step5_auto_experts_enabled(moe_config, vllm_config) -> bool:
    if os.getenv("VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS", "0") != "1":
        return False
    if vllm_config is None:
        return False
    p = vllm_config.parallel_config
    model = vllm_config.model_config
    hf = getattr(model, "hf_config", None)
    architectures = getattr(model, "architectures", None) or getattr(
        hf, "architectures", ()
    ) or ()
    mp = moe_config.moe_parallel_config
    return bool(
        any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
            for a in architectures)
        and (p.tensor_parallel_size, p.data_parallel_size) == (4, 2)
        and p.pipeline_parallel_size == 1
        and getattr(p, "decode_context_parallel_size", 1) == 1
        and getattr(p, "prefill_context_parallel_size", 1) == 1
        and getattr(mp, "use_deepep_auto_kernels", False)
        and mp.is_sequence_parallel and mp.sp_size == 4 and mp.ep_size == 8
        and not getattr(p, "enable_dbo", False)
        and not getattr(moe_config, "is_lora_enabled", False)
    )


class _Step5AutoDelegates(DeepEPAutoDeepGemmExperts):
    def prepare_workspace_metadata(
        self, topk_ids, local_num_experts, expert_map, expert_tokens_meta
    ):
        prepare = getattr(self._current(), "prepare_workspace_metadata", None)
        if prepare is None:
            return expert_tokens_meta
        return prepare(topk_ids, local_num_experts, expert_map, expert_tokens_meta)

    def set_expected_m(self, expected_m: int) -> None:
        # The selected child owns its tile/workspace hints for this call.
        self._current().set_expected_m(expected_m)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)


class Step5AutoFP8Experts(_Step5AutoDelegates):
    """Pack once; HT and LL consume the same gfx938 FP8 W6 storage."""

    _LAYOUT = "step5-gfx938-fp8-shared-ht-ll"
    _PACKING = "step5-gfx938-fp8-shared-packing"

    def _bind_weights(self, layer) -> None:
        w13, w2 = layer.w13_weight, layer.w2_weight
        ll13 = getattr(layer, "_dsv4_channel_deepgemm_masked_w13", None)
        ll2 = getattr(layer, "_dsv4_channel_deepgemm_masked_w2", None)
        if (w13.ndim != 6 or w2.ndim != 6 or ll13 is None or ll2 is None
                or ll13.data_ptr() != w13.data_ptr()
                or ll2.data_ptr() != w2.data_ptr()
                or ll13.shape != w13.shape or ll2.shape != w2.shape):
            raise RuntimeError("Step5 auto experts lost their shared packed weights")
        self.ht_experts._packed_deepgemm_problem_shape(w13, w2)
        for child in (self.ht_experts, self.ll_experts):
            child._deepgemm_w13 = w13
            child._deepgemm_w2 = w2

    def process_weights_after_loading(self, layer) -> None:
        if self._fixed_use_low_latency is not None:
            raise RuntimeError("Step5 shared weights require per-forward HT/LL")
        marker = getattr(layer, "_dsv4_channel_deepgemm_layout", None)
        if marker == self._LAYOUT and layer.w13_weight.ndim == 6:
            self._bind_weights(layer)
            return
        if marker not in (None, self._LAYOUT):
            raise RuntimeError(f"Step5 auto cannot repack weight layout {marker}")
        if not self.ht_experts._can_pack_channel_weights(layer):
            raise RuntimeError("Step5 shared weights require raw channel-wise FP8")
        if self.quant_config.use_int8_w8a8:
            raise RuntimeError("Step5 shared layout is qualified for FP8 only")
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        w13, w2 = layer.w13_weight, layer.w2_weight
        if w13.dtype not in fp8_dtypes or w2.dtype != w13.dtype:
            raise RuntimeError("Step5 shared weights require matching E4M3 dtypes")
        arch = torch.cuda.get_device_properties(w13.device).gcnArchName.split(":")[0]
        if arch != "gfx938":
            raise RuntimeError("Step5 shared HT/LL FP8 layout requires gfx938")
        # Both installed gfx938 packers use pack_int8_weight_w6_ll. This path
        # is opt-in and must be qualified against both native GEMM consumers.
        # A failed in-place pack is terminal: never retry mutated raw weights.
        layer._dsv4_channel_deepgemm_layout = self._PACKING
        with torch.no_grad():
            packed13, packed2 = self.ht_experts._pack_channel_weights(w13, w2)
        replace_parameter(layer, "w13_weight", packed13)
        replace_parameter(layer, "w2_weight", packed2)
        layer._dsv4_channel_deepgemm_masked_w13 = layer.w13_weight.detach()
        layer._dsv4_channel_deepgemm_masked_w2 = layer.w2_weight.detach()
        self._bind_weights(layer)
        layer._dsv4_channel_deepgemm_layout = self._LAYOUT
        layer._dsv4_channel_deepgemm_repacked = True


class Step5AutoBF16Experts(_Step5AutoDelegates):
    """Retain raw BF16 weights while switching standard/batched activations."""

    def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers,
                 fixed_use_low_latency=None):
        if fixed_use_low_latency is not None:
            raise RuntimeError("Step5 BF16 auto requires per-forward HT/LL")
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self._fixed_use_low_latency = None
        self._use_low_latency_snapshot = False
        self.ht_experts = Step5HTChunkedTritonExperts(moe_config, quant_config)
        self.ll_experts = BatchedTritonExperts(
            moe_config, quant_config, max_num_tokens, num_dispatchers
        )
        self.activation_config = self.ht_experts.activation_config

    @property
    def max_tokens_per_chunk(self) -> int:
        # LL has static batched activations and remains unchunked in graphs.
        # HT receives a data-dependent token count. Bound BF16 intermediates
        # before allocation so skewed routing cannot outgrow the locked pool.
        return 0 if self._use_low_latency_snapshot else self.ht_experts.max_tokens_per_chunk

    def process_weights_after_loading(self, layer) -> None:
        w13, w2 = layer.w13_weight, layer.w2_weight
        if (w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
                or w13.ndim != 3 or w2.ndim != 3
                or w13.shape[0] != w2.shape[0]
                or w13.shape[2] != w2.shape[1]
                or w13.shape[1] != 2 * w2.shape[2]):
            raise RuntimeError("Step5 BF16 auto requires canonical BF16 MoE weights")

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key is None and activation_key is None
