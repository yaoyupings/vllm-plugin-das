# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 text decoder ported from OpenDAS/vllm-hcu step37-dp.

Reference commit: 0686260600ad725f3c594a6ad9f5dd0a16830091.
Vision execution is not enabled. Step5MTP provides the separate three-module
draft decoder. The target preserves FP32 Gemma normalization and residuals,
clamped SwiGLU, sigmoid expert routing, head-wise attention gates, and
layer-specific rotary dimensions.

Routed experts follow the checkpoint quantization configuration. Packed expert
names are replaced by prefix, preserving their scale names and values for
block-FP8 or channel-FP8 loading. Missing decoder parameters fail loading.

Full-attention layers use the reference CSA/DSA indexer and kernels. Sliding
layers explicitly use HCU FlashAttention with a compatible paged cache view.
VLLM_STEP4_DISABLE_DSA=1 bypasses the indexer for dense diagnostics; this is
equivalent only while topk * region_block_size covers the whole history.
"""

import os
import typing
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter


from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm_hcu.ops.step5_norm_quant import step5_gemma_norm_fp8_quant
from vllm.model_executor.layers.activation import SiluAndMul, SwigluStepAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    MoERunner,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm_hcu.model_executor.layers.step4_dsa import (
    Step4DSAGeometry,
    Step4DSAState,
    Step4Indexer,
    Step4SparseAttentionBackend,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)

from vllm_hcu.v1.attention.backends.flash_attn import HcuFlashAttentionBackend
from vllm_hcu.model_executor.layers.step5_attention_ops import fused_qknorm_rope
from vllm_hcu.model_executor.layers.step5_projection import (
    merged_projection, prepare_model_projections,
)
from vllm_hcu.model_executor.layers.step5_qkv_fp8 import Step5QKVFp8LinearMethod

logger = init_logger(__name__)
norm_quant_logger = init_logger("vllm.hcu.step5.norm_quant")

# Checkpoint tensors that belong to the DSA indexer. Only consulted when DSA is
# switched off: the dense-equivalent path does not allocate them, and they are
# counted at load time so a skip stays distinguishable from a successful load.
_DSA_INDEXER_SUBSTRINGS = (
    "sparse_indexer_q",
    "sparse_indexer_k",
    "sparse_indexer_w",
    "sparse_indexer_z",
    "sparse_indexer_q_norm",
    "sparse_indexer_k_norm",
    "ssmax_s",
)

# Escape hatch back to the pre-DSA dense path, for A/B against this backend.
DISABLE_DSA = os.environ.get("VLLM_STEP4_DISABLE_DSA", "0") == "1"


def dsa_layer_indices(config: Any) -> list[int]:
    """Loaded layer indices that run DSA, in order.

    Read from ``sparse_config.apply_to_layer_types`` rather than assumed, and
    bound indices by ``num_hidden_layers`` for the main decoder. Checkpoint
    per-layer lists may also include MTP layers, which Step5MTP loads separately.
    """
    if DISABLE_DSA:
        return []
    sparse_config = getattr(config, "sparse_config", None)
    if not sparse_config:
        return []
    if isinstance(sparse_config, dict):

        def get(name, default=None):
            return sparse_config.get(name, default)

    else:

        def get(name, default=None):
            return getattr(sparse_config, name, default)

    if not get("enabled", False):
        return []
    wanted = set(get("apply_to_layer_types", ["full_attention"]) or [])
    layer_types = getattr(config, "layer_types", None) or []
    return [
        idx
        for idx in range(min(config.num_hidden_layers, len(layer_types)))
        if layer_types[idx] in wanted
    ]


def map_step5_checkpoint_weights(weights):
    """Keep native FP8 block scale names and values; skip vision tensors."""
    for name, tensor in weights:
        if name.startswith(
            ("vision_model.", "vit_large_projector.", "vit_downsampler.")
        ):
            continue
        yield name, tensor


class FP32ReplicatedLinear(ReplicatedLinear):
    """
    Use FP32 for higher precision.
    """

    def forward(
        self, x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        assert self.params_dtype == torch.float32
        return super().forward(x.to(torch.float32))


def _fp32_gemma_rms_norm(hidden_size: int, eps: float) -> GemmaRMSNorm:
    """GemmaRMSNorm holding its weight in fp32, on the native forward path.

    Every norm weight in the Step-4 checkpoint is serialized as fp32, and the
    reference implementation keeps it that way. GemmaRMSNorm allocates with the
    default dtype instead, which rounds the weight to bf16; because the norm
    computes ``x * (1 + w)`` and these weights are centred on zero, rounding
    perturbs ``1 + w`` by up to ~8e-3 -- far above bf16 compute noise, on all
    four norms of all 92 layers.

    Device-specific GemmaRMSNorm kernels require the activation and the weight to
    share a dtype, so an fp32 weight also has to opt out of them. forward_native
    reduces in fp32 and returns the input dtype, which is the wanted semantics.
    """
    norm = GemmaRMSNorm(hidden_size, eps)
    norm.weight = nn.Parameter(
        torch.zeros(hidden_size, dtype=torch.float32), requires_grad=False
    )
    norm._forward_method = norm.forward_native
    return norm


class Step5MLP(nn.Module):
    """Dense FFN, used both for the four non-MoE layers and the shared expert."""

    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.limit = None
        # The split schedule is valid only for the BF16 shared expert without
        # a down-projection collective. Dense FFNs keep the ordinary path.
        self._hcu_split_shared_supported = (
            quant_config is None and not reduce_results
            and prefix.endswith(".share_expert")
        )
        layer_idx = extract_layer_index(prefix)
        swiglu_limits_shared = getattr(config, "swiglu_limits_shared", None)
        if (
            swiglu_limits_shared
            and layer_idx < len(swiglu_limits_shared)
            and swiglu_limits_shared[layer_idx] is not None
            and swiglu_limits_shared[layer_idx] != 0
        ):
            self.limit = swiglu_limits_shared[layer_idx]
            self.act_fn = SwigluStepAndMul(limit=self.limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output

    def _hcu_shared_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        return self.act_fn(gate_up)

    def _hcu_shared_combine(self, intermediate_act: torch.Tensor) -> torch.Tensor:
        output, _ = self.down_proj(intermediate_act)
        return output


class Step5FlashAttentionBackend(HcuFlashAttentionBackend):
    @staticmethod
    def get_builder_cls():
        from vllm_hcu.model_executor.layers.step5_window_metadata import (
            Step5WindowMetadataBuilder,
        )

        return Step5WindowMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # V2 queries cache specs outside a current-config context. Step5's
        # validated varlen path uses 64-token pages and has no Mamba layers.
        return [64]


class Step5FlashAttention(Attention):
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        from vllm_hcu.model_executor.layers.step5_dsa_compat import flash_cache_view

        self.kv_cache = flash_cache_view(kv_cache)


class Step5SparseAttention(Attention):
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view

        self.kv_cache = sparse_cache_view(kv_cache)


class Step5Attention(nn.Module):
    """GQA with qk-norm, a per-head sigmoid output gate, and partial rotary.

    Every weight here is bf16 in the checkpoint, so ``quant_config`` is
    deliberately not threaded into the projections -- see the module docstring.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: CacheConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
        sparse_config: Any = None,
        dsa_state: Step4DSAState | None = None,
        dsa_slot: int | None = None,
        total_num_attention_heads: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        if layer_types:
            enable_sliding_window = layer_types[self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0
        if yarn_only_types and layer_types[self.layer_idx] not in yarn_only_types:
            rope_scaling = None

        if sliding_window is not None and enable_sliding_window:
            if swa_num_attention_heads is not None:
                num_heads = swa_num_attention_heads
                self.total_num_heads = swa_num_attention_heads
        else:
            sliding_window = None

        if isinstance(rope_theta, list):
            rope_theta = rope_theta[self.layer_idx]

        self.rank = get_tensor_model_parallel_rank()
        self.partial_rotary_factor = partial_rotary_factor
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Fewer KV heads than ranks: replicate. TP8 with 4 groups gives one
            # KV head per rank and two replicas per group.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.qkv_fp8_w8a8 = os.environ.get("VLLM_STEP5_QKV_FP8_W8A8", "0") == "1"
        if self.qkv_fp8_w8a8:
            self.qkv_proj.quant_method = Step5QKVFp8LinearMethod(prefix=prefix)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            raise ValueError("rope_scaling must be a dict for Step5Attention.")

        rope_parameters: dict[str, Any] = (
            dict(rope_scaling) if rope_scaling is not None else {}
        )
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters["rope_theta"] = self.rope_theta
        rope_parameters["partial_rotary_factor"] = partial_rotary_factor

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.q_norm = _fp32_gemma_rms_norm(self.head_dim, rms_norm_eps)
        self.k_norm = _fp32_gemma_rms_norm(self.head_dim, rms_norm_eps)
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        if use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.g_proj",
            )

        self.use_rope = True
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]

        # Pair each per-layer cache view with its validated reader. V2's
        # automatic selection can otherwise choose Triton for sliding layers.
        self.is_sparse_layer = dsa_state is not None and dsa_slot is not None
        dcp_size = get_current_vllm_config().parallel_config.decode_context_parallel_size
        self.dcp2_attention_enabled = dcp_size == 2
        if dcp_size != 1:
            if (
                dcp_size != 2 or tp_size != 2
                or os.environ.get("VLLM_STEP5_DCP2_RUNTIME", "0") != "1"
                or self.total_num_heads != 64 or self.total_num_kv_heads != 4
                or self.head_dim != 192
                or (not self.is_sparse_layer and sliding_window != 512)
                or (dsa_state is not None and dsa_state.context_parallel_size != 2)
            ):
                raise ValueError(
                    "Step5 DCP requires experimental TP2/DCP2, global64Q/4KV/HD192 "
                    "and sparse or window512 Attention"
                )
            from vllm_hcu.model_executor.layers.step5_dcp2_attention import (
                Step5DCP2Attention, Step5DCP2AttentionBackend,
            )
            attention_cls, attention_backend = Step5DCP2Attention, Step5DCP2AttentionBackend
            cache_kv_heads = 4
        else:
            attention_cls = (
                Step5SparseAttention if self.is_sparse_layer else Step5FlashAttention
            )
            attention_backend = (
                Step4SparseAttentionBackend if self.is_sparse_layer else Step5FlashAttentionBackend
            )
            cache_kv_heads = self.num_kv_heads
        self.attn = attention_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=cache_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            attn_backend=attention_backend,
        )

        self.indexer: Step4Indexer | None = None
        if self.is_sparse_layer:
            if sliding_window is not None:
                raise ValueError(
                    f"layer {self.layer_idx} is both sliding and sparse; DSA "
                    f"applies to full_attention layers only"
                )
            # The indexer's parameters are attached to *this* module rather than
            # to a submodule, because that is where the checkpoint puts them.
            self.indexer = Step4Indexer(
                self,
                geometry=Step4DSAGeometry.from_config(sparse_config),
                state=dsa_state,
                dsa_slot=dsa_slot,
                num_kv_groups=self.num_kv_heads,
                hidden_size=hidden_size,
                tp_rank=self.rank,
                tp_size=tp_size,
                rms_norm_eps=rms_norm_eps,
                rope_theta=self.rope_theta,
                # The indexer rotates with the same theta and the same llama3
                # scaling as the layer it feeds, so it has to see the scaling
                # dict after the yarn_only_types filter above, not before.
                rope_scaling=rope_scaling,
                max_position=max_position,
                params_dtype=torch.get_default_dtype(),
                num_attention_heads=total_num_attention_heads or self.total_num_heads,
            )

        self.merge_attention_projection = os.environ.get("VLLM_STEP5_MERGED_ATTENTION_PROJECTION", "0") == "1"
        self.register_buffer("step5_merged_weight", None, persistent=False)
        self.register_buffer("step5_merged_bias", None, persistent=False)
        self.step5_merged_offsets = {}
        self.step5_merged_ready = False
        norm_exchange_requested = os.environ.get("VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE", "0")
        if norm_exchange_requested not in ("0", "1"):
            raise ValueError("VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE must be 0 or 1")
        pack_norm_requested = os.environ.get("VLLM_STEP5_DSA_PACK_QKNORM", "0")
        if pack_norm_requested not in ("0", "1"):
            raise ValueError("VLLM_STEP5_DSA_PACK_QKNORM must be 0 or 1")
        compiled_native_rope = False
        if (os.environ.get("VLLM_STEP5_FUSED_QK_NORM_ROPE", "0") == "1"
                or norm_exchange_requested == "1" or pack_norm_requested == "1"):
            fusion_config = get_current_vllm_config()
            norm_priority = fusion_config.kernel_config.ir_op_priority.rms_norm
            compiled_native_rope = (
                fusion_config.compilation_config.mode != 0
                and fusion_config.compilation_config.backend == "inductor"
                and bool(norm_priority) and norm_priority[0] == "native"
                and getattr(self.rotary_emb._forward_method, "__name__", "") == "forward_native"
            )
        self.fused_qk_norm_rope = (
            os.environ.get("VLLM_STEP5_FUSED_QK_NORM_ROPE", "0") == "1"
            and compiled_native_rope
            and self.head_dim == 192 and self.use_rope
            and self.rotary_emb.is_neox_style
            and self.rotary_emb.rotary_dim in (64, 192)
        )
        self.fused_dsa_pack_norm = (
            pack_norm_requested == "1" and compiled_native_rope
            and self.head_dim == 192 and self.use_rope
            and self.rotary_emb.is_neox_style
            and self.rotary_emb.rotary_dim in (64, 192)
            and self.indexer is not None and not self.dcp2_attention_enabled
            and (self.num_heads, self.num_kv_heads) in ((32, 2), (16, 1))
        )
        if self.fused_dsa_pack_norm:
            logger.info_once(
                "Step5 joint verification pack/QK-norm enabled: Q heads=%d; KV heads=%d",
                self.num_heads, self.num_kv_heads,
            )
        if self.fused_qk_norm_rope:
            logger.info_once("Step5 main Attention: fused FP32-weight QK-norm + RoPE enabled.")
        self.fused_dcp2_norm_exchange = (
            norm_exchange_requested == "1" and self.dcp2_attention_enabled
            and compiled_native_rope and self.use_rope
            and self.rotary_emb.is_neox_style and self.head_dim == 192
            and self.num_heads == 32 and self.num_kv_heads == 2
            and self.rotary_emb.rotary_dim in (64, 192)
        )
        if norm_exchange_requested == "1" and not self.fused_dcp2_norm_exchange:
            raise ValueError("Step5 fused norm exchange requires compiled native TP2/DCP2 NeoX norm/rope")
        self.max_position_embeddings = max_position
        # Step-4 uses 1/3 on full-attention layers, so take the rotary width from
        # get_rope (int(head_size * factor)) instead of assuming 1 or 1/2.
        self.rotary_dim = getattr(
            self.rotary_emb,
            "rotary_dim",
            int(self.head_dim * self.partial_rotary_factor),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor,
        qkv_quantized: torch.Tensor | None = None,
        qkv_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_index = None
        projected_gate = None
        if (qkv_quantized is None) != (qkv_scale is None):
            raise ValueError("Step5 prequantized QKV requires both values and scales")
        if qkv_quantized is not None:
            if (
                not self.qkv_fp8_w8a8 or self.merge_attention_projection
                or getattr(self, "step5_full_projection_ready", False)
                or self.qkv_proj.gather_output
            ):
                raise ValueError("Step5 norm/quant fusion requires local FP8 QKV")
            bias = self.qkv_proj.bias if not self.qkv_proj.skip_bias_add else None
            qkv = self.qkv_proj.quant_method.apply_prequantized(
                self.qkv_proj, qkv_quantized, qkv_scale, bias
            )
        elif getattr(self, "step5_full_projection_ready", False):
            from vllm_hcu.model_executor.layers.step5_full_projection import full_projection
            qkv, projected_index, projected_gate = full_projection(self, hidden_states)
        elif self.merge_attention_projection:
            qkv, projected_index, projected_gate = merged_projection(self, hidden_states)
            if qkv is None:
                qkv, _ = self.qkv_proj(hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        if self.fused_dcp2_norm_exchange:
            if self.indexer is not None:
                self.indexer.forward(hidden_states, positions, projected_index)
            cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            attn_output = self.attn.forward_qkv(
                qkv, self.q_norm.weight, self.k_norm.weight, cache,
                positions=positions, rotary_pairs=self.rotary_emb.rotary_dim // 2,
                epsilon=self.q_norm.variance_epsilon,
                plan_token=self.step4_dsa_plan_token if self.indexer is not None else None,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            joint_indexer = self.fused_dsa_pack_norm and qkv.dtype == torch.bfloat16
            if joint_indexer:
                cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
                q, k = torch.ops.vllm.step5_dsa_index_qknorm(
                    hidden_states, positions, qkv,
                    self.q_norm.weight, self.k_norm.weight, cache,
                    self.indexer.layer_name, self.step4_dsa_plan_token,
                    self.num_heads, self.num_kv_heads, self.rotary_dim // 2,
                    self.q_norm.variance_epsilon, projected_index,
                )
            elif self.fused_qk_norm_rope and qkv.dtype == torch.bfloat16:
                cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
                q, k = fused_qknorm_rope(
                    qkv, self.q_norm.weight, self.k_norm.weight, cache, positions,
                    self.num_heads, self.num_kv_heads, self.rotary_dim // 2,
                    self.q_norm.variance_epsilon,
                )
            else:
                q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
                q_by_head = self.q_norm(q_by_head.contiguous())
                q = q_by_head.view(q.shape)

                k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
                k_by_head = self.k_norm(k_by_head.contiguous())
                k = k_by_head.view(k.shape)
                if self.use_rope:
                    q, k = self.rotary_emb(positions, q, k)
            if self.indexer is not None and not joint_indexer:
                # Publishes layer.attn.step4_sparse_plan, which the DSA backend
                # reads. It needs hidden_states, which Attention.forward never sees.
                self.indexer.forward(hidden_states, positions, projected_index)
            if self.dcp2_attention_enabled:
                attn_output = self.attn(
                    q, k, v, positions=positions,
                    plan_token=self.step4_dsa_plan_token if self.indexer is not None else None,
                )
            else:
                attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            if projected_gate is None:
                extra_dims, _ = self.g_proj(hidden_states)
            else:
                extra_dims = projected_gate
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class FusedMoEBlock(nn.Module):
    """Router + shared expert + 352 routed experts.

    ``quant_config`` reaches ``FusedMoE`` and nothing else: the expert weights are
    the only fp8 tensors in the checkpoint. Layers 88-90 keep bf16 experts; the
    block checkpoint lists them in ``modules_to_not_convert`` and the channel one
    in the compressed-tensors ``ignore`` list. Either way the layer resolves to an
    unquantized method for those three ``FusedMoE`` instances.

    Note the ``ignore`` list is matched against the *unfused* expert names vLLM
    synthesises (``...moe.experts.0.gate_proj``), not against this module's
    ``...moe.experts`` prefix.
    """

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)

        self.ep_size = get_ep_group().device_group.size()
        self.ep_rank = get_ep_group().device_group.rank()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.is_sequence_parallel = (
            os.environ.get("VLLM_STEP5_SEQUENCE_PARALLEL_MOE", "0") == "1"
            and parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )
        if self.is_sequence_parallel:
            logger.info_once(
                "Step5 MoE sequence parallel: shard tokens within each TP group "
                "and replicate shared-expert weights."
            )

        self.hidden_size = config.hidden_size
        self.enable_eplb = parallel_config.enable_eplb
        self.n_routed_experts = config.moe_num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}."
            )

        self.gate = FP32ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,  # Use FP32 for higher precision.
            prefix=f"{prefix}.gate",
        )
        self.use_moe_router_bias = config.use_moe_router_bias
        assert self.use_moe_router_bias, "Only support use_moe_router_bias is true."
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.need_fp32_gate = config.need_fp32_gate
        assert (
            self.need_fp32_gate
        ), "Router logits must use FP32 precision for numerical stability."

        activation = "silu"
        swiglu_limits = config.swiglu_limits or []
        swiglu_limit = (
            swiglu_limits[self.layer_idx]
            if self.layer_idx < len(swiglu_limits)
            else None
        )
        if swiglu_limit not in (None, 0):
            swiglu_limit = float(swiglu_limit)
            assert (
                swiglu_limit == 7.0
            ), "Swiglu limit in fused moe block only support 7.0 now."
            activation = "swiglustep"

        self.share_expert = Step5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.share_expert_dim,
            hidden_act="silu",
            reduce_results=False,
            is_sequence_parallel=self.is_sequence_parallel,
            quant_config=None,
            prefix=f"{prefix}.share_expert",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.share_expert,
            gate=self.gate,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            activation=activation,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
            e_score_correction_bias=self.router_bias,
            routed_scaling_factor=config.moe_router_scaling_factor,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )[:num_tokens]
        return final_hidden_states.view(num_tokens, hidden_dim)


class Step5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        dsa_state: Step4DSAState | None = None,
        dsa_slots: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )
        self.compute_dtype = vllm_config.model_config.dtype
        if cache_config is not None:
            cache_config.sliding_window = None
        if config.att_impl_type != "GQA":
            raise ValueError(
                f"Unsupported attention implementation: {config.att_impl_type}"
            )

        num_attention_heads = None
        num_attention_groups = None
        head_dim = None
        if (
            getattr(config, "attention_other_setting", None)
            and getattr(config, "layer_types", [])
            and config.layer_types[layer_idx]
            == config.attention_other_setting["attention_type"]
        ):
            num_attention_heads = config.attention_other_setting["num_attention_heads"]
            num_attention_groups = config.attention_other_setting[
                "num_attention_groups"
            ]
            head_dim = config.attention_other_setting["head_dim"]
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        self.self_attn = Step5Attention(
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads
            if num_attention_heads
            else config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=num_attention_groups
            if num_attention_groups
            else config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
            cache_config=cache_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=getattr(config, "sliding_window", None),
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=getattr(config, "layer_types", []),
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            partial_rotary_factor=partial_rotary_factors[layer_idx]
            if partial_rotary_factors
            else 1.0,
            prefix=f"{prefix}.self_attn",
            sparse_config=getattr(config, "sparse_config", None),
            dsa_state=dsa_state if dsa_slots and layer_idx in dsa_slots else None,
            dsa_slot=(dsa_slots or {}).get(layer_idx),
            total_num_attention_heads=config.num_attention_heads,
        )

        self.use_moe = False

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        if layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(vllm_config, prefix=f"{prefix}.moe",)
            self.use_moe = True
        else:
            self.mlp = Step5MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act="silu",
                quant_config=None,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = _fp32_gemma_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = _fp32_gemma_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.prefix = prefix
        fusion = os.environ.get("VLLM_STEP5_QKV_NORM_FP8_FUSION", "0")
        if fusion not in {"0", "1"}:
            raise ValueError("VLLM_STEP5_QKV_NORM_FP8_FUSION must be 0 or 1")
        self.fused_input_norm_fp8 = fusion == "1"
        if self.fused_input_norm_fp8:
            if (
                not self.fp32_residual_connection
                or self.compute_dtype != torch.bfloat16 or self.hidden_size != 4096
                or not self.self_attn.qkv_fp8_w8a8
                or self.self_attn.merge_attention_projection
                or os.environ.get("VLLM_STEP5_FULL_MIXED_PROJECTION", "0") == "1"
                or self.self_attn.qkv_proj.gather_output
            ):
                raise ValueError(
                    "Step5 norm/FP8 fusion requires BF16 compute, FP32 norms, "
                    "4096 hidden size and local FP8 QKV without merged projections"
                )
            norm_quant_logger.info_once(
                "Step5 fused input norm/FP8 candidate enabled for M4/8/16/32/64; BF16 hidden is "
                "retained for indexer/gate; prequantized QKV uses LightOp GEMM.",
                scope="process",
            )

    def _feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_moe:
            return self.moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if not self.fp32_residual_connection:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            return self._feed_forward(hidden_states) + residual

        # fp32 residual stream. GemmaRMSNorm reduces in fp32 and returns the
        # input dtype, so an fp32 input yields an fp32 norm output that has to be
        # narrowed explicitly before the GEMMs.
        residual = hidden_states
        # Limit model adoption to qualified Decode Graph shapes. Other
        # shapes retain native normalization and the normal QKV quantizer.
        if self.fused_input_norm_fp8 and residual.shape[0] in (4, 8, 16, 32, 64):
            hidden, quantized, scale = step5_gemma_norm_fp8_quant(
                residual, self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
            attention = self.self_attn(
                positions=positions, hidden_states=hidden,
                qkv_quantized=quantized, qkv_scale=scale,
            )
        else:
            hidden = self.input_layernorm(residual).to(self.compute_dtype)
            attention = self.self_attn(positions=positions, hidden_states=hidden)
        residual = residual + attention.float()
        hidden = self.post_attention_layernorm(residual).to(self.compute_dtype)
        return residual + self._feed_forward(hidden).float()


@support_torch_compile
class Step5Model(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.vocab_size = config.vocab_size
        self.config = config
        self.compute_dtype = vllm_config.model_config.dtype
        self.fp32_residual_connection = getattr(
            config, "fp32_residual_connection", False
        )

        self.moe_num_experts = config.moe_num_experts
        self.skipped_dsa_weights = 0

        # One Step4DSAState is shared by every DSA layer on this rank: the
        # region bookkeeping it does is layer-independent, so the first layer of
        # a step pays for it and the other 22 reuse the result.
        sparse_layers = dsa_layer_indices(config)
        self.dsa_slots = {idx: slot for slot, idx in enumerate(sparse_layers)}
        self.dsa_state: Step4DSAState | None = None
        if sparse_layers:
            speculative = vllm_config.speculative_config
            num_speculative_tokens = (
                speculative.num_speculative_tokens if speculative is not None else 0
            )
            self.dsa_state = Step4DSAState(
                num_layers=len(sparse_layers),
                geometry=Step4DSAGeometry.from_config(config.sparse_config),
                # A running request holds at most one half-filled region, so
                # max_num_seqs would do; the slack absorbs rows left behind by
                # requests that finished mid-region, which are only reclaimed
                # least-recently-touched.
                max_pending=(4 if num_speculative_tokens else 2)
                * vllm_config.scheduler_config.max_num_seqs + 16,
                num_speculative_tokens=num_speculative_tokens,
                context_parallel_size=(
                    vllm_config.parallel_config.decode_context_parallel_size
                ),
                dtype=self.compute_dtype,
                # Capture bounds: the decode schedule lives in [max_num_reqs]
                # buffers at a score width covering max_model_len, so that both
                # its addresses and its shapes survive a CUDA graph replay.
                max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
                max_model_len=vllm_config.model_config.max_model_len,
            )
            logger.info(
                "step4: DSA enabled on %d of %d layers (%d...%d)",
                len(sparse_layers),
                config.num_hidden_layers,
                sparse_layers[0],
                sparse_layers[-1],
            )

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size, config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step5DecoderLayer(
                vllm_config,
                prefix=prefix,
                dsa_state=self.dsa_state,
                dsa_slots=self.dsa_slots,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = _fp32_gemma_rms_norm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.fp32_residual_connection:
            hidden_states = hidden_states.float()

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states,})

        # Narrow before the final RMSNorm, not after: normalising in fp32 and
        # casting only at lm_head is measurably different.
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(self.compute_dtype)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        assert config.num_attention_groups > 1, "Only support GQA"
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        base_layer = (
            "routed_experts.base_layer."
            if any(".base_layer." in name for name in params_dict)
            else "routed_experts."
        )
        skipped_dsa = 0

        # Prefix replacement preserves scale suffixes in the packed 3D format.
        expert_params_mapping = [
            (f".moe.experts.{base_layer}w13_weight", ".moe.gate_proj.weight", "w1"),
            (f".moe.experts.{base_layer}w13_weight", ".moe.up_proj.weight", "w3"),
            (f".moe.experts.{base_layer}w2_weight", ".moe.down_proj.weight", "w2"),
        ]

        # Per-expert format: .moe.experts.E.{gate,up,down}_proj.*
        per_expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.moe_num_experts,
        )

        disable_moe_stacked_params = [data[1] for data in expert_params_mapping]

        for name, loaded_weight in weights:
            if name.startswith("model."):
                local_name = name[len("model.") :]
                full_name = name
            else:
                local_name = name
                full_name = f"model.{name}" if name else "model"

            spec_layer = get_spec_layer_idx_from_weight_name(config, full_name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # Skip any layers beyond the main model's depth (e.g., MTP layers)
            if full_name.startswith("model.layers."):
                parts = full_name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= config.num_hidden_layers:
                        continue

            # With DSA switched off there is nothing to load the indexer
            # tensors into. Count instead of silently dropping so that a load
            # problem stays visible.
            if DISABLE_DSA and any(
                sub in local_name for sub in _DSA_INDEXER_SUBSTRINGS
            ):
                skipped_dsa += 1
                continue

            if ".moe.experts." in local_name:
                is_expert_weight = False
                for mapping in per_expert_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in local_name:
                        continue
                    is_expert_weight = True
                    name_mapped = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    if (
                        not is_expert_weight
                        and not is_pp_missing_parameter(local_name, self)
                        and local_name in params_dict
                    ):
                        param = params_dict[local_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader,
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(local_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in local_name:
                    continue
                if any(
                    disable_moe_stacked_param in local_name
                    for disable_moe_stacked_param in disable_moe_stacked_params
                ):
                    continue
                replaced_name = local_name.replace(weight_name, param_name)
                if is_pp_missing_parameter(replaced_name, self):
                    continue
                if replaced_name not in params_dict:
                    continue
                param = params_dict[replaced_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(replaced_name)
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    if weight_name not in local_name:
                        continue
                    replaced_name = local_name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(replaced_name, self):
                        continue
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    param = params_dict[replaced_name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.moe_num_experts
                    assert loaded_weight.shape[0] == moe_expert_num
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            replaced_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(replaced_name)
                    break
                else:
                    if is_pp_missing_parameter(local_name, self):
                        continue
                    if "expert_bias" in local_name:
                        logger.warning_once("ignore expert_bias")
                        continue
                    if local_name not in params_dict:
                        raise RuntimeError(
                            f"Step5 has no parameter for checkpoint tensor {local_name}"
                        )
                    param = params_dict[local_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(local_name)

        self.skipped_dsa_weights = skipped_dsa
        if skipped_dsa:
            logger.info(
                "step4: VLLM_STEP4_DISABLE_DSA is set, so %d DSA indexer "
                "tensor(s) were skipped; dense attention is exact only for "
                "sequences up to topk * region_block_size tokens.",
                skipped_dsa,
            )
        return loaded_params


class Step5ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    # Required so quantization exclude lists match fused module prefixes.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".share_expert.": ".moe.share_expert."}
    )

    def process_weights_after_loading(self) -> None:
        prepare_model_projections(self)

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "",
    ):
        super().__init__()
        from vllm_hcu.model_executor.layers.fused_moe.step5_bf16_tiles import (
            configure_step5_bf16_tiles,
        )
        configure_step5_bf16_tiles(vllm_config)
        config = vllm_config.model_config.hf_config
        self.compute_dtype = vllm_config.model_config.dtype
        # The final three expert layers are BF16; exclusion entries in the
        # checkpoint name fused parameters instead of the owning module.
        quant_config = vllm_config.quant_config
        if quant_config is not None and hasattr(quant_config, "ignored_layers"):
            ignored = list(quant_config.ignored_layers or [])
            for name in tuple(ignored):
                if name.endswith((".experts.w13_weight", ".experts.w2_weight")):
                    module_name = name.rsplit(".", 1)[0] + ".routed_experts"
                    if module_name not in ignored:
                        ignored.append(module_name)
            quant_config.ignored_layers = ignored
        self.model = Step5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.moe_layers: list[MoERunner] = []
        self.moe_mlp_layers: list[FusedMoEBlock] = []
        example_layer: FusedMoEBlock | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Step5DecoderLayer)
            if hasattr(layer, "moe") and isinstance(layer.moe, FusedMoEBlock):
                example_layer = layer.moe
                self.moe_layers.append(layer.moe.experts)
                self.moe_mlp_layers.append(layer.moe)

        assert len(self.moe_layers) > 0, "No MoE layers found in the model."
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.norm(hidden_states).to(self.compute_dtype)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.moe_mlp_layers:
            layer.n_local_physical_experts = num_local_physical_experts
            layer.n_physical_experts = num_physical_experts
            layer.n_redundant_experts = self.num_redundant_experts
            layer.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            map_step5_checkpoint_weights(weights), mapper=self.hf_to_vllm_mapper
        )
        # A shard can interleave model.*, lm_head.*, then model.* again. Validate
        # once the complete stream is consumed, after all decoder callbacks.
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise RuntimeError(
                f"Step5 checkpoint did not load parameters: {sorted(missing)}"
            )
        return loaded


class Step5ForConditionalGeneration(Step5ForCausalLM):
    """Text-only execution of a composite checkpoint; no image API is advertised."""

    def __init__(self, *, vllm_config, prefix=""):
        text_config = vllm_config.model_config.hf_config.text_config
        super().__init__(
            vllm_config=vllm_config.with_hf_config(
                text_config, architectures=["Step5ForCausalLM"]
            ),
            prefix=prefix,
        )
