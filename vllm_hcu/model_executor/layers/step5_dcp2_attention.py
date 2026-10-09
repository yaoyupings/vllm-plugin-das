# SPDX-License-Identifier: Apache-2.0
"""Experimental Step5 Attention adapter for the qualified DCP2 execution bridge.

This adapter is explicitly selected by Step5, never registered over the CUSTOM
backend slot. Stock DCP GQA validation and hybrid-cache coordinator integration
remain separate requirements before whole-model activation.
"""
from __future__ import annotations

import os
import torch

from vllm.config import get_current_vllm_config, CUDAGraphMode
from vllm.logger import init_logger
from vllm.distributed import get_dcp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.model_executor.layers.step4_dsa import (
    Step4DSAMetadata, Step4DSAMetadataBuilder,
    Step4DSAImpl, Step4SparseAttentionBackend,
)
from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view
from vllm_hcu.ops.step5_dcp2_runtime import dcp2_attention
from vllm_hcu.ops.step5_dcp2_tuning import select_short_query_splits
from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec


logger = init_logger("vllm.hcu.step5.dcp2_attention")


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


class Step5DCP2MetadataBuilder(Step4DSAMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        parallel = vllm_config.parallel_config
        if (parallel.tensor_parallel_size != 2
            or parallel.decode_context_parallel_size != 2
            or parallel.prefill_context_parallel_size != 1
            or kv_cache_spec.block_size != 64):
            raise ValueError("Step5 DCP2 metadata requires TP2/DCP2/PCP1/local64")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # SWA/MTP layers have no indexer state from which to infer CP size.
        # Their table is still indexed by global128 pages, just like DSA layers.
        md.block_size = 128
        return md


class Step5DCP2Impl(Step4DSAImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type="decoder",
                 kv_sharing_target_layer_name=None, **kwargs):
        if (num_heads != 32 or num_kv_heads != 4 or head_size != 192
            or sliding_window not in (None, 512)
            or kv_cache_dtype not in ("auto", "bfloat16")
            or attn_type != "decoder"):
            raise ValueError("Step5 DCP2 backend requires local32Q/cache4KV/BF16 HD192/SWA512")
        # The gathered-head bridge implements both sparse and window attention.
        # Reuse the base's ALiBi/soft-cap/KV-sharing/splitting checks without
        # routing its DCP1-only sliding implementation into this backend.
        super().__init__(
            num_heads, head_size, scale, num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes, sliding_window=None,
            kv_cache_dtype=kv_cache_dtype, logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name, **kwargs)
        self.sliding_window = sliding_window
        self.tuned_short_splits = _env_flag("VLLM_STEP5_DCP2_TUNED_SHORT_SPLITS", "1")
        self.fused_exchange = _env_flag("VLLM_STEP5_DCP2_FUSED_EXCHANGE", "0")
        self.fused_norm_exchange = _env_flag("VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE", "0")
        if self.fused_norm_exchange and not self.fused_exchange:
            raise ValueError("Step5 norm exchange requires fused DCP2 exchange")

        logger.info_once(
            "Step5 DCP2 Attention: tuned_short_splits=%s fused_exchange=%s fused_norm_exchange=%s",
            self.tuned_short_splits, self.fused_exchange, self.fused_norm_exchange,
            scope="process",
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Step5 DCP2 must execute through its gathered-head custom op")


class Step5DCP2AttentionBackend(Step4SparseAttentionBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_builder_cls():
        return Step5DCP2MetadataBuilder

    @staticmethod
    def get_impl_cls():
        return Step5DCP2Impl

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [64]


class Step5DCP2Attention(Attention):
    def __init__(self, *args, **kwargs):
        compilation = get_current_vllm_config().compilation_config
        if (
            compilation.cudagraph_mode != CUDAGraphMode.NONE
            and "vllm::step5_dcp2_attention" not in (compilation.splitting_ops or [])
        ):
            raise RuntimeError(
                "Step5 DCP2 requires vllm::step5_dcp2_attention in splitting_ops"
            )
        super().__init__(*args, **kwargs)

    def bind_kv_cache(self, kv_cache):
        # SWA and sparse layers both use an affine, aliased LBNHC cache view.
        # Keep SlidingWindowSpec recycling for SWA; layout does not imply
        # full-context allocation.
        self.kv_cache = sparse_cache_view(kv_cache)

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        if self.sliding_window is not None:
            return Step5DCP2SlidingWindowSpec.from_spec(spec)
        return spec

    def forward_qkv(self, qkv, q_weight, k_weight, rope_cache, *, positions,
                    rotary_pairs, epsilon, plan_token=None):
        tokens = qkv.shape[0]
        q,k,v = qkv.split([6144,384,384],dim=-1)
        result = torch.ops.vllm.step5_dcp2_attention(
            q.view(tokens,32,192),k.view(tokens,2,192),v.view(tokens,2,192),
            positions,plan_token,self.layer_name,
            qkv,q_weight,k_weight,rope_cache,rotary_pairs,epsilon)
        return result.flatten(1)

    def forward(self, query, key, value, *, positions, plan_token=None):
        tokens = query.shape[0]
        result = torch.ops.vllm.step5_dcp2_attention(
            query.view(tokens, 32, 192),
            key.view(tokens, 2, 192),
            value.view(tokens, 2, 192),
            positions, plan_token, self.layer_name,
        )
        return result.flatten(1)


def step5_dcp2_attention(query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor, positions: torch.Tensor,
                         plan_token: torch.Tensor | None,
                         layer_name: str,
                         raw_qkv: torch.Tensor | None = None,
                         q_weight: torch.Tensor | None = None,
                         k_weight: torch.Tensor | None = None,
                         rope_cache: torch.Tensor | None = None,
                         rotary_pairs: int = 32,
                         epsilon: float = 1e-6) -> torch.Tensor:
    md, layer, cache, slots = get_attention_context(layer_name)
    if not isinstance(layer, Step5DCP2Attention):
        raise RuntimeError("DCP2 op reached an incompatible Attention layer")
    if not isinstance(md, Step4DSAMetadata) or cache.numel() == 0:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot capture Step5 DCP2 without metadata and KV cache")
        return torch.zeros_like(query)
    if md.block_size != 128:
        raise RuntimeError("Step5 DCP2 requires global128 metadata pages")
    group = get_dcp_group()
    tp = get_tp_group()
    if (group.world_size != 2 or group.ranks != tp.ranks
        or group.rank_in_group != tp.rank_in_group):
        raise RuntimeError("Step5 DCP2 and TP2 must have identical ordered membership")
    tokens = query.shape[0]
    slots = slots if slots is not None else md.slot_mapping
    if slots.numel() < tokens:
        raise RuntimeError("Step5 DCP2 slot mapping does not include padded queries")
    k, v = cache[0], cache[1]
    if (k.shape[1:] != (64, 4, 192)
        or k.stride(0) != 64*k.stride(1)
        or v.stride(0) != 64*v.stride(1)):
        raise ValueError("Step5 DCP2 requires affine local64/four-head cache pages")
    k, v = k.flatten(0, 1), v.flatten(0, 1)
    window = int(layer.sliding_window or 0)
    if not window and plan_token is None:
        raise RuntimeError("Step5 DCP2 sparse op requires the indexer's dependency token")
    # Select only from static shapes/host metadata; no Graph replay sync.
    splits = (select_short_query_splits(tokens, md.max_query_len, window)
              if layer.impl.tuned_short_splits else 2)
    norm_data = None
    if raw_qkv is not None:
        if (not layer.impl.fused_norm_exchange
            or any(x is None for x in (q_weight,k_weight,rope_cache))):
            raise ValueError("Step5 raw QKV requires enabled norm exchange and its parameters")
        if layer.impl.fused_exchange and 1 <= md.max_query_len <= 4 and tokens <= 64:
            norm_data = (raw_qkv,q_weight,k_weight,rope_cache,rotary_pairs,epsilon)
        else:
            from vllm_hcu.model_executor.layers.step5_attention_ops import qknorm_rope_impl
            query,key = qknorm_rope_impl(
                raw_qkv,q_weight,k_weight,rope_cache,positions,
                32,2,rotary_pairs,epsilon)
            query=query.view(tokens,32,192)
            key=key.view(tokens,2,192)
    return dcp2_attention(
        query, key, value, k, v, slots[:tokens], positions,
        md.query_start_loc, md.seq_lens, md.block_table, group,
        window=window, plan=getattr(layer, "step4_sparse_plan", None),
        metadata=md, splits=splits, fused_exchange_enabled=layer.impl.fused_exchange,
        norm_data=norm_data,
    )


def validate_dcp2_attention_layers(layers):
    """The bridge combines FP32 shard LSE internally, returning local TP heads."""
    cached = 0
    for name, layer in layers.items():
        impl = getattr(layer, "impl", None)
        if impl is None:
            continue
        if (type(layer) is not Step5DCP2Attention or type(impl) is not Step5DCP2Impl
            or impl.num_heads != 32 or impl.num_kv_heads != 4
            or impl.head_size != 192 or abs(impl.scale - 192**-.5) > 1e-8
            or impl.kv_cache_dtype not in ("auto", "bfloat16")
            or impl.sliding_window not in (None, 512)):
            raise ValueError(f"Step5 DCP2 has an incompatible Attention layer: {name}")
        cached += 1
    if not cached:
        raise ValueError("Step5 DCP2 requires gathered-head Attention layers")


def step5_dcp2_attention_fake(query,key,value,positions,plan_token,layer_name,
                             raw_qkv=None,q_weight=None,k_weight=None,
                             rope_cache=None,rotary_pairs=32,epsilon=1e-6):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="step5_dcp2_attention", op_func=step5_dcp2_attention,
    mutates_args=[], fake_impl=step5_dcp2_attention_fake,
)
