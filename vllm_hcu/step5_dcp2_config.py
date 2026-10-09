# SPDX-License-Identifier: Apache-2.0
"""Configuration contract for Step5's gathered-head TP2/DP4/DCP2 path."""
from __future__ import annotations

import copy
import functools
import os

_STEP5_ARCHITECTURES = frozenset({
    "Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP",
})


def validate_step5_dcp2_parallel(model_config, parallel_config) -> bool:
    architectures = getattr(model_config, "architectures", ()) or ()
    if len(architectures) != 1 or architectures[0] not in _STEP5_ARCHITECTURES:
        return False
    if parallel_config.decode_context_parallel_size == 1:
        return False
    if os.getenv("VLLM_STEP5_DCP2_RUNTIME", "0") != "1":
        raise ValueError("Step5 DCP2 requires VLLM_STEP5_DCP2_RUNTIME=1")
    topology = (
        parallel_config.tensor_parallel_size,
        parallel_config.data_parallel_size,
        parallel_config.decode_context_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.pipeline_parallel_size,
    )
    if topology != (2, 4, 2, 1, 1):
        raise ValueError(f"Step5 DCP2 requires TP2/DP4/DCP2/PCP1/PP1, got {topology}")
    if (model_config.model_arch_config.total_num_attention_heads != 64
        or model_config.get_total_num_kv_heads() != 4
        or model_config.get_head_size() != 192 or model_config.use_mla):
        raise ValueError("Step5 DCP2 requires GQA global64Q/4KV/HD192")
    if (parallel_config.cp_kv_cache_interleave_size != 1
        or parallel_config.dcp_comm_backend != "ag_rs"
        or parallel_config.dcp_q_replicate):
        raise ValueError("Step5 DCP2 requires AG/RS, interleave1, and TP-sharded QKV")
    if (not parallel_config.enable_expert_parallel
        or parallel_config.all2all_backend != "deepep_low_latency"
        or parallel_config.enable_dbo or parallel_config.enable_eplb):
        raise ValueError("Step5 DCP2 requires EP8/DeepEP-LL without DBO/EPLB")
    return True


def wrap_step5_parallel_verification(original):
    @functools.wraps(original)
    def verify(self, parallel_config):
        if not validate_step5_dcp2_parallel(self, parallel_config):
            return original(self, parallel_config)
        # Stock GQA DCP only supports replicated KV heads (TP>total KV).
        # Step5 explicitly gathers all four heads before context sharding.
        # Preserve every TP/DP/EP/PP/MM check in the original validator.
        stock_view = copy.copy(parallel_config)
        stock_view.decode_context_parallel_size = 1
        return original(self, stock_view)
    return verify


def validate_step5_dcp2_config(config) -> bool:
    if not validate_step5_dcp2_parallel(config.model_config, config.parallel_config):
        return False
    import torch

    model = config.model_config
    text = model.hf_text_config
    if (model.dtype != torch.bfloat16 or text.hidden_size != 4096
        or text.sliding_window != 512
        or getattr(text, "swa_num_attention_heads", None) not in (None, 64)):
        raise ValueError("Step5 DCP2 requires BF16 hidden4096 and global64-head SWA512")
    sparse = text.sparse_config
    required = {
        "enabled": True, "proxy_dim": 256, "topk": 512,
        "region_block_size": 8, "num_provider_groups": 4,
        "sparse_indexer_num_heads": 16,
    }
    if any(sparse.get(key) != value for key, value in required.items()):
        raise ValueError("Step5 DCP2 requires checkpoint-native CSA/SSMax fine8 geometry")
    cache = config.cache_config
    if (cache.block_size != 64 or cache.cache_dtype not in ("auto", "bfloat16")
        or cache.enable_prefix_caching or cache.kv_offloading_size is not None):
        raise ValueError("Step5 DCP2 requires BF16 local64 cache without prefix caching/offload")
    if not config.use_v2_model_runner or config.kernel_config.moe_backend != "deep_gemm":
        raise ValueError("Step5 DCP2 requires Model Runner V2 and DeepGEMM")
    if (os.getenv("VLLM_KV_CACHE_LAYOUT") != "LBNHC"
        or os.getenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE", "0") != "1"
        or os.getenv("VLLM_STEP5_SP_LL_WORKSPACE", "0") != "1"):
        raise ValueError("Step5 DCP2 requires LBNHC cache and bounded SP DeepEP-LL workspace")
    if (os.getenv("VLLM_STEP4_DSA_FP8_MATH") != "off"
        or os.getenv("VLLM_STEP4_DISABLE_DSA", "0") == "1"
        or os.getenv("VLLM_STEP4_DSA_DENSE_EQUIV", "0") == "1"):
        raise ValueError("Step5 DCP2 requires exact sparse DSA with BF16 Attention math")
    speculative = config.speculative_config
    if speculative is not None and (
        speculative.method != "mtp" or speculative.num_speculative_tokens != 3
    ):
        raise ValueError("Step5 DCP2 supports checkpoint MTP3 or no speculation")
    if config.lora_config is not None:
        raise ValueError("Step5 DCP2 LoRA is not supported")
    transfer = config.kv_transfer_config
    if transfer is not None and getattr(transfer, "kv_connector", None) is not None:
        raise ValueError("Step5 DCP2 P/D disaggregation is not supported")
    return True


def step5_context_parallel_supported(config) -> bool:
    """Reuse the proven DCP1 paths or the explicitly validated DCP2 envelope."""
    if config.parallel_config.decode_context_parallel_size == 1:
        return True
    if os.getenv("VLLM_STEP5_DCP2_RUNTIME", "0") != "1":
        return False
    return validate_step5_dcp2_config(config)


def step5_dcp2_full_graph_allowed(config) -> bool:
    """Opt in only the qualified gathered-head adapter to FULL Decode graphs."""
    if (config.parallel_config.decode_context_parallel_size != 2
        or os.getenv("VLLM_STEP5_DCP2_FULL_GRAPH", "0") != "1"):
        return False
    if not validate_step5_dcp2_config(config):
        return False
    from vllm.config.compilation import CUDAGraphMode
    if config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        raise ValueError("Step5 DCP2 full Graph opt-in requires FULL_DECODE_ONLY")
    return True
