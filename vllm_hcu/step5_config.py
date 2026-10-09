# SPDX-License-Identifier: Apache-2.0
"""Configuration for the September 2026 Step5 preview checkpoint.

Registration does not assert that the sparse attention algorithm is supported.
The decoder uses the CSA backend ported from OpenDAS/vllm-hcu step37-dp.
"""
import os
from transformers import PretrainedConfig


class Step5TextConfig(PretrainedConfig):
    model_type = "step5"

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=13824,
        num_hidden_layers=92,
        num_attention_heads=64,
        num_attention_groups=4,
        head_dim=192,
        vocab_size=128896,
        max_position_embeddings=1048576,
        rms_norm_eps=1e-5,
        moe_num_experts=352,
        moe_top_k=8,
        moe_intermediate_size=1536,
        share_expert_dim=1536,
        moe_layer_list=None,
        moe_layers_enum=None,
        layer_types=None,
        rope_theta=10000.0,
        partial_rotary_factors=None,
        rope_scaling=None,
        sparse_config=None,
        swiglu_limits=None,
        swiglu_limits_shared=None,
        bos_token_id=0,
        eos_token_id=None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        kwargs.setdefault("architectures", ["Step5ForCausalLM"])
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("att_impl_type", "GQA")
        kwargs.setdefault("fp32_residual_connection", True)
        kwargs.setdefault("norm_dtype", "float32")
        kwargs.setdefault("use_moe", True)
        kwargs.setdefault("use_moe_router_bias", True)
        kwargs.setdefault("need_fp32_gate", True)
        kwargs.setdefault("norm_expert_weight", True)
        kwargs.setdefault("moe_router_scaling_factor", 3.0)
        kwargs.setdefault("moe_router_activation", "sigmoid")
        kwargs.setdefault("use_head_wise_attn_gate", True)
        kwargs.setdefault("num_nextn_predict_layers", 3)
        kwargs.setdefault("sliding_window", 512)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=[1, 2] if eos_token_id is None else eos_token_id,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_key_value_heads = num_attention_groups
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.moe_num_experts = moe_num_experts
        self.num_experts_per_tok = self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        if moe_layers_enum is None:
            if moe_layer_list is None:
                moe_layer_list = list(range(3, num_hidden_layers - 1))
            moe_layers_enum = ",".join(map(str, moe_layer_list))
        self.moe_layers_enum = moe_layers_enum
        self.moe_layer_list = [int(i) for i in moe_layers_enum.split(",") if i]
        self.layer_types = layer_types or [
            "full_attention" if i % 4 == 3 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.rope_theta = rope_theta
        self.partial_rotary_factors = (
            partial_rotary_factors or [1.0] * num_hidden_layers
        )
        self.rope_scaling = rope_scaling
        self.sparse_config = {} if sparse_config is None else dict(sparse_config)
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        if (
            self.sparse_config.get("enabled")
            and os.environ.get("VLLM_STEP4_DISABLE_DSA") != "1"
        ):
            from vllm.config.compilation import CompilationConfig

            op = "vllm::step4_dsa_index"
            if op not in CompilationConfig._attention_ops:
                CompilationConfig._attention_ops.append(op)
            if os.environ.get("VLLM_STEP5_DSA_PACK_QKNORM", "0") == "1":
                op = "vllm::step5_dsa_index_qknorm"
                if op not in CompilationConfig._attention_ops:
                    CompilationConfig._attention_ops.append(op)
        if os.environ.get("VLLM_STEP5_DCP2_RUNTIME", "0") == "1":
            from vllm.config.compilation import CompilationConfig

            op = "vllm::step5_dcp2_attention"
            if op not in CompilationConfig._attention_ops:
                CompilationConfig._attention_ops.append(op)


class Step5VisionConfig(PretrainedConfig):
    model_type = "perception_encoder"


class Step5Config(PretrainedConfig):
    model_type = "step5_vl"
    is_composition = True
    sub_configs = {"text_config": Step5TextConfig, "vision_config": Step5VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        kwargs.pop("model_type", None)
        kwargs.setdefault("architectures", ["Step5ForConditionalGeneration"])
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.text_config = (
            text_config
            if isinstance(text_config, PretrainedConfig)
            else Step5TextConfig(**(text_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, PretrainedConfig)
            else Step5VisionConfig(**(vision_config or {}))
        )


def register_step5_configs(module):
    from transformers import AutoConfig

    for name, cls in (("step5", Step5TextConfig), ("step5_vl", Step5Config)):
        AutoConfig.register(name, cls, exist_ok=True)
        module._CONFIG_REGISTRY[name] = cls
