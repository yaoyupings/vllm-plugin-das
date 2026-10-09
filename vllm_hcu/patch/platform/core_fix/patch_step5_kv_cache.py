# SPDX-License-Identifier: Apache-2.0
# Discard temporary Step5 DSA state when V2 replaces its KV allocation.

from functools import wraps
import inspect
from types import ModuleType

TARGET_MODULE = "vllm.v1.worker.gpu.model_runner"
PATCH_ID = "platform.core_fix.step5_kv_cache"
TARGETS = (
    f"{TARGET_MODULE}.GPUModelRunner.initialize_kv_cache",
    f"{TARGET_MODULE}.check_attention_cp_compatibility",
)


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, "_vllm_hcu_step5_kv_cache", False):
        return False
    cls = module.GPUModelRunner
    original = cls.initialize_kv_cache
    cp_check = module.check_attention_cp_compatibility
    if tuple(inspect.signature(cp_check).parameters) != ("vllm_config",):
        from vllm_hcu.patch.platform.framework_opt._common import PatchCompatibilityError
        raise PatchCompatibilityError("MRV2 Attention CP compatibility signature changed")

    @wraps(cp_check)
    def check_cp(vllm_config):
        from vllm_hcu.step5_dcp2_config import validate_step5_dcp2_config
        if not validate_step5_dcp2_config(vllm_config):
            return cp_check(vllm_config)
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm_hcu.model_executor.layers.step5_dcp2_attention import (
            validate_dcp2_attention_layers,
        )
        # Stock CP expects an LSE-returning backend. Step5's custom op instead
        # computes local FP32 LSE, combines shards and scatters TP heads inside
        # the qualified bridge. Do not pretend the base forward returns LSE.
        validate_dcp2_attention_layers(
            get_layers_from_vllm_config(vllm_config, AttentionLayerBase))

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if getattr(self.model_config.hf_text_config, "model_type", None) == "step5":
            # The same model survives graph-memory profiling and real cache init.
            # Reset before builders can reuse a schedule from the previous pool.
            seen = set()
            for layer in self.model.modules():
                state = getattr(layer, "dsa_state", None)
                if state is not None and id(state) not in seen:
                    seen.add(id(state))
                    state.reset_cache()
        return original(self, *args, **kwargs)

    module.check_attention_cp_compatibility = check_cp
    cls.initialize_kv_cache = initialize
    module._vllm_hcu_step5_kv_cache = True
    return True
