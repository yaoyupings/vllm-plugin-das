# SPDX-License-Identifier: Apache-2.0
"""Register preview configuration types before vLLM parses config.json."""

from types import ModuleType

TARGET_MODULE = "vllm.transformers_utils.config"
PATCH_ID = "platform.core_fix.step5_config"
TARGETS = (f"{TARGET_MODULE}._CONFIG_REGISTRY",)


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, "_vllm_hcu_step5_config_registered", False):
        return False
    from vllm_hcu.step5_config import register_step5_configs

    register_step5_configs(module)
    module._vllm_hcu_step5_config_registered = True
    return True
