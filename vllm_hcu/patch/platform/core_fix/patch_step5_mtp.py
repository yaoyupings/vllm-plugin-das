# SPDX-License-Identifier: Apache-2.0
"""Resolve Step5 draft configuration without modifying the target config."""

from copy import deepcopy
from types import ModuleType
from typing import Literal, get_args

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.step5_mtp"
TARGETS = (f"{TARGET_MODULE}.SpeculativeConfig.hf_config_override",)
_original_override = None
_original_step_mtp = None


def hf_config_override(config):
    # Module-level callable: the draft hf_overrides travels to spawned DP ranks.
    architectures = getattr(config, "architectures", None) or []
    if getattr(config, "model_type", None) != "step5" and not any(
        arch in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
        for arch in architectures
    ):
        return _original_override(config)
    draft = deepcopy(getattr(config, "text_config", config))
    quantization = getattr(config, "quantization_config", None)
    if quantization is not None and getattr(draft, "quantization_config", None) is None:
        draft.quantization_config = deepcopy(quantization)
    count = getattr(draft, "num_nextn_predict_layers", 0)
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Step5 checkpoint must declare its MTP layer count")
    draft.model_type = "step5_mtp"
    draft.update({"n_predict": count, "architectures": ["Step5MTP"]})
    return draft


def use_step_mtp(self):
    # The legacy runner uses this predicate to select the Step MTP proposer.
    return _original_step_mtp(self) or (
        self.method == "mtp"
        and self.draft_model_config is not None
        and self.draft_model_config.hf_config.model_type == "step5_mtp"
    )


def apply_to_module(module: ModuleType) -> bool:
    global _original_override, _original_step_mtp
    if getattr(module, "_vllm_hcu_step5_mtp_registered", False):
        return False
    cls = module.SpeculativeConfig
    _original_override = cls.hf_config_override
    _original_step_mtp = cls.use_step3p5_mtp
    module.MTPModelTypes = Literal[get_args(module.MTPModelTypes) + ("step5_mtp",)]
    cls.hf_config_override = staticmethod(hf_config_override)
    cls.use_step3p5_mtp = use_step_mtp
    module._vllm_hcu_step5_mtp_registered = True
    return True
