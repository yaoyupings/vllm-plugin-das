# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add an opt-in LightOp W16A16 backend to vLLM's unquantized MoE oracle."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.worker.op_opt.moe._common import (
    PatchCompatibilityError,
    check_module_marker,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.unquantized"
PATCH_ID = "worker.op_opt.moe.oracle.unquantized_lightop_w16a16"
TARGETS = (
    f"{TARGET_MODULE}.UnquantizedMoeBackend",
    f"{TARGET_MODULE}.backend_to_kernel_cls",
    f"{TARGET_MODULE}.map_unquantized_backend",
    f"{TARGET_MODULE}.select_unquantized_moe_backend",
    f"{TARGET_MODULE}.convert_to_unquantized_kernel_format",
    f"{TARGET_MODULE}.make_unquantized_moe_kernel",
)
_MARKER = "_vllm_hcu_lightop_w16a16_oracle_applied"
_BINDING_MARKER = "_vllm_hcu_lightop_w16a16_oracle_binding"


def _normalize_backend(backend, old_enum, new_enum):
    if isinstance(backend, old_enum):
        return new_enum[backend.name]
    return backend


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    bindings = tuple(
        (target, name, _BINDING_MARKER)
        for name in (
            "UnquantizedMoeBackend",
            "backend_to_kernel_cls",
            "map_unquantized_backend",
            "select_unquantized_moe_backend",
            "convert_to_unquantized_kernel_format",
            "make_unquantized_moe_kernel",
        )
    )
    if check_module_marker(target, _MARKER, bindings):
        return False

    old_enum = require_class(target, "UnquantizedMoeBackend", TARGETS[0])
    backend_to_cls = require_callable(target, "backend_to_kernel_cls", TARGETS[1])
    map_backend = require_callable(target, "map_unquantized_backend", TARGETS[2])
    select_backend = require_callable(
        target, "select_unquantized_moe_backend", TARGETS[3]
    )
    convert = require_callable(
        target, "convert_to_unquantized_kernel_format", TARGETS[4]
    )
    make_kernel = require_callable(target, "make_unquantized_moe_kernel", TARGETS[5])
    require_parameter_names(backend_to_cls, TARGETS[1], ("backend",))
    require_parameter_names(map_backend, TARGETS[2], ("runner_backend",))
    require_parameter_names(select_backend, TARGETS[3], ("moe_config",))
    require_parameter_names(
        convert,
        TARGETS[4],
        ("unquantized_backend", "moe_config", "w13_weight", "w2_weight"),
    )
    require_parameter_names(
        make_kernel,
        TARGETS[5],
        ("quant_config", "moe_config", "backend", "experts_cls", "routing_tables"),
    )

    values = {member.name: member.value for member in old_enum}
    if "HCU_LIGHTOP_W16A16" in values:
        raise PatchCompatibilityError(
            "HCU LightOp W16A16 backend is already present outside the HCU adapter"
        )
    values["HCU_LIGHTOP_W16A16"] = "HCU LightOp W16A16"
    hcu_enum = target.Enum("UnquantizedMoeBackend", values, module=target.__name__)
    target._vllm_hcu_original_unquantized_moe_backend = old_enum
    target.UnquantizedMoeBackend = hcu_enum

    @functools.wraps(backend_to_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == hcu_enum.HCU_LIGHTOP_W16A16:
            from vllm_hcu.model_executor.layers.fused_moe.experts import (
                lightop_w16a16_moe,
            )

            return [lightop_w16a16_moe.LightopW16A16Experts]
        return backend_to_cls(_normalize_backend(backend, old_enum, hcu_enum))

    @functools.wraps(map_backend)
    def hcu_map_unquantized_backend(runner_backend):
        # Some vLLM consumers retain the original selector during import, but
        # resolve this mapping dynamically. Cover that path for mixed Step5.
        if runner_backend == "deep_gemm":
            from vllm.config import get_current_vllm_config_or_none
            from vllm_hcu.deepseek_v4_runtime import model_architectures

            cfg = get_current_vllm_config_or_none()
            if any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
                   for a in model_architectures(cfg)):
                backend = cfg.parallel_config.all2all_backend
                runner_backend = (
                    "batched_triton" if backend == "deepep_low_latency" else "triton"
                )
        mapped = map_backend(runner_backend)
        return _normalize_backend(mapped, old_enum, hcu_enum)

    @functools.wraps(select_backend)
    def hcu_select_unquantized_moe_backend(moe_config):
        from vllm_hcu.model_executor.layers.fused_moe import (
            lightop_w16a16_runtime,
        )
        from vllm_hcu.model_executor.layers.fused_moe.experts import (
            lightop_w16a16_moe,
        )
        from vllm_hcu.platforms import envs as henvs

        # Step5 intentionally leaves its last three routed layers in BF16.
        # An FP8-only DeepGEMM selection must not force these layers to quantize.
        if getattr(moe_config, "moe_backend", "auto") == "deep_gemm":
            from dataclasses import replace
            from vllm.config import get_current_vllm_config_or_none
            from vllm_hcu.deepseek_v4_runtime import model_architectures

            architectures = model_architectures(get_current_vllm_config_or_none())
            if any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
                   for a in architectures):
                moe_config = replace(moe_config, moe_backend="auto")

        if (
            getattr(moe_config, "moe_backend", "auto") == "auto"
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_W16A16_MOE
        ):
            activation_format = (
                target.mk.FusedMoEActivationFormat.BatchedExperts
                if moe_config.moe_parallel_config.use_batched_activation_format
                else target.mk.FusedMoEActivationFormat.Standard
            )
            experts_cls = lightop_w16a16_moe.LightopW16A16Experts
            supported, _ = experts_cls.is_supported_config(
                experts_cls,
                moe_config,
                None,
                None,
                activation_format,
            )
            if supported:
                # Backend selection precedes layout mutation.  A missing M=1
                # decode config therefore delegates while weights are still
                # canonical and safe for official Triton/AITER paths.
                max_tokens = int(getattr(moe_config, "max_num_tokens", 0))
                configs_available = all(
                    lightop_w16a16_runtime.select_lightop_w16a16_config(
                        moe_config, expected_m=tokens, device="cuda"
                    )
                    is not None
                    for tokens in range(1, max_tokens + 1)
                )
                if configs_available:
                    return hcu_enum.HCU_LIGHTOP_W16A16, experts_cls

        from vllm.config import get_current_vllm_config_or_none
        from vllm_hcu.model_executor.layers.fused_moe.step5_deepep_auto import (
            step5_auto_experts_enabled,
        )
        if step5_auto_experts_enabled(
            moe_config, get_current_vllm_config_or_none()
        ):
            from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
                BatchedTritonExperts,
            )
            # Both Triton children consume canonical BF16 weights.
            return hcu_enum.BATCHED_TRITON, BatchedTritonExperts
        backend, experts = select_backend(moe_config)
        return _normalize_backend(backend, old_enum, hcu_enum), experts

    @functools.wraps(convert)
    def hcu_convert_to_unquantized_kernel_format(
        unquantized_backend,
        moe_config,
        w13_weight,
        w2_weight,
    ):
        if unquantized_backend == hcu_enum.HCU_LIGHTOP_W16A16:
            from vllm_hcu.model_executor.layers.fused_moe import (
                lightop_w16a16_runtime,
            )

            packed13, packed2, _ = (
                lightop_w16a16_runtime.pack_lightop_w16a16_weights(
                    w13_weight, w2_weight
                )
            )
            return packed13, packed2
        return convert(
            _normalize_backend(unquantized_backend, old_enum, hcu_enum),
            moe_config,
            w13_weight,
            w2_weight,
        )

    @functools.wraps(make_kernel)
    def hcu_make_unquantized_moe_kernel(
        quant_config,
        moe_config,
        backend,
        experts_cls,
        routing_tables=None,
    ):
        # Bound BF16 scratch space only for the opted-in Step5 DP8/TP1 HT path.
        import os
        from vllm.config import get_current_vllm_config_or_none
        from vllm_hcu.deepseek_v4_runtime import model_architectures
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

        cfg = get_current_vllm_config_or_none()
        from vllm_hcu.model_executor.layers.fused_moe.step5_sp_workspace import (
            bound_sequence_parallel_workspace,
        )
        moe_config = bound_sequence_parallel_workspace(moe_config, cfg)
        from vllm_hcu.model_executor.layers.fused_moe.step5_deepep_auto import (
            Step5AutoBF16Experts, step5_auto_experts_enabled,
        )
        if step5_auto_experts_enabled(moe_config, cfg):
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                _make_deepep_auto_deepgemm_moe_kernel,
            )
            return _make_deepep_auto_deepgemm_moe_kernel(
                moe_quant_config=quant_config,
                moe_config=moe_config,
                routing_tables=routing_tables,
                experts_cls=Step5AutoBF16Experts,
            )
        if (
            os.getenv("VLLM_STEP5_HT_COMPACT_WORKSPACE", "0") == "1"
            and experts_cls is TritonExperts
            and cfg is not None
            and any(a in ("Step5ForCausalLM", "Step5ForConditionalGeneration")
                    for a in model_architectures(cfg))
            and cfg.parallel_config.tensor_parallel_size == 1
            and cfg.parallel_config.data_parallel_size == 8
            and cfg.parallel_config.all2all_backend == "deepep_high_throughput"
        ):
            from vllm_hcu.model_executor.layers.fused_moe.step5_ht_workspace import (
                Step5HTChunkedTritonExperts,
            )

            experts_cls = Step5HTChunkedTritonExperts
        return make_kernel(
            quant_config,
            moe_config,
            _normalize_backend(backend, old_enum, hcu_enum),
            experts_cls,
            routing_tables,
        )

    for replacement in (
        hcu_enum,
        hcu_backend_to_kernel_cls,
        hcu_map_unquantized_backend,
        hcu_select_unquantized_moe_backend,
        hcu_convert_to_unquantized_kernel_format,
        hcu_make_unquantized_moe_kernel,
    ):
        setattr(replacement, _BINDING_MARKER, True)
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target.map_unquantized_backend = hcu_map_unquantized_backend
    target.select_unquantized_moe_backend = hcu_select_unquantized_moe_backend
    target.convert_to_unquantized_kernel_format = (
        hcu_convert_to_unquantized_kernel_format
    )
    target.make_unquantized_moe_kernel = hcu_make_unquantized_moe_kernel
    import sys

    consumer = sys.modules.get(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    )
    if consumer is not None:
        for name in ("UnquantizedMoeBackend", "select_unquantized_moe_backend",
                     "convert_to_unquantized_kernel_format",
                     "make_unquantized_moe_kernel"):
            if hasattr(consumer, name):
                setattr(consumer, name, getattr(target, name))
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
