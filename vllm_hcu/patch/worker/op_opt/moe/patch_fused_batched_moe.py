# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Use the ROCm capture API for batched MoE's current-stream query.

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    require_callable,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.fused_batched_moe"
PATCH_ID = "worker.op_opt.moe.experts.batched_stream_capture"
TARGETS = (f"{TARGET_MODULE}._is_capturing_or_compiling",)
_MARKER = "_vllm_hcu_batched_stream_capture_applied"
_BINDING_MARKER = "_vllm_hcu_batched_stream_capture_binding"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    name = "_is_capturing_or_compiling"
    if check_module_marker(target, _MARKER, ((target, name, _BINDING_MARKER),)):
        return False
    original = require_callable(target, name, TARGETS[0])
    require_parameter_names(original, TARGETS[0], ())

    @functools.wraps(original)
    def hcu_is_capturing_or_compiling() -> bool:
        if target.current_platform.is_rocm():
            # torch.accelerator.current_stream() returns torch.Stream on this
            # HCU runtime, which has no is_capturing method. This backend API
            # queries the same current stream and works inside HIP capture.
            return (
                target.torch.compiler.is_compiling()
                or target.torch.cuda.is_current_stream_capturing()
            )
        return original()

    setattr(hcu_is_capturing_or_compiling, _BINDING_MARKER, True)
    setattr(target, name, hcu_is_capturing_or_compiling)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
