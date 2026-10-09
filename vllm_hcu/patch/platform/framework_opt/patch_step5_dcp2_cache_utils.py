# SPDX-License-Identifier: Apache-2.0
"""Use context-sharded block geometry only for Step5's typed DCP2 SWA."""
from __future__ import annotations

import functools
import inspect
import os

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.step5_dcp2_cache_utils"
TARGETS = (f"{TARGET_MODULE}.dcp_world_size_for_kv_cache_spec",)
_MARKER = "_vllm_hcu_step5_dcp2_cache_world_size"


def wrap_dcp_cache_world_size(original):
    @functools.wraps(original)
    def world_size(spec, dcp_world_size):
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec
        specs = (tuple(spec.kv_cache_specs.values())
                 if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,))
        typed = [type(s) is Step5DCP2SlidingWindowSpec for s in specs]
        if not any(typed):
            return original(spec, dcp_world_size)
        if (not all(typed) or dcp_world_size != 2
                or os.getenv("VLLM_STEP5_DCP2_RUNTIME", "0") != "1"):
            raise ValueError("Step5 DCP2 SWA requires homogeneous typed groups and DCP2 opt-in")
        return 2
    return world_size


def apply_to_module(module):
    module = load_exact_module(TARGET_MODULE, module)
    if getattr(module, _MARKER, False):
        return False
    original = module.dcp_world_size_for_kv_cache_spec
    if tuple(inspect.signature(original).parameters) != ("spec", "dcp_world_size"):
        raise PatchCompatibilityError("Step5 DCP2 cache world-size signature changed")
    module.dcp_world_size_for_kv_cache_spec = wrap_dcp_cache_world_size(original)
    setattr(module, _MARKER, True)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
