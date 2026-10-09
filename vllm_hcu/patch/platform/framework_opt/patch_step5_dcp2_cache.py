# SPDX-License-Identifier: Apache-2.0
"""Admit only Step5's typed DCP2 SWA groups to the stock hybrid coordinator."""
from __future__ import annotations

import ast
import functools
import inspect
import os
import textwrap

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.v1.core.kv_cache_coordinator"
PATCH_ID = "platform.framework_opt.step5_dcp2_cache"
TARGETS = (f"{TARGET_MODULE}.HybridKVCacheCoordinator.__init__",)
_MARKER = "_vllm_hcu_step5_dcp2_hybrid_cache"


def _extend_type_guard(original, swa_class, owner_class):
    """Change the one DCP allowed-type assertion; retain all other stock code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assert)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Name) and n.test.func.id == "isinstance"
        and len(n.test.args) == 2 and isinstance(n.test.args[1], ast.Tuple)
        and [getattr(t, "id", None) for t in n.test.args[1].elts]
            == ["FullAttentionSpec", "MambaSpec"]
        and isinstance(n.msg, ast.JoinedStr)
        and "DCP with hybrid" in ast.unparse(n.msg)
    ]
    if len(matches) != 1:
        raise PatchCompatibilityError("Step5 DCP2 hybrid cache type guard changed")
    matches[0].test.args[1].elts.append(ast.Name(id="_Step5DCP2SWA", ctx=ast.Load()))
    # Module-level compilation cannot retain the implicit __class__ cell.
    # Bind the original super() call explicitly to its exact owning class.
    super_calls = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "super" and not n.args and not n.keywords]
    if len(super_calls) != 1:
        raise PatchCompatibilityError("Step5 DCP2 hybrid superclass contract changed")
    super_calls[0].args = [
        ast.Name(id="_Step5HybridOwner", ctx=ast.Load()),
        ast.Name(id="self", ctx=ast.Load()),
    ]
    globals_ = dict(original.__globals__)
    globals_["_Step5DCP2SWA"] = swa_class
    globals_["_Step5HybridOwner"] = owner_class
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"),
         globals_, namespace)
    return namespace[original.__name__]


def apply_to_module(module):
    module = load_exact_module(TARGET_MODULE, module)
    hybrid = module.HybridKVCacheCoordinator
    if getattr(hybrid, _MARKER, False):
        return False
    # A coordinator imported before plugin discovery holds a stale from-import.
    from vllm.v1.core import kv_cache_utils
    from . import patch_step5_dcp2_cache_utils
    patch_step5_dcp2_cache_utils.apply_to_module(kv_cache_utils)
    module.dcp_world_size_for_kv_cache_spec = (
        kv_cache_utils.dcp_world_size_for_kv_cache_spec
    )
    original = hybrid.__init__
    signature = inspect.signature(original)
    if tuple(signature.parameters) != (
        "self", "kv_cache_config", "max_model_len", "max_in_flight_tokens",
        "use_eagle", "enable_caching", "enable_kv_cache_events", "dcp_world_size",
        "pcp_world_size", "scheduler_block_size", "hash_block_size",
        "metrics_collector", "num_prefill_lookahead",
    ):
        raise PatchCompatibilityError("Step5 DCP2 hybrid constructor changed")
    from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec
    admitted = _extend_type_guard(original, Step5DCP2SlidingWindowSpec, hybrid)

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        specs = [g.kv_cache_spec
                 for g in arguments["kv_cache_config"].kv_cache_groups]
        if not any(isinstance(s, Step5DCP2SlidingWindowSpec) for s in specs):
            return original(self, *args, **kwargs)
        if (os.getenv("VLLM_STEP5_DCP2_RUNTIME", "0") != "1"
            or arguments["dcp_world_size"] != 2 or arguments["pcp_world_size"] != 1
            or arguments["enable_caching"]):
            raise ValueError("Step5 DCP2 hybrid cache requires opt-in, DCP2/PCP1/no prefix")
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
        import torch
        for spec in specs:
            if (type(spec) not in (FullAttentionSpec, Step5DCP2SlidingWindowSpec)
                or spec.block_size != 64 or spec.num_kv_heads != 4
                or spec.head_size != 192 or spec.head_size_v != 192
                or spec.dtype != torch.bfloat16 or spec.kv_quant_mode != KVQuantMode.NONE
                or spec.tokens_per_state != 1):
                raise ValueError("Step5 DCP2 hybrid cache has an incompatible group")
        return admitted(self, *args, **kwargs)

    hybrid.__init__ = init
    setattr(hybrid, _MARKER, True)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
