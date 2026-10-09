# SPDX-License-Identifier: Apache-2.0
"""CSA/DSA driver ported from OpenDAS/vllm-hcu step37-dp.

Reference commit: 0686260600ad725f3c594a6ad9f5dd0a16830091.
The vendored step4_dsa_kernels module is kept byte-identical to that branch.
This driver adapts its region summaries, pending partial regions, and sparse
attention plans to vLLM's paged cache. Step5's layer wrappers bind V2 cache
views; its sliding attention layers use a separate HCU FlashAttention backend.

Each query scores 8-token regions and attends to the top 512 regions. When
4096 tokens cover the complete causal history, this matches dense attention.
Above that length the selected regions determine the sparse attention result.
Optional boltops and FlashAttention DSA stages are inherited from the source
branch; only the vendored BF16 path is validated for this Step5 adaptation.
"""

from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
    REGION_BLOCK_SIZE,
    build_rope_cache,
    csa_compress_regions,
    decode_sparse_meta,
    indexer_logits,
    indexer_norm_rope,
    merge_split_states,
    prefill_sparse_meta,
    round_activations_e4m3,
    sparse_attention_decode,
    sparse_attention_prefill,
)
from vllm_hcu.model_executor.layers.step5_dsa_norm_rope import (
    fused_indexer_qk_norm_rope,
    short_index_query_rounding_enabled,
)
from vllm_hcu.model_executor.layers.step5_dsa_shortquery import (
    sparse_attention_short_prefill,
    prefill_request_indices_device,
    short_query_num_splits,
    is_short_query_batch,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
attention_route_logger = init_logger("vllm.hcu.step5.sparse_attention")
indexer_norm_cast_logger = init_logger("vllm.hcu.step5.indexer")

_STEP5_DSA_SELECTOR = os.environ.get("VLLM_STEP5_DSA_RADIX_SELECTOR", "")
if _STEP5_DSA_SELECTOR in {"1", "radix_flash"}:
    from vllm_hcu.model_executor.layers.step5_dsa_selector import (
        install_native_radix_selectors,
        prefill_sparse_meta_fused,
    )

    install_native_radix_selectors()
    logger.info(
        "Step5 DSA: native radix selectors enabled with gfx938 workaround"
    )
elif os.environ.get("VLLM_STEP5_DSA_TORCH_SELECTOR") == "1":
    from vllm_hcu.model_executor.layers.step5_dsa_compat import install_torch_selectors

    install_torch_selectors()
    logger.info(
        "Step5 DSA: exact Torch GPU selectors enabled for gfx938 compiler compatibility"
    )

# Step-4's head_dim. Declared as a supported-head-size list so that a checkpoint
# with different geometry fails at layer construction rather than inside a
# Triton kernel with a shape error.
STEP4_HEAD_DIM = 192

# The qualified name of the indexer op. Duplicated as a literal in
# transformers_utils/configs/step4.py, which cannot import this module (it runs
# during config resolution, long before torch/Triton would be wanted). If the
# two ever drift apart, _require_splitting_op fails at startup and says so.
STEP4_DSA_INDEX_OP = "vllm::step4_dsa_index"

# Every Step4Indexer on this rank, keyed by the name of the attention layer it
# feeds. The custom op below can only be handed tensors and strings, so the
# object itself has to be looked up rather than passed.
_INDEXERS: dict[str, "Step4Indexer"] = {}

# Bootstrap/debug switch, see the module docstring. Selecting every region makes
# DSA mathematically equal to full causal attention, at the cost of a
# [tokens, all_regions] metadata tensor, so this is for validation only.
DENSE_EQUIV = os.environ.get("VLLM_STEP4_DSA_DENSE_EQUIV", "0") == "1"

# How many query tokens are scored against their region summaries at a time.
# The scoring tensor is [rows, regions] fp32, and at 64K a request has 8192
# regions: a whole 16384-token prefill chunk scored at once would be 536 MB.
# 2048 rows brings that to 67 MB, which is the only reason this knob exists --
# selection is per row, so the chunking changes nothing about the result.
# Query-axis chunking for prefill indexer scoring. Off by default, and the default
# is the point: it used to be 2048, which at a 65536-token prefill meant 32
# ``indexer_logits`` launches per request per DSA layer -- 6716 launches in one
# recorded 9-step trace -- plus two activation quantizations and four group copies
# each. The chunk existed only to bound the ``[groups * count, nscore]`` fp32
# temporary that ``indexer_logits`` used to return; now that the kernel scores
# directly into ``prefill_logits`` there is no temporary to bound, and the query
# axis was always grid dimension 0 anyway, so one launch does the same work.
#
# Measured on one 65536-token, 1023-region request (groups=4), per DSA layer:
#
#     old, chunk=2048   9.60 ms   +47 MiB transient
#     new, no chunk     6.57 ms  +359 MiB
#     new, chunk=8192   6.72 ms   +23 MiB
#
# So nearly all of the 3.0 ms is the copies and the redundant key quantization,
# not the launch count -- collapsing 32 launches to 1 is only the last 0.15 ms,
# and it costs ~340 MiB of transient peak because the e4m3 copy of ``index_q`` is
# then made at full length. That peak is bounded by --max-num-batched-tokens and
# freed immediately, which is why the default is still 0; set
# VLLM_STEP4_DSA_Q_CHUNK to a positive token count to trade it back.
INDEXER_Q_CHUNK = int(os.environ.get("VLLM_STEP4_DSA_Q_CHUNK", "0"))

# Decode score width, in regions, for the CUDA-graph-captured path. A graph
# replays the launch arguments it recorded, so the [rows, regions] score tensor
# and the selector's scan bound must be the same on every step rather than
# following the batch's longest context. The default covers max_model_len, which
# is correct but pessimistic: the selector scans the whole declared width, so a
# server started with --max-model-len 65536 does an eighth of the region scan a
# 512K one does. Set this only to go *below* max_model_len deliberately -- a
# batch whose real history exceeds it is rejected loudly, never truncated.
GRAPH_REGIONS_OVERRIDE = int(os.environ.get("VLLM_STEP4_DSA_GRAPH_REGIONS", "0"))

# Debug switch for the host-side slot derivation in _decode_schedule. When set,
# the slots computed from the block table's host mirror are checked against the
# device tensor they replace -- which reintroduces exactly the device-to-host
# sync the mirror exists to remove, so this is for bring-up and for bisecting a
# suspected mirror/device divergence, never for a timed run.
CHECK_SLOTS = os.environ.get("VLLM_STEP4_DSA_CHECK_SLOTS", "0") == "1"

# Depth of the pinned staging ring used to upload the per-step pending rows.
# One buffer per vector is not enough: the upload is asynchronous, so the next
# step's host write can land in the buffer before the previous step's copy has
# read it out of it. Four is a step of run-ahead well past anything the
# scheduler currently sustains; the events below make the depth a bound rather
# than an assumption.
_STAGING_DEPTH = 4

# Which sparse-attention stages to run out of ``boltops.step4_dsa`` (the
# operator team's library, contract ``step4-dsa-ops-v1``) instead of the
# vendored reference copy. Comma-separated, empty means "vendored everywhere":
#
#   VLLM_STEP4_DSA_BOLTOPS=decode,prefill
#
# Only these two stages are switchable. The selector, indexer, CSA compression
# and metadata builders stay on the vendored copy in either case: they produce
# the *inputs* both libraries consume, so keeping one producer means a
# difference in output is attributable to the attention kernel alone.
#
# Two things the two libraries do not share:
#
# * boltops reads an fp8 cache but does not do fp8 *arithmetic*. As of its
#   commit 28239f4 both stages take ``k_scale``/``v_scale`` and dequantize a
#   ``float8_e4m3fn`` cache on load -- ``k_scale`` folded into ``softmax_scale``
#   on the host, ``v_scale`` applied to the fp32 output under an
#   ``APPLY_V_SCALE`` constexpr -- then widen to the query dtype before
#   ``tl.dot``. That matches the vendored copy at
#   ``VLLM_STEP4_DSA_FP8_MATH=off`` and nothing else: boltops has no equivalent
#   of ``qk_fp8``/``pv_fp8``, so an A/B against the server default compares two
#   different arithmetics, not two implementations of one.
#   boltops also keys its occupancy JSON on the *cache* dtype, because the query
#   stays bfloat16 on the fp8 path and the cache dtype is the only thing that
#   distinguishes it. ``_boltops_decode_config`` therefore has to pass both.
# * boltops decode owns its split count. ``num_splits=1`` there means "split to
#   fill the CUs and merge for me", so the ``_decode_num_splits`` policy and the
#   explicit ``merge_split_states`` pass are skipped on that path.
_BOLTOPS_STAGE_NAMES = frozenset({"decode", "prefill"})
BOLTOPS_STAGES = frozenset(
    name.strip()
    for name in os.environ.get("VLLM_STEP4_DSA_BOLTOPS", "").split(",")
    if name.strip()
)
if BOLTOPS_STAGES - _BOLTOPS_STAGE_NAMES:
    raise ValueError(
        f"VLLM_STEP4_DSA_BOLTOPS: unknown stage(s) "
        f"{sorted(BOLTOPS_STAGES - _BOLTOPS_STAGE_NAMES)}; "
        f"expected a comma-separated subset of {sorted(_BOLTOPS_STAGE_NAMES)}"
    )

# Which sparse-attention stages to run through
# flash-attention-cutlass's own sparse kernels (``fa_sparse_attention_prefill``/
# ``fa_sparse_attention_decode``, and their ``_fp8`` variants) instead of the
# vendored Triton reference copy. Same shape as ``VLLM_STEP4_DSA_BOLTOPS``:
#
#   VLLM_STEP4_DSA_FLASH_ATTN=decode,prefill
#
# The selector, indexer, CSA compression and metadata builders stay on the
# vendored copy regardless -- only the attention-compute call is swapped, same
# as the BoltOps switch above.
#
# Sliding-window (dense) attention on HCU already routes through flash_attn
# when ``VLLM_HCU_USE_FLASH_ATTN``/``VLLM_HCU_USE_FLASH_ATTN_UNIFIED`` is set
# (see ``vllm_hcu.platforms.hcu._get_backend_priorities``), but that selector
# hardcodes the sparse-MLA backends whenever ``use_sparse`` is true and never
# consults those two flags for DSA layers -- and DSA layers bypass backend
# auto-selection entirely, wiring straight to ``Step4SparseAttentionBackend``
# in ``step4.py``. So without this switch, setting those two env vars makes
# sliding layers call flash_attn but leaves DSA on Triton. Default DSA to
# flash_attn under exactly the same two flags, so one launch config moves both
# attention families together; ``VLLM_STEP4_DSA_FLASH_ATTN`` still overrides
# explicitly (e.g. to stage prefill out while decode is validated).
_FLASH_ATTN_STAGE_NAMES = frozenset({"decode", "prefill", "short_prefill"})
_flash_attn_stages_raw = os.environ.get("VLLM_STEP4_DSA_FLASH_ATTN")
if _flash_attn_stages_raw is not None:
    FLASH_ATTN_DSA_STAGES = frozenset(
        name.strip() for name in _flash_attn_stages_raw.split(",") if name.strip()
    )
elif _STEP5_DSA_SELECTOR == "radix_flash":
    # EngineCore startup only carries the established Step5 selector through
    # its sanitized worker environment. Keep the alternate decode backend on
    # the same selector so the model-loading process observes the choice.
    FLASH_ATTN_DSA_STAGES = frozenset({"decode", "short_prefill"})
else:
    import vllm_hcu.platforms.envs as _henvs

    FLASH_ATTN_DSA_STAGES = (
        frozenset(_FLASH_ATTN_STAGE_NAMES)
        if (_henvs.VLLM_HCU_USE_FLASH_ATTN or _henvs.VLLM_HCU_USE_FLASH_ATTN_UNIFIED)
        else frozenset()
    )
if FLASH_ATTN_DSA_STAGES - _FLASH_ATTN_STAGE_NAMES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FLASH_ATTN: unknown stage(s) "
        f"{sorted(FLASH_ATTN_DSA_STAGES - _FLASH_ATTN_STAGE_NAMES)}; "
        f"expected a comma-separated subset of {sorted(_FLASH_ATTN_STAGE_NAMES)}"
    )
if FLASH_ATTN_DSA_STAGES & BOLTOPS_STAGES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FLASH_ATTN and VLLM_STEP4_DSA_BOLTOPS both claim "
        f"stage(s) {sorted(FLASH_ATTN_DSA_STAGES & BOLTOPS_STAGES)}; a stage "
        f"can only run through one alternate sparse-attention implementation"
    )

# How much of the sparse attention arithmetic runs in fp8 on an fp8 cache. The
# vendored kernels honour this for both decode and prefill; boltops decode reads
# an fp8 cache but always widens before the dot, and boltops prefill has no fp8
# MMA path at all, so a boltops stage behaves as ``off`` regardless.
#
#   VLLM_STEP4_DSA_FP8_MATH=off             K/V widen to the query dtype
#   VLLM_STEP4_DSA_FP8_MATH=qk              Q quantized, QK dot runs on the fp8 MMA
#   VLLM_STEP4_DSA_FP8_MATH=full  (default) PV dot runs on the fp8 MMA as well
#
# ``full`` is the fastest setting and, as of 2026-09-21, the server default by
# explicit instruction. Per-layer cost at the production shape:
#
#   decode (topk=512):    117 us off, 83 us qk, 53 us full
#   prefill (8K queries): 9.2 ms off, 11.5 ms qk, 6.3 ms full
#
# ``full`` is the only one where an fp8 cache costs less latency than bf16 (77 us
# decode, 7.3 ms prefill) rather than more, and for prefill ``qk`` alone is the
# slowest of the four because the loop body holds both an fp8 K tile and a
# bf16-widened V tile and blows through the 256-VGPR budget (91 spills). ``full``
# is what removes bf16 from the loop, and only then does it fit (0 spills).
#
# Read this before trusting that default. ``off`` is the only setting inside the
# acceptance envelopes (stage 6 for decode, stage 5 for prefill), so the operator
# suite does not bless what now runs by default. The fp8 cache by itself stays in
# the envelope because kernel and reference read the same fp8 bytes and the
# storage error cancels; quantizing Q, and much more so P, is error no fp32
# reference models -- measured max relative error against an fp32 reference goes
# 4.0e-3 (off) -> 7.1e-3 (qk) -> 3.2e-2 (full) for decode, similar for prefill.
# No end-to-end score has been run at this setting. ``VLLM_STEP4_DSA_FP8_MATH=off``
# restores the gated path.
_FP8_MATH_MODES = {"off": (False, False), "qk": (True, False), "full": (True, True)}
_fp8_math = os.environ.get("VLLM_STEP4_DSA_FP8_MATH", "full").strip().lower() or "full"
if _fp8_math not in _FP8_MATH_MODES:
    raise ValueError(
        f"VLLM_STEP4_DSA_FP8_MATH={_fp8_math!r}: expected one of "
        f"{sorted(_FP8_MATH_MODES)}"
    )
DECODE_QK_FP8, DECODE_PV_FP8 = _FP8_MATH_MODES[_fp8_math]
PREFILL_QK_FP8, PREFILL_PV_FP8 = _FP8_MATH_MODES[_fp8_math]
if _fp8_math != "off":
    # Loud, once, at import: the thing running is not the thing the ten gates
    # accepted, and a reader of these logs should not have to know that from a
    # source comment.
    logger.warning(
        "step4 DSA: fp8 math = %r. This is faster than 'off' but sits "
        "outside the acceptance envelopes (stage 6 decode, stage 5 prefill), "
        "so operator_parity does not cover it and no end-to-end score has been "
        "recorded at this setting. Set VLLM_STEP4_DSA_FP8_MATH=off for the gated path.",
        _fp8_math,
    )


@functools.lru_cache(maxsize=1)
def _boltops_attention() -> tuple[Any, Any]:
    """``(prefill, decode)`` from boltops, checked for an API version we know.

    Imported lazily: the module is only needed when a stage is switched to it,
    and an unavailable library should fail where the switch was asked for.
    """
    try:
        import boltops.step4_dsa as ops
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "VLLM_STEP4_DSA_BOLTOPS asked for boltops sparse attention but "
            "boltops.step4_dsa could not be imported"
        ) from exc
    version = getattr(ops, "STEP4_DSA_OPS_API_VERSION", None)
    if version != "step4-dsa-ops-v1":
        raise RuntimeError(
            f"boltops.step4_dsa speaks {version!r}, this backend calls "
            f"'step4-dsa-ops-v1'. The prefill/decode signatures may have moved."
        )
    logger.info(
        "step4 DSA: %s from boltops.step4_dsa (%s)",
        ", ".join(sorted(BOLTOPS_STAGES)),
        version,
    )
    return ops.triton_sparse_attention_prefill, ops.triton_sparse_attention_decode


@functools.lru_cache(maxsize=1)
def _flash_attn_dsa_ops() -> tuple[Any, Any, Any, Any]:
    """``(prefill_bf16, prefill_fp8, decode_bf16, decode_fp8)`` from flash_attn.

    Imported lazily, same reasoning as ``_boltops_attention``: only needed once
    a stage is actually switched to it, and an unavailable/stale build should
    fail at the switch, not at import time for everyone who never sets
    ``VLLM_STEP4_DSA_FLASH_ATTN``/``VLLM_HCU_USE_FLASH_ATTN*``.

    These four ops come from flash-attention-cutlass's own Stage-5/6 sparse
    port (``fa_sparse_attention_{prefill,decode}[_fp8]`` in
    ``flash_attn_interface.py``), which speaks the same ``packed_regions``/
    ``region_counts`` bit-packing (``phys_region | (valid_tokens << 24)``) and
    ``[tokens, num_kv_groups, head_dim]`` flat KV-cache layout as the vendored
    Triton kernels here, at the same ``head_dim=192`` Step-4 uses throughout.
    """
    try:
        from flash_attn import flash_attn_interface
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "VLLM_STEP4_DSA_FLASH_ATTN (or VLLM_HCU_USE_FLASH_ATTN/"
            "VLLM_HCU_USE_FLASH_ATTN_UNIFIED) asked for flash_attn sparse "
            "attention but flash_attn.flash_attn_interface could not be "
            "imported"
        ) from exc
    required = {
        "prefill": "fa_sparse_attention_prefill",
        "decode": "fa_sparse_attention_decode",
    }
    missing = [
        name
        for stage, name in required.items()
        if (
            stage in FLASH_ATTN_DSA_STAGES
            or (stage == "prefill" and "short_prefill" in FLASH_ATTN_DSA_STAGES)
        )
        and not hasattr(flash_attn_interface, name)
    ]
    if missing:
        raise ImportError(
            "flash_attn does not expose the enabled sparse attention op(s): "
            + ", ".join(missing)
        )
    logger.info(
        "step4 DSA: %s from flash_attn.flash_attn_interface",
        ", ".join(sorted(FLASH_ATTN_DSA_STAGES)),
    )
    return (
        getattr(flash_attn_interface, "fa_sparse_attention_prefill", None),
        getattr(flash_attn_interface, "fa_sparse_attention_prefill_fp8", None),
        getattr(flash_attn_interface, "fa_sparse_attention_decode", None),
        getattr(flash_attn_interface, "fa_sparse_attention_decode_fp8", None),
    )


def _flash_attn_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float | None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter matching ``sparse_attention_prefill``'s call in ``forward()``.

    ``_ignored`` swallows the vendored kernel's tiled-prefill-only kwargs
    (``logical_regions``, ``q_positions``, ``request_indices``, ``block_q``)
    and the fp8-math toggles (``qk_fp8``, ``pv_fp8``, ``q_scale``): the
    flash-attention-cutlass port has no query-tiled path (it raises
    ``NotImplementedError`` if ``logical_regions`` is passed) and always
    quantizes/dequantizes K/V by ``k_scale``/``v_scale`` on its own fp8 MMA
    path, with no separate query-quantization knob.
    """
    prefill_bf16, prefill_fp8, _, _ = _flash_attn_dsa_ops()
    if key_cache.dtype == torch.float8_e4m3fn:
        if prefill_fp8 is None:
            raise RuntimeError(
                "installed flash_attn lacks fa_sparse_attention_prefill_fp8"
            )
        return prefill_fp8(
            query,
            key_cache,
            value_cache,
            packed_regions,
            region_counts,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            softmax_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    result = prefill_bf16(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
    )
    attention_route_logger.info_once(
        "Step5 native sparse FA prefill active: KV dtype=%s, stride=%s, "
        "query heads=%d, KV groups=%d, head_dim=%d",
        key_cache.dtype, tuple(key_cache.stride()), query.shape[1],
        num_kv_groups, query.shape[-1], scope="process",
    )
    return result


def _flash_attn_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_regions: torch.Tensor,
    region_counts: torch.Tensor,
    kv_seqlens: torch.Tensor,
    *,
    num_kv_groups: int,
    region_size: int,
    softmax_scale: float | None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter matching ``sparse_attention_decode``'s call in ``forward()``.

    Always calls with ``num_splits=1``: the C++ entry point ignores the
    argument and always returns final merged BF16 output plus natural-log
    LSE internally (see its docstring), so unlike the vendored Triton decode
    path this adapter never calls ``merge_split_states`` -- there is nothing
    left to merge. ``_ignored`` swallows ``qk_fp8``/``pv_fp8``/``q_scale``,
    which have no equivalent here (see ``_flash_attn_prefill``).
    """
    _, _, decode_bf16, decode_fp8 = _flash_attn_dsa_ops()
    if key_cache.dtype == torch.float8_e4m3fn:
        if decode_fp8 is None:
            raise RuntimeError(
                "installed flash_attn lacks fa_sparse_attention_decode_fp8"
            )
        return decode_fp8(
            query,
            key_cache,
            value_cache,
            packed_regions,
            region_counts,
            kv_seqlens,
            num_kv_groups=num_kv_groups,
            region_size=region_size,
            softmax_scale=softmax_scale,
            num_splits=1,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    result = decode_bf16(
        query,
        key_cache,
        value_cache,
        packed_regions,
        region_counts,
        kv_seqlens,
        num_kv_groups=num_kv_groups,
        region_size=region_size,
        softmax_scale=softmax_scale,
        num_splits=1,
    )
    attention_route_logger.info_once(
        "Step5 native sparse FA decode active: KV dtype=%s, stride=%s, "
        "query heads=%d, KV groups=%d, head_dim=%d",
        key_cache.dtype, tuple(key_cache.stride()), query.shape[1],
        num_kv_groups, query.shape[-1], scope="process",
    )
    return result


@functools.lru_cache(maxsize=64)
def _boltops_decode_config(
    num_reqs: int,
    num_heads: int,
    head_dim: int,
    num_kv_groups: int,
    topk: int,
    region_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> dict[str, Any]:
    """boltops' tuned decode launch config, resolved without reading the device.

    boltops keys its decode JSON partly on the batch's longest context and
    reads that with ``kv_seqlens.max().item()`` -- a blocking device-to-host
    sync on every call, and illegal inside a CUDA graph capture, which is
    exactly the path decode takes here. Passing ``config=`` explicitly skips
    that lookup, so do the lookup on the host instead: everything else the
    selector reads is a shape, and the longest context is already on the host in
    ``seq_lens_cpu``. Meta tensors stand in for the real ones.

    ``cache_dtype`` is passed separately from the query ``dtype`` because the
    selector keys its ``dtype`` entry off the KV cache: the query stays bfloat16
    on the fp8 serving path, so folding the two would resolve the bfloat16 tile
    for an fp8 cache.

    Returns ``{}`` -- which boltops reads as "keep the built-in tile" -- when
    nothing matches or the private selector has moved. Note that nothing matches
    on a device with no shipped JSON: the files are named after
    ``torch.cuda.get_device_name()``, and boltops ships ``BW1100`` -- aliased
    from ``BW1101`` since its commit 28239f4 -- plus ``BW200`` and ``K100_AI``.
    """
    from boltops.step4_dsa.triton import decode as bolt_decode

    def meta(*shape: int, dt: torch.dtype = dtype) -> torch.Tensor:
        return torch.empty(shape, device="meta", dtype=dt)

    try:
        config = bolt_decode._load_sparse_attn_decode_config(
            meta(num_reqs, num_heads, head_dim),
            meta(1, num_kv_groups, head_dim, dt=cache_dtype),
            meta(num_kv_groups * num_reqs, topk, dt=torch.int64),
            torch.tensor([max_seq_len], dtype=torch.int32),  # host-side on purpose
            1,
            region_size,
        )
    except Exception:  # noqa: BLE001 - private API, any failure means "untuned"
        logger.warning_once(
            "step4 DSA: could not resolve a boltops decode config; running its "
            "built-in tile. Check whether _load_sparse_attn_decode_config still "
            "exists in boltops.step4_dsa.triton.decode."
        )
        return {}
    return config or {}


# --------------------------------------------------------------------------
# Driver-local state updates
# --------------------------------------------------------------------------


@triton.jit
def _store_live_summary_rows_kernel(
    dst_ptr,
    src_ptr,
    slots_ptr,
    live_ptr,
    dst_row_stride,
    dst_proxy_stride,
    src_row_stride,
    src_proxy_stride,
    PROXY_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
) -> None:
    """Copy completed region summaries without compacting live rows on host."""
    row = tl.program_id(0)
    if tl.load(live_ptr + row) == 0:
        return

    slot = tl.load(slots_ptr + row).to(tl.int64)
    offsets = tl.arange(0, BLOCK_P)
    mask = offsets < PROXY_DIM
    values = tl.load(
        src_ptr + row * src_row_stride + offsets * src_proxy_stride, mask=mask,
    )
    tl.store(
        dst_ptr + slot * dst_row_stride + offsets * dst_proxy_stride, values, mask=mask,
    )


def _store_live_summary_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    slots: torch.Tensor,
    live: torch.Tensor,
) -> None:
    """Store fixed-shape FP8 rows; dead records leave the cache untouched."""
    rows = source.shape[0]
    if rows == 0:
        return
    proxy = source.shape[-1]
    _store_live_summary_rows_kernel[(rows,)](
        destination,
        source,
        slots,
        live,
        destination.stride(0),
        destination.stride(2),
        source.stride(0),
        source.stride(2),
        PROXY_DIM=proxy,
        BLOCK_P=triton.next_power_of_2(proxy),
        num_warps=4,
    )


@triton.jit
def _insert_decode_tokens_kernel(
    key_dst_ptr,
    z_dst_ptr,
    key_src_ptr,
    z_src_ptr,
    heads_ptr,
    dst_row_stride: tl.int64,
    dst_region_stride: tl.int64,
    dst_proxy_stride: tl.int64,
    src_row_stride: tl.int64,
    src_proxy_stride: tl.int64,
    PROXY_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
) -> None:
    """Write each decode token into its request's partial-region buffer.

    PyTorch implements ``scatter_`` through TensorIterator's generic
    scatter/gather assignment kernel. That kernel is safe in eager execution,
    but its replayed launch was the faulting command when the Step5 target model
    used a full decode CUDA graph on gfx938. This operation has a much simpler
    contract than general scatter: one contiguous proxy vector per request and
    exactly one destination row. Keeping that contract explicit gives graph
    replay stable pointers and launch geometry.
    """
    row = tl.program_id(0)
    proxy_offset = tl.arange(0, BLOCK_P)
    mask = proxy_offset < PROXY_DIM
    head = tl.load(heads_ptr + row).to(tl.int64)
    src_offset = row * src_row_stride + proxy_offset * src_proxy_stride
    dst_offset = (
        row * dst_row_stride
        + head * dst_region_stride
        + proxy_offset * dst_proxy_stride
    )
    key = tl.load(key_src_ptr + src_offset, mask=mask)
    z = tl.load(z_src_ptr + src_offset, mask=mask)
    tl.store(key_dst_ptr + dst_offset, key, mask=mask)
    tl.store(z_dst_ptr + dst_offset, z, mask=mask)


def _insert_decode_tokens(
    buffer_key: torch.Tensor,
    buffer_z: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    heads: torch.Tensor,
) -> None:
    """Graph-stable specialized replacement for two decode ``scatter_`` calls."""
    rows = heads.shape[0]
    if rows == 0:
        return
    proxy = buffer_key.shape[-1]
    _insert_decode_tokens_kernel[(rows,)](
        buffer_key,
        buffer_z,
        index_k,
        index_z,
        heads,
        buffer_key.stride(0),
        buffer_key.stride(1),
        buffer_key.stride(2),
        index_k.stride(0),
        index_k.stride(-1),
        PROXY_DIM=proxy,
        BLOCK_P=triton.next_power_of_2(proxy),
        num_warps=4,
    )


@triton.jit
def _assemble_verification_tokens_kernel(
    key_out_ptr,
    z_out_ptr,
    key_pending_ptr,
    z_pending_ptr,
    key_src_ptr,
    z_src_ptr,
    heads_ptr,
    pending_in_ptr,
    prefix_lengths_ptr,
    out_row_stride: tl.int64,
    out_token_stride: tl.int64,
    out_proxy_stride: tl.int64,
    pending_row_stride: tl.int64,
    pending_token_stride: tl.int64,
    pending_proxy_stride: tl.int64,
    src_row_stride: tl.int64,
    src_proxy_stride: tl.int64,
    REGION: tl.constexpr,
    QUERIES: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
) -> None:
    """Assemble two fixed compression regions for one MTP verification row."""
    row = tl.program_id(0)
    token_pos = tl.program_id(1)
    proxy_offset = tl.arange(0, BLOCK_P)
    proxy_mask = proxy_offset < PROXY_DIM

    head = tl.load(heads_ptr + row).to(tl.int64)
    pending_row = tl.load(pending_in_ptr + row).to(tl.int64)
    prefix_total = (
        tl.load(prefix_lengths_ptr + 2 * row)
        + tl.load(prefix_lengths_ptr + 2 * row + 1)
    )
    live = prefix_total > 0
    from_pending = live & (token_pos < head)
    source_token = tl.maximum(0, tl.minimum(QUERIES - 1, token_pos - head))
    from_source = live & (token_pos >= head) & (token_pos < head + QUERIES)

    pending_offset = (
        pending_row * pending_row_stride
        + token_pos * pending_token_stride
        + proxy_offset * pending_proxy_stride
    )
    source_offset = (
        (row * QUERIES + source_token) * src_row_stride
        + proxy_offset * src_proxy_stride
    )
    key_pending = tl.load(
        key_pending_ptr + pending_offset,
        mask=proxy_mask & from_pending,
        other=0.0,
    )
    z_pending = tl.load(
        z_pending_ptr + pending_offset,
        mask=proxy_mask & from_pending,
        other=0.0,
    )
    key_source = tl.load(
        key_src_ptr + source_offset,
        mask=proxy_mask & from_source,
        other=0.0,
    )
    z_source = tl.load(
        z_src_ptr + source_offset,
        mask=proxy_mask & from_source,
        other=0.0,
    )
    key = tl.where(from_source, key_source, key_pending)
    z = tl.where(from_source, z_source, z_pending)
    out_offset = (
        row * out_row_stride
        + token_pos * out_token_stride
        + proxy_offset * out_proxy_stride
    )
    tl.store(key_out_ptr + out_offset, key, mask=proxy_mask)
    tl.store(z_out_ptr + out_offset, z, mask=proxy_mask)


def _assemble_verification_tokens(
    pending_key: torch.Tensor,
    pending_z: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    heads: torch.Tensor,
    pending_in: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    queries_per_request: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build graph-stable two-region K/Z buffers for MTP target verification."""
    rows = heads.shape[0]
    proxy = index_k.shape[-1]
    buffer_key = torch.empty(
        (rows, 2 * region_size, proxy), dtype=index_k.dtype, device=index_k.device
    )
    buffer_z = torch.empty_like(buffer_key)
    if rows == 0:
        return buffer_key, buffer_z
    _assemble_verification_tokens_kernel[(rows, 2 * region_size)](
        buffer_key,
        buffer_z,
        pending_key,
        pending_z,
        index_k,
        index_z,
        heads,
        pending_in,
        prefix_lengths,
        buffer_key.stride(0),
        buffer_key.stride(1),
        buffer_key.stride(2),
        pending_key.stride(0),
        pending_key.stride(1),
        pending_key.stride(2),
        index_k.stride(0),
        index_k.stride(-1),
        REGION=region_size,
        QUERIES=queries_per_request,
        PROXY_DIM=proxy,
        BLOCK_P=triton.next_power_of_2(proxy),
        num_warps=4,
    )
    return buffer_key, buffer_z


@triton.jit
def _store_verification_prefixes_kernel(
    key_dst_ptr,
    z_dst_ptr,
    key_src_ptr,
    z_src_ptr,
    rows_ptr,
    prefix_lengths_ptr,
    dst_row_stride: tl.int64,
    dst_token_stride: tl.int64,
    dst_proxy_stride: tl.int64,
    src_row_stride: tl.int64,
    src_token_stride: tl.int64,
    src_proxy_stride: tl.int64,
    REGION: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
) -> None:
    """Store every rollback prefix without indexed PyTorch assignments."""
    region_row = tl.program_id(0)
    token_offset = tl.program_id(1)
    prefix = tl.load(prefix_lengths_ptr + region_row)
    live = token_offset < prefix
    proxy_offset = tl.arange(0, BLOCK_P)
    mask = (proxy_offset < PROXY_DIM) & live
    dst_row = tl.load(rows_ptr + region_row).to(tl.int64)
    request = region_row // 2
    within_request = (region_row % 2) * REGION + token_offset
    src_offset = (
        request * src_row_stride
        + within_request * src_token_stride
        + proxy_offset * src_proxy_stride
    )
    dst_offset = (
        dst_row * dst_row_stride
        + token_offset * dst_token_stride
        + proxy_offset * dst_proxy_stride
    )
    key = tl.load(key_src_ptr + src_offset, mask=mask)
    z = tl.load(z_src_ptr + src_offset, mask=mask)
    tl.store(key_dst_ptr + dst_offset, key, mask=mask)
    tl.store(z_dst_ptr + dst_offset, z, mask=mask)


def _store_verification_prefixes(
    pending_key: torch.Tensor,
    pending_z: torch.Tensor,
    buffer_key: torch.Tensor,
    buffer_z: torch.Tensor,
    rows: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    region_size: int,
) -> None:
    """Write the one or two prefixes that MTP rejection may reopen."""
    region_rows = rows.shape[0]
    if region_rows == 0:
        return
    proxy = buffer_key.shape[-1]
    _store_verification_prefixes_kernel[(region_rows, region_size)](
        pending_key,
        pending_z,
        buffer_key,
        buffer_z,
        rows,
        prefix_lengths,
        pending_key.stride(0),
        pending_key.stride(1),
        pending_key.stride(2),
        buffer_key.stride(0),
        buffer_key.stride(1),
        buffer_key.stride(2),
        REGION=region_size,
        PROXY_DIM=proxy,
        BLOCK_P=triton.next_power_of_2(proxy),
        num_warps=4,
    )


@triton.jit
def _store_kv_rows_kernel(
    k_dst_ptr,
    v_dst_ptr,
    k_src_ptr,
    v_src_ptr,
    slots_ptr,
    k_dst_row_stride: tl.int64,
    v_dst_row_stride: tl.int64,
    k_dst_head_stride: tl.int64,
    v_dst_head_stride: tl.int64,
    k_src_row_stride: tl.int64,
    v_src_row_stride: tl.int64,
    k_scale,
    v_scale,
    HEAD_SIZE: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    QUANTIZE: tl.constexpr,
) -> None:
    """Store dense K/V rows while ignoring padded slots on device."""
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot < 0:
        return

    offsets = tl.arange(0, BLOCK_ELEMS)
    mask = offsets < ROW_ELEMS
    key = tl.load(k_src_ptr + row * k_src_row_stride + offsets, mask=mask,)
    value = tl.load(v_src_ptr + row * v_src_row_stride + offsets, mask=mask,)
    if QUANTIZE:
        # Divide by the scale, then clamp before the cast. Letting tl.store do the
        # narrowing on its own is what makes an fp8 cache quietly wrong: e4m3
        # saturates at 448, so every activation above it would land on the same
        # number with no indication anything happened. This is the same division
        # the sparse kernels undo, and the same clamp DSALayerCache._store_dtype
        # applies in the gated source. The bound is spelled as a literal rather
        # than FP8_E4M3_MAX because Triton cannot close over a plain float global;
        # act_quant in the kernel source writes -448.0/448.0 for the same reason.
        key = tl.clamp(key.to(tl.float32) / k_scale, -448.0, 448.0)
        value = tl.clamp(value.to(tl.float32) / v_scale, -448.0, 448.0)
    tl.store(
        k_dst_ptr
        + slot * k_dst_row_stride
        + (offsets // HEAD_SIZE) * k_dst_head_stride
        + offsets % HEAD_SIZE,
        key,
        mask=mask,
    )
    tl.store(
        v_dst_ptr
        + slot * v_dst_row_stride
        + (offsets // HEAD_SIZE) * v_dst_head_stride
        + offsets % HEAD_SIZE,
        value,
        mask=mask,
    )


def _store_kv_rows(
    k_destination: torch.Tensor,
    v_destination: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Store K/V rows; negative slots leave the cache untouched.

    Each head must be internally dense; packed K/V may stride between heads. K and V
    may have different token-row strides, as they do when unpacked from Step-4's
    fused QKV projection.

    When the destination is ``float8_e4m3`` the rows are quantized on the way in,
    dividing by the scale the sparse kernels later multiply back. The destination
    dtype is what selects that path, not a flag: the cache tensor is allocated by
    the runner from ``kv_cache_dtype``, so it is the one thing that cannot get out
    of step with what the kernels will read.
    """
    rows = key.shape[0]
    if rows == 0:
        return
    row_elems = key.shape[1] * key.shape[2]
    _store_kv_rows_kernel[(rows,)](
        k_destination,
        v_destination,
        key,
        value,
        slots,
        k_destination.stride(0),
        v_destination.stride(0),
        k_destination.stride(1),
        v_destination.stride(1),
        key.stride(0),
        value.stride(0),
        k_scale,
        v_scale,
        HEAD_SIZE=key.shape[2],
        ROW_ELEMS=row_elems,
        BLOCK_ELEMS=triton.next_power_of_2(row_elems),
        QUANTIZE=k_destination.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )


@triton.jit
def _decode_indexer_logits_kernel(
    index_q_ptr,
    summary_ptr,
    block_table_ptr,
    weights_ptr,
    out_ptr,
    nscore_ptr,
    num_reqs,
    stride_q_req,
    stride_q_group,
    stride_q_head,
    stride_s_row,
    stride_bt_req,
    stride_w_req,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    regions_per_page: tl.constexpr,
    query_dtype: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Batched decode form of ``_indexer_logits_kernel``.

        score[r, j] = sum_h relu(dot(q_rh, k_rj)) * w_rh

    Same maths, same accumulation order, one difference: the key set is indexed
    per request rather than shared, so requests move from the *host* loop that
    ``_score`` runs into the program grid. That is the whole point -- at decode
    every request contributes one query, so the released ``indexer_logits`` gets
    called once per request with ``seq_q == 1``, and 23 layers x num_reqs of those
    launches is most of the DSA path's eager-launch count.

    **Why this is bit-identical to the reference and not merely close.** Each
    output element is an independent dot product over ``proxy_dim``, so
    ``tl.dot``'s accumulation order along K does not depend on the M extent. The
    reference runs M = BLOCK_Q = 64 with one valid query and 63 masked rows; this
    runs M = 16 the same way. Keeping the per-head loop (rather than folding heads
    into M and reducing) preserves the fp32 add order of ``accumulator``, which is
    what a tree reduction over heads would have changed. Selection is a top-k over
    these scores, so a last-ULP difference is not cosmetic: it reorders ties.
    """
    req = tl.program_id(0)
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = tl.arange(0, BLOCK_Q)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    # One query per request at decode; the remaining rows exist only so that
    # tl.dot has a legal M extent, exactly as in the reference.
    query_ok = queries < 1
    key_ok = keys < tl.load(nscore_ptr + req)

    # Address the fp8 summary pool the same way the selector does: physical KV
    # block `b` owns summary rows [b * rpp, (b + 1) * rpp). Reading it in place
    # rather than taking a pre-gathered key tensor is what keeps every argument
    # here a persistent buffer -- and it drops a [num_reqs, width, proxy] gather
    # plus its fp8->bf16 cast from every one of the 23 layers.
    pages = tl.load(
        block_table_ptr + req * stride_bt_req + keys // regions_per_page,
        mask=key_ok,
        other=0,
    ).to(tl.int64)
    slots = pages * regions_per_page + keys % regions_per_page
    key_tile = tl.load(
        summary_ptr + slots[None, :] * stride_s_row + dims[:, None],
        mask=key_ok[None, :],
        other=0.0,
    ).to(query_dtype)

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + req * stride_q_req
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :]
            + queries[:, None] * 0,
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr + req * stride_w_req + group * stride_w_group + head
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight

    rows = group * num_reqs + req
    tl.store(
        out_ptr + rows * stride_out_row + keys[None, :] + queries[:, None] * 0,
        accumulator,
        mask=query_ok[:, None] & key_ok[None, :],
    )


def decode_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    nscore: torch.Tensor,
    out: torch.Tensor,
    *,
    regions_per_page: int,
    block_k: int = 128,
) -> None:
    """Score every decode request against its own key set, in one launch.

    Args:
        index_q: ``[num_reqs, groups, heads_per_group, proxy_dim]``.
        weights: ``[num_reqs, groups, heads_per_group]``, prescale already applied.
        summary: the fp8 region-summary pool, ``[rows, 1, proxy_dim]``. Read in
            place through ``block_table``; nothing is gathered.
        block_table: ``[num_reqs, pages]`` logical->physical map, the same one
            the region selector reads.
        nscore: ``[num_reqs]`` int32, complete regions visible to each request.
        out: ``[groups * num_reqs, width]`` float32, group-major. Only the first
            ``nscore[req]`` columns of a row are written. That is exactly the
            range the region selector then reads -- its ``history_lengths`` is
            the same ``(seq_len - 1) // region_size`` -- and every load past it
            is masked off, so the columns this kernel skips are never observed
            and the caller does not have to clear them. Do not narrow either
            bound on the assumption that the other one still covers you.

    Only ``index_q`` is rounded here. The pool is already e4m3, so the
    reference's key-side ``round_activations_e4m3`` is the identity on it: every
    finite e4m3 value is exact in bf16, and widening then re-rounding returns
    the same byte.
    """
    num_reqs, groups, heads_per_group, proxy_dim = index_q.shape
    width = out.shape[1]
    if summary.shape[-1] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} pool={summary.shape[-1]}")
    if block_table.shape[0] < num_reqs:
        raise ValueError(
            f"block table has {block_table.shape[0]} rows, need {num_reqs}"
        )
    if out.shape[0] != groups * num_reqs:
        raise ValueError(f"out has {out.shape[0]} rows, expected {groups * num_reqs}")
    if width > block_table.shape[1] * regions_per_page:
        raise ValueError(
            f"score width {width} exceeds the {block_table.shape[1]} pages x "
            f"{regions_per_page} regions the block table can address"
        )

    quantized_q = round_activations_e4m3(index_q)

    _decode_indexer_logits_kernel[(num_reqs, triton.cdiv(width, block_k), groups)](
        quantized_q,
        summary,
        block_table,
        weights,
        out,
        nscore,
        num_reqs,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        summary.stride(0),
        block_table.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        regions_per_page=regions_per_page,
        query_dtype=tl.bfloat16 if quantized_q.dtype is torch.bfloat16 else tl.float16,
        BLOCK_Q=16,
        BLOCK_K=block_k,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _short_prefill_indexer_logits_kernel(
    index_q_ptr,
    summary_ptr,
    block_table_ptr,
    request_indices_ptr,
    positions_ptr,
    weights_ptr,
    out_ptr,
    num_tokens,
    stride_q_token,
    stride_q_group,
    stride_q_head,
    stride_s_row,
    stride_bt_req,
    stride_w_token,
    stride_w_group,
    stride_out_row,
    heads_per_group: tl.constexpr,
    proxy_dim: tl.constexpr,
    regions_per_page: tl.constexpr,
    region_size: tl.constexpr,
    query_dtype: tl.constexpr,
    QUERIES_PER_REQUEST: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Score every short-prefill query against its request's summary pool.

    MTP verification contributes four queries per request. The reference path
    loops over requests on the host because ``indexer_logits`` accepts one key
    set; moving the request lookup into this grid removes 8 or 16 launches per
    DSA layer at total BS32 or BS64 while retaining the same dot-product and
    head accumulation order.
    """
    token_base = tl.program_id(0) * QUERIES_PER_REQUEST
    k_block = tl.program_id(1)
    group = tl.program_id(2)

    queries = tl.arange(0, BLOCK_Q)
    tokens = token_base + queries
    query_ok = (queries < QUERIES_PER_REQUEST) & (tokens < num_tokens)
    keys = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, proxy_dim)
    request = tl.load(request_indices_ptr + token_base).to(tl.int64)
    visible = (
        tl.load(positions_ptr + tokens, mask=query_ok, other=0).to(tl.int32)
        // region_size
    )
    max_visible = tl.max(visible, axis=0)
    key_load_ok = keys < max_visible

    pages = tl.load(
        block_table_ptr
        + request * stride_bt_req
        + keys // regions_per_page,
        mask=key_load_ok,
        other=0,
    ).to(tl.int64)
    slots = pages * regions_per_page + keys % regions_per_page
    key_tile = tl.load(
        summary_ptr + slots[None, :] * stride_s_row + dims[:, None],
        mask=key_load_ok[None, :],
        other=0.0,
    ).to(query_dtype)

    accumulator = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for head in tl.static_range(heads_per_group):
        query_tile = tl.load(
            index_q_ptr
            + tokens[:, None] * stride_q_token
            + group * stride_q_group
            + head * stride_q_head
            + dims[None, :],
            mask=query_ok[:, None],
            other=0.0,
        )
        head_weight = tl.load(
            weights_ptr
            + tokens * stride_w_token
            + group * stride_w_group
            + head,
            mask=query_ok,
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(query_tile, key_tile, out_dtype=tl.float32)
        accumulator += tl.maximum(scores, 0.0) * head_weight[:, None]

    rows = group * num_tokens + tokens
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + keys[None, :],
        accumulator,
        mask=query_ok[:, None] & (keys[None, :] < visible[:, None]),
    )


def short_prefill_indexer_logits(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    *,
    regions_per_page: int,
    region_size: int = REGION_BLOCK_SIZE,
    block_k: int = 128,
    queries_per_request: int = 1,
    q_is_e4m3_rounded: bool = False,
) -> None:
    """Batched request-aware indexer scoring for MTP verification queries.

    Each ``queries_per_request`` group must be contiguous and belong to one
    request so its rows can reuse the same summary tile.
    """
    num_tokens, groups, heads_per_group, proxy_dim = index_q.shape
    width = out.shape[1]
    if queries_per_request not in (1, 2, 3, 4):
        raise ValueError("queries_per_request must be between one and four")
    if num_tokens % queries_per_request:
        raise ValueError("query count must be divisible by queries_per_request")
    if request_indices.numel() != num_tokens or positions.numel() != num_tokens:
        raise ValueError(
            "short-prefill indexer needs one request and position per query; "
            f"got tokens={num_tokens}, requests={request_indices.numel()}, "
            f"positions={positions.numel()}"
        )
    if summary.shape[-1] != proxy_dim:
        raise ValueError(f"proxy_dim mismatch: q={proxy_dim} pool={summary.shape[-1]}")
    if out.shape[0] != groups * num_tokens:
        raise ValueError(
            f"out has {out.shape[0]} rows, expected {groups * num_tokens}"
        )
    if width > block_table.shape[1] * regions_per_page:
        raise ValueError(
            f"score width {width} exceeds the {block_table.shape[1]} pages x "
            f"{regions_per_page} regions the block table can address"
        )

    if q_is_e4m3_rounded and index_q.dtype != torch.bfloat16:
        raise ValueError("Pre-rounded indexer queries must be BF16")
    quantized_q = index_q if q_is_e4m3_rounded else round_activations_e4m3(index_q)
    if (
        os.environ.get("VLLM_STEP5_GROUPED_HEAD_INDEXER", "0") == "1"
        and num_tokens <= 128 and groups in (1, 2, 4)
        and heads_per_group in (4, 8) and proxy_dim == 256
    ):
        from .step5_grouped_head_indexer import grouped_head_indexer_logits
        grouped_head_indexer_logits(
            quantized_q, weights, summary, block_table, request_indices,
            positions, out, regions_per_page=regions_per_page,
            queries_per_request=queries_per_request, region_size=region_size,
        )
        return
    _short_prefill_indexer_logits_kernel[
        (num_tokens // queries_per_request, triton.cdiv(width, block_k), groups)
    ](
        quantized_q,
        summary,
        block_table,
        request_indices,
        positions,
        weights,
        out,
        num_tokens,
        quantized_q.stride(0),
        quantized_q.stride(1),
        quantized_q.stride(2),
        summary.stride(0),
        block_table.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        heads_per_group=heads_per_group,
        proxy_dim=proxy_dim,
        regions_per_page=regions_per_page,
        region_size=region_size,
        query_dtype=(
            tl.bfloat16 if quantized_q.dtype is torch.bfloat16 else tl.float16
        ),
        QUERIES_PER_REQUEST=queries_per_request,
        BLOCK_Q=16,
        BLOCK_K=block_k,
    )

# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@dataclass
class Step4DSAMetadata:
    """Everything the DSA path needs that vLLM already computes.

    This is a repackaging of ``CommonAttentionMetadata``, not a computation. The
    region selection itself is *not* here: it depends on ``hidden_states``, which
    the attention layer never sees, so the indexer computes it model-side and
    hands it over through the layer object (see ``Step4DSAImpl.forward``).
    """

    # Batch split. With reorder_batch_threshold=1 decodes are at the front.
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    num_actual_tokens: int

    block_table: torch.Tensor  # [num_reqs, max_blocks] int32
    slot_mapping: torch.Tensor  # [num_tokens] int64, -1 = padding
    seq_lens: torch.Tensor  # [num_reqs] int32, context incl. current token
    query_start_loc: torch.Tensor  # [num_reqs + 1] int32
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    # Absolute position of every query token; the region selector needs it to
    # derive each row's causal horizon. May be None on a dummy/profiling run.
    positions: torch.Tensor | None
    max_seq_len: int
    block_size: int
    # Host mirror of ``block_table``, when the runner has one. The block table is
    # built on the host and uploaded, so this is the same data one step earlier
    # in the pipeline, not a read-back -- which is the whole point: the pending
    # bookkeeping below is keyed by physical summary slot, and deriving the slot
    # from the device tensor costs a sync that stalls the step. ``None`` means
    # "no mirror available", and every consumer keeps its device-side fallback.
    # Rows at or past num_reqs are stale rather than NULL_BLOCK_ID; they carry
    # seq_len == 0, which is what the masks below key on.
    block_table_np: np.ndarray | None = None
    # The decode-only compression schedule, when the metadata builder was able
    # to construct it. Building it there rather than inside the indexer op is
    # what makes the decode path capturable: the schedule is where all the host
    # control flow and the one device-to-host read live, and the builder runs
    # eagerly on every step -- including the steps that are only a graph replay.
    schedule: Step4DSASchedule | None = None
    prefill_request_indices: torch.Tensor | None = None
    max_query_len: int = 0


class Step4DSAMetadataBuilder(AttentionMetadataBuilder[Step4DSAMetadata]):
    # Single-token decode and uniform MTP target verification both use
    # persistent schedules. Mixed batches and ordinary prefills remain on the
    # piecewise path where the indexer op runs eagerly.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Pull single-token queries to the front so the batch splits cleanly into
    # the reference implementation's two paths: decode reads one query per
    # request, prefill reads a ragged run.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self._state: Step4DSAState | None = None

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if DENSE_EQUIV:
            # The dense-equivalence bypass sets topk to the batch's own score
            # width, so the packed region list is [rows, width + 1] -- a shape
            # that moves with the longest context in the batch and cannot be
            # recorded. Pinning it to the graph width instead would mean a
            # 64K-region top-k per layer at the default max_model_len, and this
            # mode exists to check the selector against full attention, not to
            # be fast. Refuse capture and say so rather than quietly pay that.
            logger.info_once(
                "step4 DSA: VLLM_STEP4_DSA_DENSE_EQUIV=1 disables CUDA graph "
                "capture; its region count follows the batch."
            )
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def _dsa_state(self) -> Step4DSAState | None:
        """The model-side state, reached through any of this backend's layers.

        The builder is constructed by vLLM from a config and a device; it has no
        handle on the model. ``_INDEXERS`` is the same registry the custom op
        uses to get from a layer name back to its indexer, so this is not a new
        coupling, just the existing one read from the other end.
        """
        if self._state is None:
            for name in self.layer_names:
                indexer = _INDEXERS.get(name)
                if indexer is not None:
                    self._state = indexer.state
                    break
        return self._state

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step4DSAMetadata:
        m = common_attn_metadata
        state = self._dsa_state()
        logical_block_size = self.block_size * (
            state.context_parallel_size if state is not None else 1
        )
        if state is not None and state.context_parallel_size == 2 and self.block_size != 64:
            raise ValueError("Step5 DCP2 summaries require local cache pages64")
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_decodes_and_prefills(m, decode_threshold=self.reorder_batch_threshold)
        md = Step4DSAMetadata(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_actual_tokens=m.num_actual_tokens,
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping,
            seq_lens=m.seq_lens,
            query_start_loc=m.query_start_loc,
            query_start_loc_cpu=m.query_start_loc_cpu,
            seq_lens_cpu=m.seq_lens_cpu,
            positions=m.positions,
            max_seq_len=m.max_seq_len,
            block_size=logical_block_size,
            max_query_len=m.max_query_len,
            block_table_np=getattr(m, "block_table_np", None),
        )

        # Build graph schedules here rather than in the indexer op. The builder
        # runs eagerly before every replay and writes current metadata into the
        # fixed addresses captured by the target model graph.
        sched = None
        if (
            state is not None
            and state.graph_decode
            and state.summary is not None  # first warmup allocates the pools
        ):
            if num_prefills == 0 and num_decodes > 0:
                sched = state._decode_schedule(md, persistent=True)
            elif num_decodes == 0 and num_prefills > 0 and state.num_speculative_tokens:
                queries = state.num_speculative_tokens + 1
                qlens = (
                    m.query_start_loc_cpu[1 : num_prefills + 1]
                    - m.query_start_loc_cpu[:num_prefills]
                )
                uniform_verification = bool(
                    torch.all((qlens == queries) | (qlens == 0)).item()
                    and md.num_actual_tokens == num_prefills * queries
                )
                if uniform_verification:
                    sched = state._verification_schedule(md, persistent=True)
                    md.prefill_request_indices = sched.request_indices

        if (
            md.prefill_request_indices is None
            and os.environ.get("VLLM_STEP5_DEVICE_REQUEST_INDICES", "0") == "1"
            and num_prefills > 0
        ):
            # Ordinary prefills keep the device-built ragged map. Uniform MTP
            # verification instead uses the persistent fixed-address map above.
            md.prefill_request_indices = prefill_request_indices_device(
                m.query_start_loc, num_decodes, num_prefills, num_prefill_tokens,
            )

        if sched is not None:
            sched.builder_owned = True
            md.schedule = sched
            # Adopt it now, so end_step() on the next build releases the prior
            # step even when the model forward itself was only a graph replay.
            state._schedule_key = md
            state._schedule = sched
            state._layers_done = 0
        return md


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


@register_backend(AttentionBackendEnum.CUSTOM)
class Step4SparseAttentionBackend(AttentionBackend):
    """DSA backend for Step-4's 23 full_attention layers.

    Registered into the ``CUSTOM`` enum slot, which is the only one left open
    for out-of-tree backends and is otherwise unused here -- ``vllm_hcu``
    registers five backends and all of them take named slots. The slot matters
    because ``Attention.__init__`` resolves ``AttentionBackendEnum[get_name()]``
    even when the backend class was passed in explicitly.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The sparse kernels read the main K/V cache either as raw bf16 or as
    # float8_e4m3, widening to the query dtype right after the load and folding
    # k_scale/v_scale into the fp32 side. (The indexer's proxy keys are fp8
    # independently of this and live in Step4DSAState.summary, not here.)
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16", "fp8_e4m3"]

    # Keep the K/V write inside this impl. The alternative (declaring False and
    # implementing do_kv_cache_update for unified_kv_cache_update) splits the
    # write away from the layout assumption it has to match, for no gain.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Step4DSAImpl"]:
        return Step4DSAImpl

    @staticmethod
    def get_builder_cls() -> type[Step4DSAMetadataBuilder]:
        return Step4DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [STEP4_HEAD_DIM]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # A region must not straddle a page: the sparse kernels address a
        # region's tokens as one contiguous run in the flattened cache.
        return [MultipleOf(REGION_BLOCK_SIZE)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # NHD, and no transpose of V: kv_cache[i].flatten(0, 1) must give
        # [num_blocks * block_size, num_kv_heads, head_size], which is the
        # "paged cache already flattened across pages" contract of
        # sparse_attention_prefill / sparse_attention_decode.
        return (2, num_blocks, block_size, num_kv_heads, head_size)


# --------------------------------------------------------------------------
# Impl
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _device_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:  # no device yet, or a platform without the field
        return 64


def _decode_num_splits(rows: int, topk: int) -> int:
    """How many ways to split the decode slot window across compute units.

    The decode kernel's grid is ``(num_reqs, num_kv_groups, num_splits)``, so at
    the recorded serving config -- 4 requests, one local KV group -- an unsplit
    launch occupies 4 of the device's 64 CUs and leaves the other 60 idle.
    Splitting the 512 selected regions K ways multiplies the grid by K and costs
    one extra log-sum-exp merge pass, which is why ``merge_split_states`` exists.

    Measured on one BW1101 at the trace_s6 decode shape: 1.594 ms unsplit,
    0.180 ms at 16 splits, then flat -- the knee is exactly where the grid fills
    the CUs. Beyond that the merge and the per-split prologue eat the gain, so
    aim for full occupancy and stop.
    """
    device_cus = _device_cu_count()
    splits = -(-device_cus // max(rows, 1))
    # A split thinner than one region block does redundant prologue work for
    # nothing; keep at least 16 slots each.
    return max(1, min(splits, topk // 16))


def _require_splitting_op() -> None:
    """Fail at startup if the indexer op would be captured into a *piecewise* graph.

    A captured graph replays recorded kernels, not Python. For a decode-only
    step that is now the intended path: the builder writes this step's schedule
    into fixed addresses before the model runs, so the recorded kernels read
    current values and the op's Python is only needed at capture. See
    ``Step4DSAMetadataBuilder.get_cudagraph_support``.

    Piecewise is the case this guards. There the op has to sit on a split
    boundary so it runs eagerly between the compiled subgraphs; swallowed into
    one, it executes exactly once -- during warm-up, where there is no metadata
    and it publishes no plan -- and every real step afterwards attends to
    whatever that single execution left behind. The three ways it has actually
    gone wrong: no plan at all, a plan from the previous step, and (before the
    op existed) a traced-away no-op. None of them are quiet, but only because
    the plan is checked; a coarser design would have returned plausible text.

    Being in ``splitting_ops`` serves both: it makes the op a boundary for
    piecewise compilation, and a full-graph capture takes the whole forward
    including the eager gaps between subgraphs.

    Step4Config.__init__ normally arranges this by appending the op to
    CompilationConfig._attention_ops before the config is finalised. It is
    bypassed by passing --compilation-config with an explicit splitting_ops
    list, which is why this check exists at all.
    """
    try:
        config = get_current_vllm_config()
    except AssertionError:
        # No config in scope. The offline gates (test_dsa_indexer.py,
        # test_dsa_vllm_wiring.py) construct the impl directly and call it
        # eagerly, so there is no graph to be captured into.
        return
    compilation = config.compilation_config
    if compilation.cudagraph_mode == CUDAGraphMode.NONE:
        return  # nothing is captured, so the op runs every step wherever it sits
    required = [STEP4_DSA_INDEX_OP]
    if os.environ.get("VLLM_STEP5_DSA_PACK_QKNORM", "0") == "1":
        required.append("vllm::step5_dsa_index_qknorm")
    if all(op in (compilation.splitting_ops or []) for op in required):
        return
    raise RuntimeError(
        f"Step-4 DSA requires {required!r} in "
        f"CompilationConfig.splitting_ops, but the resolved list is "
        f"{compilation.splitting_ops!r} with cudagraph_mode="
        f"{compilation.cudagraph_mode}. The indexer would be captured into a "
        f"CUDA graph and its region plan would never be recomputed. Either drop "
        f"the explicit --compilation-config splitting_ops (Step4Config registers "
        f"the op automatically), add the op to that list, or run with "
        f"enforce_eager=True."
    )


class Step4DSAImpl(AttentionImpl[Step4DSAMetadata]):
    """Writes K/V, then runs sparse attention over the indexer's chosen regions.

    The region plan is produced model-side by the indexer and attached to the
    ``Attention`` layer object as ``step4_sparse_plan`` immediately before this
    runs. Passing it through the layer rather than a module-level dict keeps the
    producer and consumer in the same object graph, so a missing plan is a
    construction error rather than a silent fallback.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("Step-4 DSA does not use ALiBi")
        if sliding_window is not None:
            raise NotImplementedError(
                "Step-4's sliding layers do not use DSA; only full_attention "
                "layers are routed to this backend."
            )
        if logits_soft_cap:
            raise NotImplementedError("Step-4 DSA does not use logit soft-capping")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("Step-4 DSA does not share KV across layers")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        if BOLTOPS_STAGES:
            stages = _boltops_attention()
            if kv_cache_dtype not in ("auto", "bfloat16"):
                # boltops gained fp8 cache reads in its commit 28239f4. Probing
                # the signature rather than a version string keeps this honest
                # against a library that is installed editable from a checkout:
                # without the scales the kernels silently attend to raw fp8
                # bytes, which is wrong by a factor of k_scale, not noisy.
                missing = [
                    name
                    for name, fn in zip(("prefill", "decode"), stages)
                    if name in BOLTOPS_STAGES
                    and not {"k_scale", "v_scale"}.issubset(
                        inspect.signature(fn).parameters
                    )
                ]
                if missing:
                    raise NotImplementedError(
                        f"VLLM_STEP4_DSA_BOLTOPS={','.join(sorted(BOLTOPS_STAGES))} "
                        f"with --kv-cache-dtype {kv_cache_dtype}: the installed "
                        f"boltops takes no k_scale/v_scale on "
                        f"{', '.join(missing)}, so it cannot dequantize an fp8 "
                        f"cache. Update boltops to 28239f4 or later, run the "
                        f"comparison at bfloat16 on both sides, or keep the "
                        f"vendored kernels for an fp8 cache."
                    )

        _require_splitting_op()

    # -- KV write ---------------------------------------------------------

    @staticmethod
    def _write_kv(
        kv_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter K/V into the paged cache and return the flattened views.

        ``slot_mapping`` uses -1 for padded rows. Indexing with -1 would silently
        write to the last slot of the cache and corrupt whichever request owns
        it, so the mask is load-bearing, not defensive.

        An fp8 cache arrives here as ``uint8``: vLLM maps every fp8 kv-cache dtype
        to a byte buffer in ``STR_DTYPE_TO_TORCH_DTYPE`` and leaves interpretation
        to the backend. The ``view`` below is that interpretation, and it has to
        happen before either the store or the sparse kernels see the tensor --
        they widen K/V with ``.to(query.dtype)``, which on a uint8 tensor would
        convert the *integer* byte values instead of reinterpreting the fp8 bits,
        and produce plausible-looking garbage rather than an error.
        """
        num_blocks, block_size = kv_cache.shape[1], kv_cache.shape[2]
        num_kv_heads, head_size = kv_cache.shape[3], kv_cache.shape[4]
        k_flat = kv_cache[0].view(num_blocks * block_size, num_kv_heads, head_size)
        v_flat = kv_cache[1].view(num_blocks * block_size, num_kv_heads, head_size)
        if kv_cache.dtype == torch.uint8:
            k_flat = k_flat.view(torch.float8_e4m3fn)
            v_flat = v_flat.view(torch.float8_e4m3fn)

        slots = slot_mapping[: key.shape[0]]
        _store_kv_rows(k_flat, v_flat, key, value, slots, k_scale, v_scale)
        return k_flat, v_flat

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Step4DSAMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Step-4 DSA does not support fused output quantization"
            )

        # Dummy/profiling run: no metadata, no cache to write, nothing to
        # attend to. The profiler only needs the allocation high-water mark.
        if attn_metadata is None:
            return output

        num_tokens = attn_metadata.num_actual_tokens
        # Both default to 1.0 unless --calculate-kv-scales is on. Keeping them at
        # 1.0 is what puts this path on the same footing as Step-4's 69 sliding
        # layers: their fp8 flash-attention kernels are compiled with
        # FP8_NO_SCALES and ignore descales entirely, so a DSA layer quantizing at
        # some other scale would make the two families disagree about what the
        # cache bytes mean.
        k_scale = float(getattr(layer, "_k_scale_float", 1.0))
        v_scale = float(getattr(layer, "_v_scale_float", 1.0))
        k_flat, v_flat = self._write_kv(
            kv_cache,
            key[:num_tokens],
            value[:num_tokens],
            attn_metadata.slot_mapping,
            k_scale,
            v_scale,
        )

        plan = getattr(layer, "step4_sparse_plan", None)
        if plan is None:
            raise RuntimeError(
                f"{layer.layer_name}: no DSA region plan. The indexer must run "
                f"before the attention layer it feeds."
            )
        if plan.md is not attn_metadata:
            raise RuntimeError(
                f"{layer.layer_name}: the DSA region plan belongs to a different "
                f"forward step. The indexer op and the attention op have been "
                f"reordered, and attending to a stale region set would be silent."
            )

        num_kv_groups = self.num_kv_heads
        nd, ndt = attn_metadata.num_decodes, attn_metadata.num_decode_tokens

        # Decode rows come first (reorder_batch_threshold=1).
        if nd > 0:
            packed, counts = plan.decode
            seq_lens_i32 = attn_metadata.seq_lens[:nd].to(torch.int32)
            if "decode" in FLASH_ATTN_DSA_STAGES:
                # flash-attention-cutlass's C++ entry point ignores num_splits
                # and always returns final merged state (see _flash_attn_decode),
                # so neither _decode_num_splits nor merge_split_states runs here.
                out_d, lse_d = _flash_attn_decode(
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            elif "decode" in BOLTOPS_STAGES:
                # num_splits=1 is boltops' "give me final states": it picks the
                # split count from device occupancy and merges internally, so
                # neither _decode_num_splits nor merge_split_states runs here.
                # config= is resolved on the host to keep boltops' own JSON
                # lookup -- which reads kv_seqlens.max() off the device -- out of
                # a graph-captured step.
                out_d, lse_d = _boltops_attention()[1](
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    num_splits=1,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    config=_boltops_decode_config(
                        nd,
                        self.num_heads,
                        self.head_size,
                        num_kv_groups,
                        int(packed.shape[1]),
                        REGION_BLOCK_SIZE,
                        int(attn_metadata.seq_lens_cpu[:nd].max()),
                        query.dtype,
                        k_flat.dtype,
                    ),
                )
            else:
                splits = _decode_num_splits(nd * num_kv_groups, packed.shape[1])
                out_d, lse_d = sparse_attention_decode(
                    query[:ndt],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    seq_lens_i32,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    num_splits=splits,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    qk_fp8=DECODE_QK_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                    pv_fp8=DECODE_PV_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                )
                if splits > 1:
                    out_d, lse_d = merge_split_states(out_d, lse_d)
            del lse_d
            output[:ndt] = out_d

        if attn_metadata.num_prefills > 0:
            packed, counts, logical_ids, request_indices = plan.prefill
            wrote_prefill_output = False
            prefill_tokens = num_tokens - ndt
            # The decode launcher captures up to max_num_seqs=32 target rows.
            # MTP3 verifies four queries per row, so the largest uniform target
            # graph contains 128 tokens. Keep that whole verification range on
            # the device-only short path; the general prefill path constructs
            # query tiles on the host and cannot be recorded in a full graph.
            # Ragged MTP batches also stay on the short path when every
            # request has at most four queries. Ordinary 65--128 token
            # prefills retain the tiled kernel.
            verification = bool(
                attn_metadata.schedule is not None
                and attn_metadata.schedule.verification
            )
            short_query = is_short_query_batch(
                prefill_tokens, attn_metadata.max_query_len,
                verification=verification,
                capturing=torch.cuda.is_current_stream_capturing(),
            )
            use_flash_prefill = (
                "prefill" in FLASH_ATTN_DSA_STAGES
                or (
                    "short_prefill" in FLASH_ATTN_DSA_STAGES
                    and short_query
                )
            )
            if not use_flash_prefill:
                # Native FA consumes packed regions only; avoid an unused
                # position conversion in each sparse Attention layer.
                q_positions = attn_metadata.positions[ndt:num_tokens].to(torch.int32)
                prefill_kwargs = dict(
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    logical_regions=logical_ids,
                    q_positions=q_positions,
                    request_indices=request_indices,
                    block_q=128,
                    block_regions=16,
                )
            if use_flash_prefill:
                # Native FA consumes the already-packed causal regions;
                # query-tiled positions and request indices are unnecessary.
                out_p, lse_p = _flash_attn_prefill(
                    query[ndt:num_tokens],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    num_kv_groups=num_kv_groups,
                    region_size=REGION_BLOCK_SIZE,
                    softmax_scale=self.scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            elif "prefill" in BOLTOPS_STAGES:
                out_p, lse_p = _boltops_attention()[0](
                    query[ndt:num_tokens],
                    k_flat,
                    v_flat,
                    packed,
                    counts,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    **prefill_kwargs,
                )
            else:
                # MTP verification has only a few query tokens. The untiled
                # per-query kernel consumes the same causal region metadata
                # without building a large union bitmap on the host.
                if (
                    short_query
                    and not PREFILL_QK_FP8
                    and not PREFILL_PV_FP8
                ):
                    prefill_kwargs["logical_regions"] = None
                if (
                    os.environ.get("VLLM_STEP5_DSA_SHORT_QUERY", "0") == "1"
                    and short_query
                    and self.head_size == 192
                    and not PREFILL_QK_FP8
                    and not PREFILL_PV_FP8
                ):
                    if os.environ.get(
                        "VLLM_STEP5_DSA_TUNED_SHORT_SPLITS", "0"
                    ) == "1":
                        splits = short_query_num_splits(
                            prefill_tokens, num_kv_groups, packed.shape[1]
                        )
                    else:
                        splits = _decode_num_splits(
                            prefill_tokens * num_kv_groups, packed.shape[1]
                        )
                    out_p, lse_p = sparse_attention_short_prefill(
                        query[ndt:num_tokens], k_flat, v_flat, packed, counts,
                        num_kv_groups=num_kv_groups,
                        region_size=REGION_BLOCK_SIZE, softmax_scale=self.scale,
                        num_splits=splits, k_scale=k_scale, v_scale=v_scale,
                        output=output[ndt:num_tokens], return_lse=False,
                    )
                    wrote_prefill_output = True
                else:
                    out_p, lse_p = sparse_attention_prefill(
                        query[ndt:num_tokens],
                        k_flat,
                        v_flat,
                        packed,
                        counts,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        qk_fp8=PREFILL_QK_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                        pv_fp8=PREFILL_PV_FP8 and k_flat.dtype == torch.float8_e4m3fn,
                        **prefill_kwargs,
                    )
            del lse_p
            if not wrote_prefill_output:
                output[ndt:num_tokens] = out_p

        return output


@dataclass
class Step4SparsePlan:
    """Selected regions for one layer, one forward step.

    ``prefill`` additionally carries logical region ids and request indices for the
    query-tiled prefill kernel. The packed words remain physical cache addresses and are
    not interchangeable with those logical ids.
    """

    prefill: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
    ] | None = None
    decode: tuple[torch.Tensor, torch.Tensor] | None = None
    # The metadata this plan was built from. The impl compares it by identity:
    # the indexer and the attention layer are two separate ops in the compiled
    # graph, and if anything ever reordered them the layer would silently attend
    # to the previous step's regions instead of failing.
    md: Step4DSAMetadata | None = None
    main_qk: tuple[torch.Tensor, torch.Tensor] | None = None


def build_plan(
    attn_metadata: Step4DSAMetadata,
    prefill_logits: torch.Tensor | None,
    decode_logits: torch.Tensor | None,
    *,
    topk: int,
    num_kv_groups: int,
    regions_per_page: int,
    context_parallel_size: int = 1,
    main_qknorm: tuple | None = None,
) -> Step4SparsePlan:
    """Turn indexer scores into the two kernels' metadata.

    Both halves are pass-throughs to the reference selectors, which already
    speak vLLM's block table: they take ``[num_reqs, pages]`` logical->physical
    maps plus a per-row request index, which is exactly what
    ``CommonAttentionMetadata`` carries.
    """
    if context_parallel_size not in (1, 2):
        raise ValueError("Step5 sparse selections support CP1 or experimental CP2")
    plan = Step4SparsePlan()
    plan.md = attn_metadata
    nd = attn_metadata.num_decodes
    ndt = attn_metadata.num_decode_tokens

    if nd > 0 and decode_logits is not None:
        decode_meta = decode_sparse_meta
        decode_seqlens = attn_metadata.seq_lens[:nd].to(torch.int32)
        if os.environ.get("VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512", "0") == "1":
            from .step5_dsa_lightop_selector import (
                decode_sparse_meta_lightop, supports_native_decode_selector,
            )
            if supports_native_decode_selector(
                decode_logits, decode_seqlens, topk=topk,
            ):
                decode_meta = decode_sparse_meta_lightop
        packed, counts = decode_meta(
            decode_logits,
            decode_seqlens,
            attn_metadata.block_table[:nd],
            None,  # group-major arange(num_reqs), the default, is what we want
            topk=topk,
            region_size=REGION_BLOCK_SIZE,
            regions_per_page=regions_per_page,
        )
        plan.decode = (packed, counts)

    if attn_metadata.num_prefills > 0 and prefill_logits is not None:
        positions = attn_metadata.positions
        assert positions is not None, (
            "CommonAttentionMetadata.positions is required by the DSA region "
            "selector; it derives each row's causal horizon from it."
        )
        fused_selector = _STEP5_DSA_SELECTOR in {"1", "radix_flash"}
        q_positions = positions[ndt : attn_metadata.num_actual_tokens]
        if not fused_selector:
            q_positions = q_positions.to(torch.int32)
        # request_indices maps each score row to its block-table row. Rows are
        # group-major, so the per-token request index tiles once per KV group.
        req_of_token = attn_metadata.prefill_request_indices
        if req_of_token is None:
            starts = attn_metadata.query_start_loc_cpu
            req_of_token = torch.repeat_interleave(
                torch.arange(
                    nd,
                    nd + attn_metadata.num_prefills,
                    dtype=torch.int32,
                    device=positions.device,
                ),
                (starts[nd + 1 :] - starts[nd:-1]).to(positions.device),
            )
        group_request_indices = req_of_token.repeat(num_kv_groups)
        prefill_meta = (
            prefill_sparse_meta_fused
            if fused_selector
            else prefill_sparse_meta
        )
        prefill_tokens = attn_metadata.num_actual_tokens - ndt
        verification = bool(
            attn_metadata.schedule is not None
            and attn_metadata.schedule.verification
        )
        capturing = (
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        short_query = is_short_query_batch(
            prefill_tokens, attn_metadata.max_query_len,
            verification=verification, capturing=capturing,
        )
        # DCP2 exchanges logical selections before addressing striped cache;
        # short-query packed physical words cannot replace these logical IDs.
        need_logical_ids = context_parallel_size == 2 or not (
            os.environ.get("VLLM_STEP5_DSA_SHORT_QUERY", "0") == "1"
            and short_query
            and not PREFILL_QK_FP8
            and not PREFILL_PV_FP8
        )
        if (
            fused_selector and short_query
            and os.environ.get("VLLM_STEP5_DSA_LIGHTOP_TOP512", "0") == "1"
        ):
            from .step5_dsa_lightop_selector import (
                prefill_sparse_meta_lightop, supports_native_selector,
            )
            if supports_native_selector(
                prefill_logits, q_positions, group_request_indices,
                topk=topk, num_kv_groups=num_kv_groups,
            ):
                prefill_meta = prefill_sparse_meta_lightop
        selector_kwargs = {"num_kv_groups": num_kv_groups} if fused_selector else {}
        joint_norm = False
        if (
            main_qknorm is not None and fused_selector and short_query
            and prefill_tokens == attn_metadata.num_actual_tokens
            and not need_logical_ids and context_parallel_size == 1
            and prefill_meta is not prefill_sparse_meta_fused
            and os.environ.get("VLLM_STEP5_DSA_LIGHTOP_TOP512", "0") == "1"
        ):
            from .step5_dsa_norm_selector import (
                supports_main_norm_selector, prefill_meta_with_main_norm,
            )
            joint_norm = supports_main_norm_selector(
                prefill_logits, q_positions, group_request_indices, main_qknorm
            )
        if joint_norm:
            packed, counts, q, k = prefill_meta_with_main_norm(
                prefill_logits, q_positions, attn_metadata.block_table,
                group_request_indices, main_qknorm,
                regions_per_page=regions_per_page,
            )
            plan.main_qk = (q, k)
            selected = (packed, counts)
        else:
            selected = prefill_meta(
                prefill_logits,
                q_positions if fused_selector else q_positions.repeat(num_kv_groups),
                attn_metadata.block_table,
                group_request_indices,
                topk=topk,
                region_size=REGION_BLOCK_SIZE,
                regions_per_page=regions_per_page,
                return_logical_ids=need_logical_ids,
                **selector_kwargs,
            )
        if need_logical_ids:
            packed, counts, logical_ids = selected
        else:
            packed, counts = selected
            logical_ids = None
        plan.prefill = (
            packed,
            counts,
            logical_ids,
            group_request_indices,
        )
        _maybe_dump_prefill_plan(packed, counts, q_positions)

    return plan


def _maybe_dump_prefill_plan(packed, counts, q_positions) -> None:
    """Save one real prefill region plan when VLLM_STEP4_DUMP_PLAN names a path.

    Temporary instrumentation for the query-tiling work. Tiling queries only pays
    if neighbouring queries select overlapping regions, and how much they overlap
    is a property of the trained indexer that no synthetic plan can stand in for.
    Writes the first long chunk it sees and then stays quiet.
    """
    import os

    path = os.environ.get("VLLM_STEP4_DUMP_PLAN")
    if not path or getattr(_maybe_dump_prefill_plan, "_done", False):
        return
    if packed.shape[0] < 8192:  # skip warm-up and short chunks
        return
    _maybe_dump_prefill_plan._done = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = f"{path}.rank{rank}"
    torch.save(
        {
            "packed": packed.cpu(),
            "counts": counts.cpu(),
            "positions": q_positions.cpu(),
        },
        path,
    )
    logger.info("step4 DSA: dumped prefill plan %s -> %s", tuple(packed.shape), path)


# --------------------------------------------------------------------------
# Indexer geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step4DSAGeometry:
    """The DSA shape constants, read from the checkpoint's ``sparse_config``.

    Nothing here is hardcoded except the fallbacks, which match the released
    config. ``regions_per_page`` is deliberately absent: it is
    ``kv_block_size // region_size`` and therefore a property of the *runtime*
    cache, not of the checkpoint, so it is derived where the block size is known.
    """

    proxy_dim: int = 256
    topk: int = 512
    region_size: int = REGION_BLOCK_SIZE
    # The full rotated width of an indexer head. The cos/sin tables are half
    # this, because one frequency drives the pair (d, d + span // 2), and it is
    # that halved value the kernels call ``rotary_dim``.
    rope_span: int = 32
    num_indexer_heads: int = 16
    num_provider_groups: int = 4

    @classmethod
    def from_config(cls, sparse_config: Any) -> "Step4DSAGeometry":
        def pick(name: str, default: int) -> int:
            if isinstance(sparse_config, dict):
                return int(sparse_config.get(name, default))
            return int(getattr(sparse_config, name, default))

        geometry = cls(
            proxy_dim=pick("proxy_dim", 256),
            topk=pick("topk", 512),
            region_size=pick("region_block_size", REGION_BLOCK_SIZE),
            rope_span=pick("sparse_indexer_rope_dim", 32),
            num_indexer_heads=pick("sparse_indexer_num_heads", 16),
            num_provider_groups=pick("num_provider_groups", 4),
        )
        if geometry.region_size != REGION_BLOCK_SIZE:
            raise ValueError(
                f"sparse_config.region_block_size={geometry.region_size} but the "
                f"released kernels are compiled around {REGION_BLOCK_SIZE}"
            )
        if geometry.rope_span % 2:
            raise ValueError(f"rope span must be even, got {geometry.rope_span}")
        return geometry


# --------------------------------------------------------------------------
# Pending partial regions
# --------------------------------------------------------------------------


class Step4PendingTable:
    """Which pending buffer row holds which region's not-yet-complete tokens.

    Keyed by the region's *physical* summary slot, never by a batch row. vLLM
    swaps batch rows when it reorders decodes to the front, and a preempted
    request loses its row outright, so a row index is not a stable identity for
    a request across steps. The physical slot is: it stays valid exactly as long
    as the request holds the block, which is exactly as long as its half-filled
    region means anything.

    Rows are recycled least-recently-touched. That is safe because every live
    partial region is touched on every step its owner is scheduled, and a
    request that stops being scheduled has either finished (pending is dead) or
    been preempted (vLLM recomputes it from position 0, so pending is dead
    again). Evicting a row that was touched *this* step would be a capacity bug,
    so it raises instead.

    The table is shared by all 23 DSA layers -- the slot-to-row assignment is
    layer-independent, only the buffered values differ -- so it is stepped once
    per forward pass, not once per layer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._row_of_slot: dict[int, int] = {}
        self._slot_of_row: list[int] = [-1] * capacity
        self._touched: list[int] = [-1] * capacity
        self.step = 0

    def begin_step(self) -> None:
        self.step += 1

    def get(self, slot: int) -> int:
        """Row holding ``slot``'s partial tokens; raises if there is none."""
        row = self._row_of_slot.get(slot)
        if row is None:
            raise RuntimeError(
                f"Step-4 DSA: no pending tokens for region slot {slot}. A request "
                f"resumed in the middle of a region whose leading tokens were "
                f"never buffered -- the summary for that region would be wrong, "
                f"so this fails loudly rather than silently attending to noise."
            )
        self._touched[row] = self.step
        return row

    def acquire(self, slot: int) -> int:
        """Row for ``slot``, allocating or recycling one if needed."""
        row = self._row_of_slot.get(slot)
        if row is None:
            row = self._free_row()
            self._row_of_slot[slot] = row
            self._slot_of_row[row] = slot
        self._touched[row] = self.step
        return row

    def release(self, slot: int) -> None:
        row = self._row_of_slot.pop(slot, None)
        if row is not None:
            self._slot_of_row[row] = -1
            self._touched[row] = -1

    def _free_row(self) -> int:
        oldest_row, oldest_step = -1, self.step
        for row in range(self.capacity):
            if self._slot_of_row[row] < 0:
                return row
            if self._touched[row] < oldest_step:
                oldest_row, oldest_step = row, self._touched[row]
        if oldest_row < 0:
            raise RuntimeError(
                f"Step-4 DSA: all {self.capacity} pending region buffers are in use "
                f"this step. Capacity is sized from max_num_seqs, so this means "
                f"more concurrent sequences reached the indexer than the scheduler "
                f"declared."
            )
        # Drop the recycled row's forward mapping too. Leaving it behind would
        # leave a stale slot pointing at a row another region now owns, and
        # get() would hand back that other request's buffered tokens instead of
        # raising -- the one failure this table exists to make impossible.
        self._row_of_slot.pop(self._slot_of_row[oldest_row], None)
        self._slot_of_row[oldest_row] = -1
        return oldest_row


# --------------------------------------------------------------------------
# Per-step schedule
# --------------------------------------------------------------------------


@dataclass
class _RequestPlan:
    """Where one request's tokens land in the compression buffer, this step."""

    row: int  # block-table row
    past: int  # tokens already in the KV cache before this step
    qlen: int  # tokens contributed this step
    tok_start: int  # offset of those tokens in the packed batch
    head: int  # leading tokens taken from pending (== past % region_size)
    buf_start: int  # offset in the assembled compression buffer
    pending_row_in: int  # pending row to read the head from, -1 if head == 0
    pending_row_out: int  # pending row to write the tail to, -1 if no tail
    tail: int  # trailing tokens that do not complete a region
    # Complete regions this request's *last* query may score against, i.e.
    # (past + qlen - 1) // region_size. Earlier queries in the same chunk see
    # fewer; the selector bounds each row by its own position, so scoring the
    # whole width and letting it clip is both correct and simpler.
    nscore: int
    score_slots: torch.Tensor | None  # [nscore] int64 physical summary slots


@dataclass
class Step4DSASchedule:
    """Layer-independent bookkeeping for one forward step.

    The 23 DSA layers see the same batch, so which regions complete, where their
    summaries go, and which pending rows are involved is computed once and
    reused. Only the values differ per layer.
    """

    requests: list[_RequestPlan] = field(default_factory=list)
    buf_len: int = 0
    # Region compression arguments, over the assembled buffer.
    starts: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    slots: torch.Tensor | None = None  # physical summary slot per region
    live: torch.Tensor | None = None  # counts == region_size
    # Regions whose summary is now final; their pending rows are released after
    # every layer has consumed them.
    completed_slots: list[int] = field(default_factory=list)
    # (pending row, assembled-buffer offset, prefix length). Verification can
    # reopen a completed region after draft rejection, so retain its prefix.
    rollback_prefixes: list[tuple[int, int, int]] = field(default_factory=list)

    # -- pure-decode fast path ---------------------------------------------
    # Set only when the step is decode-only (num_prefills == 0), where every
    # request contributes exactly one token and therefore touches exactly one
    # region. That collapses the per-request Python above into fixed-shape
    # tensors, which is what lets the 23 layers run batched instead of looping.
    # ``requests`` is left empty in that case: the generic consumers must not
    # silently fall through to a zero-iteration loop, so they check ``decode``.
    decode: bool = False
    num_decodes: int = 0
    head: torch.Tensor | None = None  # [nd] int64, tokens already in the region
    tail: torch.Tensor | None = None  # [nd] int64, tokens to re-buffer, 0 if closed
    nscore: torch.Tensor | None = None  # [nd] int32, complete regions visible
    max_nscore: int = 0
    pending_in: torch.Tensor | None = None  # [nd] int64 row to read the head from
    pending_out: torch.Tensor | None = None  # [nd] int64 row to write the tail to
    # Built by Step4DSAMetadataBuilder rather than by the first layer to reach
    # state.schedule(). The pending rows have then already been assigned by the
    # time the layers run, so finish_layer() must not release them a second time.
    builder_owned: bool = False
    # Fixed score width this schedule was built against, when it came from
    # persistent buffers. Zero means "size the score tensor to max_nscore", the
    # eager behaviour.
    graph_regions: int = 0

    # -- speculative target-verification fast path ------------------------
    # MTP3 verifies four consecutive queries per request. A four-query run can
    # cross at most two 8-token compression regions, so this schedule uses two
    # fixed region records per request and keeps every replayed pointer stable.
    verification: bool = False
    queries_per_request: int = 0
    num_requests: int = 0
    request_indices: torch.Tensor | None = None  # [num_requests * q] int32
    pending_rows: torch.Tensor | None = None  # [num_requests * 2] int64
    prefix_lengths: torch.Tensor | None = None  # [num_requests * 2] int32


# --------------------------------------------------------------------------
# Model-side DSA state
# --------------------------------------------------------------------------


class Step4DSAState:
    """Summaries, pending buffers, and the per-step schedule for one model.

    One instance per ``Step4Model``; every DSA layer holds a reference and its
    own slot index into the layer-major tensors. Allocation is lazy because the
    number of KV blocks is not known until vLLM has profiled memory and built
    the cache -- which is also why this pool comes out of the leftover headroom
    rather than out of the profiled budget.

    **Why the summaries are not an ``MLAAttentionSpec``.** The obvious route --
    registering them with vLLM's KV cache manager at ``compress_ratio =
    region_size``, the way ``DeepseekV4IndexerCache`` does -- does not survive
    Step-4's mix of specs. Page sizes are only unified through the
    DeepSeek-specific branch when a ``SlidingWindowMLASpec`` is present; Step-4
    has none, so it falls through to ``unify_kv_cache_spec_page_size``, which
    pads every spec up to a common page. A 512 B summary page padded to the
    12288 B DSA/SWA page is a 24x blowup: the summary pool would cost as much as
    the entire KV pool. Allocating here instead costs ``proxy_dim / (2 *
    head_dim * region_size)`` of the DSA KV pool -- about 4.2% at Step-4
    geometry -- and the exact figure is logged.

    **Addressing.** Physical KV block ``b`` owns summary rows ``[b *
    regions_per_page, (b + 1) * regions_per_page)``. Reusing the KV block id
    means no second block table and no second allocator.

    **Why block reuse needs no invalidation.** When a block is recycled its
    summary rows are stale, but only rows for *incomplete* regions can be: a
    region that fills is always overwritten as it fills. And an incomplete
    region's summary is never read -- the selector bounds each row's candidates
    to ``(past + len - 1) // region_size``, i.e. complete regions only, and the
    query's own partial region is force-included without consulting a summary.
    That is precisely why the reference refuses to store partial summaries.
    """

    def __init__(
        self,
        num_layers: int,
        geometry: Step4DSAGeometry,
        max_pending: int,
        dtype: torch.dtype,
        max_num_reqs: int = 0,
        max_model_len: int = 0,
        num_speculative_tokens: int = 0,
        context_parallel_size: int = 1,
    ) -> None:
        if context_parallel_size not in (1, 2):
            raise ValueError("DSA summary geometry supports context sizes 1 and 2")
        self.context_parallel_size = context_parallel_size
        self.num_layers = num_layers
        self.geometry = geometry
        self.dtype = dtype
        self.pending = Step4PendingTable(max_pending)
        self.max_pending = max_pending
        self.num_speculative_tokens = int(num_speculative_tokens)
        # One row past the allocator's capacity, never handed out by the table.
        # The vectorized decode path writes every request's tail unconditionally
        # and sends the requests that have no tail here, which keeps the scatter
        # a fixed-shape tensor op instead of a data-dependent row selection.
        self.pending_trash_row = max_pending

        # Capture parameters. Both must be known for the decode schedule to live
        # in persistent buffers at a fixed score width; when either is zero the
        # path still works, it just reallocates per step at the batch's own
        # width and cannot be captured.
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_len = int(max_model_len)
        self.graph_decode = self.max_num_reqs > 0 and self.max_model_len > 0
        # Set by the first Step4Indexer to be constructed; the score pool is
        # group-major and cannot be sized without it.
        self.num_kv_groups = 0
        self.heads_per_group = 0
        self.graph_regions = 0

        self.summary: torch.Tensor | None = None
        self.pending_key: torch.Tensor | None = None
        self.pending_z: torch.Tensor | None = None
        self.regions_per_page = 0
        self.block_size = 0

        # Persistent decode schedule, filled in place every step. Allocated in
        # ensure_allocated; None until then, and always None when graph_decode
        # is off.
        self._buf: dict[str, torch.Tensor] = {}
        self._host: dict[str, list[torch.Tensor]] = {}
        self._stage = 0
        self._stage_ev: list[torch.cuda.Event] = []
        self._decode_logits: torch.Tensor | None = None
        self._verification_logits: torch.Tensor | None = None

        self._rope: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._schedule_key: object | None = None
        self._schedule: Step4DSASchedule | None = None
        # Which layer slots actually exist on this rank, so end_step() fires
        # after the last one rather than after a hardcoded count (pipeline
        # parallel would give a rank only some of the 23).
        self._registered: set[int] = set()
        self._layers_done = 0

    def register_layer(self, slot: int) -> None:
        self._registered.add(slot)

    # -- allocation -------------------------------------------------------

    def reset_cache(self) -> None:
        # V2 graph-memory profiling binds a minimal, temporary KV cache.
        # Its block IDs and graph buffers must not survive cache replacement.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot replace Step5 DSA cache during graph capture")
        self.summary = None
        self.pending_key = None
        self.pending_z = None
        self.pending = Step4PendingTable(self.max_pending)
        self.block_size = 0
        self.regions_per_page = 0
        self.graph_regions = 0
        self._buf = {}
        self._host = {}
        self._stage = 0
        self._stage_ev = []
        self._decode_logits = None
        self._verification_logits = None
        self._schedule_key = None
        self._schedule = None
        self._layers_done = 0

    def ensure_allocated(self, kv_cache: torch.Tensor, device: torch.device) -> None:
        num_blocks = int(kv_cache.shape[1])
        physical_block_size = int(kv_cache.shape[2])
        if self.context_parallel_size == 2 and physical_block_size != 64:
            raise ValueError("Step5 DCP2 summaries require local cache pages64")
        # Summary compression always sees the complete global token stream.
        # Physical DCP2 page64 corresponds to logical page128/16 region8 rows.
        block_size = physical_block_size * self.context_parallel_size
        if self.summary is not None:
            expected_rows = num_blocks * (block_size // self.geometry.region_size)
            if self.block_size == block_size and self.summary.shape[1] == expected_rows:
                return
            self.reset_cache()
        if torch.cuda.is_current_stream_capturing():
            # Anything allocated here would come out of the graph's private
            # memory pool and be freed with it. vLLM runs cudagraph_num_of_warmups
            # eager steps before capturing, so reaching this under capture means
            # the warmup never touched the DSA path -- fail loudly rather than
            # hand back memory that goes away.
            raise RuntimeError(
                "Step-4 DSA: summary pool is still unallocated at CUDA graph "
                "capture time. The capture warmup must run at least one eager "
                "step through the DSA layers first."
            )
        if block_size % self.geometry.region_size:
            raise ValueError(
                f"KV block size {block_size} is not a multiple of the DSA region "
                f"size {self.geometry.region_size}; a region straddling two pages "
                f"would break the contiguity the sparse kernels rely on."
            )
        self.block_size = block_size
        self.regions_per_page = block_size // self.geometry.region_size
        rows = num_blocks * self.regions_per_page
        proxy = self.geometry.proxy_dim

        self.summary = torch.zeros(
            (self.num_layers, rows, 1, proxy), dtype=torch.float8_e4m3fn, device=device,
        )
        self.pending_key = torch.zeros(
            (self.num_layers, self.max_pending + 1, self.geometry.region_size, proxy),
            dtype=self.dtype,
            device=device,
        )
        self.pending_z = torch.zeros_like(self.pending_key)
        # Measured against the real cache rather than derived from the geometry:
        # the summary is one proxy vector per region regardless of how many KV
        # heads a rank holds, so the ratio moves with the TP degree.
        kv_bytes = kv_cache.numel() * kv_cache.element_size()
        logger.info(
            "step4 DSA: %d region summaries x %d layers = %.1f MiB fp8 "
            "(%.1f%% of one DSA layer's KV pool), plus %.1f MiB of pending buffers.",
            rows,
            self.num_layers,
            self.summary.numel() / 2 ** 20,
            100.0 * self.summary[0].numel() / max(kv_bytes, 1),
            2 * self.pending_key.numel() * self.pending_key.element_size() / 2 ** 20,
        )
        self._allocate_graph_buffers(rows, device)

    def _allocate_graph_buffers(self, pool_rows: int, device: torch.device) -> None:
        """Persistent decode-schedule buffers and the fixed-width score pool.

        A CUDA graph replays the pointers and launch arguments it recorded, so
        every tensor the decode path touches has to keep both its address and
        its shape from one step to the next. These are allocated once, outside
        capture, and refilled in place by ``_decode_schedule``.

        The score pool is deliberately *not* zeroed, then or ever. Each row is
        written for its own ``nscore`` columns and the selector reads exactly
        ``(seq_len - 1) // region_size`` of them -- the same quantity -- so no
        read ever reaches a column this step did not write. Zeroing 64K fp32
        columns per row per layer would cost more than the scoring.
        """
        if not self.graph_decode:
            return
        if self.num_kv_groups <= 0:
            raise RuntimeError(
                "Step-4 DSA: num_kv_groups was never published to the state; "
                "Step4Indexer.__init__ must set it before the first forward."
            )
        region = self.geometry.region_size
        # A request can never see more complete regions than its own history,
        # nor more than the summary pool holds.
        need = -(-self.max_model_len // region)
        self.graph_regions = min(GRAPH_REGIONS_OVERRIDE or need, need, pool_rows)

        nr = self.max_num_reqs
        i64 = dict(dtype=torch.int64, device=device)
        verify_q = self.num_speculative_tokens + 1
        self._buf = {
            "head": torch.zeros(nr, **i64),
            "tail": torch.zeros(nr, **i64),
            "slots": torch.zeros(nr, **i64),
            "pending_in": torch.zeros(nr, **i64),
            "pending_out": torch.zeros(nr, **i64),
            "nscore": torch.zeros(nr, dtype=torch.int32, device=device),
            "counts": torch.zeros(nr, dtype=torch.int32, device=device),
            "starts": torch.arange(nr, dtype=torch.int32, device=device) * region,
            "live": torch.zeros(nr, dtype=torch.bool, device=device),
            "verification_head": torch.zeros(nr, **i64),
            "verification_pending_in": torch.zeros(nr, **i64),
            "verification_starts": (
                torch.arange(2 * nr, dtype=torch.int32, device=device) * region
            ),
            "verification_counts": torch.zeros(
                2 * nr, dtype=torch.int32, device=device
            ),
            "verification_slots": torch.zeros(2 * nr, **i64),
            "verification_live": torch.zeros(
                2 * nr, dtype=torch.bool, device=device
            ),
            "verification_pending_rows": torch.zeros(2 * nr, **i64),
            "verification_prefix_lengths": torch.zeros(
                2 * nr, dtype=torch.int32, device=device
            ),
            "verification_request_indices": torch.arange(
                nr, dtype=torch.int32, device=device
            ).repeat_interleave(max(verify_q, 1)),
        }
        # Staged on pinned host memory so the per-step upload of the pending
        # rows is one async copy rather than two torch.tensor(list) round trips.
        #
        # A *ring* of them, because the copy is async: stream ordering protects
        # the device buffer, not this one. With a single buffer per vector, step
        # N+1's host write can overwrite the bytes step N's copy has not read
        # yet -- a write-after-read hazard that shows up as a request picking up
        # another request's pending row, and the best candidate for the
        # non-determinism the copy below used to be made blocking to avoid. The
        # events make reuse wait for the copy that last read the slot, so the
        # depth is a bound rather than a bet on how far ahead the host runs.
        pin = torch.cuda.is_available()
        host_lengths = {
            "pending_in": nr,
            "pending_out": nr,
            "verification_pending_in": nr,
            "verification_pending_rows": 2 * nr,
        }
        self._host = {
            name: [
                torch.zeros(length, dtype=torch.int64, pin_memory=pin)
                for _ in range(_STAGING_DEPTH)
            ]
            for name, length in host_lengths.items()
        }
        self._stage = 0
        self._stage_ev = (
            [torch.cuda.Event() for _ in range(_STAGING_DEPTH)] if pin else []
        )
        for ev in self._stage_ev:
            # Recorded once so the first _STAGING_DEPTH steps can wait on them
            # uniformly instead of branching on "has this slot been used yet".
            ev.record()
        self._decode_logits = torch.empty(
            (self.num_kv_groups * nr, self.graph_regions),
            dtype=torch.float32,
            device=device,
        )
        self._verification_logits = torch.empty(
            (self.num_kv_groups * nr * max(verify_q, 1), self.graph_regions),
            dtype=torch.float32,
            device=device,
        )
        mib = (
            self._decode_logits.numel() + self._verification_logits.numel()
        ) * 4 / 2 ** 20
        logger.info(
            "step4 DSA: decode graph width %d regions (max_model_len %d, "
            "region %d), %d x %d fp32 score pool = %.1f MiB. The region selector "
            "scans the full width every step, so lowering --max-model-len (or "
            "VLLM_STEP4_DSA_GRAPH_REGIONS) cuts decode selection cost "
            "proportionally.",
            self.graph_regions,
            self.max_model_len,
            region,
            self._decode_logits.shape[0],
            self.graph_regions,
            mib,
        )
        self._warm_graph_kernels(device)

    def _warm_graph_kernels(self, device: torch.device) -> None:
        """Compile the width-dependent decode kernels before capture can reach them.

        Triton compiles on first launch, and a compile inside ``torch.cuda.graph``
        loads a module on the capturing stream -- which is how this model's
        trace bubbles were traced to ``hipModuleLoadDataEx`` in the first place.

        vLLM's own capture warmup does not cover this. The warmup and the capture
        share a batch size, so everything specialized on ``num_decodes`` is
        already compiled by the time capture runs; but the *first* warmup step is
        also the step that allocates this pool, so it necessarily ran at the old
        dynamic width. That width picks a different ``BLOCK_R`` in the region
        selector (``min(1024, next_pow2(seq_regions))``), which is a constexpr,
        so the graph-width variant would compile for the first time under
        capture. One throwaway launch at the real width settles it.
        """
        region = self.geometry.region_size
        proxy = self.geometry.proxy_dim
        groups, hpg = self.num_kv_groups, self.heads_per_group
        pages = -(-self.graph_regions // self.regions_per_page)
        # Zero seq lens: every row is masked out and the kernels return without
        # touching the pool. Compilation is the whole point, not the result.
        seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
        table = torch.zeros((1, pages), dtype=torch.int32, device=device)
        logits = self._decode_logits[:groups, : self.graph_regions]
        decode_indexer_logits(
            torch.zeros((1, groups, hpg, proxy), dtype=self.dtype, device=device),
            torch.zeros((1, groups, hpg), dtype=torch.float32, device=device),
            self.summary[0],
            table,
            torch.zeros(1, dtype=torch.int32, device=device),
            logits,
            regions_per_page=self.regions_per_page,
        )
        decode_sparse_meta(
            logits,
            seq_lens,
            table,
            None,
            topk=self.geometry.topk,
            region_size=region,
            regions_per_page=self.regions_per_page,
        )

        if os.environ.get("VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512", "0") == "1":
            from .step5_dsa_lightop_selector import (
                decode_sparse_meta_lightop, supports_native_decode_selector,
            )
            if supports_native_decode_selector(
                logits, seq_lens, topk=self.geometry.topk,
            ):
                decode_sparse_meta_lightop(
                    logits, seq_lens, table, topk=self.geometry.topk,
                    region_size=region, regions_per_page=self.regions_per_page,
                )

        if os.environ.get("VLLM_STEP5_BATCHED_SHORT_INDEXER", "0") == "1":
            # MTP3 target verification reaches the four-query scorer only after
            # serving begins.  Its live token count changes as requests finish;
            # compile one non-specialized variant here so neither the first
            # target step nor a later batch contraction performs Triton JIT.
            num_tokens = 4
            short_prefill_indexer_logits(
                torch.zeros(
                    (num_tokens, groups, hpg, proxy),
                    dtype=self.dtype,
                    device=device,
                ),
                torch.zeros(
                    (num_tokens, groups, hpg),
                    dtype=torch.float32,
                    device=device,
                ),
                self.summary[0],
                table,
                torch.zeros(num_tokens, dtype=torch.int32, device=device),
                torch.zeros(num_tokens, dtype=torch.int32, device=device),
                self._decode_logits[: groups * num_tokens, : self.graph_regions],
                regions_per_page=self.regions_per_page,
                queries_per_request=(
                    4
                    if os.environ.get(
                        "VLLM_STEP5_GROUPED_MTP_INDEXER", "0"
                    ) == "1"
                    else 1
                ),
            )

    def rope_tables(
        self,
        *,
        theta: float,
        scaling: dict | None,
        max_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer cos/sin, shared between layers with identical rope geometry.

        Built with the released ``build_rope_cache`` rather than vLLM's
        ``get_rope`` so that the table is bit-identical to the reference's: the
        two agree on the maths but not on the intermediate precision, and the
        difference shows up in bf16 table entries from position 24 onward.
        """
        key = (float(theta), repr(scaling), int(max_position), str(device), dtype)
        cached = self._rope.get(key)
        if cached is None:
            cached = build_rope_cache(
                rotary_span=self.geometry.rope_span,
                theta=theta,
                max_position=max_position,
                device=device,
                dtype=dtype,
                scaling=scaling,
            )
            self._rope[key] = cached
        return cached

    # -- per-step schedule ------------------------------------------------

    def schedule(self, md: Step4DSAMetadata) -> Step4DSASchedule:
        """Build (or reuse) this step's compression schedule.

        Keyed on the metadata object, which vLLM rebuilds once per forward pass
        and hands unchanged to all 23 layers. The first layer pays for it; the
        other 22 get it for free, including the pending-row assignments, which
        must not be made twice.
        """
        if self._schedule_key is md:
            assert self._schedule is not None
            return self._schedule

        if md.schedule is not None:
            # Built by the metadata builder, before the model ran. Adopt it as
            # this step's schedule so the 23 layers share it exactly as they
            # would a locally built one.
            self._schedule_key = md
            self._schedule = md.schedule
            self._layers_done = 0
            return md.schedule

        if md.num_prefills == 0 and md.num_decodes > 0:
            sched = self._decode_schedule(md)
        else:
            sched = self._general_schedule(md)
        self._schedule_key = md
        self._schedule = sched
        self._layers_done = 0
        return sched

    def _general_schedule(self, md: Step4DSAMetadata) -> Step4DSASchedule:
        """The chunk-shaped schedule: any mix of prefill and decode rows.

        Loops in Python because a prefill row's region count, buffer offset and
        pending rows all depend on its own query length. ``_decode_schedule``
        handles the case where they cannot.
        """
        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        starts_cpu = md.query_start_loc_cpu.tolist()
        seq_lens_cpu = md.seq_lens_cpu.tolist()
        num_reqs = md.num_decodes + md.num_prefills
        # One host copy per step instead of two device reads per request: the
        # slot arithmetic below is host-side control flow either way. Prefer the
        # block table's host mirror, which is the copy vLLM uploaded from and so
        # costs no synchronisation at all; the device read stays as the fallback
        # for the cases that have no mirror (see Step4DSAMetadata.block_table_np).
        table_np = md.block_table_np
        if table_np is not None and table_np.shape[0] >= num_reqs:
            table_cpu = table_np
        else:
            table_cpu = md.block_table[:num_reqs].to("cpu", non_blocking=False).tolist()

        # Release the previous step's closed regions before claiming this
        # step's. finish_layer() normally does it, but a step that was a graph
        # replay never reached finish_layer, so the release has to be anchored
        # here -- at the one point every step passes through. end_step() is
        # idempotent, so doing it twice costs nothing.
        self.end_step()
        self.pending.begin_step()
        sched = Step4DSASchedule()
        starts: list[int] = []
        counts: list[int] = []
        slots: list[int] = []
        buf_off = 0

        for row in range(num_reqs):
            tok_start = starts_cpu[row]
            qlen = starts_cpu[row + 1] - tok_start
            if qlen <= 0:
                continue
            past = seq_lens_cpu[row] - qlen
            head = past % region
            first_region = past // region
            nbuf = head + qlen
            nregions = (nbuf + region - 1) // region
            pages = table_cpu[row]

            def slot_of(region_id: int, _pages=pages) -> int:
                return int(_pages[region_id // rpp]) * rpp + region_id % rpp

            for j in range(nregions):
                filled = min(region, nbuf - j * region)
                starts.append(buf_off + j * region)
                # A count below region_size marks the ragged tail; zero makes
                # the compression kernel skip the region entirely. Storing a
                # partial summary is what pending exists to avoid.
                counts.append(region if filled == region else 0)
                slots.append(slot_of(first_region + j))

            first_slot = slots[-nregions]
            last_slot = slots[-1]
            # The head tokens were buffered by a previous step; that buffer is
            # only free once the region it belongs to has actually closed.
            pending_in = self.pending.get(first_slot) if head else -1
            if head and nbuf >= region and not self.num_speculative_tokens:
                sched.completed_slots.append(first_slot)

            tail = nbuf % region
            pending_out = self.pending.acquire(last_slot) if tail else -1
            if self.num_speculative_tokens:
                # Retain every region the next verified prefix can reopen.
                # At least one token advances; up to all draft tokens reject.
                earliest = max(0, (head + qlen - self.num_speculative_tokens) // region)
                for j in range(earliest, nregions):
                    prefix_len = min(region, nbuf - j * region)
                    pending_row = self.pending.acquire(slot_of(first_region + j))
                    sched.rollback_prefixes.append(
                        (pending_row, buf_off + j * region, prefix_len)
                    )

            sched.requests.append(
                _RequestPlan(
                    row=row,
                    past=past,
                    qlen=qlen,
                    tok_start=tok_start,
                    head=head,
                    buf_start=buf_off,
                    pending_row_in=pending_in,
                    pending_row_out=pending_out,
                    tail=tail,
                    nscore=(past + qlen - 1) // region,
                    score_slots=None,
                )
            )
            buf_off += nbuf

        # Summary slots the scorer gathers, in logical region order. Built once
        # per step rather than once per layer; the gather itself is per layer.
        for plan in sched.requests:
            if plan.nscore <= 0:
                continue
            ids = torch.arange(plan.nscore, device=device, dtype=torch.int64)
            plan.score_slots = (
                md.block_table[plan.row, ids // rpp].to(torch.int64) * rpp + ids % rpp
            )

        sched.buf_len = buf_off
        if starts:
            sched.starts = torch.tensor(starts, dtype=torch.int32, device=device)
            sched.counts = torch.tensor(counts, dtype=torch.int32, device=device)
            sched.slots = torch.tensor(slots, dtype=torch.int64, device=device)
            sched.live = sched.counts > 0

        return sched

    def _decode_slots_host(
        self, md: Step4DSAMetadata, nd: int, seq_np: np.ndarray, slots: torch.Tensor,
    ) -> np.ndarray:
        """This step's physical summary slot per decode row, without a sync.

        The same arithmetic that produced the device ``slots`` tensor, run on
        the block table's host mirror. This is not a shortcut with a different
        answer: ``BlockTable`` maintains the table on the host and uploads it,
        so the mirror is the source and the device tensor is the copy.

        Falls back to reading ``slots`` back when there is no mirror, or when it
        is too short for this batch, so nothing here depends on the mirror
        existing -- it is a fast path, not a new requirement.
        """
        table = md.block_table_np
        if table is None or table.shape[0] < nd:
            return np.asarray(slots.tolist(), dtype=np.int64)

        region = self.geometry.region_size
        rpp = self.regions_per_page
        past = np.maximum(seq_np - 1, 0)
        first_region = past // region
        page = table[np.arange(nd), first_region // rpp].astype(np.int64, copy=False)
        # Block 0 is NULL_BLOCK_ID and is never allocated to a real request, so
        # a zero page is padding however the row got here. Rows past num_reqs
        # hold a stale page rather than NULL_BLOCK_ID -- the runner's padding
        # fill is applied to the device tensor only -- which is why seq_len is
        # part of the mask and not a redundant check.
        real = (seq_np > 0) & (page > 0)
        out = np.where(real, page * rpp + first_region % rpp, -1)

        if CHECK_SLOTS:
            device_slots = np.asarray(slots.tolist(), dtype=np.int64)
            if not np.array_equal(device_slots, out):
                bad = int(np.flatnonzero(device_slots != out)[0])
                raise RuntimeError(
                    "Step-4 DSA: host slot derivation disagrees with the block "
                    f"table on device at decode row {bad}: host "
                    f"{int(out[bad])} vs device {int(device_slots[bad])} "
                    f"(seq_len {int(seq_np[bad])}). The host mirror is supposed "
                    "to be the copy vLLM uploaded from; a mismatch means it was "
                    "reindexed or committed out of order."
                )
        return out

    def _verification_schedule(
        self, md: Step4DSAMetadata, *, persistent: bool = False
    ) -> Step4DSASchedule:
        """Fixed two-region schedule for uniform MTP target verification.

        MTP3 contributes four consecutive queries per request. With an
        eight-token compression region, ``head + 4`` spans at most two regions.
        Two records per request therefore cover every completion and every
        rollback prefix with graph-constant shapes. The metadata builder fills
        these persistent tensors before each target graph replay.
        """
        if md.num_decodes:
            raise RuntimeError("MTP verification schedule requires a uniform batch")
        queries = self.num_speculative_tokens + 1
        if queries <= 1 or queries > self.geometry.region_size:
            raise RuntimeError(
                "Step5 verification schedule expects 2..region_size queries, "
                f"got {queries}"
            )
        num_requests = md.num_prefills
        expected_tokens = num_requests * queries
        if md.num_actual_tokens != expected_tokens:
            raise RuntimeError(
                "Step5 full target graph requires a padded uniform verification "
                f"batch ({num_requests} requests x {queries} queries), got "
                f"{md.num_actual_tokens} tokens"
            )
        if persistent and num_requests > self.max_num_reqs:
            raise RuntimeError(
                f"Step5 verification batch {num_requests} exceeds max_num_reqs "
                f"{self.max_num_reqs}"
            )

        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        buf = self._buf if persistent else {}
        qlens = (
            md.query_start_loc[1 : num_requests + 1]
            - md.query_start_loc[:num_requests]
        ).to(torch.int64)
        seq_lens = md.seq_lens[:num_requests].to(torch.int64)
        real = (qlens == queries) & (seq_lens > 0)
        past = (seq_lens - qlens).clamp_min(0)
        head = past % region
        first_region = torch.div(past, region, rounding_mode="floor")
        region_ids = first_region[:, None] + torch.arange(
            2, dtype=torch.int64, device=device
        )[None, :]
        page_columns = torch.div(region_ids, rpp, rounding_mode="floor").clamp(
            max=md.block_table.shape[1] - 1
        )
        pages = torch.gather(
            md.block_table[:num_requests].to(torch.int64), 1, page_columns
        )
        real = real & (pages[:, 0] > 0)
        used = torch.stack(
            (real, real & (head + queries > region)), dim=1
        )
        slots = torch.where(
            used,
            pages * rpp + region_ids % rpp,
            -torch.ones_like(region_ids),
        )
        closes_first = real & (head + queries >= region)
        live = torch.stack((closes_first, torch.zeros_like(closes_first)), dim=1)
        counts = torch.where(
            live,
            torch.full_like(region_ids, region),
            torch.zeros_like(region_ids),
        ).to(torch.int32)
        head = torch.where(real, head, torch.zeros_like(head))
        nscore = torch.where(
            real,
            torch.div(past + queries - 1, region, rounding_mode="floor"),
            torch.zeros_like(past),
        ).to(torch.int32)

        # Every region that can be reopened after rejecting up to three draft
        # tokens retains its prefix. For MTP3, the first region can be dropped
        # only at head==7; the second exists only at head>=5.
        nbuf = head + queries
        earliest = torch.div(
            (nbuf - self.num_speculative_tokens).clamp_min(0),
            region,
            rounding_mode="floor",
        )
        prefix_first = torch.where(
            real & (earliest == 0),
            torch.minimum(nbuf, torch.full_like(nbuf, region)),
            torch.zeros_like(nbuf),
        )
        prefix_second = torch.where(
            used[:, 1] & (earliest <= 1),
            nbuf - region,
            torch.zeros_like(nbuf),
        )
        prefix_lengths = torch.stack((prefix_first, prefix_second), dim=1).to(
            torch.int32
        )

        sched = Step4DSASchedule(
            verification=True,
            queries_per_request=queries,
            num_requests=num_requests,
            buf_len=num_requests * 2 * region,
        )

        def out(name: str, value: torch.Tensor) -> torch.Tensor:
            dst = buf.get(name)
            if dst is None:
                return value
            view = dst[: value.numel()].view(value.shape)
            view.copy_(value)
            return view

        sched.starts = (
            buf["verification_starts"][: 2 * num_requests]
            if buf
            else torch.arange(
                2 * num_requests, dtype=torch.int32, device=device
            ) * region
        )
        sched.counts = out("verification_counts", counts.flatten())
        sched.slots = out("verification_slots", slots.flatten())
        sched.live = out("verification_live", live.flatten())
        sched.head = out("verification_head", head)
        sched.nscore = nscore
        sched.prefix_lengths = out(
            "verification_prefix_lengths", prefix_lengths.flatten()
        )
        if buf:
            sched.request_indices = buf["verification_request_indices"][:expected_tokens]
        else:
            sched.request_indices = torch.arange(
                num_requests, dtype=torch.int32, device=device
            ).repeat_interleave(queries)

        qlens_np = np.diff(
            md.query_start_loc_cpu[: num_requests + 1].numpy()
        ).astype(np.int64, copy=False)
        seq_np = md.seq_lens_cpu[:num_requests].numpy().astype(np.int64, copy=False)
        past_np = np.maximum(seq_np - qlens_np, 0)
        head_np = past_np % region
        first_np = past_np // region
        table_np = md.block_table_np
        if table_np is not None and table_np.shape[0] >= num_requests:
            region_np = first_np[:, None] + np.arange(2, dtype=np.int64)[None, :]
            columns_np = np.minimum(region_np // rpp, table_np.shape[1] - 1)
            pages_np = table_np[np.arange(num_requests)[:, None], columns_np].astype(
                np.int64, copy=False
            )
            used_np = np.stack(
                (
                    (qlens_np == queries) & (seq_np > 0) & (pages_np[:, 0] > 0),
                    (qlens_np == queries)
                    & (seq_np > 0)
                    & (pages_np[:, 0] > 0)
                    & (head_np + queries > region),
                ),
                axis=1,
            )
            slots_np = np.where(
                used_np, pages_np * rpp + region_np % rpp, -1
            )
        else:
            slots_np = np.asarray(slots.tolist(), dtype=np.int64)

        self.end_step()
        self.pending.begin_step()
        trash = self.pending_trash_row
        rows_in = np.full(num_requests, trash, dtype=np.int64)
        pending_rows = np.full((num_requests, 2), trash, dtype=np.int64)
        widest = 0
        for row in range(num_requests):
            if qlens_np[row] != queries or slots_np[row, 0] < 0:
                continue
            head_row = int(head_np[row])
            if head_row:
                rows_in[row] = self.pending.get(int(slots_np[row, 0]))
            nbuf_row = head_row + queries
            nregions = 1 + int(nbuf_row > region)
            earliest_row = max(
                0, (nbuf_row - self.num_speculative_tokens) // region
            )
            for region_index in range(earliest_row, nregions):
                pending_rows[row, region_index] = self.pending.acquire(
                    int(slots_np[row, region_index])
                )
            widest = max(widest, int((past_np[row] + queries - 1) // region))

        if buf:
            stage = self._stage
            self._stage = (stage + 1) % _STAGING_DEPTH
            ev = self._stage_ev[stage] if self._stage_ev else None
            if ev is not None:
                ev.synchronize()
            host_in = self._host["verification_pending_in"][stage]
            host_rows = self._host["verification_pending_rows"][stage]
            host_in.numpy()[:num_requests] = rows_in
            host_rows.numpy()[: 2 * num_requests] = pending_rows.reshape(-1)
            sched.pending_in = buf["verification_pending_in"][:num_requests]
            sched.pending_rows = buf["verification_pending_rows"][: 2 * num_requests]
            sched.pending_in.copy_(host_in[:num_requests], non_blocking=ev is not None)
            sched.pending_rows.copy_(
                host_rows[: 2 * num_requests], non_blocking=ev is not None
            )
            if ev is not None:
                ev.record()
        else:
            sched.pending_in = torch.tensor(rows_in, dtype=torch.int64, device=device)
            sched.pending_rows = torch.tensor(
                pending_rows.reshape(-1), dtype=torch.int64, device=device
            )

        sched.max_nscore = widest
        if persistent:
            if widest > self.graph_regions:
                raise RuntimeError(
                    f"Step5 verification needs {widest} region scores but graph "
                    f"width is {self.graph_regions}"
                )
            sched.graph_regions = self.graph_regions
        return sched

    def _decode_schedule(
        self, md: Step4DSAMetadata, *, persistent: bool = False
    ) -> Step4DSASchedule:
        """Fixed-shape schedule for a decode-only step.

        Every request contributes exactly one token, so it touches exactly one
        region and ``nregions == 1`` unconditionally. That removes the two things
        the general path loops for: the region count per request, and the packed
        buffer offset. Requests get a fixed stride of ``region_size`` in the
        compression buffer instead of being packed end to end -- the compression
        kernel addresses regions through ``starts``/``counts``, so the padding
        between them is never read, and in exchange the layout stops depending on
        the batch's sequence lengths.

        A region only becomes live when this token fills its last slot, i.e.
        ``(seq_len - 1) % region_size == region_size - 1``; otherwise ``counts``
        is zero and the compression kernel skips it, exactly as before.

        The physical summary slot per request comes from the block table, and
        slot-keying is what lets a partial region follow its request across
        batch reordering, so it has to be known on the host. It is *not* read
        back from the device: the block table is built on the host and uploaded,
        so ``md.block_table_np`` is the same data, and the slot arithmetic is
        repeated there. Reading it from the device tensor instead used to be the
        one blocking sync left in a decode step, which drained everything queued
        ahead of it and capped run-ahead at a single step. The device read
        remains as the fallback when no mirror is available.

        **Padded rows.** In a CUDA-graph batch the trailing rows are not
        requests: vLLM gives them ``seq_len = 0`` and a block table of
        ``NULL_BLOCK_ID`` (0, permanently reserved). Naively, ``past = -1`` makes
        ``first_region = -1`` and the gather indexes out of bounds. Every row is
        therefore masked by ``real = (seq_len > 0) & (page > 0)`` and a dead row
        is driven to a state that is inert everywhere downstream: ``counts = 0``
        so compression skips it, ``live = False`` so no summary is stored,
        ``nscore = 0`` so it scores nothing, pending routed to the trash row, and
        no entry in the host loop at all. That one mask covers dummy runs,
        capture, and replay padding alike, which is why
        ``build_for_cudagraph_capture`` needs no override.

        With ``persistent``, the results are written into the state's
        pre-allocated ``[max_num_reqs]`` buffers instead of freshly allocated
        ones, so their addresses survive from capture to replay.
        """
        region = self.geometry.region_size
        rpp = self.regions_per_page
        device = md.block_table.device
        nd = md.num_decodes
        buf = self._buf if persistent else {}
        if persistent and nd > self.max_num_reqs:
            raise RuntimeError(
                f"Step-4 DSA: decode batch of {nd} exceeds max_num_reqs "
                f"{self.max_num_reqs}; the persistent schedule buffers are sized "
                f"from the scheduler's own limit, so this should be unreachable."
            )
        if persistent:
            # The score width is also bounded by what the block table can
            # address: both the scorer and the region selector reach the summary
            # pool through it, so a column past its last page has no page to
            # resolve. vLLM sizes the table from max_model_len, the same figure
            # graph_regions comes from, so in a server this is a no-op -- but the
            # table is the authority, and it is only visible from here.
            cap = md.block_table.shape[1] * rpp
            if cap < self.graph_regions:
                logger.info(
                    "step4 DSA: decode graph width trimmed %d -> %d regions, the "
                    "limit of a %d-page block table.",
                    self.graph_regions,
                    cap,
                    md.block_table.shape[1],
                )
                self.graph_regions = cap

        seq_lens = md.seq_lens[:nd].to(torch.int64)
        past = (seq_lens - 1).clamp_min(0)
        head = past % region
        first_region = torch.div(past, region, rounding_mode="floor")
        page = torch.gather(
            md.block_table[:nd].to(torch.int64),
            1,
            torch.div(first_region, rpp, rounding_mode="floor").unsqueeze(1),
        ).squeeze(1)
        # Block 0 is NULL_BLOCK_ID and is never allocated to a real request, so
        # a zero page is padding however the row got here.
        real = (seq_lens > 0) & (page > 0)
        zero = torch.zeros_like(head)

        slots = torch.where(
            real, page * rpp + first_region % rpp, -torch.ones_like(head)
        )
        closes = real & (head == region - 1)
        head = torch.where(real, head, zero)

        sched = Step4DSASchedule()
        sched.decode = True
        sched.num_decodes = nd
        sched.buf_len = nd * region

        def out(name: str, value: torch.Tensor) -> torch.Tensor:
            """Land ``value`` in the persistent buffer, or pass it through."""
            dst = buf.get(name)
            if dst is None:
                return value
            view = dst[:nd]
            view.copy_(value)
            return view

        # `starts` is the same arange every step, so the persistent copy is
        # prebuilt in _allocate_graph_buffers and only sliced here.
        sched.starts = (
            buf["starts"][:nd]
            if buf
            else torch.arange(nd, device=device, dtype=torch.int32) * region
        )
        sched.counts = out(
            "counts",
            torch.where(closes, torch.full_like(head, region), zero).to(torch.int32),
        )
        sched.slots = out("slots", slots)
        sched.live = out("live", closes)
        sched.head = out("head", head)
        sched.tail = out("tail", torch.where(closes, zero, head + real.to(head.dtype)))
        sched.nscore = out(
            "nscore", torch.where(real, first_region, zero).to(torch.int32)
        )

        # Host-side pending bookkeeping. Both inputs are already on the host:
        # the lengths from vLLM's CPU mirror, the slots derived from the block
        # table's, so the step no longer synchronises here.
        seq_np = md.seq_lens_cpu[:nd].numpy().astype(np.int64, copy=False)
        slots_np = self._decode_slots_host(md, nd, seq_np, slots)
        past_np = np.maximum(seq_np - 1, 0)
        head_np = past_np % region
        first_np = past_np // region
        trash = self.pending_trash_row
        self.end_step()  # see the note in _general_schedule
        self.pending.begin_step()
        rows_in: list[int] = [trash] * nd
        rows_out: list[int] = [trash] * nd
        widest = 0
        for row in range(nd):
            slot = int(slots_np[row])
            if slot < 0:  # padding, not a request
                continue
            head_row = int(head_np[row])
            rows_in[row] = self.pending.get(slot) if head_row else trash
            if head_row == region - 1 and not self.num_speculative_tokens:
                # The region closed this step; its tokens are now in a summary
                # and the pending row is released once every layer has read it.
                sched.completed_slots.append(slot)
            else:
                rows_out[row] = self.pending.acquire(slot)
            widest = max(widest, int(first_np[row]))
        if buf:
            # Staged through pinned memory rather than torch.tensor(list), whose
            # pageable copy is much slower.
            #
            # These two vectors pick the pending row each region's head is read
            # from and its tail written to, and they land in buffers the
            # captured decode graph reads. This copy was blocking for a long
            # time, because with non_blocking=True the run was not reproducible:
            # two identical greedy runs at four concurrent decodes diverged
            # within a few tokens, while single-request decode stayed clean and
            # the piecewise path was bit-exact. That is the signature of a
            # write-after-read hazard on the *host* staging buffer rather than a
            # missing stream order on the device: the copy is issued on the
            # stream that later replays the graph, so the device side was always
            # ordered, but with one buffer per vector the next step's host write
            # could reach it before the previous step's copy had read it out --
            # and a single decode never noticed, because its pending row does
            # not change from step to step.
            #
            # So the copy is async again, against a ring of staging buffers with
            # an event per slot: reuse waits for the copy that last read it.
            # Blocking here is no longer free either, now that the slot
            # derivation above does not already synchronise. The hazard is
            # closed by construction, but it was not re-bisected against the
            # original divergence -- if non-determinism returns at concurrency,
            # set _STAGING_DEPTH to 1 to put the old configuration back and
            # confirm before looking elsewhere.
            stage = self._stage
            self._stage = (stage + 1) % _STAGING_DEPTH
            ev = self._stage_ev[stage] if self._stage_ev else None
            if ev is not None:
                ev.synchronize()
            h_in = self._host["pending_in"][stage]
            h_out = self._host["pending_out"][stage]
            h_in.numpy()[:nd] = rows_in
            h_out.numpy()[:nd] = rows_out
            sched.pending_in = buf["pending_in"][:nd]
            sched.pending_out = buf["pending_out"][:nd]
            sched.pending_in.copy_(h_in[:nd], non_blocking=ev is not None)
            sched.pending_out.copy_(h_out[:nd], non_blocking=ev is not None)
            if ev is not None:
                ev.record()
        else:
            sched.pending_in = torch.tensor(rows_in, dtype=torch.int64, device=device)
            sched.pending_out = torch.tensor(rows_out, dtype=torch.int64, device=device)

        # Widest history in the batch. The scorer reads the summary pool in
        # place through the block table, so this only sizes the logit tensor;
        # columns past a request's own ``nscore`` are masked by the scorer and
        # discarded by the selector's ``history_lengths``.
        sched.max_nscore = widest
        if persistent:
            if widest > self.graph_regions:
                raise RuntimeError(
                    f"Step-4 DSA: a request needs {widest} region scores but the "
                    f"captured decode width is {self.graph_regions}. Raise "
                    f"VLLM_STEP4_DSA_GRAPH_REGIONS or --max-model-len; the width "
                    f"is a graph constant and cannot be grown per step."
                )
            sched.graph_regions = self.graph_regions
        return sched

    def finish_layer(self) -> None:
        """Called by each DSA layer once it has consumed this step's schedule."""
        self._layers_done += 1
        if self._layers_done >= len(self._registered) and not (
            self._schedule is not None and self._schedule.builder_owned
        ):
            # A builder-owned schedule is released at the *start* of the next
            # one instead. Releasing here would be wrong the moment a step is a
            # graph replay, because then no layer calls finish_layer at all and
            # the counter this branch reads is whatever the capture left behind.
            self.end_step()

    def end_step(self) -> None:
        """Release pending rows for regions that closed."""
        if self._schedule is None:
            return
        for slot in self._schedule.completed_slots:
            self.pending.release(slot)
        self._schedule.completed_slots.clear()


# --------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------


class Step4IndexerNorm(nn.Module):
    """Parameter holder for one indexer norm; the maths happens in the kernel.

    ``indexer_norm_rope`` fuses normalisation, the affine, and the partial
    rotation into a single pass, so nothing here has a ``forward``. It exists
    because the checkpoint stores these as ``...sparse_indexer_q_norm.weight``
    and ``...sparse_indexer_k_norm.{weight,bias}``, and a module is the only way
    to make ``named_parameters()`` produce those exact names.

    fp32, like every other norm parameter in this checkpoint. The q-norm is
    applied as ``x * (1 + w)``, so rounding a zero-centred ``w`` to bf16 would
    perturb the multiplier by far more than bf16 compute noise -- the same
    reason ``_fp32_gemma_rms_norm`` exists on the attention side.
    """

    def __init__(self, dim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))


def _group_slice_loader(first_group: int, num_groups: int, total_groups: int):
    """Weight loader for a tensor sliced along dim 0 by *provider group*.

    ``sparse_indexer_q`` and ``sparse_indexer_w`` are the only indexer tensors
    that shard, and they shard by provider group rather than by rank: at TP8
    with 4 groups, ranks 2k and 2k+1 both hold group k's rows, which is the same
    replica convention ``QKVParallelLinear`` uses for Step-4's 4 KV heads. This
    mirrors ``inference/convert.py``'s ``_slice_grouped_dim``.
    """

    def loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        chunk = loaded_weight.shape[0] // total_groups
        if chunk * total_groups != loaded_weight.shape[0]:
            raise ValueError(
                f"indexer tensor of {loaded_weight.shape[0]} rows does not split "
                f"into {total_groups} provider groups"
            )
        shard = loaded_weight[first_group * chunk : (first_group + num_groups) * chunk]
        if tuple(shard.shape) != tuple(param.data.shape):
            raise ValueError(
                f"indexer shard {tuple(shard.shape)} does not fit parameter "
                f"{tuple(param.data.shape)}"
            )
        param.data.copy_(shard)

    return loader


class Step4Indexer:
    """Scores every completed region of the context and picks the top ``topk``.

    Deliberately not an ``nn.Module``. Its parameters live directly on the
    owning ``Step4Attention`` because that is where the checkpoint puts them
    (``model.layers.N.self_attn.sparse_indexer_q.weight``); wrapping them in a
    submodule would rename every one of them and force a mapping table. The
    reference implementation makes the same call for the same reason.

    It also has to run model-side rather than inside the attention backend: it
    consumes ``hidden_states``, and ``Attention.forward`` only ever sees q/k/v.
    The result is handed to the backend through ``layer.step4_sparse_plan``.

    Per step, in this order -- the order matters, because a region that fills
    with *this* chunk's tokens must be scorable by later tokens of the same
    chunk:

    1. project and normalise/rotate the proxy q/k/z,
    2. fold the pending tail of the previous step in, compress every region
       that is now complete, buffer the new tail,
    3. score each request's queries against its own summaries,
    4. select and pack, and publish the plan.
    """

    def __init__(
        self,
        owner: nn.Module,
        *,
        geometry: Step4DSAGeometry,
        state: Step4DSAState,
        dsa_slot: int,
        num_kv_groups: int,
        hidden_size: int,
        tp_rank: int,
        tp_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: dict | None,
        max_position: int,
        params_dtype: torch.dtype,
        num_attention_heads: int,
    ) -> None:
        self.owner = owner
        norm_cast_mode = os.environ.get("VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS", "0")
        if norm_cast_mode not in ("0", "1"):
            raise ValueError("VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS must be 0 or 1")
        self.fuse_norm_weight_casts = norm_cast_mode == "1"
        self.geometry = geometry
        self.state = state
        self.slot = dsa_slot
        self.num_kv_groups = num_kv_groups
        # The state sizes its group-major score pool from this and cannot know
        # it on its own. Every DSA layer on a rank holds the same number of KV
        # groups, so the first one to be built settles it.
        if state.num_kv_groups not in (0, num_kv_groups):
            raise ValueError(
                f"Step-4 DSA: layer declares {num_kv_groups} KV groups but the "
                f"shared state was already sized for {state.num_kv_groups}."
            )
        state.num_kv_groups = num_kv_groups
        self.eps = rms_norm_eps
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.max_position = int(max_position)

        groups_total = geometry.num_provider_groups
        heads_per_group = geometry.num_indexer_heads // groups_total
        if heads_per_group * groups_total != geometry.num_indexer_heads:
            raise ValueError(
                f"{geometry.num_indexer_heads} indexer heads do not divide into "
                f"{groups_total} provider groups"
            )
        if tp_size <= groups_total:
            local_groups = groups_total // tp_size
            first_group = tp_rank * local_groups
        else:
            # More ranks than groups: each group is replicated across
            # tp_size // groups_total ranks, matching QKVParallelLinear's KV
            # replica rule (tp_rank // num_kv_head_replicas).
            local_groups = 1
            first_group = tp_rank // (tp_size // groups_total)
        if local_groups != num_kv_groups:
            raise ValueError(
                f"indexer resolves {local_groups} provider group(s) on this rank "
                f"but attention has {num_kv_groups} KV group(s); the two index the "
                f"same group axis and must agree"
            )
        self.heads_per_group = heads_per_group
        state.heads_per_group = heads_per_group
        self.num_heads = local_groups * heads_per_group
        # The divisor is the heads inside ONE provider group, not every head
        # resident on this rank. At TP8 those differ (4 vs 4 only because a rank
        # holds exactly one group); deriving it from local head count on a
        # different TP degree gives the wrong scale and silently changes which
        # regions get selected.
        self.weight_prescale = float(heads_per_group) ** -0.5

        proxy = geometry.proxy_dim
        owner.sparse_indexer_q = nn.Linear(
            hidden_size, self.num_heads * proxy, bias=False, dtype=params_dtype
        )
        owner.sparse_indexer_k = nn.Linear(
            hidden_size, proxy, bias=False, dtype=params_dtype
        )
        owner.sparse_indexer_z = nn.Linear(
            hidden_size, proxy, bias=False, dtype=params_dtype
        )
        # fp32 in the checkpoint and kept that way; it is rounded to the
        # activation dtype at call time instead, which is the rounding boundary
        # the deployed kernel has.
        owner.sparse_indexer_w = nn.Linear(
            hidden_size, self.num_heads, bias=False, dtype=torch.float32
        )
        owner.sparse_indexer_q_norm = Step4IndexerNorm(proxy, bias=False)
        owner.sparse_indexer_k_norm = Step4IndexerNorm(proxy, bias=True)
        # Present on every head of every layer at its initialisation value and
        # unused by the deployed kernels: the indexer scores with a weighted
        # ReLU, so there is no softmax for a scalable-softmax scale to act on.
        # Registered only so that loading the checkpoint stays exhaustive.
        owner.ssmax_s = nn.Parameter(
            torch.zeros(num_attention_heads, dtype=torch.float32), requires_grad=False,
        )

        sharded = _group_slice_loader(first_group, local_groups, groups_total)
        owner.sparse_indexer_q.weight.weight_loader = sharded
        owner.sparse_indexer_w.weight.weight_loader = sharded

        state.register_layer(dsa_slot)

        # The op is dispatched by name, because a Python object cannot cross a
        # custom-op boundary. Names are unique per layer per process, and vLLM
        # already relies on that for the attention layers themselves.
        self.layer_name = owner.attn.layer_name
        if self.fuse_norm_weight_casts:
            indexer_norm_cast_logger.info_once(
                "Step5 indexer fused norm weight casts enabled",
                scope="process",
            )
        _INDEXERS[self.layer_name] = self
        # Mutated by the op, and the reason the op is not optimised away: see
        # step4_dsa_index. Non-persistent so it stays out of the state dict, but
        # a real buffer so that .cuda() moves it with the rest of the layer.
        owner.register_buffer(
            "step4_dsa_plan_token", torch.zeros(1, dtype=torch.int32), persistent=False,
        )

    # -- forward ----------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                projected_index: torch.Tensor | None = None) -> None:
        """Score and select, via the opaque op so that ``torch.compile`` cannot
        trace into this. See ``step4_dsa_index`` for why that matters."""
        torch.ops.vllm.step4_dsa_index(
            hidden_states,
            positions,
            self.layer_name,
            # Read through the owner, not cached: .cuda() rebinds the buffer.
            self.owner.step4_dsa_plan_token,
            projected_index,
        )

    def run(self, hidden_states: torch.Tensor, positions: torch.Tensor,
            projected_index: torch.Tensor | None = None,
            main_qknorm: tuple | None = None) -> int:
        attn = self.owner.attn
        raw = get_forward_context().attn_metadata
        if isinstance(raw, dict):
            md = raw.get(attn.layer_name)
        elif isinstance(raw, list):
            md = raw[0].get(attn.layer_name)
        else:
            md = raw
        kv_cache = attn.kv_cache
        if not isinstance(md, Step4DSAMetadata) or kv_cache.numel() == 0:
            # Profiling or a dummy run: there is no cache to summarise and the
            # impl short-circuits on the same condition.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{attn.layer_name}: the DSA indexer is being captured into "
                    f"a CUDA graph with no attention metadata (or an unallocated "
                    f"KV cache). The recorded graph would contain no DSA work at "
                    f"all and every replay would attend to nothing, silently."
                )
            attn.step4_sparse_plan = None
            return 0

        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and md.schedule is None:
            # Everything a captured step needs must have been written by
            # Step4DSAMetadataBuilder.build(), into buffers whose addresses the
            # recorded kernels keep pointing at. Without that the schedule below
            # is built here, from tensors allocated inside the capture, and
            # every replay would re-read this one step's values forever. The
            # host round-trip in _general_schedule would fail first -- but with
            # a stream-capture error, not a reason.
            raise RuntimeError(
                f"{attn.layer_name}: CUDA graph capture reached the DSA indexer "
                f"without a builder-supplied schedule "
                f"(num_prefills={md.num_prefills}, num_decodes={md.num_decodes}, "
                f"graph_decode={self.state.graph_decode}). Only decode-only steps "
                f"are capturable; see Step4DSAMetadataBuilder.build."
            )

        if md.positions is None:
            # V2 common metadata omits positions; the model passes the exact
            # position tensor to its indexer before compression and selection.
            md.positions = positions
        state = self.state
        state.ensure_allocated(kv_cache, hidden_states.device)
        sched = state.schedule(md)

        num_tokens = md.num_actual_tokens
        index_q, index_k, index_z, weights = self._project(
            hidden_states[:num_tokens], positions[:num_tokens], num_tokens,
            projected_index[:num_tokens] if projected_index is not None else None,
        )
        self._update_summaries(sched, index_k, index_z)
        prefill_logits, decode_logits, width = self._score(md, sched, index_q, weights)

        topk = max(width, 1) if DENSE_EQUIV else self.geometry.topk
        attn.step4_sparse_plan = build_plan(
            md,
            prefill_logits,
            decode_logits,
            topk=topk,
            num_kv_groups=self.num_kv_groups,
            regions_per_page=state.regions_per_page,
            context_parallel_size=state.context_parallel_size,
            main_qknorm=main_qknorm,
        )
        state.finish_layer()
        return width

    # -- stage 1: projection ----------------------------------------------

    def _project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, num_tokens: int,
        projected_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        owner = self.owner
        proxy = self.geometry.proxy_dim

        if projected_index is None:
            index_q = owner.sparse_indexer_q(hidden_states)
            index_k = owner.sparse_indexer_k(hidden_states)
            index_z = owner.sparse_indexer_z(hidden_states)
            raw_weights = F.linear(
                hidden_states, owner.sparse_indexer_w.weight.to(hidden_states.dtype)
            )
        else:
            sizes = (self.num_heads * proxy, proxy, proxy, self.num_heads)
            index_q, index_k, index_z, raw_weights = projected_index.split(sizes, dim=-1)

        cos, sin = self.state.rope_tables(
            theta=self.rope_theta,
            scaling=self.rope_scaling,
            max_position=self.max_position,
            device=hidden_states.device,
            dtype=index_q.dtype,
        )
        # Q uses RMSNorm and K uses LayerNorm; both then receive partial
        # RoPE. Index Z is returned untouched because it is a per-dimension
        # softmax logit rather than a key.
        fuse_weight_casts = (
            self.fuse_norm_weight_casts and 0 < num_tokens <= 128
            and proxy == 256 and index_q.dtype == torch.bfloat16
            and index_k.dtype == torch.bfloat16
        )
        if fuse_weight_casts:
            # The fused kernel performs the same BF16 parameter rounding.
            q_norm_weight = owner.sparse_indexer_q_norm.weight
            k_norm_weight = owner.sparse_indexer_k_norm.weight
            k_norm_bias = owner.sparse_indexer_k_norm.bias
        else:
            q_norm_weight = owner.sparse_indexer_q_norm.weight.to(dtype=index_q.dtype)
            k_norm_weight = owner.sparse_indexer_k_norm.weight.to(dtype=index_k.dtype)
            k_norm_bias = owner.sparse_indexer_k_norm.bias.to(dtype=index_k.dtype)
        # The one-wave joint Q/K grid removes a launch and remains exact through
        # 128 target-verification tokens. Larger prefill shapes retain the two
        # reference launches because their reduction result starts to diverge.
        if num_tokens <= 128:
            index_q, index_k = fused_indexer_qk_norm_rope(
                index_q,
                index_k,
                q_norm_weight,
                k_norm_weight,
                k_norm_bias,
                cos,
                sin,
                positions,
                head_dim=proxy,
                num_q_heads=self.num_heads,
                rotary_dim=cos.shape[1],
                eps=self.eps,
                round_q_e4m3=short_index_query_rounding_enabled(
                    num_tokens, proxy, index_q.dtype,
                ),
                round_norm_weights=fuse_weight_casts,
            )
        else:
            index_q, index_k, index_z = indexer_norm_rope(
                index_q,
                index_k,
                index_z,
                q_norm_weight,
                k_norm_weight,
                k_norm_bias,
                cos,
                sin,
                positions,
                head_dim=proxy,
                num_q_heads=self.num_heads,
                num_k_heads=1,
                rotary_dim=cos.shape[1],
                eps=self.eps,
                q_norm_weight_bias=1.0,
            )

        # Round the fp32 weight to the activation dtype *before* the GEMM and
        # widen the result back to fp32 afterwards. Both boundaries are part of
        # the deployed semantics: doing the GEMM in fp32 gives different scores.
        weights = raw_weights.float()
        weights = weights.view(num_tokens, self.num_kv_groups, self.heads_per_group)
        weights = (weights * self.weight_prescale).contiguous()

        return (
            index_q.view(num_tokens, self.num_kv_groups, self.heads_per_group, proxy),
            index_k.view(num_tokens, 1, proxy),
            index_z.view(num_tokens, 1, proxy),
            weights,
        )

    # -- stage 2: summaries -------------------------------------------------

    def _update_summaries(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """Compress every region that completed this step; re-buffer the tails.

        The compression buffer is assembled rather than compressed in place
        because a region can span a chunk boundary: its leading tokens were
        produced by an earlier forward pass and only survive in pending. The
        reference gets a contiguous per-request buffer for free; here it has to
        be built, which is the one structural difference between the two.

        The summary is a per-dimension softmax-weighted mean over the region's
        tokens with a shared shift, so it is not incrementally computable
        byte-for-byte -- holding the tokens and compressing once is what keeps
        the streaming result identical to the batched one rather than merely
        close to it.
        """
        if sched.verification:
            self._update_summaries_verification(sched, index_k, index_z)
            return
        if sched.decode:
            self._update_summaries_decode(sched, index_k, index_z)
            return
        if sched.starts is None:
            return
        state = self.state
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]
        proxy = self.geometry.proxy_dim

        buffer_key = index_k.new_zeros((sched.buf_len, 1, proxy))
        buffer_z = index_z.new_zeros((sched.buf_len, 1, proxy))
        for plan in sched.requests:
            base = plan.buf_start
            if plan.head:
                rows = slice(0, plan.head)
                buffer_key[base : base + plan.head] = pending_key[
                    plan.pending_row_in, rows
                ].unsqueeze(1)
                buffer_z[base : base + plan.head] = pending_z[
                    plan.pending_row_in, rows
                ].unsqueeze(1)
            start = base + plan.head
            tokens = slice(plan.tok_start, plan.tok_start + plan.qlen)
            buffer_key[start : start + plan.qlen] = index_k[tokens]
            buffer_z[start : start + plan.qlen] = index_z[tokens]

        _, summary_fp8 = csa_compress_regions(
            buffer_key,
            buffer_z,
            sched.starts,
            sched.counts,
            region_size=self.geometry.region_size,
        )
        _store_live_summary_rows(
            state.summary[self.slot], summary_fp8, sched.slots, sched.live,
        )

        for pending_row, offset, prefix_len in sched.rollback_prefixes:
            pending_key[pending_row, :prefix_len] = buffer_key[
                offset : offset + prefix_len, 0
            ]
            pending_z[pending_row, :prefix_len] = buffer_z[
                offset : offset + prefix_len, 0
            ]

        for plan in sched.requests:
            if not plan.tail:
                continue
            end = plan.buf_start + plan.head + plan.qlen
            pending_key[plan.pending_row_out, : plan.tail] = buffer_key[
                end - plan.tail : end, 0
            ]
            pending_z[plan.pending_row_out, : plan.tail] = buffer_z[
                end - plan.tail : end, 0
            ]

    # -- stage 2b: summaries, graph-captured MTP verification ---------------

    def _update_summaries_verification(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """Compress and retain a fixed two-region MTP verification buffer."""
        state = self.state
        region = self.geometry.region_size
        num_requests = sched.num_requests
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]
        buffer_key, buffer_z = _assemble_verification_tokens(
            pending_key,
            pending_z,
            index_k[: num_requests * sched.queries_per_request],
            index_z[: num_requests * sched.queries_per_request],
            sched.head,
            sched.pending_in,
            sched.prefix_lengths,
            queries_per_request=sched.queries_per_request,
            region_size=region,
        )
        _, summary_fp8 = csa_compress_regions(
            buffer_key.view(num_requests * 2 * region, 1, self.geometry.proxy_dim),
            buffer_z.view(num_requests * 2 * region, 1, self.geometry.proxy_dim),
            sched.starts,
            sched.counts,
            region_size=region,
        )
        _store_live_summary_rows(
            state.summary[self.slot], summary_fp8, sched.slots, sched.live
        )
        _store_verification_prefixes(
            pending_key,
            pending_z,
            buffer_key,
            buffer_z,
            sched.pending_rows,
            sched.prefix_lengths,
            region_size=region,
        )

    # -- stage 2c: summaries, decode-only fast path -------------------------

    def _update_summaries_decode(
        self, sched: Step4DSASchedule, index_k: torch.Tensor, index_z: torch.Tensor
    ) -> None:
        """The same compression, with the request loop replaced by a batch axis.

        Every decode request contributes exactly one token, so the buffer can be
        given a fixed ``region_size`` stride per request instead of the packed
        variable stride the general path computes. Compression addresses regions
        through ``starts``/``counts``, so the padding between them is never read
        and the fixed stride is result-neutral -- but it makes every shape here a
        function of ``num_decodes`` alone, which is what capture needs.

        Requests that do not close a region still write their tail; they are
        routed to ``pending_trash_row`` so the write is unconditional and the
        host never has to branch per request.
        """
        state = self.state
        region = self.geometry.region_size
        proxy = self.geometry.proxy_dim
        num_decodes = sched.num_decodes
        pending_key = state.pending_key[self.slot]
        pending_z = state.pending_z[self.slot]

        positions = torch.arange(region, device=index_k.device)
        # Slots [0, head) come from pending; the rest is padding until the
        # current token is scattered in at `head`.
        head_mask = (positions[None, :] < sched.head[:, None]).unsqueeze(-1)
        zero = pending_key.new_zeros(())
        buffer_key = torch.where(head_mask, pending_key[sched.pending_in], zero)
        buffer_z = torch.where(head_mask, pending_z[sched.pending_in], zero)

        _insert_decode_tokens(
            buffer_key,
            buffer_z,
            index_k[:num_decodes],
            index_z[:num_decodes],
            sched.head,
        )

        _, summary_fp8 = csa_compress_regions(
            buffer_key.view(num_decodes * region, 1, proxy),
            buffer_z.view(num_decodes * region, 1, proxy),
            sched.starts,
            sched.counts,
            region_size=region,
        )
        _store_live_summary_rows(
            state.summary[self.slot], summary_fp8, sched.slots, sched.live,
        )

        retained = sched.head + 1 if state.num_speculative_tokens else sched.tail
        tail_mask = (positions[None, :] < retained[:, None]).unsqueeze(-1)
        rows = sched.pending_out
        pending_key[rows] = torch.where(tail_mask, buffer_key, pending_key[rows])
        pending_z[rows] = torch.where(tail_mask, buffer_z, pending_z[rows])

    # -- stage 3: scoring ---------------------------------------------------

    def _score(
        self,
        md: Step4DSAMetadata,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Per-request indexer scores, in the selector's group-major row order.

        Two tensors rather than one because the two selectors disagree about
        what a row is: decode has one row per request, prefill one per token.

        The width is the largest number of complete regions any row can see, not
        the summary pool's capacity. The difference is not cosmetic -- at a 64K
        context the capacity-sized tensor for a 16K chunk would be tens of GB
        for a live corner a thousandth of that size.

        The loop is over requests because each scores against a different key
        set and the kernel takes a single shared key head. There is no longer a
        loop over the query axis: the kernel writes into ``prefill_logits`` /
        ``decode_logits`` in place, so nothing has to be chunked to bound a
        temporary. See ``INDEXER_Q_CHUNK`` for the valve that restores chunking.
        """
        state = self.state
        groups = self.num_kv_groups
        num_decodes = md.num_decodes
        num_decode_tokens = md.num_decode_tokens
        num_prefill_tokens = md.num_actual_tokens - num_decode_tokens
        device = index_q.device
        summary = state.summary[self.slot]

        if sched.verification:
            return self._score_verification(md, sched, index_q, weights)
        if sched.decode:
            return self._score_decode(md, sched, index_q, weights)

        width_decode = max(
            (p.nscore for p in sched.requests if p.row < num_decodes), default=0
        )
        width_prefill = max(
            (p.nscore for p in sched.requests if p.row >= num_decodes), default=0
        )

        decode_logits = None
        if num_decodes > 0:
            decode_logits = torch.zeros(
                (groups * num_decodes, max(width_decode, 1)),
                dtype=torch.float32,
                device=device,
            )
        prefill_logits = None
        if md.num_prefills > 0:
            prefill_logits = torch.zeros(
                (groups * num_prefill_tokens, max(width_prefill, 1)),
                dtype=torch.float32,
                device=device,
            )

        prefill_plans = [
            plan for plan in sched.requests if plan.row >= num_decodes
        ]
        batched_short_prefill = (
            os.environ.get("VLLM_STEP5_BATCHED_SHORT_INDEXER", "0") == "1"
            and not DENSE_EQUIV
            and prefill_logits is not None
            and md.prefill_request_indices is not None
            and max((plan.qlen for plan in prefill_plans), default=0) <= 4
            and width_prefill > 0
        )
        if batched_short_prefill:
            assert md.positions is not None
            common_qlen = prefill_plans[0].qlen if prefill_plans else 1
            queries_per_request = (
                common_qlen
                if os.environ.get(
                    "VLLM_STEP5_GROUPED_MTP_INDEXER", "0"
                ) == "1"
                and 1 <= common_qlen <= 4
                and all(plan.qlen == common_qlen for plan in prefill_plans)
                and num_prefill_tokens % common_qlen == 0
                else 1
            )
            short_prefill_indexer_logits(
                index_q[num_decode_tokens:].contiguous(),
                weights[num_decode_tokens:].contiguous(),
                summary,
                md.block_table,
                md.prefill_request_indices,
                md.positions[num_decode_tokens:md.num_actual_tokens],
                prefill_logits,
                regions_per_page=state.regions_per_page,
                queries_per_request=queries_per_request,
                q_is_e4m3_rounded=short_index_query_rounding_enabled(
                    md.num_actual_tokens, self.geometry.proxy_dim, index_q.dtype,
                ),
            )

        if DENSE_EQUIV:
            # All-zero scores plus topk >= every row's region count selects
            # every visible region, which makes DSA exactly full causal
            # attention. Any output difference against the dense backend is then
            # a paging/metadata bug and nothing else.
            return prefill_logits, decode_logits, max(width_decode, width_prefill)

        for plan in sched.requests:
            if plan.nscore == 0:
                continue
            decode = plan.row < num_decodes
            if batched_short_prefill and not decode:
                continue
            keys = summary[plan.score_slots].to(index_q.dtype)
            out = decode_logits if decode else prefill_logits
            stride = num_decodes if decode else num_prefill_tokens
            base = plan.row if decode else plan.tok_start - num_decode_tokens

            for begin in range(0, plan.qlen, INDEXER_Q_CHUNK or plan.qlen):
                count = min(INDEXER_Q_CHUNK or plan.qlen, plan.qlen - begin)
                first = plan.tok_start + begin
                indexer_logits(
                    index_q[first : first + count].contiguous(),
                    weights[first : first + count].contiguous(),
                    keys,
                    out=out,
                    out_row_base=base + begin,
                    out_group_stride=stride,
                    # The pool is e4m3, so the key-side rounding is the identity --
                    # the same argument ``decode_indexer_logits`` already makes.
                    quantize_keys=False,
                )

        return prefill_logits, decode_logits, max(width_decode, width_prefill)

    def _score_verification(
        self,
        md: Step4DSAMetadata,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, None, int]:
        """Score a fixed-width uniform MTP batch through graph-stable buffers."""
        state = self.state
        num_tokens = sched.num_requests * sched.queries_per_request
        rows = self.num_kv_groups * num_tokens
        if sched.graph_regions:
            width = sched.graph_regions
            prefill_logits = state._verification_logits[:rows, :width]
        else:
            width = sched.max_nscore
            prefill_logits = torch.empty(
                (rows, max(width, 1)), dtype=torch.float32, device=index_q.device
            )
        if DENSE_EQUIV or width == 0:
            prefill_logits.zero_()
            return prefill_logits, None, width
        assert md.positions is not None
        short_prefill_indexer_logits(
            index_q[:num_tokens].contiguous(),
            weights[:num_tokens].contiguous(),
            state.summary[self.slot],
            md.block_table[: sched.num_requests],
            sched.request_indices,
            md.positions[:num_tokens],
            prefill_logits,
            regions_per_page=state.regions_per_page,
            queries_per_request=sched.queries_per_request,
            q_is_e4m3_rounded=short_index_query_rounding_enabled(
                md.num_actual_tokens, self.geometry.proxy_dim, index_q.dtype,
            ),
        )
        return prefill_logits, None, width

    def _score_decode(
        self,
        md: Step4DSAMetadata,
        sched: Step4DSASchedule,
        index_q: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[None, torch.Tensor, int]:
        """One scoring launch for the whole decode batch instead of one per request.

        The general path must call ``indexer_logits`` per request because each
        one scores against a different key set and that kernel takes a single
        shared key head. Here the request axis moves into the launch grid and
        the kernel reads the summary pool in place through the block table, so
        nothing is gathered.

        Under ``graph_regions`` the score tensor is a slice of a pool allocated
        once, at a width that covers ``max_model_len`` rather than this batch's
        longest context, because a captured launch cannot resize. The slice is
        deliberately not cleared: a row is written for its own ``nscore``
        columns, the selector reads exactly ``(seq_len - 1) // region_size`` of
        them, and those are the same number -- so nothing this step did not
        write is ever read, and clearing 64K columns per row per layer would
        cost more than the scoring does.
        """
        state = self.state
        num_decodes = sched.num_decodes
        rows = self.num_kv_groups * num_decodes

        if sched.graph_regions:
            width = sched.graph_regions
            decode_logits = state._decode_logits[:rows, :width]
        else:
            width = sched.max_nscore
            decode_logits = torch.zeros(
                (rows, max(width, 1)), dtype=torch.float32, device=index_q.device,
            )
        if DENSE_EQUIV or width == 0:
            return None, decode_logits, width

        decode_indexer_logits(
            index_q[:num_decodes].contiguous(),
            weights[:num_decodes].contiguous(),
            state.summary[self.slot],
            md.block_table[:num_decodes],
            sched.nscore,
            decode_logits,
            regions_per_page=state.regions_per_page,
        )
        return None, decode_logits, width


# ---------------------------------------------------------------------------
# The indexer as an opaque op
# ---------------------------------------------------------------------------
#
# Calling Step4Indexer.run() directly from Step4Attention.forward does not
# survive torch.compile. The model is traced once, during the profiling run,
# where attn_metadata is None -- so Dynamo takes run()'s early-return branch,
# bakes "do nothing" into the graph, and every real step afterwards reaches the
# attention layer with no plan at all. Even without that, the body is untraceable
# in principle: it branches on CPU sequence lengths and allocates tensors whose
# shapes depend on them.
#
# Wrapping it in a custom op fixes both. The body is never traced, it runs
# eagerly at the point the op appears in the graph, and the Python attribute it
# writes (layer.step4_sparse_plan) is set at execution time rather than trace
# time. This is exactly how vLLM's own DeepSeek indexer is wired
# (model_executor/layers/sparse_attn_indexer.py:395).
#
# plan_token is what keeps the op alive. An op with no outputs and no declared
# mutations is dead code to the functionaliser; declaring the write makes the
# buffer a graph input that has to be updated, so the call is kept and stays
# ordered against the attention op that consumes its plan. The value written is
# the real one -- the number of scored regions this step -- so the buffer is
# also readable when debugging a layer that selected nothing.
#
# Being a custom op is necessary but NOT sufficient: the op must also be listed
# in CompilationConfig.splitting_ops. vLLM compiles piecewise and captures each
# piece between splitting ops into a CUDA graph, and a replayed graph re-runs
# recorded kernels, not Python -- so an unsplit indexer runs once, at capture
# time, and every subsequent step silently reuses that plan. vLLM's in-tree
# indexer is in the default list ("vllm::sparse_attn_indexer",
# config/compilation.py:750); this one is out of tree, so Step4Config.__init__
# appends it to CompilationConfig._attention_ops (see
# transformers_utils/configs/step4.py for why the registration lives there and
# not here). _require_splitting_op below is what turns a bypassed registration
# into a startup error instead of a wrong answer at the first request.


def step4_dsa_index(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    plan_token: torch.Tensor,
    projected_index: torch.Tensor | None = None,
) -> None:
    indexer = _INDEXERS.get(layer_name)
    if indexer is None:
        raise RuntimeError(
            f"{layer_name}: no Step4Indexer registered. The op is dispatched by "
            f"layer name, so the indexer must be constructed before the first "
            f"forward pass."
        )
    plan_token.fill_(indexer.run(hidden_states, positions, projected_index))


def step4_dsa_index_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    plan_token: torch.Tensor,
    projected_index: torch.Tensor | None = None,
) -> None:
    return None


direct_register_custom_op(
    op_name="step4_dsa_index",
    op_func=step4_dsa_index,
    mutates_args=["plan_token"],
    fake_impl=step4_dsa_index_fake,
    dispatch_key=current_platform.dispatch_key,
)

def step5_dsa_index_qknorm(
    hidden_states: torch.Tensor, positions: torch.Tensor,
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, layer_name: str, plan_token: torch.Tensor,
    num_q_heads: int, num_kv_heads: int, rotary_pairs: int, epsilon: float,
    projected_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    indexer = _INDEXERS.get(layer_name)
    if indexer is None:
        raise RuntimeError(f"{layer_name}: no Step4Indexer registered")
    main_norm = (qkv, q_weight, k_weight, cos_sin_cache, rotary_pairs, epsilon)
    # Keep QKV head geometry aligned with the indexer provider groups.
    width = indexer.run(
        hidden_states, positions, projected_index,
        main_qknorm=main_norm if (num_q_heads, num_kv_heads) in ((32, 2), (16, 1)) else None,
    )
    plan_token.fill_(width)
    plan = indexer.owner.attn.step4_sparse_plan
    if plan is not None and plan.main_qk is not None:
        indexer_norm_cast_logger.info_once(
            "Step5 joint causal pack/QK-norm/RoPE active for target verification",
            scope="process",
        )
        return plan.main_qk
    from .step5_attention_ops import qknorm_rope_impl
    return qknorm_rope_impl(
        qkv, q_weight, k_weight, cos_sin_cache, positions,
        num_q_heads, num_kv_heads, rotary_pairs, epsilon,
    )


def step5_dsa_index_qknorm_fake(
    hidden_states: torch.Tensor, positions: torch.Tensor,
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, layer_name: str, plan_token: torch.Tensor,
    num_q_heads: int, num_kv_heads: int, rotary_pairs: int, epsilon: float,
    projected_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        qkv.new_empty((qkv.shape[0], num_q_heads * 192)),
        qkv.new_empty((qkv.shape[0], num_kv_heads * 192)),
    )


direct_register_custom_op(
    op_name="step5_dsa_index_qknorm", op_func=step5_dsa_index_qknorm,
    mutates_args=["plan_token"], fake_impl=step5_dsa_index_qknorm_fake,
    dispatch_key=current_platform.dispatch_key,
)
