# SPDX-License-Identifier: Apache-2.0
"""Pack the TP-local Step5 attention and indexer projections after weight loading."""
from __future__ import annotations

import torch
from torch import nn


def _parallel_weight_nn(layer: nn.Module, hidden: int, outputs: int) -> torch.Tensor:
    """HCU linear defaults to [K,N]; other backends store [N,K]."""
    w = layer.weight
    if tuple(w.shape) == (hidden, outputs):
        return w
    if tuple(w.shape) == (outputs, hidden):
        return w.t()
    raise ValueError(f"Step5 projection has unexpected local weight shape {tuple(w.shape)}")


@torch.no_grad()
def prepare_merged_projection(owner: nn.Module) -> None:
    if not owner.merge_attention_projection:
        return
    hidden = owner.qkv_proj.input_size
    qkv_width = owner.q_size + 2 * owner.kv_size
    items = []
    if not getattr(owner, "qkv_fp8_w8a8", False):
        items.append(("qkv", owner.qkv_proj, _parallel_weight_nn(owner.qkv_proj, hidden, qkv_width)))
    if owner.indexer is not None:
        for name in ("q", "k", "z", "w"):
            layer = getattr(owner, "sparse_indexer_" + name)
            # These are torch.nn.Linear, including the FP32 W parameter.
            # W is rounded before the GEMM in the reference implementation.
            items.append(("index_" + name, layer, layer.weight.t()))
    if owner.use_head_wise_attn_gate:
        items.append(("gate", owner.g_proj, _parallel_weight_nn(owner.g_proj, hidden, owner.num_heads)))
    # With FP8 QKV, a sliding layer has only the BF16 gate left. Keep its
    # original narrow GEMM; packing one projection only adds padding.
    if len(items) < 2:
        owner.merge_attention_projection = False
        return
    if items[0][2].dtype != torch.bfloat16:
        raise ValueError("Merged Step5 projection currently requires BF16 weights")
    widths = [int(w.shape[1]) for _, _, w in items]
    total = sum(widths)
    padded = (total + 255) // 256 * 256
    # The BF16 indexer layers are native nn.Linear with [N,K] weights.
    # In FP8-QKV mode, retain that layout for the residual merged GEMM.
    # BF16-QKV mode retains HCU's [K,N] layout for the full projection.
    nk_layout = getattr(owner, "qkv_fp8_w8a8", False)
    packed_shape = (padded, hidden) if nk_layout else (hidden, padded)
    packed = torch.zeros(packed_shape, device=items[0][2].device, dtype=torch.bfloat16)
    bias = None
    if any(getattr(layer, "bias", None) is not None for _, layer, _ in items):
        bias = torch.zeros((padded,), device=packed.device, dtype=packed.dtype)
    offset = 0
    offsets = {}
    for (name, layer, w), width in zip(items, widths):
        end = offset + width
        if nk_layout:
            packed[offset:end, :].copy_(w.t())
        else:
            packed[:, offset:end].copy_(w)
        b = getattr(layer, "bias", None)
        if b is not None:
            bias[offset:end].copy_(b)
        # Reuse the storage for BF16 parameters; retain the original FP32 W
        # parameter so checkpoints/reloads never lose its serialized precision.
        if layer.weight.dtype == packed.dtype:
            view = packed[offset:end, :].t() if nk_layout else packed[:, offset:end]
            if name.startswith("index_"):
                layer.weight.data = view.t()
            elif tuple(layer.weight.shape) == (hidden, width):
                layer.weight.data = view
            else:
                layer.weight.data = view.t()
        offsets[name] = (offset, end)
        offset = end
    owner.step5_merged_layout = "NK" if nk_layout else "KN"
    owner.step5_merged_weight = packed
    owner.step5_merged_bias = bias
    owner.step5_merged_offsets = offsets
    owner.step5_merged_ready = True


def prepare_model_projections(model: nn.Module) -> None:
    import os
    full = os.getenv("VLLM_STEP5_FULL_MIXED_PROJECTION", "0") == "1"
    for module in model.modules():
        if hasattr(module, "merge_attention_projection"):
            requested = module.merge_attention_projection
            prepare_merged_projection(module)
            if full and requested:
                from .step5_full_projection import prepare_full_projection
                prepare_full_projection(module)


def merged_projection(owner: nn.Module, hidden_states: torch.Tensor):
    if not owner.step5_merged_ready:
        raise RuntimeError("Step5 merged projections must be prepared after loading all weights")
    if getattr(owner, "step5_merged_layout", "KN") == "NK":
        projected = torch.nn.functional.linear(
            hidden_states, owner.step5_merged_weight, owner.step5_merged_bias
        )
    elif owner.step5_merged_bias is None:
        projected = torch.matmul(hidden_states, owner.step5_merged_weight)
    else:
        projected = torch.addmm(owner.step5_merged_bias, hidden_states, owner.step5_merged_weight)
    ranges = owner.step5_merged_offsets
    qkv = None
    if "qkv" in ranges:
        qstart, qend = ranges["qkv"]
        qkv = projected[:, qstart:qend]
    index = None
    if owner.indexer is not None:
        istart = ranges["index_q"][0]
        iend = ranges["index_w"][1]
        index = projected[:, istart:iend]
    gate = None
    if owner.use_head_wise_attn_gate:
        start, end = ranges["gate"]
        gate = projected[:, start:end]
    return qkv, index, gate
