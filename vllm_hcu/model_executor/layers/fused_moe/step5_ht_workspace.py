# SPDX-License-Identifier: Apache-2.0
# Size Step5 HT expert buffers from fresh post-dispatch routing counts.
from dataclasses import dataclass
from contextvars import ContextVar
import torch
from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter

@dataclass
class Step5HTTokensMetadata(ExpertTokensMetadata):
    aligned_rows: int = 0
    routed_shape: tuple[int, ...] = ()
    topk_ptr: int = 0
    topk_version: int | None = 0
    map_ptr: int = 0
    map_version: int | None = 0

def aligned_expert_rows(counts: torch.Tensor, alignment: int = 256) -> int:
    if counts.device.type != "cpu" or counts.ndim != 1:
        raise ValueError("Expert counts must be a one-dimensional CPU tensor")
    if alignment <= 0 or bool((counts < 0).any()):
        raise ValueError("Invalid expert count or alignment")
    values = counts.to(torch.int64)
    return int((((values + alignment - 1) // alignment) * alignment).sum())

def compact_workspace_parallelism_supported(tp_size: int, dp_size: int) -> bool:
    # Both layouts use all eight ranks for EP, so post-dispatch expert counts
    # provide the same exact workspace bound on TP1/DP8, TP2/DP4 and TP4/DP2.
    return (tp_size, dp_size) in ((1, 8), (2, 4), (4, 2))

@dataclass
class Step5HTDispatchTokensMetadata(ExpertTokensMetadata):
    # Only the HT receiver creates this marker, after its local-to-global remap.
    routed_ids: torch.Tensor | None = None
    routed_version: int | None = None
    rank_expert_offset: int = 0
    global_num_experts: int = 0


def mark_fresh_dispatch_metadata(meta, routed_ids, offset, global_num_experts):
    return Step5HTDispatchTokensMetadata(
        expert_num_tokens=meta.expert_num_tokens,
        expert_num_tokens_cpu=meta.expert_num_tokens_cpu,
        routed_ids=routed_ids,
        routed_version=_tensor_version(routed_ids),
        rank_expert_offset=offset,
        global_num_experts=global_num_experts,
    )


def _dispatch_counts_match_local_map(meta, topk_ids, local_num_experts,
                                    expert_map, map_cache):
    if not isinstance(meta, Step5HTDispatchTokensMetadata):
        return False
    if meta.routed_ids is not topk_ids:
        return False
    if meta.routed_version != _tensor_version(topk_ids):
        return False
    cpu = meta.expert_num_tokens_cpu
    if (cpu is None or cpu.device.type != "cpu" or cpu.ndim != 1
            or cpu.numel() != local_num_experts
            or meta.expert_num_tokens.shape != cpu.shape):
        return False
    if expert_map is None or expert_map.numel() != meta.global_num_experts:
        return False
    version = _tensor_version(expert_map)
    if version is None:
        # Inference tensors have no mutation counter. Retain the recount path.
        return False
    key = (version, meta.rank_expert_offset, local_num_experts,
           meta.global_num_experts)
    if map_cache.get("tensor") is not expert_map or map_cache.get("key") != key:
        expected = torch.full((meta.global_num_experts,), -1,
                              device="cpu", dtype=expert_map.dtype)
        begin = meta.rank_expert_offset
        if begin < 0 or begin + local_num_experts > expected.numel():
            return False
        expected[begin:begin + local_num_experts] = torch.arange(
            local_num_experts, dtype=expected.dtype
        )
        actual = expert_map.detach().cpu()
        map_cache.update(tensor=expert_map, key=key,
                         matches=torch.equal(actual, expected))
    return map_cache["matches"]


def _tensor_version(tensor):
    # vLLM creates routing tensors under inference_mode, without a version counter.
    try:
        return tensor._version
    except RuntimeError:
        return None

def prepare_compact_metadata(topk_ids, local_num_experts, expert_map, previous,
                             dispatch_map_cache=None):
    # Count after dispatch/remapping; never shrink from stale dispatch metadata.
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return previous
    if (dispatch_map_cache is not None
            and _dispatch_counts_match_local_map(
                previous, topk_ids, local_num_experts, expert_map,
                dispatch_map_cache)):
        counts = previous.expert_num_tokens
        cpu = previous.expert_num_tokens_cpu
        if bool((cpu < 0).any()):
            raise RuntimeError("Fresh dispatch contains negative expert counts")
    else:
        counts = count_expert_num_tokens(topk_ids, local_num_experts, expert_map)
        cpu = counts.cpu()
    total = int(cpu.to(torch.int64).sum())
    if total == 0:
        return previous  # Preserve the established all-invalid GEMM path.
    if total > topk_ids.numel():
        raise RuntimeError("Post-dispatch expert counts exceed routed assignments")
    return Step5HTTokensMetadata(
        expert_num_tokens=counts, expert_num_tokens_cpu=cpu,
        aligned_rows=aligned_expert_rows(cpu), routed_shape=tuple(topk_ids.shape),
        topk_ptr=topk_ids.data_ptr(), topk_version=_tensor_version(topk_ids),
        map_ptr=0 if expert_map is None else expert_map.data_ptr(),
        map_version=0 if expert_map is None else _tensor_version(expert_map),
    )

def compact_aligned_rows(meta, M, topk, experts):
    if not isinstance(meta, Step5HTTokensMetadata):
        return None
    if meta.routed_shape != (M, topk) or meta.expert_num_tokens.numel() != experts:
        raise RuntimeError("Compact HT metadata does not match routed shape")
    if meta.aligned_rows <= 0 or meta.aligned_rows % 256:
        raise RuntimeError("Invalid compact HT workspace alignment")
    return meta.aligned_rows

def compact_moe_permute(aq, aq_scale, topk_ids, local_num_experts,
                        expert_map, expert_tokens_meta, aq_out):
    meta = expert_tokens_meta
    rows = compact_aligned_rows(meta, *topk_ids.shape, local_num_experts)
    if (meta.topk_ptr != topk_ids.data_ptr()
            or meta.topk_version != _tensor_version(topk_ids)
            or meta.map_ptr != (0 if expert_map is None else expert_map.data_ptr())
            or meta.map_version != (0 if expert_map is None else _tensor_version(expert_map))):
        raise RuntimeError("Routing changed after compact HT workspace sizing")
    if (aq.ndim != 2 or aq_scale.ndim != 2
            or aq_scale.dtype != torch.float32
            or not topk_ids.dtype.is_signed
            or aq_out.shape != (rows, aq.shape[1])):
        raise RuntimeError("Unsupported compact HT activation layout")
    scales = torch.empty((rows, aq_scale.shape[1]), device=aq.device, dtype=torch.float32)
    indices = torch.full((rows,), -1, device=aq.device, dtype=torch.int32)
    inverse = torch.full(topk_ids.shape, -1, device=aq.device, dtype=torch.int32)
    starts = torch.empty(local_num_experts, device=aq.device, dtype=torch.int32)
    ep_scatter(recv_x=aq, recv_x_scale=aq_scale, recv_topk=topk_ids,
               num_recv_tokens_per_expert=meta.expert_num_tokens,
               expert_start_loc=starts, expert_map=expert_map,
               output_tensor=aq_out, output_tensor_scale=scales,
               m_indices=indices, output_index=inverse, align_m=256,
               block_size=128, pack_ue8m0=False)
    return aq_out, scales, indices, inverse, 256


from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.logger import init_logger


_bf16_route_mask = ContextVar("step5_ht_bf16_route_mask", default=None)


class Step5HTChunkedTritonExperts(TritonExperts):
    # Dispatch and combine still handle the full token batch. Only the BF16
    # expert scratch buffers are bounded; the model's quantization is unchanged.
    max_tokens_per_chunk = 4096

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        init_logger("vllm.hcu.step5_ht_workspace").info_once(
            "Step5 HT BF16 experts compute in at most 4096 received-token chunks."
        )


    def apply(
        self, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, global_num_experts, expert_map, a1q_scale, a2_scale,
        workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input,
    ):
        valid = topk_ids >= 0
        if expert_map is not None:
            valid = valid & (expert_map[topk_ids.clamp_min(0).long()] >= 0)
        token = _bf16_route_mask.set(valid)
        try:
            return super().apply(
                output, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                workspace13, workspace2, expert_tokens_meta,
                apply_router_weight_on_input,
            )
        finally:
            _bf16_route_mask.reset(token)

    def moe_sum(self, input, output):
        valid = _bf16_route_mask.get()
        if valid is None or tuple(valid.shape) != tuple(input.shape[:2]):
            raise RuntimeError("Missing BF16 HT routed-assignment mask")
        # Triton omits -1 assignments from its sorted expert work list. Their
        # scratch rows can retain gate/up values or a previous chunk's data.
        # They must contribute zero to the top-k reduction.
        input.masked_fill_(~valid.unsqueeze(-1), 0)
        return super().moe_sum(input, output)
