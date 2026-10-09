# SPDX-License-Identifier: Apache-2.0
"""Preserve Step5 stage selection and FP32 feedback in the V2 MTP runner."""

import os
from contextvars import ContextVar
from functools import wraps
from types import ModuleType

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator"
PATCH_ID = "platform.core_fix.step5_multi_mtp"
TARGETS = (
    f"{TARGET_MODULE}.MultiModuleMTPSpeculator",
    f"{TARGET_MODULE}.dispatch_cg_and_sync_dp",
)

# Scope the adjustment to this speculator's idle-rank proposal. The target's
# DP state is immutable and already agreed by all ranks; no new collective or
# change to the common dispatcher is needed.
_DUMMY_DP_SYNC = ContextVar("step5_mtp_dummy_dp_sync", default=None)


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, "_vllm_hcu_step5_multi_mtp", False):
        return False
    cls = module.MultiModuleMTPSpeculator
    original_init = cls.__init__
    original_run = cls._run_model
    original_propose = cls.propose
    original_dispatch = module.dispatch_cg_and_sync_dp
    original_build = cls._build_draft_attn_metadata
    original_cg_init = cls.init_cudagraph_manager

    @wraps(original_init)
    def init(self, vllm_config, device):
        original_init(self, vllm_config, device)
        draft = vllm_config.speculative_config.draft_model_config.hf_config
        if draft.model_type == "step5_mtp":
            # Each Step5 decoder returns an FP32 residual. Keep it until the
            # next module's normalization/projection rather than round early.
            self.hidden_states = self.hidden_states.float()
            self.cached_target_hidden_states = self.cached_target_hidden_states.float()

    @wraps(original_run)
    def run(self, num_tokens, attn_metadata, slot_mappings,
            num_tokens_across_dp, spec_module_idx, *args, **kwargs):
        if getattr(getattr(self.model, "config", None), "model_type", None) == "step5_mtp":
            # Set at the runner boundary even if the forward uses a graph.
            # Python stage selection is captured into each graph's head GEMM.
            self.model.set_mtp_step(spec_module_idx)
        return original_run(self, num_tokens, attn_metadata, slot_mappings,
                            num_tokens_across_dp, spec_module_idx, *args, **kwargs)

    @wraps(original_build)
    def build_draft_metadata(
        self, num_reqs, num_reqs_padded, num_tokens_padded,
        seq_lens_cpu_upper_bound, step, num_query_per_req=1, causal=True,
        query_start_loc_np=None, dcp_local_seq_lens=None,
    ):
        if (
            getattr(getattr(self.model, "config", None), "model_type", None)
            == "step5_mtp"
            and getattr(self, "_step5_draft_full_graph_enabled", False)
            and query_start_loc_np is not None
            and num_reqs_padded > num_reqs
        ):
            active_tokens = int(query_start_loc_np[num_reqs])
            if not 0 <= active_tokens <= num_tokens_padded:
                raise ValueError("Step5 MTP query offsets exceed padded tokens")
            # The native prepare kernel visits only real requests. FULL replay
            # also samples padded rows, whose old prefill indices can exceed
            # the current graph's token count. Clear only draft-owned padding.
            self.last_token_indices[num_reqs:num_reqs_padded].zero_()
            self.input_buffers.query_start_loc[
                num_reqs + 1:num_reqs_padded + 1
            ].fill_(active_tokens)
            self.input_buffers.seq_lens[num_reqs:num_reqs_padded].zero_()
            self.draft_input_id_overrides[
                num_reqs:num_reqs_padded
            ].fill_(-1)
        if (
            getattr(getattr(self.model, "config", None), "model_type", None)
            == "step5_mtp"
            and query_start_loc_np is not None
            and num_reqs_padded == num_reqs
        ):
            # DP/piecewise dispatch can pad tokens without adding requests.
            # Keep that padding in the model/EP forward, but attention must
            # describe only rows owned by the actual local query offsets.
            # HCU paged attention reshapes Q using those request dimensions.
            active_tokens = int(query_start_loc_np[num_reqs])
            if not 0 <= active_tokens <= num_tokens_padded:
                raise ValueError("Step5 MTP query offsets exceed padded tokens")
            num_tokens_padded = active_tokens
        return original_build(
            self, num_reqs, num_reqs_padded, num_tokens_padded,
            seq_lens_cpu_upper_bound, step,
            num_query_per_req=num_query_per_req, causal=causal,
            query_start_loc_np=query_start_loc_np,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )

    @wraps(original_cg_init)
    def init_cudagraph_manager(self, cudagraph_mode):
        self._step5_draft_full_graph_enabled = False
        if os.getenv("VLLM_STEP5_MTP_DRAFT_FULL_GRAPH", "0") == "1":
            from vllm.config.compilation import CUDAGraphMode
            from vllm.v1.attention.backend import AttentionCGSupport

            from vllm_hcu.step5_dcp2_config import step5_context_parallel_supported
            parallel = self.vllm_config.parallel_config
            draft = self.vllm_config.speculative_config.draft_model_config.hf_config
            scoped = (
                draft.model_type == "step5_mtp"
                and self.num_speculative_steps == 3
                and (parallel.tensor_parallel_size, parallel.data_parallel_size)
                in ((2, 4), (4, 2))
                and parallel.pipeline_parallel_size == 1
                and step5_context_parallel_supported(self.vllm_config)
                and parallel.prefill_context_parallel_size == 1
                and cudagraph_mode in (
                    CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY,
                    CUDAGraphMode.FULL_AND_PIECEWISE,
                )
            )
            if scoped:
                # Preserve padding initialization for all FULL draft modes,
                # including a target that already selected FULL_DECODE_ONLY.
                # DCP2 uses its qualified window bridge and logical-page cache.
                support = self.attn_cg_support.min_cg_support
                if support.value < AttentionCGSupport.UNIFORM_BATCH.value:
                    raise ValueError(
                        "Step5 MTP draft attention cannot capture a uniform batch"
                    )
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
                self._step5_draft_full_graph_enabled = True
                module.logger.info(
                    "Step5 MTP draft graph: FULL_DECODE_ONLY; TP=%s; DP=%s; "
                    "DCP=%s; draft attention support=%s",
                    parallel.tensor_parallel_size, parallel.data_parallel_size,
                    parallel.decode_context_parallel_size, support.name,
                )
        return original_cg_init(self, cudagraph_mode)

    @wraps(original_propose)
    def propose(
        self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
        aux_hidden_states, num_sampled, num_rejected, last_sampled,
        next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
        skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
    ):
        is_step5 = (
            getattr(getattr(self.model, "config", None), "model_type", None)
            == "step5_mtp"
        )
        token = _DUMMY_DP_SYNC.set(dp_sync if is_step5 and dummy_run else None)
        try:
            return original_propose(
                self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature,
                seeds, dp_sync=dp_sync, dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs, is_profile=is_profile,
            )
        finally:
            _DUMMY_DP_SYNC.reset(token)

    @wraps(original_dispatch)
    def dispatch(
        cudagraph_manager, num_reqs, num_tokens, uniform_token_count,
        dp_size, dp_rank, max_query_len=None, need_eager=False,
        num_active_loras=0, parallel_config=None, allow_ubatching=False,
        uniform_decode=False, dp_sync=None,
    ):
        if dp_sync is not None and dp_sync is _DUMMY_DP_SYNC.get():
            # PIECEWISE target dispatch pads tokens without padding requests.
            # InputBatch.make_dummy redistributes those padded tokens, so its
            # artificial query length can differ from the agreed decode length.
            # Dummy outputs are discarded; retain the target's global agreement.
            # The original dispatcher still checks padded tokens and graph mode.
            uniform_token_count = dp_sync.uniform_token_count
        return original_dispatch(
            cudagraph_manager, num_reqs, num_tokens, uniform_token_count,
            dp_size, dp_rank, max_query_len=max_query_len,
            need_eager=need_eager, num_active_loras=num_active_loras,
            parallel_config=parallel_config, allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode, dp_sync=dp_sync,
        )

    cls.init_cudagraph_manager = init_cudagraph_manager
    cls.__init__ = init
    cls._run_model = run
    cls._build_draft_attn_metadata = build_draft_metadata
    cls.propose = propose
    module.dispatch_cg_and_sync_dp = dispatch
    module._vllm_hcu_step5_multi_mtp = True
    return True
