# SPDX-License-Identifier: Apache-2.0
"""Real vLLM validation/allocator gates for Step5's DCP2 integration."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config.model import ModelConfig
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec,
)
from vllm_hcu.step5_dcp2_config import (
    validate_step5_dcp2_config, wrap_step5_parallel_verification,
)
from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec


def envelope():
    text = SimpleNamespace(
        hidden_size=4096, sliding_window=512, swa_num_attention_heads=None,
        sparse_config=dict(enabled=True,proxy_dim=256,topk=512,region_block_size=8,
                           num_provider_groups=4,sparse_indexer_num_heads=16))
    model = SimpleNamespace(
        architectures=["Step5ForCausalLM"], use_mla=False, dtype=torch.bfloat16,
        model_arch_config=SimpleNamespace(total_num_attention_heads=64),
        get_total_num_kv_heads=lambda:4, get_head_size=lambda:192,
        hf_text_config=text, multimodal_config=None)
    model._verify_with_expert_parallelism = lambda: setattr(model,"expert_verified",True)
    parallel = SimpleNamespace(
        tensor_parallel_size=2,data_parallel_size=4,decode_context_parallel_size=2,
        prefill_context_parallel_size=1,pipeline_parallel_size=1,
        cp_kv_cache_interleave_size=1,dcp_comm_backend="ag_rs",dcp_q_replicate=False,
        enable_expert_parallel=True,all2all_backend="deepep_low_latency",
        enable_dbo=False,enable_eplb=False,world_size_across_dp=8)
    return SimpleNamespace(
        model_config=model,parallel_config=parallel,
        cache_config=SimpleNamespace(block_size=64,cache_dtype="auto",
                                     enable_prefix_caching=False,kv_offloading_size=None),
        kernel_config=SimpleNamespace(moe_backend="deep_gemm"),use_v2_model_runner=True,
        speculative_config=SimpleNamespace(method="mtp",num_speculative_tokens=3),
        lora_config=None,kv_transfer_config=None)


@pytest.fixture(autouse=True)
def runtime_patches(monkeypatch):
    # The global test harness disables plugin discovery during collection.
    # Apply the production adapters explicitly and restore all targets.
    from vllm.v1.core import kv_cache_utils, kv_cache_coordinator
    from vllm_hcu.patch.platform.framework_opt import (
        patch_step5_dcp2_cache, patch_step5_dcp2_cache_utils,
    )
    monkeypatch.setattr(ModelConfig, "verify_with_parallel_config",
                       wrap_step5_parallel_verification(
                           ModelConfig.verify_with_parallel_config))
    monkeypatch.setattr(kv_cache_utils, "dcp_world_size_for_kv_cache_spec",
                       kv_cache_utils.dcp_world_size_for_kv_cache_spec)
    monkeypatch.setattr(kv_cache_utils, patch_step5_dcp2_cache_utils._MARKER,
                       False, raising=False)
    monkeypatch.setattr(kv_cache_coordinator, "dcp_world_size_for_kv_cache_spec",
                       kv_cache_coordinator.dcp_world_size_for_kv_cache_spec)
    monkeypatch.setattr(HybridKVCacheCoordinator, "__init__",
                       HybridKVCacheCoordinator.__init__)
    monkeypatch.setattr(HybridKVCacheCoordinator, patch_step5_dcp2_cache._MARKER,
                       False, raising=False)
    assert patch_step5_dcp2_cache.apply_to_module(kv_cache_coordinator)


@pytest.fixture
def opt_in(monkeypatch):
    flags=dict(VLLM_STEP5_DCP2_RUNTIME="1", VLLM_KV_CACHE_LAYOUT="LBNHC",
               VLLM_STEP5_SEQUENCE_PARALLEL_MOE="1",VLLM_STEP5_SP_LL_WORKSPACE="1",
               VLLM_STEP4_DSA_FP8_MATH="off",VLLM_STEP4_DISABLE_DSA="0",
               VLLM_STEP4_DSA_DENSE_EQUIV="0")
    for key,value in flags.items():
        monkeypatch.setenv(key,value)


def test_real_stock_verifier_keeps_ep_validation_and_parallel_config(opt_in):
    cfg = envelope()
    assert validate_step5_dcp2_config(cfg)
    ModelConfig.verify_with_parallel_config(cfg.model_config,cfg.parallel_config)
    assert cfg.model_config.expert_verified
    assert cfg.parallel_config.decode_context_parallel_size == 2
    assert cfg.model_config.get_total_num_kv_heads() == 4
    assert not cfg.model_config.use_mla


def test_other_gqa_model_still_rejected(opt_in):
    cfg = envelope()
    cfg.model_config.architectures = ["LlamaForCausalLM"]
    with pytest.raises(ValueError,match="greater than"):
        ModelConfig.verify_with_parallel_config(cfg.model_config,cfg.parallel_config)


def test_stock_mm_ipc_rejection_is_preserved(opt_in):
    cfg = envelope()
    cfg.model_config.multimodal_config=SimpleNamespace(mm_tensor_ipc="torch_shm")
    with pytest.raises(ValueError,match="torch_shm"):
        ModelConfig.verify_with_parallel_config(cfg.model_config,cfg.parallel_config)


@pytest.mark.parametrize("owner,key,value", [
    ("parallel_config","tensor_parallel_size",4),
    ("parallel_config","data_parallel_size",2),
    ("parallel_config","decode_context_parallel_size",4),
    ("parallel_config","prefill_context_parallel_size",2),
    ("parallel_config","pipeline_parallel_size",2),
    ("parallel_config","cp_kv_cache_interleave_size",64),
    ("parallel_config","dcp_q_replicate",True),
    ("parallel_config","all2all_backend","deepep_high_throughput"),
    ("parallel_config","enable_expert_parallel",False),
    ("parallel_config","enable_dbo",True),
    ("cache_config","enable_prefix_caching",True),
    ("cache_config","block_size",128),
    ("cache_config","cache_dtype","fp8_e4m3"),
    ("cache_config","kv_offloading_size",1),
    ("kernel_config","moe_backend","aiter"),
    ("model_config","dtype",torch.float16),
    ("speculative_config","num_speculative_tokens",2),
])
def test_unsupported_envelopes_fail(opt_in, owner,key,value):
    cfg=envelope()
    setattr(getattr(cfg,owner),key,value)
    with pytest.raises(ValueError):
        validate_step5_dcp2_config(cfg)


def make_specs(extra=3):
    full=FullAttentionSpec(block_size=64,num_kv_heads=4,head_size=192,
                           dtype=torch.bfloat16)
    window=Step5DCP2SlidingWindowSpec(block_size=64,num_kv_heads=4,head_size=192,
                                     dtype=torch.bfloat16,sliding_window=512,
                                     extra_retained_tokens=extra)
    return full,window


def coordinator(specs, *, blocks=1024, dcp=2, pcp=1, prefix=False, hash_block_size=128):
    config=KVCacheConfig(
        num_blocks=blocks,kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer-{i}"],s) for i,s in enumerate(specs)])
    return HybridKVCacheCoordinator(
        config,max_model_len=65536,max_in_flight_tokens=384,use_eagle=True,
        enable_caching=prefix,enable_kv_cache_events=False,
        dcp_world_size=dcp,pcp_world_size=pcp,
        scheduler_block_size=128,hash_block_size=hash_block_size,num_prefill_lookahead=3)


@pytest.mark.parametrize("batch",[1,32,64])
def test_real_hybrid_coordinator_recycles_window_and_keeps_sparse_history(opt_in,batch):
    full,window=make_specs()
    cap=window.max_admission_blocks_per_request(384,65536)
    # Enough capacity for the retained full history and worst-case SWA cap only.
    c=coordinator((full,window),blocks=batch*(512+cap)+1)
    full_manager,window_manager=c.single_type_managers
    assert full_manager.block_size == window_manager.block_size == 128
    assert c.full_attention_group_id == 0
    assert len(c.attention_groups)==1
    requests=[f"request-{i}" for i in range(batch)]
    for req in requests:
        for manager in c.single_type_managers:
            manager.add_local_computed_blocks(req,(),0,0)
    for computed in range(0,65536,384):
        end=min(computed+384,65536)
        for req in requests:
            c.remove_skipped_blocks(req,computed)
            needed=c.get_num_blocks_to_allocate(req,end,((),()),0,computed,computed,end)
            before=c.block_pool.get_num_free_blocks()
            assert needed<=before
            allocated=c.allocate_new_blocks(req,end,end)
            assert sum(map(len,allocated))==needed
            assert c.block_pool.get_num_free_blocks()==before-needed
            full_blocks=full_manager.req_to_blocks[req]
            window_blocks=window_manager.req_to_blocks[req]
            assert len(full_blocks)==(end+127)//128
            assert all(b is not c.block_pool.null_block for b in full_blocks)
            first=max(0,computed-511-3)//128
            assert all(b is c.block_pool.null_block for b in window_blocks[:first])
            assert all(b is not c.block_pool.null_block for b in window_blocks[first:])
            assert sum(b is not c.block_pool.null_block for b in window_blocks)<=cap
    for req in requests:
        c.free(req)
    assert c.block_pool.get_num_free_blocks()==batch*(512+cap)


def test_plain_sliding_window_dcp_still_rejected(opt_in):
    full,window=make_specs()
    stock=SlidingWindowSpec(block_size=64,num_kv_heads=4,head_size=192,
                            dtype=torch.bfloat16,sliding_window=512)
    with pytest.raises(AssertionError,match="only supports"):
        coordinator((full,stock),hash_block_size=64)


@pytest.mark.parametrize("kwargs",[{"prefix":True},{"dcp":1},{"dcp":4},{"pcp":2}])
def test_typed_window_rejects_unqualified_coordinator_modes(opt_in,kwargs):
    with pytest.raises(ValueError):
        coordinator(make_specs(),**kwargs)


def test_mixed_stock_window_cannot_bypass_type_guard(opt_in):
    full,window=make_specs()
    stock=SlidingWindowSpec(block_size=64,num_kv_heads=4,head_size=192,
                            dtype=torch.bfloat16,sliding_window=512)
    with pytest.raises(ValueError,match="incompatible"):
        coordinator((full,window,stock))


def test_wrong_full_cache_geometry_rejected(opt_in):
    full,window=make_specs()
    with pytest.raises(ValueError,match="incompatible"):
        coordinator((replace(full,num_kv_heads=2),window))


def test_actual_world_size_helper_scales_only_typed_window(opt_in):
    from vllm.v1.core import kv_cache_utils, kv_cache_coordinator
    full, window = make_specs()
    stock = SlidingWindowSpec(block_size=64, num_kv_heads=4, head_size=192,
                              dtype=torch.bfloat16, sliding_window=512)
    assert kv_cache_utils.dcp_world_size_for_kv_cache_spec(window, 2) == 2
    assert kv_cache_utils.dcp_world_size_for_kv_cache_spec(stock, 2) == 1
    assert kv_cache_utils.dcp_world_size_for_kv_cache_spec(full, 2) == 2
    assert kv_cache_coordinator.dcp_world_size_for_kv_cache_spec is (
        kv_cache_utils.dcp_world_size_for_kv_cache_spec)
    with pytest.raises(ValueError):
        kv_cache_utils.dcp_world_size_for_kv_cache_spec(window, 1)


def test_dcp2_sp_workspace_covers_target_draft_and_graph_padding(opt_in):
    from vllm_hcu.model_executor.layers.fused_moe.step5_sp_workspace import (
        sequence_parallel_dispatch_bound,
    )
    cfg = envelope()
    cfg.scheduler_config = SimpleNamespace(max_num_batched_tokens=384, max_num_seqs=64)
    cfg.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[4,8,16,32,64,128,512])
    moe = SimpleNamespace(max_num_tokens=384,
        moe_parallel_config=SimpleNamespace(is_sequence_parallel=True,sp_size=2,ep_size=8))
    assert sequence_parallel_dispatch_bound(moe,cfg)==256
    assert moe.max_num_tokens==384


def test_dcp2_summary_budget_uses_global_pages_and_includes_retained_state(opt_in,monkeypatch):
    from vllm.v1.core import kv_cache_utils
    from vllm_hcu.v1.step5_dsa_memory import reserve_step5_dsa_memory
    cfg = envelope()
    full,window=make_specs()
    state=SimpleNamespace(context_parallel_size=2,dtype=torch.bfloat16,num_layers=2,
        max_pending=3,graph_decode=False,
        geometry=SimpleNamespace(region_size=8,proxy_dim=256,rope_span=32))
    indexer=SimpleNamespace(layer_name="sparse",state=state,rope_theta=10000,
                           rope_scaling=None,max_position=128)
    worker=SimpleNamespace(vllm_config=cfg,
        get_kv_cache_spec=lambda:{"sparse":full,"window":window},
        model_runner=SimpleNamespace(model=SimpleNamespace(
            modules=lambda:[SimpleNamespace(indexer=indexer)])))
    monkeypatch.setenv("VLLM_STEP5_DSA_MEMORY_BUDGET","1")
    monkeypatch.setenv("VLLM_STEP5_SWA_PREFILL_CROP","0")
    monkeypatch.setattr(kv_cache_utils,"get_kv_cache_groups",lambda *a:[])
    monkeypatch.setattr(kv_cache_utils,"_pool_bytes_per_block",lambda *a:1024)
    available=10**8
    cache=reserve_step5_dsa_memory(worker,available)
    budget=worker._step5_dsa_memory_budget
    assert budget["summary_per_block"]==2*16*256
    assert budget["states"][0]["pending"]==2*2*4*8*256*2
    assert cache+budget["reserve"]<=available
    assert cache==budget["blocks"]*1024
    assert available-cache-budget["reserve"]<1024+2*16*256
    state.context_parallel_size=1
    with pytest.raises(ValueError,match="sizes differ"):
        reserve_step5_dsa_memory(worker,available)


def test_typed_world_size_requires_opt_in_and_homogeneous_uniform_group(opt_in,monkeypatch):
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    _,window=make_specs()
    uniform=UniformTypeKVCacheSpecs(block_size=64,kv_cache_specs={"a":window,"b":window})
    assert kv_cache_utils.dcp_world_size_for_kv_cache_spec(uniform,2)==2
    monkeypatch.setenv("VLLM_STEP5_DCP2_RUNTIME","0")
    with pytest.raises(ValueError,match="opt-in"):
        kv_cache_utils.dcp_world_size_for_kv_cache_spec(uniform,2)


def test_full_graph_scope_requires_explicit_decode_only_opt_in(opt_in,monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm_hcu.step5_dcp2_config import step5_dcp2_full_graph_allowed
    cfg=envelope()
    cfg.compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    monkeypatch.delenv("VLLM_STEP5_DCP2_FULL_GRAPH",raising=False)
    assert not step5_dcp2_full_graph_allowed(cfg)
    monkeypatch.setenv("VLLM_STEP5_DCP2_FULL_GRAPH","1")
    assert step5_dcp2_full_graph_allowed(cfg)
    cfg.compilation_config.cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    with pytest.raises(ValueError,match="FULL_DECODE_ONLY"):
        step5_dcp2_full_graph_allowed(cfg)
    cfg.model_config.architectures=["LlamaForCausalLM"]
    assert not step5_dcp2_full_graph_allowed(cfg)
    cfg=envelope()
    cfg.parallel_config.decode_context_parallel_size=1
    assert not step5_dcp2_full_graph_allowed(cfg)


@pytest.mark.parametrize("dcp",[1,2])
@pytest.mark.parametrize("mode",["PIECEWISE","FULL_DECODE_ONLY","FULL_AND_PIECEWISE"])
def test_mtp_full_padding_guard_is_armed_for_existing_full_modes(opt_in,monkeypatch,dcp,mode):
    from types import ModuleType
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm_hcu.patch.platform.core_fix import patch_step5_multi_mtp
    monkeypatch.setenv("VLLM_STEP5_MTP_DRAFT_FULL_GRAPH","1")
    class Speculator:
        def __init__(self,*a,**kw): pass
        def _run_model(self,*a,**kw): pass
        def propose(self,*a,**kw): pass
        def _build_draft_attn_metadata(self,*a,**kw): pass
        def init_cudagraph_manager(self,mode): self.seen_mode=mode
    module=ModuleType(patch_step5_multi_mtp.TARGET_MODULE)
    module.MultiModuleMTPSpeculator=Speculator
    module.dispatch_cg_and_sync_dp=lambda *a,**kw:None
    module.logger=SimpleNamespace(info=lambda *a:None)
    assert patch_step5_multi_mtp.apply_to_module(module)
    cfg=envelope()
    cfg.parallel_config.decode_context_parallel_size=dcp
    cfg.speculative_config.draft_model_config=SimpleNamespace(
        hf_config=SimpleNamespace(model_type="step5_mtp"))
    obj=Speculator.__new__(Speculator)
    obj.vllm_config=cfg
    obj.num_speculative_steps=3
    obj.attn_cg_support=SimpleNamespace(min_cg_support=AttentionCGSupport.UNIFORM_BATCH)
    obj.init_cudagraph_manager(CUDAGraphMode[mode])
    assert obj.seen_mode==CUDAGraphMode.FULL_DECODE_ONLY
    assert obj._step5_draft_full_graph_enabled
