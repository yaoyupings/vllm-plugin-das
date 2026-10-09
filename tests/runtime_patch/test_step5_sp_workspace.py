# SPDX-License-Identifier: Apache-2.0
"""Capacity/padding policy tests; native EP8 qualification is recorded separately."""
from pathlib import Path
from types import SimpleNamespace as NS
import runpy
import pytest
import torch

impl = runpy.run_path(str(Path(__file__).resolve().parents[2] /
    "vllm_hcu/model_executor/layers/fused_moe/step5_sp_workspace.py"))
bound = impl["sequence_parallel_dispatch_bound"]
copy_config = impl["bound_sequence_parallel_workspace"]

def config(tokens=384, captures=(4,8,16,32,64,128,256), seqs=16, draft=3, tp=2, dp=4):
    parallel=NS(tensor_parallel_size=tp,data_parallel_size=dp,pipeline_parallel_size=1,
                decode_context_parallel_size=1,prefill_context_parallel_size=1,
                all2all_backend="deepep_low_latency")
    moe=NS(max_num_tokens=tokens,moe_parallel_config=NS(is_sequence_parallel=True,
                       sp_size=tp,ep_size=8))
    cfg=NS(parallel_config=parallel,model_config=NS(architectures=["Step5ForCausalLM"]),
           scheduler_config=NS(max_num_batched_tokens=tokens,max_num_seqs=seqs),
           compilation_config=NS(cudagraph_capture_sizes=list(captures)),
           speculative_config=NS(num_speculative_tokens=draft))
    return moe,cfg

@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_SP_LL_WORKSPACE","1")
    monkeypatch.setenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE","1")
    monkeypatch.delenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT",raising=False)
    monkeypatch.setattr(torch.cuda,"current_device",lambda:0)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:NS(gcnArchName="gfx938:xnack-"))

def test_default_keeps_existing_alignment():
    moe,cfg=config()
    assert bound(moe,cfg)==256

def test_qualified_alignment_preserves_original_config(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT","64")
    moe,cfg=config()
    new=copy_config(moe,cfg)
    assert new is not moe and new.max_num_tokens==192
    assert moe.max_num_tokens==384
    assert new.moe_parallel_config is moe.moe_parallel_config

@pytest.mark.parametrize("tokens,captures,seqs,draft,tp,dp",[
    (384,(256,),16,3,2,4),(384,(512,),16,3,2,4),
    (33,(),17,3,2,4),(513,(256,),16,3,2,4),
    (768,(1024,),32,3,4,2),(1,(),1,0,2,4),
])
def test_every_capacity_source_is_covered(monkeypatch,tokens,captures,seqs,draft,tp,dp):
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT","64")
    moe,cfg=config(tokens,captures,seqs,draft,tp,dp)
    result=bound(moe,cfg)
    assert result%64==0
    assert result*tp>=max(tokens,max(captures,default=0),seqs*(1+draft))
    assert (result-64)*tp<max(tokens,max(captures,default=0),seqs*(1+draft))

@pytest.mark.parametrize("field,value",[("pipeline_parallel_size",2),
    ("decode_context_parallel_size",2),("prefill_context_parallel_size",2),
    ("all2all_backend","deepep_high_throughput"),("data_parallel_size",2)])
def test_unqualified_parallel_paths_are_untouched(field,value):
    moe,cfg=config();setattr(cfg.parallel_config,field,value)
    assert bound(moe,cfg) is None
    assert copy_config(moe,cfg) is moe

@pytest.mark.parametrize("change",["model","sp","ep","sp_env","bound_env"])
def test_unqualified_model_and_expert_paths_are_untouched(monkeypatch,change):
    moe,cfg=config()
    if change=="model":cfg.model_config.architectures=["OtherForCausalLM"]
    elif change=="sp":moe.moe_parallel_config.sp_size=1
    elif change=="ep":moe.moe_parallel_config.ep_size=4
    elif change=="sp_env":monkeypatch.setenv("VLLM_STEP5_SEQUENCE_PARALLEL_MOE","0")
    else:monkeypatch.setenv("VLLM_STEP5_SP_LL_WORKSPACE","0")
    assert copy_config(moe,cfg) is moe

@pytest.mark.parametrize("raw",["0","32","65","192","nan","64.0"])
def test_invalid_alignment_is_rejected(monkeypatch,raw):
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT",raw)
    with pytest.raises(ValueError,match="alignment must be"):bound(*config())

def test_unqualified_arch_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT","64")
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:NS(gcnArchName="gfx92a"))
    with pytest.raises(ValueError,match="requires gfx938"):bound(*config())

def test_carried_vllm_config_uses_same_policy(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT","64")
    moe,cfg=config();moe._hcu_vllm_config=cfg
    assert bound(moe,None)==192
