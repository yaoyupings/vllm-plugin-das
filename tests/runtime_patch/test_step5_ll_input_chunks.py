# SPDX-License-Identifier: Apache-2.0
"""Collective order, padding, router preservation and reusable-buffer contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
import runpy
import pytest
import torch

module = runpy.run_path(str(Path(__file__).resolve().parents[2] /
    "vllm_hcu/model_executor/layers/fused_moe/step5_ll_chunk.py"))
plan = module["uniform_chunk_count"]
apply_chunks = module["apply_input_chunks"]
policy = module["configured_chunk_tokens"]

@pytest.mark.parametrize("counts", [[0,16384], [17,8193], [65536,3], [0,0], [255,256]])
@pytest.mark.parametrize("chunk", [128,256,512])
def test_collective_calls_match_for_all_sp_ranks(counts, chunk):
    # Enumerate token assignment instead of repeating the planner's arithmetic.
    assignments = [[len(range(rank, n, 4)) for rank in range(4)] for n in counts]
    padded = [max(sizes) for sizes in assignments]
    calls = [plan(n, torch.tensor(counts), 4, chunk) for n in padded for _ in range(4)]
    assert len(set(calls)) == 1
    assert calls[0] * chunk >= max(padded)
    assert calls[0] == 1 or (calls[0] - 1) * chunk < max(padded)

@pytest.mark.parametrize("rows", [0,1,127,128,129,513])
@pytest.mark.parametrize("shared", [False,True])
def test_routed_and_shared_results_survive_reused_outputs(rows, shared):
    x = torch.arange(rows*8,dtype=torch.float32).reshape(rows,8)
    logits = torch.arange(rows*3,dtype=torch.float32).reshape(rows,3)
    original = x.clone()
    scratch = torch.empty(128,8)
    ss = torch.empty_like(scratch)
    observed = []
    def callback(**kw):
        h, r, s = (kw[k] for k in ("hidden_states","router_logits","shared_experts_input"))
        assert h.shape[0] == 128
        observed.append(r.clone())
        scratch.copy_(h*2 + r[:,0,None])
        if s is not None: ss.copy_(s*3)
        h.zero_()  # An inplace expert must not modify the caller or next chunk.
        return (ss if shared else None),scratch
    count = plan(rows,torch.tensor([rows*4,4096]),4,128)
    s,f = apply_chunks(callback,hidden_states=x,router_logits=logits,
                       shared_experts_input=x if shared else None,
                       chunk=128,num_chunks=count)
    torch.testing.assert_close(f,original*2+logits[:,0,None],rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    assert len(observed)==count
    torch.testing.assert_close(torch.cat(observed)[:rows],logits,rtol=0,atol=0)
    assert torch.count_nonzero(torch.cat(observed)[rows:]) == 0
    if shared: torch.testing.assert_close(s,original*3,rtol=0,atol=0)
    else: assert s is None

def config():
    p=NS(tensor_parallel_size=4,data_parallel_size=2,pipeline_parallel_size=1,
         decode_context_parallel_size=1,prefill_context_parallel_size=1,
         all2all_backend="deepep_low_latency",enable_dbo=False)
    m=NS(moe_parallel_config=NS(is_sequence_parallel=True,sp_size=4,ep_size=8))
    c=NS(parallel_config=p,model_config=NS(architectures=["Step5ForCausalLM"]))
    return m,c

@pytest.fixture
def enabled(monkeypatch):
    for k,v in {"VLLM_STEP5_LL_MOE_CHUNK_TOKENS":"512",
                "VLLM_STEP5_SEQUENCE_PARALLEL_MOE":"1",
                "VLLM_STEP5_SP_LL_WORKSPACE":"1",
                "VLLM_DISABLE_SHARED_EXPERTS_STREAM":"1",
                "VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP":"0"}.items():
        monkeypatch.setenv(k,v)

def test_policy_is_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_STEP5_LL_MOE_CHUNK_TOKENS",raising=False)
    assert policy(*config()) == 0

def test_supported_and_carried_config(enabled):
    m,c=config();assert policy(m,c)==512
    m._hcu_vllm_config=c;assert policy(m,None)==512

@pytest.mark.parametrize("field,value",[("tensor_parallel_size",2),("data_parallel_size",4),
    ("pipeline_parallel_size",2),("decode_context_parallel_size",2),
    ("prefill_context_parallel_size",2),("all2all_backend","deepep_high_throughput")])
def test_other_parallel_paths_unchanged(enabled,field,value):
    m,c=config();setattr(c.parallel_config,field,value);assert policy(m,c)==0

def test_overlap_and_dbo_are_rejected(enabled,monkeypatch):
    m,c=config();c.parallel_config.enable_dbo=True
    with pytest.raises(ValueError,match="DBO"):policy(m,c)
    c.parallel_config.enable_dbo=False
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM","0")
    with pytest.raises(ValueError,match="overlap disabled"):policy(m,c)

def test_bad_scheduler_counts_are_rejected():
    with pytest.raises(ValueError,match="negative"):plan(0,torch.tensor([-1,0]),4,128)
    with pytest.raises(ValueError,match="exceeds scheduler"):plan(5,torch.tensor([4,0]),4,128)
    with pytest.raises(ValueError,match="CPU scheduler"):
        plan(1,NS(device=NS(type="cuda")),4,128)


@pytest.mark.parametrize("chunk",[128,256,512])
def test_workspace_cap_keeps_scheduler_budget(enabled,monkeypatch,chunk):
    from vllm_hcu.model_executor.layers.fused_moe.step5_sp_workspace import sequence_parallel_dispatch_bound
    monkeypatch.setenv("VLLM_STEP5_LL_MOE_CHUNK_TOKENS",str(chunk))
    monkeypatch.setenv("VLLM_STEP5_LL_DISPATCH_ALIGNMENT","128")
    m,c=config();m.max_num_tokens=16384
    c.scheduler_config=NS(max_num_batched_tokens=16384,max_num_seqs=64)
    c.compilation_config=NS(cudagraph_capture_sizes=[64,128,256])
    c.speculative_config=NS(num_speculative_tokens=3)
    assert sequence_parallel_dispatch_bound(m,c)==chunk
    assert m.max_num_tokens==16384 and c.scheduler_config.max_num_batched_tokens==16384
    monkeypatch.setenv("VLLM_STEP5_LL_MOE_CHUNK_TOKENS","0")
    assert sequence_parallel_dispatch_bound(m,c)==4096
