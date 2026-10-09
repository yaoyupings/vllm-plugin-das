# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_hcu.model_executor.layers.fused_moe.experts import dpsk_v4_deep_gemm_moe as impl

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA/HIP unavailable")

def expert(max_tokens,dispatchers=1):
    obj=object.__new__(impl.DeepEPDeepGemmMaskedExperts)
    obj.max_num_tokens=max_tokens
    obj.num_dispatchers=dispatchers
    obj._step5_ll_tuning=True
    obj.quant_config=NS(use_int8_w8a8=False,gemm1_clamp_limit=7.,
                        w1_scale=None,w2_scale=None)
    return obj

@pytest.mark.parametrize("enabled",[False,True])
@pytest.mark.parametrize("tuned,int8,experts,n,k",[
    (True,False,44,3072,4096),(False,False,44,3072,4096),
    (True,True,44,3072,4096),(True,False,43,3072,4096),
    (True,False,44,4096,3072),
])
def test_only_qualified_shape_reduces_gateup_workspace(monkeypatch,enabled,tuned,int8,experts,n,k):
    obj=expert(128,8);obj._step5_ll_tuning=tuned;obj.quant_config.use_int8_w8a8=int8
    monkeypatch.setenv("VLLM_STEP5_LL_COMPACT_WORKSPACE","1" if enabled else "0")
    w13,w2,out=obj.workspace_shapes(32,n,k,8,352,experts,None,MoEActivation.SWIGLUSTEP)
    expected=n if enabled and tuned and not int8 and (experts,n,k)==(44,3072,4096) else max(n,k)
    assert w2==(experts,1024,expected)
    assert out==(experts,1024,k)
    assert w13==(experts,1024,max(k,n//2))

@pytest.mark.parametrize("capacity",[128,512])
def test_native_apply_graph_counts_and_storage_canaries(monkeypatch,capacity):
    torch.manual_seed(717)
    e,n,k=44,3072,4096
    obj=expert(capacity)
    w13=(torch.randn((e,n,k),device="cuda",dtype=torch.bfloat16)*.05).to(torch.float8_e4m3fn)
    w2=(torch.randn((e,k,n//2),device="cuda",dtype=torch.bfloat16)*.05).to(torch.float8_e4m3fn)
    obj._deepgemm_w13=impl.marlin_fp8_masked_weight(w13)
    obj._deepgemm_w2=impl.marlin_fp8_masked_weight(w2)
    obj.quant_config.w1_scale=torch.ones((e,n),device="cuda",dtype=torch.float32)
    obj.quant_config.w2_scale=torch.ones((e,k),device="cuda",dtype=torch.float32)
    a=(torch.randn((e,capacity,k),device="cuda",dtype=torch.bfloat16)*.1).to(torch.float8_e4m3fn)
    scales=torch.ones((e,capacity),device="cuda",dtype=torch.float32)
    counts=torch.zeros(e,device="cuda",dtype=torch.int32)
    ctx=NS(dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([64]*4)))
    monkeypatch.setattr(impl,"get_forward_context",lambda:ctx)
    # Preserve the native allocation contract: workspace13 and down output
    # share an arena, while workspace2 is simultaneous and independent.
    sentinel=-123.
    old_common=torch.full((e*capacity*k+4096,),sentinel,device="cuda",dtype=torch.bfloat16)
    new_common=torch.full_like(old_common,sentinel)
    old_gate=torch.full_like(old_common,sentinel)
    new_gate=torch.full((e*capacity*n+4096,),sentinel,device="cuda",dtype=torch.bfloat16)
    def run(common,gate):
        output=common[:e*capacity*k].view(e,capacity,k)
        obj.apply(output,a,obj._deepgemm_w13,obj._deepgemm_w2,None,
          torch.zeros((1,8),device="cuda",dtype=torch.int32),
          MoEActivation.SWIGLUSTEP,352,None,scales,None,
          common[:e*capacity*k].view(e,capacity,k),gate[:-4096],
          NS(expert_num_tokens=counts),False)
        return output
    # Read actual shape policies and ensure narrow allocations are used.
    monkeypatch.setenv("VLLM_STEP5_LL_COMPACT_WORKSPACE","1")
    assert obj.workspace_shapes(capacity,n,k,8,352,e,None,MoEActivation.SWIGLUSTEP)[1]==(e,capacity,n)
    counts[:8]=torch.tensor([1,2,3,4,8,16,32,64],device="cuda",dtype=torch.int32)
    run(old_common,old_gate);run(new_common,new_gate)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=run(new_common,new_gate)
    cases=[[0,1,2,3,4,8,16,64], [0,capacity,capacity//2,capacity-1,1,2,4,8], [0]*8]
    for values in cases:
        counts.zero_();counts[:8]=torch.tensor(values,device="cuda",dtype=torch.int32)
        a.copy_((torch.randn(a.shape,device="cuda",dtype=torch.bfloat16)*.1).to(torch.float8_e4m3fn))
        old_common.fill_(sentinel);new_common.fill_(sentinel)
        old_gate.fill_(sentinel);new_gate.fill_(sentinel)
        graph.replay()
        reference=run(old_common,old_gate)
        active=torch.arange(capacity,device="cuda")[None,:]<counts[:,None]
        torch.testing.assert_close(captured[active],reference[active],rtol=0,atol=0)
        assert (new_gate[-4096:]==sentinel).all()
        assert (new_common[-4096:]==sentinel).all()
