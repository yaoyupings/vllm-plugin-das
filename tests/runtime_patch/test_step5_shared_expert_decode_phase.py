# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
from vllm.forward_context import BatchDescriptor
from vllm_hcu.model_executor.layers.fused_moe.step5_overlap import is_short_uniform_decode
from vllm_hcu.model_executor.layers.fused_moe.shared_experts import SharedExperts, SharedExpertsOrder

@pytest.mark.parametrize("uniform,tokens,requests,expected", [
    (True,16,16,True),(True,32,16,True),(True,48,16,True),(True,64,16,True),
    (False,64,16,False),(True,512,16,False),(True,65,16,False),
    (True,64,None,False),(True,0,0,False),
])
def test_phase_uses_real_host_descriptor(uniform,tokens,requests,expected):
    context=NS(batch_descriptor=BatchDescriptor(tokens,requests,uniform))
    assert is_short_uniform_decode(context) is expected
    assert is_short_uniform_decode(context) is expected

def test_missing_descriptor_does_not_enable_overlap():
    assert not is_short_uniform_decode(None)
    assert not is_short_uniform_decode(NS(batch_descriptor=None))

@pytest.mark.parametrize("uniform,can_mk,multistream_safe,expected", [
    (False,True,True,SharedExpertsOrder.MK_INTERNAL_OVERLAPPED),
    (False,False,True,SharedExpertsOrder.NO_OVERLAP),
    (True,True,True,SharedExpertsOrder.MULTI_STREAM_OVERLAPPED),
    (True,False,True,SharedExpertsOrder.MULTI_STREAM_OVERLAPPED),
    (True,True,False,SharedExpertsOrder.MK_INTERNAL_OVERLAPPED),
    (True,False,False,SharedExpertsOrder.NO_OVERLAP),
])
def test_phase_guard_limits_aux_stream_and_preserves_native_mk(monkeypatch,uniform,can_mk,multistream_safe,expected):
    from vllm_hcu.model_executor.layers.fused_moe import shared_experts as module
    import vllm.forward_context
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE","1")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD","128")
    monkeypatch.setattr(module.current_platform,"is_cuda_alike",lambda:True)
    context=NS(batch_descriptor=BatchDescriptor(64,16,uniform))
    monkeypatch.setattr(vllm.forward_context,"is_forward_context_available",lambda:True)
    monkeypatch.setattr(vllm.forward_context,"get_forward_context",lambda:context)
    item=SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(item)
    item._step5_decode_only=True
    item._stream=object()
    item._mk_can_overlap_shared_experts=lambda:can_mk
    item._is_multistream_safe=lambda:multistream_safe
    item._moe_config=NS(moe_parallel_config=NS(enable_eplb=False,use_fi_nvl_two_sided_kernels=False))
    assert item._determine_shared_experts_order(torch.empty(64,8))==expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP unavailable")
@pytest.mark.parametrize("uniform",[False,True])
@pytest.mark.parametrize("capture_descriptor",[False,True])
def test_native_aux_stream_and_graph_preserve_outputs(monkeypatch,uniform,capture_descriptor):
    import vllm.config
    import vllm.forward_context
    from vllm_hcu.model_executor.layers.fused_moe import shared_experts as module
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE","1")
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM","0")
    monkeypatch.setenv("VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY","1")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD","128")
    config=NS(model_config=NS(hf_text_config=NS(model_type="step5")))
    monkeypatch.setattr(vllm.config,"get_current_vllm_config_or_none",lambda:config)
    starts = torch.arange(17,dtype=torch.int32) * (4 if uniform else 32)
    metadata=NS(query_start_loc_cpu=starts,max_query_len=4 if uniform else 32,
                num_actual_tokens=int(starts[-1]),num_decodes=0,num_prefills=16)
    context=NS(batch_descriptor=BatchDescriptor(64,16,uniform) if capture_descriptor else None,
               attn_metadata={"dsa":metadata})
    monkeypatch.setattr(vllm.forward_context,"is_forward_context_available",lambda:True)
    monkeypatch.setattr(vllm.forward_context,"get_forward_context",lambda:context)
    torch.manual_seed(107)
    layer=torch.nn.Sequential(
        torch.nn.Linear(4096,1536,bias=False),torch.nn.SiLU(),
        torch.nn.Linear(1536,4096,bias=False),
    ).cuda().bfloat16()
    item=SharedExperts(layer,NS(moe_parallel_config=NS(
        enable_eplb=False,use_fi_nvl_two_sided_kernels=False)),False,lambda:False,lambda:True)
    x=torch.randn(64,4096,device="cuda",dtype=torch.bfloat16)
    out=torch.empty_like(x)
    routed=torch.empty_like(x)
    order=(SharedExpertsOrder.MULTI_STREAM_OVERLAPPED if uniform
           else SharedExpertsOrder.NO_OVERLAP)
    assert item._determine_shared_experts_order(x)==order
    assert item.requires_input_preservation(x)==uniform
    assert item.allows_inplace_routed_output(x,x)==(not uniform)
    def call():
        item.maybe_sync_shared_experts_stream(x)
        routed.copy_(x+1)
        item.forward(x,order)
        out.copy_(item.output)
    for _ in range(3):call()
    torch.cuda.synchronize()
    assert torch.equal(out,layer(x))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out,layer(x))
        assert torch.equal(routed,x+1)

@pytest.mark.parametrize("starts,max_query,tokens,expected", [
    ([0,4,8,12,16],4,16,True),
    ([0,1,2,3,4],1,4,True),
    ([0,4,8,8,8],4,8,True),
    ([0,1,5],4,5,False),
    ([0,128,256],128,256,False),
    ([0,4,3],4,3,False),
    ([0,4,8],4,9,False),
    ([1,5,9],4,9,False),
])
def test_full_capture_without_descriptor_uses_cpu_query_starts(starts,max_query,tokens,expected):
    metadata=NS(query_start_loc_cpu=torch.tensor(starts),max_query_len=max_query,
                num_actual_tokens=tokens,num_decodes=0,num_prefills=len(starts)-1)
    ctx=NS(batch_descriptor=None,attn_metadata={"dsa":metadata})
    assert is_short_uniform_decode(ctx) is expected

def test_metadata_fallback_never_reads_gpu_counts():
    class Poison:
        device=NS(type="cuda")
        def tolist(self):raise AssertionError("Device read would synchronize")
    metadata=NS(query_start_loc_cpu=Poison(),max_query_len=4,num_actual_tokens=16)
    assert not is_short_uniform_decode(NS(batch_descriptor=None,attn_metadata={"dsa":metadata}))

def test_explicit_nonuniform_descriptor_overrides_uniform_metadata():
    metadata=NS(query_start_loc_cpu=torch.tensor([0,4,8]),max_query_len=4,num_actual_tokens=8)
    ctx=NS(batch_descriptor=BatchDescriptor(8,2,False),attn_metadata={"dsa":metadata})
    assert not is_short_uniform_decode(ctx)

def test_mixed_attention_groups_do_not_enable_overlap():
    short=NS(query_start_loc_cpu=torch.tensor([0,4,8]),max_query_len=4,num_actual_tokens=8)
    long=NS(query_start_loc_cpu=torch.tensor([0,128,256]),max_query_len=128,num_actual_tokens=256)
    ctx=NS(batch_descriptor=None,attn_metadata={"short":short,"long":long})
    assert not is_short_uniform_decode(ctx)

@pytest.mark.parametrize('blocked', ['eplb', 'two_sided'])
def test_nonshort_fallback_respects_native_overlap_restrictions(monkeypatch,blocked):
    import vllm.forward_context
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','1')
    monkeypatch.setenv('VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE','1')
    monkeypatch.setattr(vllm.forward_context,'is_forward_context_available',lambda:True)
    monkeypatch.setattr(vllm.forward_context,'get_forward_context',lambda:NS(
        batch_descriptor=BatchDescriptor(512,16,False)))
    item=SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(item)
    item._step5_decode_only=True
    item._stream=object()
    item._mk_can_overlap_shared_experts=lambda:True
    item._moe_config=NS(moe_parallel_config=NS(
        enable_eplb=blocked=='eplb',all2all_backend='deepep_low_latency',
        use_fi_nvl_two_sided_kernels=blocked=='two_sided'))
    assert item._determine_shared_experts_order(torch.empty(512,8))==SharedExpertsOrder.NO_OVERLAP


def test_native_async_finalize_computes_shared_before_receive(monkeypatch):
    import vllm.forward_context
    from vllm_hcu.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','1')
    monkeypatch.setenv('VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE','1')
    monkeypatch.setattr(vllm.forward_context,'is_forward_context_available',lambda:True)
    monkeypatch.setattr(vllm.forward_context,'get_forward_context',lambda:NS(
        batch_descriptor=BatchDescriptor(512,16,False)))
    events=[]
    class Layer(torch.nn.Module):
        def forward(self,x):
            events.append('shared_compute')
            return x*2
    item=SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(item)
    item._step5_decode_only=True
    item._stream=object()  # Any aux-stream use would fail on this CPU test.
    item._mk_can_overlap_shared_experts=lambda:True
    item._moe_config=NS(moe_parallel_config=NS(
        enable_eplb=False,use_fi_nvl_two_sided_kernels=False))
    item._layer=Layer()
    item.enable_dbo=False
    item._output=[None,None]
    item._output_pending_on_stream=[False,False]
    class PrepareFinalize:
        def supports_async(self):return True
        def finalize_async(self,output,fused_out,*args):
            events.append('combine_launched')
            def hook():
                assert item._output[0] is not None
                events.append('recv_hook')
            def receiver():
                events.append('recv_complete')
                output.copy_(fused_out)
            return hook,receiver
    impl=object.__new__(FusedMoEKernelModularImpl)
    impl.prepare_finalize=PrepareFinalize()
    impl.fused_experts=NS(finalize_weight_and_reduce_impl=lambda:None)
    x=torch.randn(64,8)
    fused=x+1
    output=torch.empty_like(x)
    impl._finalize(output,fused,x,torch.ones(64,1),torch.zeros(64,1,dtype=torch.int32),
                   False,item,x)
    assert events==['combine_launched','shared_compute','recv_hook','recv_complete']
    torch.testing.assert_close(output,fused,rtol=0,atol=0)
    torch.testing.assert_close(item.output,x*2,rtol=0,atol=0)
