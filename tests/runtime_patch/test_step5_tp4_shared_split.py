from types import SimpleNamespace as NS
import pytest
import torch
import vllm.config
import vllm.forward_context
from vllm.forward_context import BatchDescriptor
from vllm_hcu.model_executor.layers.fused_moe.shared_experts import SharedExperts, SharedExpertsOrder
from vllm_hcu.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
from vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto import DeepEPAutoPrepareAndFinalize

class Layer(torch.nn.Module):
    _hcu_split_shared_supported = True
    def __init__(self, events):
        super().__init__(); self.events=events
        self.up=torch.nn.Linear(16,24,bias=False,dtype=torch.bfloat16)
        self.down=torch.nn.Linear(12,16,bias=False,dtype=torch.bfloat16)
    def _hcu_shared_dispatch(self,x):
        self.events.append("shared_up")
        a,b=self.up(x).chunk(2,-1)
        return torch.nn.functional.silu(a)*b
    def _hcu_shared_combine(self,x):
        self.events.append("shared_down")
        return self.down(x)
    def forward(self,x):
        return self._hcu_shared_combine(self._hcu_shared_dispatch(x))

@pytest.fixture
def setup(monkeypatch):
    cfg=NS(model_config=NS(hf_text_config=NS(model_type="step5")),
        additional_config={"hcu":{"deepep_auto":True}},
        parallel_config=NS(tensor_parallel_size=4,data_parallel_size=2,
            pipeline_parallel_size=1,prefill_context_parallel_size=1,
            decode_context_parallel_size=1,enable_expert_parallel=True,
            all2all_backend="deepep_low_latency",enable_dbo=False,
            _vllm_hcu_deepep_auto=False))
    context=NS(batch_descriptor=BatchDescriptor(64,16,True),
        deepep_auto_use_low_latency=True)
    monkeypatch.setenv("VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP","1")
    monkeypatch.setenv("VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY","0")
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM","1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS","0")
    monkeypatch.setattr(vllm.config,"get_current_vllm_config_or_none",lambda:cfg)
    monkeypatch.setattr(vllm.forward_context,"is_forward_context_available",lambda:True)
    monkeypatch.setattr(vllm.forward_context,"get_forward_context",lambda:context)
    pc=NS(enable_eplb=False,use_fi_nvl_two_sided_kernels=False)
    return cfg,context,pc

@pytest.mark.parametrize("tp,dp,dcp,auto",[(4,2,1,True),(4,2,1,False),(2,4,1,False),(2,4,2,False)])
def test_constructor_profiles_and_sidecar_fallback(setup,tp,dp,dcp,auto):
    cfg,ctx,pc=setup; p=cfg.parallel_config
    p.tensor_parallel_size=tp;p.data_parallel_size=dp;p.decode_context_parallel_size=dcp
    cfg.additional_config["hcu"]["deepep_auto"]=auto
    item=SharedExperts(Layer([]),NS(moe_parallel_config=pc),False,lambda:True,lambda:True)
    assert item._step5_split_auto is auto

@pytest.mark.parametrize("field,value",[
    ("tensor_parallel_size",1),("data_parallel_size",4),
    ("decode_context_parallel_size",2),("pipeline_parallel_size",2),
    ("prefill_context_parallel_size",2),("enable_expert_parallel",False),
    ("all2all_backend","deepep_high_throughput"),("enable_dbo",True)])
def test_reject_unsupported_profiles(setup,field,value):
    cfg,ctx,pc=setup;setattr(cfg.parallel_config,field,value)
    with pytest.raises(ValueError):
        SharedExperts(Layer([]),NS(moe_parallel_config=pc),False,lambda:True,lambda:True)

@pytest.mark.parametrize("auto_ll,uniform,m,prequant,expected",[
    (True,True,2,False,True),(True,True,4,False,True),
    (True,True,8,False,True),(True,True,16,False,True),
    (True,True,64,False,True),(False,True,16,False,False),
    (True,False,16,False,False),(True,True,0,False,False),
    (True,True,65,False,False),(True,True,16,True,False)])
def test_global_phase_and_split_state(setup,auto_ll,uniform,m,prequant,expected):
    cfg,ctx,pc=setup;ctx.deepep_auto_use_low_latency=auto_ll
    ctx.batch_descriptor=BatchDescriptor(max(1,m)*4,max(1,m),uniform)
    events=[];layer=Layer(events)
    item=SharedExperts(layer,NS(moe_parallel_config=pc),False,lambda:True,lambda:True)
    x=torch.randn(m,16,dtype=torch.bfloat16); reference=layer(x);events.clear()
    aq=(x,torch.ones(m,1)) if prequant else None
    assert item.maybe_begin_dispatch(x,aq) is expected
    if expected:
        assert events==["shared_up"]
        with pytest.raises(AssertionError): item.output
        with pytest.raises(AssertionError): item.maybe_begin_dispatch(x)
    item.forward(x,SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
    assert torch.equal(item.output,reference)
    assert item._split_dispatch_act is None and item._output[0] is None
    assert events==["shared_up","shared_down"]

@pytest.mark.parametrize("ll",[False,True])
def test_auto_delegate_modular_dispatch_and_combine_order(setup,ll):
    cfg,ctx,pc=setup;ctx.deepep_auto_use_low_latency=ll
    events=[];layer=Layer(events)
    item=SharedExperts(layer,NS(moe_parallel_config=pc),False,lambda:True,lambda:True)
    x=torch.randn(16,16,dtype=torch.bfloat16);reference=layer(x);events.clear()
    class Delegate:
        def __init__(self,label,hook):self.label=label;self.hook=hook
        def supports_async(self):return True
        def prepare_async(self,a,tw,ti,*args,**kw):
            events.append(self.label+"_send")
            def hook():events.append(self.label+"_recv")
            def receiver():return a,None,None,ti,tw
            return (hook,receiver) if self.hook else receiver
        def finalize_async(self,out,routed,*args):
            events.append("combine_send")
            def hook():events.append("combine_recv")
            def receiver():out.copy_(routed)
            return hook,receiver
    auto=DeepEPAutoPrepareAndFinalize(Delegate("ht",False),Delegate("ll",True))
    assert auto.begin_moe_call() is ll
    impl=object.__new__(FusedMoEKernelModularImpl)
    impl.prepare_finalize=auto
    impl.moe_parallel_config=NS(use_all2all_kernels=True)
    impl.fused_experts=NS(activation_format=lambda:"standard",
        quant_config=None,expects_unquantized_inputs=False,
        finalize_weight_and_reduce_impl=lambda:None)
    tw=torch.ones(16,1);ti=torch.zeros(16,1,dtype=torch.int32)
    impl._prepare(x,tw,ti,352,None,False,shared_experts=item,shared_experts_input=x)
    assert (item._split_dispatch_act is not None) is ll
    events.append("routed");routed=x*.3;out=torch.empty_like(x)
    impl._finalize(out,routed,x,tw,ti,False,item,x)
    assert torch.equal(item.output,reference) and torch.equal(out,routed)
    assert item._split_dispatch_act is None
    if ll:
        assert events==["ll_send","shared_up","ll_recv","routed","combine_send","shared_down","combine_recv"]
    else:
        assert events==["ht_send","routed","combine_send","shared_up","shared_down","combine_recv"]
