
# SPDX-License-Identifier: Apache-2.0
"""Opt-in routing and topology contracts; GPU math is qualified separately."""
from types import SimpleNamespace as NS
import pytest
import torch
from vllm_hcu.model_executor.layers import step5_qkv_fp8 as mod

class Probe:
    def __init__(self): self.calls=[]
    def apply_scaled_mm(self, **kw):
        self.calls.append(kw)
        return torch.full(kw["output_shape"], 2., dtype=torch.bfloat16)

def setup(monkeypatch, enabled=True):
    monkeypatch.setenv("VLLM_STEP5_TP4_SMALL_QKV_FP8", "1" if enabled else "0")
    method=mod.Step5QKVFp8LinearMethod()
    method.kernel=Probe(); method.use_lightop_quant=True
    layer=NS(weight=torch.empty((4096,3456),dtype=torch.float8_e4m3fn),
             weight_scale=torch.ones((3456,1)))
    quant_calls=[]; fused_calls=[]
    def quant(x,*args):
        quant_calls.append(x.shape)
        return torch.zeros_like(x,dtype=torch.float8_e4m3fn),torch.ones((x.shape[0],1))
    def fused(x,w,s):
        fused_calls.append(x.shape)
        return torch.full((x.shape[0],w.shape[1]),3.,dtype=torch.bfloat16)
    monkeypatch.setattr(mod.lightop_fp8_runtime,"quantize",quant)
    monkeypatch.setattr(torch.ops.vllm,"step5_tp4_small_qkv_fp8",fused,raising=False)
    return method,layer,quant_calls,fused_calls

def test_small_qkv_keeps_original_shape_and_bypasses_quantizer(monkeypatch):
    method,layer,q,f=setup(monkeypatch)
    x=torch.ones((2,4,4096),dtype=torch.bfloat16)
    out=method.apply(layer,x)
    assert out.shape==(2,4,3456) and (out==3).all()
    assert f==[torch.Size([8,4096])] and q==[] and method.kernel.calls==[]
    assert (x==1).all()

@pytest.mark.parametrize("enabled,m,bias",[(False,4,False),(True,4,True)])
def test_small_qkv_falls_back_for_disabled_shape_or_bias(monkeypatch,enabled,m,bias):
    method,layer,q,f=setup(monkeypatch,enabled)
    out=method.apply(layer,torch.ones((m,4096),dtype=torch.bfloat16),
                     torch.ones(3456,dtype=torch.bfloat16) if bias else None)
    assert out.shape==(m,3456) and (out==2).all()
    assert len(q)==1 and f==[] and len(method.kernel.calls)==1

@pytest.mark.parametrize("value",["true","2",""])
def test_small_qkv_rejects_invalid_feature_value(monkeypatch,value):
    monkeypatch.setenv("VLLM_STEP5_TP4_SMALL_QKV_FP8",value)
    with pytest.raises(ValueError,match="must be 0 or 1"):
        mod.Step5QKVFp8LinearMethod()

@pytest.mark.parametrize("tp,dp,ep,dcp,width",[(2,4,True,1,3456),(4,2,True,2,3456),(4,2,False,1,3456),(4,2,True,1,6912)])
def test_small_qkv_rejects_unsupported_topology_before_loading(monkeypatch,tp,dp,ep,dcp,width):
    monkeypatch.setenv("VLLM_STEP5_TP4_SMALL_QKV_FP8","1")
    monkeypatch.setattr(mod.henvs,"custom_ops_enabled",lambda:True)
    config=NS(parallel_config=NS(tensor_parallel_size=tp,data_parallel_size=dp,
        enable_expert_parallel=ep,decode_context_parallel_size=dcp))
    monkeypatch.setattr(mod,"get_current_vllm_config",lambda:config)
    method=mod.Step5QKVFp8LinearMethod()
    layer=NS(input_size=4096,output_partition_sizes=[width])
    with pytest.raises(ValueError,match="TP4 DP2 EP8 DCP1"):
        method.process_weights_after_loading(layer)

def test_small_qkv_registers_typed_custom_op():
    from vllm_hcu.ops.step5_small_qkv_fp8 import ensure_registered
    ensure_registered()
    assert torch.ops.vllm.step5_tp4_small_qkv_fp8.default._schema.name == "vllm::step5_tp4_small_qkv_fp8"

@pytest.mark.parametrize("m", [3, 128, 16384])
def test_enabled_method_always_traces_dispatch_op(monkeypatch, m):
    method, layer, quantized, fused = setup(monkeypatch)
    out = method.apply(layer, torch.empty((m,4096), dtype=torch.bfloat16))
    assert out.shape == (m,3456)
    assert fused == [torch.Size([m,4096])] and quantized == []

@pytest.mark.parametrize("m,expected", [(3,"native"), (128,"native"), (16384,"native"), (4,"fused"), (32,"fused")])
def test_runtime_dispatch_selects_qualified_shapes(monkeypatch,m,expected):
    from vllm_hcu.ops import step5_small_qkv_fp8 as op
    monkeypatch.setattr(op, "_validate_runtime_layout", lambda *args:None)
    calls=[]
    def native(x,*args):
        calls.append("native")
        return torch.zeros((m,3456),dtype=torch.bfloat16)
    def fused(x,*args,**kwargs):
        calls.append("fused")
        return torch.zeros((m,3456),dtype=torch.bfloat16)
    monkeypatch.setattr(op,"_native_qkv",native)
    monkeypatch.setattr(op,"fused_qkv",fused)
    x=torch.empty((m,4096),dtype=torch.bfloat16)
    result=op._run_small_qkv(x,torch.empty((4096,3456)),torch.ones((3456,1)))
    assert result.shape == (m,3456) and calls == [expected]

def test_symbolic_trace_keeps_runtime_dispatch_op(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx
    from vllm_hcu.ops.step5_small_qkv_fp8 import ensure_registered
    ensure_registered()
    monkeypatch.setenv("VLLM_STEP5_TP4_SMALL_QKV_FP8","1")
    method=mod.Step5QKVFp8LinearMethod()
    method.kernel=Probe();method.use_lightop_quant=True
    layer=NS(weight=torch.empty((4096,3456),dtype=torch.float8_e4m3fn),weight_scale=torch.ones((3456,1)))
    graph=make_fx(lambda x,w,ws:method.apply(NS(weight=w,weight_scale=ws),x),
                  tracing_mode="symbolic")(
        torch.empty((128,4096),dtype=torch.bfloat16),layer.weight,layer.weight_scale)
    targets=[str(n.target) for n in graph.graph.nodes]
    assert "vllm.step5_tp4_small_qkv_fp8.default" in targets
    assert not any("lightop_per_token_quant_fp8" in t for t in targets)
