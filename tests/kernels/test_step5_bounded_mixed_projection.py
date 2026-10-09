# SPDX-License-Identifier: Apache-2.0
"""Bounded mixed projection keeps original large-M GEMMs inside an opaque op."""
from types import SimpleNamespace as NS
import torch
import pytest
from vllm_hcu.model_executor.layers import step5_full_projection as mod

@pytest.mark.parametrize("m,residual_width",[(3,16),(128,1792),(16384,16)])
def test_large_or_unqualified_shape_keeps_individual_native_projections(monkeypatch,m,residual_width):
    x=torch.zeros((m,4096),dtype=torch.bfloat16)
    aq=torch.zeros_like(x,dtype=torch.float8_e4m3fn)
    weight=torch.empty((3456,4096),dtype=torch.float8_e4m3fn).t()
    scale=torch.ones((m,1));ws=torch.ones((3456,1))
    residual=torch.empty((residual_width,4096),dtype=torch.bfloat16)
    widths=[]
    def qkv(a,b,sa,sb,dtype,bias):
        assert a is aq and b is weight and sa is scale and sb is ws
        return torch.full((m,3456),2.,dtype=dtype)
    def linear(value,w):
        assert value is x
        widths.append(w.shape[0])
        return torch.full((m,w.shape[0]),len(widths)+2.,dtype=x.dtype)
    monkeypatch.setattr(torch.ops.vllm,"hcu_channel_fp8_target_triton_scaled_mm",qkv,raising=False)
    monkeypatch.setattr(torch.nn.functional,"linear",linear)
    monkeypatch.setattr(mod,"projection_config",lambda *a:(_ for _ in ()).throw(AssertionError("Decode tuning used for Prefill")))
    out=mod._mixed_projection(aq,scale,x,weight,ws,residual,True)
    expected=[16] if residual_width==16 else [1024,256,256,4,16]
    assert widths==expected and out.shape==(m,3456+residual_width)
    assert (out[:,:3456]==2).all()
    start=3456
    for index,width in enumerate(expected):
        assert (out[:,start:start+width]==index+3).all()
        start+=width
    if start<out.shape[1]:assert (out[:,start:]==0).all()

def test_full_projection_symbolic_trace_keeps_bounded_op(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
    mod.ensure_registered()
    lightop_fp8_runtime.ensure_registered(torch.float8_e4m3fn,direct_register_custom_op)
    def forward(x,w,ws,r):
        block=NS(qkv_kn=w,qkv_scale=ws,residual_nk=r,plan=NS(qkv_width=3456,residual_width=16))
        owner=NS(step5_full_projection_ready=True,step5_full_projection_block=block,
                 step5_full_projection_offsets={"gate":(0,16)},indexer=None,
                 step5_full_projection_decode_only=True)
        return mod.full_projection(owner,x)
    graph=make_fx(forward,tracing_mode="symbolic")(
        torch.empty((128,4096),dtype=torch.bfloat16),
        torch.empty((3456,4096),dtype=torch.float8_e4m3fn).t(),
        torch.ones((3456,1)),torch.empty((16,4096),dtype=torch.bfloat16))
    nodes=[n for n in graph.graph.nodes if str(n.target)=="vllm.step5_full_mixed_projection.default"]
    assert len(nodes)==1 and nodes[0].args[-1] is True
