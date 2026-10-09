# SPDX-License-Identifier: Apache-2.0
"""Cache declarations, aliasing and compilation contract for DCP2 Attention."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_hcu.model_executor.layers.step5_dcp2_attention import (
    Step5DCP2Attention, Step5DCP2AttentionBackend,
)
from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec


def make_layer(window):
    # Cache declarations/binding do not require model weights or distributed groups.
    layer = Step5DCP2Attention.__new__(Step5DCP2Attention)
    torch.nn.Module.__init__(layer)
    layer.attn_type = AttentionType.DECODER
    layer.kv_cache_dtype = "auto"
    layer.attn_backend = Step5DCP2AttentionBackend
    layer.sliding_window = window
    layer.num_kv_heads = 4
    layer.head_size = layer.head_size_v = 192
    layer.kv_cache_torch_dtype = torch.bfloat16
    return layer


@pytest.mark.parametrize("window", [None, 512])
def test_all_dcp2_layers_declare_four_cached_heads(window):
    layer = make_layer(window)
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, skip_page_size_padded=None))
    spec = layer.get_kv_cache_spec(cfg)
    assert isinstance(spec, Step5DCP2SlidingWindowSpec if window else FullAttentionSpec)
    assert spec.block_size == 64
    assert spec.num_kv_heads == 4
    assert spec.head_size == spec.head_size_v == 192
    if window:
        assert not spec.prefix_cacheable
        assert spec.sliding_window == 512


@pytest.mark.parametrize("window", [None, 512])
def test_sparse_and_window_alias_actual_v2_cache_storage(window):
    layer = make_layer(window)
    raw = torch.zeros(2,64,4,384,dtype=torch.bfloat16)
    allocated = raw.permute(0,2,1,3)
    layer.bind_kv_cache(allocated)
    view = layer.kv_cache
    assert view.shape == (2,2,64,4,192)
    assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key, value = view[0].flatten(0,1), view[1].flatten(0,1)
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key[64,3,191] = 17
    value[63,2,0] = -19
    assert raw[1,0,3,191] == 17
    assert raw[0,63,2,192] == -19


def test_fake_op_retains_local32_head_output_and_dtype():
    with FakeTensorMode():
        query = torch.empty(64,32,192,dtype=torch.bfloat16)
        key = value = torch.empty(64,2,192,dtype=torch.bfloat16)
        positions = torch.empty(64,dtype=torch.int64)
        token = torch.empty(1,dtype=torch.int32)
        result = torch.ops.vllm.step5_dcp2_attention(
            query,key,value,positions,token,"model.layers.3.self_attn.attn")
        assert result.shape == query.shape
        assert result.dtype == query.dtype


def test_piecewise_graph_requires_runtime_op_split(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm_hcu.model_executor.layers import step5_dcp2_attention as adapter

    monkeypatch.setattr(adapter, "get_current_vllm_config", lambda: SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, splitting_ops=[])))
    with pytest.raises(RuntimeError, match="splitting_ops"):
        Step5DCP2Attention(32,192,192**-.5,num_kv_heads=4)


def test_step5_config_registers_runtime_split_before_compilation(monkeypatch):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.step5_config import Step5TextConfig

    monkeypatch.setenv("VLLM_STEP5_DCP2_RUNTIME", "1")
    monkeypatch.delenv("VLLM_STEP4_DISABLE_DSA", raising=False)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])
    Step5TextConfig(sparse_config={"enabled": True})
    assert CompilationConfig._attention_ops == [
        "vllm::step4_dsa_index", "vllm::step5_dcp2_attention"]


@pytest.mark.parametrize("window", [None,512])
def test_real_backend_initializer_admits_qualified_window(monkeypatch,window):
    from vllm_hcu.model_executor.layers import step4_dsa
    from vllm_hcu.model_executor.layers.step5_dcp2_attention import Step5DCP2Impl
    monkeypatch.setattr(step4_dsa, "_require_splitting_op",lambda:None)
    impl=Step5DCP2Impl(32,192,192**-.5,num_kv_heads=4,sliding_window=window)
    assert impl.sliding_window==window
    assert impl.num_heads==32 and impl.num_kv_heads==4
    with pytest.raises(RuntimeError,match="gathered-head"):
        impl.forward()


@pytest.mark.parametrize("change",[
    {"sliding_window":256},{"num_heads":64},{"num_kv_heads":2},
    {"head_size":128},{"kv_cache_dtype":"fp8"},{"attn_type":"encoder"},
])
def test_real_backend_rejects_unqualified_geometry(change):
    from vllm_hcu.model_executor.layers.step5_dcp2_attention import Step5DCP2Impl
    kwargs=dict(num_heads=32,head_size=192,scale=192**-.5,num_kv_heads=4)
    kwargs.update(change)
    with pytest.raises(ValueError,match="backend requires"):
        Step5DCP2Impl(**kwargs)


def test_internal_lse_bridge_validation_does_not_advertise_stock_lse(monkeypatch):
    from vllm_hcu.model_executor.layers import step4_dsa
    from vllm_hcu.model_executor.layers.step5_dcp2_attention import (
        Step5DCP2Impl, validate_dcp2_attention_layers,
    )
    monkeypatch.setattr(step4_dsa,"_require_splitting_op",lambda:None)
    sparse=make_layer(None)
    window=make_layer(512)
    for layer in (sparse,window):
        layer.impl=Step5DCP2Impl(32,192,192**-.5,num_kv_heads=4,
                                sliding_window=layer.sliding_window)
        assert not layer.impl.can_return_lse_for_decode
    validate_dcp2_attention_layers({"sparse":sparse,"window":window})
    with pytest.raises(ValueError,match="incompatible"):
        validate_dcp2_attention_layers({"stock":SimpleNamespace(impl=window.impl)})
    with pytest.raises(ValueError,match="requires"):
        validate_dcp2_attention_layers({})


@pytest.mark.parametrize("rotary_pairs", [32, 96])
def test_raw_qkv_fake_op_preserves_full_graph_output(rotary_pairs):
    # Optional raw inputs preserve the output contract used by Inductor.
    with FakeTensorMode():
        raw = torch.empty(64, 6912, dtype=torch.bfloat16)
        query, key, value = raw.split([6144, 384, 384], dim=-1)
        query = query.view(64, 32, 192)
        key, value = key.view(64, 2, 192), value.view(64, 2, 192)
        positions = torch.empty(64, dtype=torch.int64)
        weights = torch.empty(192, dtype=torch.float32)
        rope_cache = torch.empty(69632, 2 * rotary_pairs, dtype=torch.bfloat16)
        result = torch.ops.vllm.step5_dcp2_attention(
            query, key, value, positions, None, "model.layers.3.self_attn.attn",
            raw, weights, weights, rope_cache, rotary_pairs, 1e-6,
        )
        assert result.shape == (64, 32, 192)
        assert result.dtype == torch.bfloat16


def test_norm_exchange_requires_explicit_fused_exchange(monkeypatch):
    from vllm_hcu.model_executor.layers import step4_dsa
    from vllm_hcu.model_executor.layers.step5_dcp2_attention import Step5DCP2Impl

    monkeypatch.setattr(step4_dsa, "_require_splitting_op", lambda: None)
    monkeypatch.delenv("VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE", raising=False)
    monkeypatch.delenv("VLLM_STEP5_DCP2_FUSED_EXCHANGE", raising=False)
    impl = Step5DCP2Impl(32, 192, 192**-.5, num_kv_heads=4)
    assert not impl.fused_norm_exchange

    monkeypatch.setenv("VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE", "1")
    with pytest.raises(ValueError, match="requires fused"):
        Step5DCP2Impl(32, 192, 192**-.5, num_kv_heads=4)
    monkeypatch.setenv("VLLM_STEP5_DCP2_FUSED_EXCHANGE", "1")
    impl = Step5DCP2Impl(32, 192, 192**-.5, num_kv_heads=4)
    assert impl.fused_norm_exchange
