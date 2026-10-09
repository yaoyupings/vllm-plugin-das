# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_hcu.step5_config import Step5Config, Step5TextConfig
from vllm_hcu.models.step5 import (
    _fp32_gemma_rms_norm,
    dsa_layer_indices,
    map_step5_checkpoint_weights,
)
from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAGeometry


def test_config_preserves_sparse_and_mtp_metadata():
    c = Step5Config(
        text_config={
            "num_hidden_layers": 4,
            "moe_layer_list": [3],
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"] * 4,
            "partial_rotary_factors": [1.0, 1.0, 1.0, 1 / 3],
            "sparse_config": {
                "enabled": True,
                "compression_method": "csa_block_compress",
            },
        }
    )
    assert isinstance(c.text_config, Step5TextConfig)
    assert c.get_text_config().num_key_value_heads == 4
    assert c.text_config.partial_rotary_factors[3] == 1 / 3
    assert len(c.text_config.layer_types) == 7
    assert (
        Step5Config.from_dict(c.to_dict()).text_config.sparse_config
        == c.text_config.sparse_config
    )


def test_router_uses_reference_sigmoid_default():
    assert Step5TextConfig().moe_router_activation == "sigmoid"
    assert (
        Step5TextConfig(moe_router_activation="softmax").moe_router_activation
        == "softmax"
    )


def test_dsa_layers_exclude_mtp_and_sliding_attention():
    c = Step5TextConfig(
        sparse_config={"enabled": True, "apply_to_layer_types": ["full_attention"]}
    )
    assert dsa_layer_indices(c) == list(range(3, 92, 4))
    c.sparse_config["enabled"] = False
    assert dsa_layer_indices(c) == []


def test_sparse_parameters_and_fp8_scale_values_are_preserved():
    t = torch.tensor([0.125, 0.5])
    names = [
        "model.layers.3.self_attn.ssmax_s",
        "model.layers.3.self_attn.sparse_indexer_q.weight",
    ]
    for name in names:
        assert list(map_step5_checkpoint_weights([(name, t)]))[0] == (name, t)
    name = "model.layers.3.moe.down_proj.weight_scale_inv"
    assert list(map_step5_checkpoint_weights([(name, t)]))[0] == (name, t)


@pytest.mark.parametrize("omit_down", [False, True])
def test_packed_fp8_scales_load_and_missing_parameters_fail(monkeypatch, omit_down):
    import vllm_hcu.models.step5 as implementation

    # Isolate the packed-checkpoint path. Per-expert mapping is independent of it.
    monkeypatch.setattr(
        implementation, "fused_moe_make_expert_params_mapping", lambda *a, **k: []
    )
    model = implementation.Step5Model.__new__(implementation.Step5Model)
    torch.nn.Module.__init__(model)
    model.config = Step5TextConfig(num_hidden_layers=4, moe_num_experts=2)
    model.moe_num_experts = 2
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
    experts = torch.nn.Module()
    model.layers[3].moe = torch.nn.Module()
    model.layers[3].moe.experts = torch.nn.Module()
    model.layers[3].moe.experts.routed_experts = experts
    experts.w13_weight_scale_inv = torch.nn.Parameter(
        torch.zeros(2, 4, 1), requires_grad=False
    )
    experts.w2_weight_scale_inv = torch.nn.Parameter(
        torch.zeros(2, 2, 1), requires_grad=False
    )
    model.norm = torch.nn.Linear(1, 1, bias=False)
    parent = implementation.Step5ForCausalLM.__new__(implementation.Step5ForCausalLM)
    torch.nn.Module.__init__(parent)
    parent.model = model
    parent.lm_head = torch.nn.Linear(1, 1, bias=False)

    def load(param, tensor, name, shard_id, expert_id):
        if shard_id == "w2":
            param.data[expert_id].copy_(tensor)
        else:
            start = 0 if shard_id == "w1" else 2
            param.data[expert_id, start : start + 2].copy_(tensor)

    experts.w13_weight_scale_inv.weight_loader = load
    experts.w2_weight_scale_inv.weight_loader = load
    gate = torch.tensor([0.01, 0.02, 0.03, 0.04]).view(2, 2, 1)
    up, down = gate * 2, gate * 3
    weights = [
        ("model.layers.3.moe.gate_proj.weight_scale_inv", gate),
        ("model.layers.3.moe.up_proj.weight_scale_inv", up),
    ]
    if not omit_down:
        weights.append(("model.layers.3.moe.down_proj.weight_scale_inv", down))
    # Repeated model prefix is valid in the native streaming loader.
    weights += [
        ("lm_head.weight", torch.ones(1, 1)),
        ("model.norm.weight", torch.ones(1, 1)),
    ]
    if omit_down:
        with pytest.raises(
            RuntimeError, match="did not load parameters.*w2_weight_scale_inv"
        ):
            parent.load_weights(weights)
    else:
        assert len(parent.load_weights(weights)) == 4
        torch.testing.assert_close(
            experts.w13_weight_scale_inv, torch.cat([gate, up], dim=1)
        )
        torch.testing.assert_close(experts.w2_weight_scale_inv, down)


def test_geometry_matches_checkpoint():
    g = Step4DSAGeometry.from_config(
        {
            "proxy_dim": 256,
            "topk": 512,
            "region_block_size": 8,
            "num_provider_groups": 4,
        }
    )
    assert (g.proxy_dim, g.topk, g.region_size, g.num_provider_groups) == (
        256,
        512,
        8,
        4,
    )


def test_expert_metadata_updates_blocks_while_exposing_runners():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from vllm_hcu.models.step5 import FusedMoEBlock, Step5ForCausalLM

    model = Step5ForCausalLM.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(model)
    block = FusedMoEBlock.__new__(FusedMoEBlock)
    torch.nn.Module.__init__(block)
    block.experts = SimpleNamespace(update_expert_map=Mock())
    model.moe_layers = [block.experts]
    model.moe_mlp_layers = [block]
    model.num_logical_experts = 4
    model.num_local_physical_experts = 2
    model.update_physical_experts_metadata(6, 2)
    assert block.n_physical_experts == 6
    assert block.n_local_physical_experts == 2
    assert block.n_redundant_experts == 2
    block.experts.update_expert_map.assert_called_once_with()


def test_norm_keeps_fp32_checkpoint_and_reference_rounding():
    with set_current_vllm_config(VllmConfig()):
        _check_norm_rounding()


def _check_norm_rounding():
    layer = _fp32_gemma_rms_norm(4, 1e-5)
    layer.weight.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    for dtype in [torch.float32, torch.bfloat16]:
        x = torch.tensor([[1.001, 2.002, 3.003, 4.004]], dtype=dtype)
        expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
            * (1 + layer.weight)
        ).to(dtype)
        assert layer.weight.dtype == torch.float32
        assert layer(x).dtype == dtype
        torch.testing.assert_close(layer(x), expected)



@pytest.mark.parametrize("tokens", [3, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_norm_quant_fusion_preserves_residual_and_native_fallback(monkeypatch, tokens, dtype):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from vllm_hcu.models import step5 as implementation

    x = torch.randn(tokens, 4, dtype=dtype)
    original = x.clone()
    positions = torch.arange(tokens)
    native_norm = Mock(side_effect=lambda h: h.to(torch.bfloat16))
    attention = Mock(side_effect=lambda **kw: torch.full_like(kw["hidden_states"], .125))
    def fused(h, w, epsilon):
        return (h.to(torch.bfloat16),
                torch.zeros_like(h, dtype=torch.float8_e4m3fn),
                torch.ones(tokens, 1))
    quantize = Mock(side_effect=fused)
    monkeypatch.setattr(implementation, "step5_gemma_norm_fp8_quant", quantize)
    layer = SimpleNamespace(
        fp32_residual_connection=True, fused_input_norm_fp8=True,
        compute_dtype=torch.bfloat16,
        input_layernorm=Mock(side_effect=native_norm),
        post_attention_layernorm=lambda h: h.to(torch.bfloat16),
        self_attn=attention, _feed_forward=lambda h: torch.zeros_like(h),
    )
    layer.input_layernorm.weight = torch.zeros(4)
    layer.input_layernorm.variance_epsilon = 1e-5
    actual = implementation.Step5DecoderLayer.forward(layer, positions, x)
    use_fused = tokens in (4, 8, 16, 32, 64)
    assert quantize.call_count == int(use_fused)
    assert native_norm.call_count == int(not use_fused)
    kwargs = attention.call_args.kwargs
    assert ("qkv_quantized" in kwargs) == use_fused
    assert ("qkv_scale" in kwargs) == use_fused
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, original.float() + .125, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
