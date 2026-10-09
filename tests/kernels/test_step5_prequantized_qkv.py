# SPDX-License-Identifier: Apache-2.0
"""Verify prequantized QKV's no-requantization and parameter contracts."""
from types import SimpleNamespace
import pytest
import torch
from vllm_hcu.model_executor.layers.step5_qkv_fp8 import Step5QKVFp8LinearMethod


class ScaledMMProbe:
    _hcu_fp8_backend = "lightop"

    def apply_scaled_mm(self, *, A, B, As, Bs, out_dtype, bias, output_shape):
        result = (A.float() @ B.float()) * As * Bs.t()
        if bias is not None:
            result = result + bias
        return result.to(out_dtype).reshape(output_shape)


def setup_method():
    torch.manual_seed(47)
    method = Step5QKVFp8LinearMethod()
    method.kernel = ScaledMMProbe()
    method.use_lightop_quant = True
    layer = SimpleNamespace(
        weight=torch.randn((16, 32)).to(torch.float8_e4m3fn),
        weight_scale=torch.rand((32, 1)),
    )
    return method, layer


def test_prequantized_values_scales_bias_and_empty(monkeypatch):
    method, layer = setup_method()
    # Any attempt to return to the input quantizer is an error.
    from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
    monkeypatch.setattr(lightop_fp8_runtime, "quantize",
                        lambda *a, **kw: pytest.fail("requantized FP8 QKV"))
    q = torch.randn((7, 16)).to(torch.float8_e4m3fn)
    scales = torch.rand((7, 1))
    bias = torch.randn(32)
    result = method.apply_prequantized(layer, q, scales, bias)
    expected = ((q.float() @ layer.weight.float()) * scales *
                layer.weight_scale.t() + bias).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert method.apply_prequantized(
        layer, q[:0].contiguous(), scales[:0].contiguous()
    ).shape == (0, 32)


@pytest.mark.parametrize("bad", ["dtype", "scale", "width", "unfinalized"])
def test_prequantized_contract_rejections(bad):
    method, layer = setup_method()
    q = torch.zeros((7, 16), dtype=torch.float8_e4m3fn)
    scales = torch.ones((7, 1))
    if bad == "dtype":
        q = q.bfloat16()
    elif bad == "scale":
        scales = scales.squeeze(1)
    elif bad == "width":
        q = q[:, :8].contiguous()
    else:
        method.kernel = None
    with pytest.raises((ValueError, RuntimeError)):
        method.apply_prequantized(layer, q, scales)
