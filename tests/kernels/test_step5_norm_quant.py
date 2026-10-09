# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the isolated Step5 norm/quant custom-op boundary."""
import pytest
import torch

from vllm_hcu.ops.step5_norm_quant import step5_gemma_norm_fp8_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires native HCU GPU"
)


@pytest.mark.parametrize("layout", range(5))
def test_graph_replay_updates_and_preserves_inputs(layout):
    torch.manual_seed(47)
    x = torch.randn((7, 4096), device="cuda", dtype=torch.float32)
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.03
    x_before, weight_before = x.clone(), weight.clone()
    eager = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    assert eager[0].dtype == torch.bfloat16
    assert eager[1].dtype == torch.float8_e4m3fn
    assert eager[2].dtype == torch.float32 and eager[2].shape == (7, 1)
    assert all(y.data_ptr() != x.data_ptr() for y in eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
    # Includes partial row CTAs, changing weights, signs and tiny scales.
    for value in (0., 1., -1., 1e-14, -1e-14):
        x.fill_(value)
        weight.add_(0.0001)
        before = weight.clone()
        graph.replay()
        expected = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(
                actual.float(), reference.float(), rtol=0, atol=0
            )
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        if value == 0.:
            assert torch.count_nonzero(captured[1].float()) == 0
            assert torch.count_nonzero(captured[2]) == 0
        else:
            assert torch.all(captured[2] > 0)


def test_compiled_dynamic_custom_op_and_empty():
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.03
    compiled = torch.compile(
        lambda x, w: step5_gemma_norm_fp8_quant(x, w, 1e-5, 2),
        fullgraph=True, dynamic=True,
    )
    for m in (3, 9, 0):
        x = torch.randn((m, 4096), device="cuda", dtype=torch.float32)
        expected = step5_gemma_norm_fp8_quant(x, weight, 1e-5, 2)
        actual = compiled(x, weight)
        assert actual[2].shape == (m, 1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


@pytest.mark.parametrize("layout,epsilon", [
    (-2, 1e-5), (5, 1e-5), (0, 0.), (0, float("nan")),
])
def test_invalid_configuration(layout, epsilon):
    x = torch.empty((1, 4096), device="cuda", dtype=torch.float32)
    w = torch.empty((4096,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x, w, epsilon, layout)


def test_rejects_bf16_norm_weight_and_noncontiguous_input():
    x = torch.randn((2, 4096), device="cuda", dtype=torch.float32)
    w = torch.randn((4096,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x, w.bfloat16(), 1e-5, 0)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x.t().contiguous().t(), w, 1e-5, 0)

@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 64, 128, 256, 384])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_auto_profile_fp64_norm_oracle(m, dtype):
    torch.manual_seed(4747 + m)
    x = torch.randn((m, 4096), device="cuda", dtype=dtype)
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.03
    before = x.clone()
    hidden, quantized, scale = step5_gemma_norm_fp8_quant(x, weight, 1e-5)
    # Preserve the required FP32 Gamma addition, then independently evaluate
    # variance/rsqrt/products in FP64. No loose BF16 relative tolerance.
    gamma = (weight + 1.).double()
    exact = x.double() * torch.rsqrt(
        x.double().square().mean(-1, keepdim=True) + 1e-5
    ) * gamma
    nearest = exact.bfloat16()
    mask = hidden != nearest
    if mask.any():
        # Only adjacent BF16 values near an independently computed midpoint
        # are permitted for FP32 normalization reassociation.
        a = hidden.view(torch.int16)[mask].int()
        b = nearest.view(torch.int16)[mask].int()
        assert ((a - b).abs() == 1).all()
        midpoint = (hidden.double()[mask] + nearest.double()[mask]) * 0.5
        assert ((exact[mask] - midpoint).abs() <=
                8 * torch.finfo(torch.float32).eps *
                exact[mask].abs().clamp_min(1e-30)).all()
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    expected_scale = (hidden.double().abs().amax(-1, keepdim=True) / 448.).float()
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    # Compare FP8 rounding to an independent exact scale, permitting only
    # adjacent E4M3 representatives at their midpoint.
    exact_q = hidden.double() * 448. / hidden.double().abs().amax(
        -1, keepdim=True
    )
    nearest_q = exact_q.float().to(torch.float8_e4m3fn)
    mask_q = quantized.float() != nearest_q.float()
    if mask_q.any():
        a = quantized.view(torch.uint8)[mask_q].int()
        b = nearest_q.view(torch.uint8)[mask_q].int()
        assert ((a - b).abs() == 1).all()
        mid = (quantized.double()[mask_q] + nearest_q.double()[mask_q]) * .5
        assert ((exact_q[mask_q] - mid).abs() <=
                8 * torch.finfo(torch.float32).eps *
                exact_q[mask_q].abs().clamp_min(1e-30)).all()


@pytest.mark.parametrize("m", [4, 8, 16, 32, 64, 128, 256])
def test_bf16_draft_norm_matches_compiled_native_in_graph(m):
    from vllm import ir

    torch.manual_seed(4707 + m)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32) * .03
    native = ir.ops.rms_norm.impls["native"].impl_fn

    def reference(x, w):
        return native(x, w + 1., 1e-5).to(torch.bfloat16)

    compiled = torch.compile(reference, fullgraph=True, dynamic=False)
    compiled(x, weight)
    step5_gemma_norm_fp8_quant(x, weight, 1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hidden, _, _ = step5_gemma_norm_fp8_quant(x, weight, 1e-5)
    for value in (None, 0., 1., -1., 1e-14):
        if value is None:
            x.normal_()
        else:
            x.fill_(value)
        graph.replay()
        torch.testing.assert_close(hidden, compiled(x, weight), rtol=0, atol=0)
