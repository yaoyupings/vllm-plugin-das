"""CPU lifecycle checks; native HT/LL layout qualification is separate."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import pytest

ROOT = Path(os.environ.get("STEP5_AUTO_REPO", str(Path(__file__).resolve().parents[2])))
SOURCE = ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_deepep_auto.py"


class Parent:
    def _current(self):
        return self.ll_experts if self._use_low_latency_snapshot else self.ht_experts


def replace(layer, name, tensor):
    setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))


env = dict(os=os, torch=torch, DeepEPAutoDeepGemmExperts=Parent,
           replace_parameter=replace, MoEActivation=NS(SILU="silu", SWIGLUSTEP="step"))
tree = ast.parse(SOURCE.read_text())
tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), env)
enabled = env["step5_auto_experts_enabled"]
FP8 = env["Step5AutoFP8Experts"]
BF16 = env["Step5AutoBF16Experts"]


def configs():
    mp = NS(use_deepep_auto_kernels=True, is_sequence_parallel=True, sp_size=4, ep_size=8)
    p = NS(tensor_parallel_size=4, data_parallel_size=2, pipeline_parallel_size=1,
           decode_context_parallel_size=1, prefill_context_parallel_size=1,
           enable_dbo=False)
    return NS(moe_parallel_config=mp, is_lora_enabled=False), NS(
        parallel_config=p, model_config=NS(architectures=["Step5ForConditionalGeneration"]))


def test_scope(monkeypatch):
    m, c = configs()
    assert not enabled(m, c)
    monkeypatch.setenv("VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS", "1")
    assert enabled(m, c)
    assert not enabled(m, None)


@pytest.mark.parametrize("owner,field,value", [
    ("p", "tensor_parallel_size", 2), ("p", "data_parallel_size", 4),
    ("p", "pipeline_parallel_size", 2), ("p", "decode_context_parallel_size", 2),
    ("p", "prefill_context_parallel_size", 2), ("p", "enable_dbo", True),
    ("mp", "use_deepep_auto_kernels", False), ("mp", "is_sequence_parallel", False),
    ("mp", "sp_size", 2), ("mp", "ep_size", 4), ("m", "is_lora_enabled", True),
])
def test_scope_rejects_unsupported(monkeypatch, owner, field, value):
    monkeypatch.setenv("VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS", "1")
    m, c = configs()
    setattr({"p": c.parallel_config, "mp": m.moe_parallel_config, "m": m}[owner], field, value)
    assert not enabled(m, c)


def packed_shape(w13, w2):
    assert w13.ndim == w2.ndim == 6
    assert w13.shape[0] == w2.shape[0]


def expert(fail_at=None):
    obj = FP8.__new__(FP8)
    obj._fixed_use_low_latency = None
    obj._use_low_latency_snapshot = False
    obj.quant_config = NS(use_int8_w8a8=False)
    calls = []
    def pack(w13, w2):
        result = []
        for w in (w13, w2):
            calls.append(w.data_ptr())
            if fail_at == len(calls):
                raise RuntimeError("in-place pack failed")
            e, n, k = w.shape
            result.append(w.detach().view(e, k // 64, n // 16, 4, 16, 16))
        return tuple(result)
    obj.ht_experts = NS(_can_pack_channel_weights=lambda layer:
                        layer.w13_weight.ndim == layer.w2_weight.ndim == 3,
                        _pack_channel_weights=pack,
                        _packed_deepgemm_problem_shape=packed_shape)
    obj.ll_experts = NS()
    return obj, calls


def layer(dtype=torch.float8_e4m3fn):
    out = torch.nn.Module()
    out.w13_weight = torch.nn.Parameter(torch.zeros(1, 128, 128, dtype=dtype), requires_grad=False)
    out.w2_weight = torch.nn.Parameter(torch.zeros(1, 128, 64, dtype=dtype), requires_grad=False)
    return out


@patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx938"))
def test_single_storage_and_idempotence(props):
    x, calls = expert(); w = layer()
    ptrs = (w.w13_weight.data_ptr(), w.w2_weight.data_ptr())
    x.process_weights_after_loading(w)
    assert calls == list(ptrs)
    assert x.ht_experts._deepgemm_w13 is x.ll_experts._deepgemm_w13
    assert x.ht_experts._deepgemm_w2 is x.ll_experts._deepgemm_w2
    assert (w.w13_weight.data_ptr(), w.w2_weight.data_ptr()) == ptrs
    x.process_weights_after_loading(w)
    assert len(calls) == 2


@patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx938"))
def test_reload_rebinds_both_children(props):
    x, calls = expert(); w = layer(); x.process_weights_after_loading(w)
    old = w.w13_weight
    raw = layer()
    w.w13_weight, w.w2_weight = raw.w13_weight, raw.w2_weight
    x.process_weights_after_loading(w)
    assert len(calls) == 4
    assert x.ht_experts._deepgemm_w13.data_ptr() != old.data_ptr()
    assert x.ht_experts._deepgemm_w13 is x.ll_experts._deepgemm_w13


@pytest.mark.parametrize("fail_at", [1, 2])
@patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx938"))
def test_partial_in_place_pack_never_retried(props, fail_at):
    x, calls = expert(fail_at); w = layer()
    with pytest.raises(RuntimeError, match="pack failed"):
        x.process_weights_after_loading(w)
    assert w._dsv4_channel_deepgemm_layout == x._PACKING
    with pytest.raises(RuntimeError, match="cannot repack"):
        x.process_weights_after_loading(w)
    assert len(calls) == fail_at


@patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx938"))
def test_lost_or_separate_layout_fails(props):
    x, _ = expert(); w = layer(); x.process_weights_after_loading(w)
    w._dsv4_channel_deepgemm_masked_w13 = w.w13_weight.clone()
    with pytest.raises(RuntimeError, match="lost"):
        x.process_weights_after_loading(w)


@patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx942"))
def test_wrong_arch_never_packs(props):
    x, calls = expert()
    with pytest.raises(RuntimeError, match="gfx938"):
        x.process_weights_after_loading(layer())
    assert not calls


def test_bf16_retains_raw_weights_and_rejects_fp8():
    x = BF16.__new__(BF16); w = layer(torch.bfloat16)
    ids = (id(w.w13_weight), id(w.w2_weight))
    x.process_weights_after_loading(w)
    assert ids == (id(w.w13_weight), id(w.w2_weight))
    with pytest.raises(RuntimeError, match="canonical BF16"):
        x.process_weights_after_loading(layer())


def test_metadata_and_tile_hints_follow_current_phase():
    x, _ = expert(); hints = []
    x.ht_experts.prepare_workspace_metadata = lambda *args: "fresh-ht-counts"
    x.ht_experts.set_expected_m = lambda m: hints.append(("ht", m))
    x.ll_experts.set_expected_m = lambda m: hints.append(("ll", m))
    assert x.prepare_workspace_metadata(None, 44, None, "ll-counts") == "fresh-ht-counts"
    x.set_expected_m(10)
    x._use_low_latency_snapshot = True
    assert x.prepare_workspace_metadata(None, 44, None, "ll-counts") == "ll-counts"
    x.set_expected_m(20)
    assert hints == [("ht", 10), ("ll", 20)]


def test_bf16_chunk_bound_follows_selected_layout():
    class HT:
        max_tokens_per_chunk = 4096
        activation_config = None
        def __init__(self,*args):
            pass
    class LL:
        def __init__(self,*args):
            pass
    env["Step5HTChunkedTritonExperts"] = HT
    env["BatchedTritonExperts"] = LL
    x = BF16(NS(),NS(),256,8)
    assert isinstance(x.ht_experts,HT)
    assert x.max_tokens_per_chunk == 4096
    x._use_low_latency_snapshot = True
    assert x.max_tokens_per_chunk == 0
    x._use_low_latency_snapshot = False
    assert x.max_tokens_per_chunk == 4096
