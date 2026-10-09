# SPDX-License-Identifier: Apache-2.0
"""Compilation boundaries for the optional joint Step5 indexer/norm op."""
from types import SimpleNamespace
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationConfig
from vllm_hcu.model_executor.layers import step4_dsa
from vllm_hcu.step5_config import Step5TextConfig


@pytest.mark.parametrize("tokens", [4, 64, 128])
@pytest.mark.parametrize("rotary_pairs", [32, 96])
def test_joint_fake_op_preserves_tp_local_query_key_outputs(tokens, rotary_pairs):
    with FakeTensorMode():
        hidden = torch.empty(tokens, 4096, dtype=torch.bfloat16)
        packed = torch.empty(tokens, 6912, dtype=torch.bfloat16)
        positions = torch.empty(tokens, dtype=torch.int64)
        weight = torch.empty(192, dtype=torch.float32)
        cache = torch.empty(69632, rotary_pairs * 2, dtype=torch.bfloat16)
        token = torch.empty(1, dtype=torch.int32)
        q, k = torch.ops.vllm.step5_dsa_index_qknorm(
            hidden, positions, packed, weight, weight, cache,
            "model.layers.3.self_attn.attn", token, 32, 2, rotary_pairs, 1e-6,
        )
        assert q.shape == (tokens, 6144)
        assert k.shape == (tokens, 384)
        assert q.dtype == k.dtype == packed.dtype


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE])
def test_joint_indexer_requires_its_own_piecewise_boundary(monkeypatch, mode):
    monkeypatch.setenv("VLLM_STEP5_DSA_PACK_QKNORM", "1")
    ops = ["vllm::step4_dsa_index"]
    monkeypatch.setattr(step4_dsa, "get_current_vllm_config", lambda:
        SimpleNamespace(compilation_config=SimpleNamespace(
            cudagraph_mode=mode, splitting_ops=ops)))
    with pytest.raises(RuntimeError, match="splitting_ops"):
        step4_dsa._require_splitting_op()
    ops.append("vllm::step5_dsa_index_qknorm")
    step4_dsa._require_splitting_op()


def test_step5_registers_joint_boundary_before_compilation(monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_DSA_PACK_QKNORM", "1")
    monkeypatch.delenv("VLLM_STEP4_DISABLE_DSA", raising=False)
    monkeypatch.delenv("VLLM_STEP5_DCP2_RUNTIME", raising=False)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])
    Step5TextConfig(sparse_config={"enabled": True})
    assert CompilationConfig._attention_ops == [
        "vllm::step4_dsa_index", "vllm::step5_dsa_index_qknorm",
    ]
