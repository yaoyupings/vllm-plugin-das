"""CPU contracts for explicit Decode capture and phase-scope isolation."""
import importlib.util
import os
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import pytest

ROOT = Path(os.getenv("STEP5_GRAPH_REPO", str(Path(__file__).resolve().parents[2])))
source = ROOT / "vllm_hcu/step5_auto_graph.py"
spec = importlib.util.spec_from_file_location("step5_auto_graph_under_test", source)
graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph)


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3


@pytest.fixture
def cfg(monkeypatch):
    for name in ["VLLM_STEP5_AUTO_DECODE_GRAPH", "VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS",
                 "VLLM_STEP5_SEQUENCE_PARALLEL_MOE"]:
        monkeypatch.setenv(name, "1")
    module = ModuleType("vllm_hcu.patch.config")
    module.get_hcu_config = lambda c: NS(deepep_auto=c.deepep_auto)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return NS(deepep_auto=True,
        model_config=NS(architectures=["Step5ForConditionalGeneration"], enforce_eager=False),
        parallel_config=NS(tensor_parallel_size=4, data_parallel_size=2,
            pipeline_parallel_size=1, decode_context_parallel_size=1,
            prefill_context_parallel_size=1, enable_expert_parallel=True, enable_dbo=False),
        compilation_config=NS(cudagraph_mode=Mode.FULL_DECODE_ONLY),
        scheduler_config=NS(max_num_seqs=64))


def manager(cfg, query=4):
    return NS(vllm_config=cfg, cudagraph_mode=Mode.FULL_DECODE_ONLY,
              decode_query_len=query, varlen_decode=False,
              _capture_descs={Mode.FULL: [NS(num_reqs=n, uniform_token_count=query, max_query_len=None,
                                            num_tokens=n * query) for n in [1, 8, 16]]})


def test_policy_and_opt_in(cfg, monkeypatch):
    assert graph.step5_auto_decode_graph_enabled(cfg)
    monkeypatch.setenv("VLLM_STEP5_AUTO_DECODE_GRAPH", "0")
    assert not graph.step5_auto_decode_graph_enabled(cfg)


@pytest.mark.parametrize("owner, field, value", [
    ("model", "architectures", ["DeepseekV4ForCausalLM"]),
    ("model", "enforce_eager", True),
    ("parallel", "tensor_parallel_size", 2), ("parallel", "data_parallel_size", 4),
    ("parallel", "pipeline_parallel_size", 2),
    ("parallel", "decode_context_parallel_size", 2),
    ("parallel", "prefill_context_parallel_size", 2),
    ("parallel", "enable_expert_parallel", False), ("parallel", "enable_dbo", True),
    ("compile", "cudagraph_mode", Mode.FULL),
    ("compile", "cudagraph_mode", Mode.PIECEWISE),
    ("config", "deepep_auto", False),
])
def test_policy_rejects_unsupported(cfg, owner, field, value):
    setattr({"model": cfg.model_config, "parallel": cfg.parallel_config,
             "compile": cfg.compilation_config, "config": cfg}[owner], field, value)
    assert not graph.step5_auto_decode_graph_enabled(cfg)


def test_mtp_draft_policy(cfg):
    cfg.model_config.architectures = ["Step5MTP"]
    assert graph.step5_auto_decode_graph_enabled(cfg)
    assert not graph.step5_auto_decode_graph_enabled(cfg, allow_draft=False)


@pytest.mark.parametrize("query", [1, 4])
def test_capture_scope_does_not_leak(cfg, query):
    m = manager(cfg, query)
    assert graph.step5_auto_capture_uses_ll(cfg, query) is None
    with graph.step5_auto_decode_capture_scope(m):
        assert graph.step5_auto_capture_uses_ll(cfg, 16 * query) is True
    assert graph.step5_auto_capture_uses_ll(cfg, query) is None
    with pytest.raises(ValueError):
        with graph.step5_auto_decode_capture_scope(m):
            raise ValueError("capture failed")
    assert graph.step5_auto_capture_uses_ll(cfg, query) is None


def test_nested_scope_restores_outer_bound(cfg):
    with graph.step5_auto_decode_capture_scope(manager(cfg, 4)):
        assert graph.step5_auto_capture_uses_ll(cfg, 128)
        with graph.step5_auto_decode_capture_scope(manager(cfg, 1)):
            with pytest.raises(RuntimeError, match="token bound"):
                graph.step5_auto_capture_uses_ll(cfg, 128)
        assert graph.step5_auto_capture_uses_ll(cfg, 128)


@pytest.mark.parametrize("field,value", [
    ("num_reqs", 0), ("num_reqs", None), ("num_tokens", 5),
    ("max_query_len", 1), ("num_tokens", 512),
    ("uniform_token_count", None), ("uniform_token_count", 1),
])
def test_invalid_descriptor_rejected_before_capture(cfg, field, value):
    m = manager(cfg)
    setattr(m._capture_descs[Mode.FULL][0], field, value)
    with pytest.raises(RuntimeError, match="descriptor"):
        with graph.step5_auto_decode_capture_scope(m):
            pytest.fail("invalid capture entered its body")


def test_never_capture_prefill_or_variable_query(cfg):
    m = manager(cfg)
    m._capture_descs[Mode.PIECEWISE] = []
    with pytest.raises(RuntimeError, match="mixed/HT"):
        graph.validate_decode_capture_descriptors(m)
    m = manager(cfg); m.varlen_decode = True
    with pytest.raises(RuntimeError, match="fixed"):
        graph.validate_decode_capture_descriptors(m)


@pytest.mark.parametrize("tokens", [None, 0, -1, 257, 16384])
def test_capture_never_routes_large_prefill_through_ll(cfg, tokens):
    with graph.step5_auto_decode_capture_scope(manager(cfg)):
        with pytest.raises(RuntimeError, match="token bound"):
            graph.step5_auto_capture_uses_ll(cfg, tokens)


def test_capture_install_is_idempotent_and_scoped(cfg, monkeypatch):
    calls = []
    class Manager:
        def capture(self):
            calls.append(graph.step5_auto_capture_uses_ll(self.vllm_config, 4))
    target = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    target.CudaGraphManager = Manager
    monkeypatch.setitem(sys.modules, target.__name__, target)
    graph.install_step5_auto_capture()
    first = Manager.capture
    graph.install_step5_auto_capture()
    assert Manager.capture is first
    m = Manager(); m.__dict__.update(manager(cfg).__dict__)
    m.capture()
    cfg.deepep_auto = False
    m.capture()
    assert calls == [True, None]


@pytest.mark.parametrize("query", [1, 4])
def test_upstream_uniform_descriptor_contract(cfg, query):
    # Use the installed vLLM dataclass definition, including its defaults.
    # Loading this one definition avoids invoking GPU/compiler initialization.
    import ast
    from dataclasses import dataclass
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("installed vLLM source not available")
    upstream = Path(spec.origin).parent / "v1/worker/gpu/cudagraph_utils.py"
    tree = ast.parse(upstream.read_text())
    definition = next(n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name == "BatchExecutionDescriptor")
    namespace = {"dataclass": dataclass, "CUDAGraphMode": Mode}
    unit = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit),str(upstream),"exec"),namespace)
    Desc = namespace["BatchExecutionDescriptor"]
    desc = Desc(cg_mode=Mode.FULL,num_tokens=16*query,num_reqs=16,
                uniform_token_count=query)
    assert desc.max_query_len is None
    m = manager(cfg,query)
    m._capture_descs = {Mode.FULL:[desc]}
    assert graph.validate_decode_capture_descriptors(m) == 64*query
    with graph.step5_auto_decode_capture_scope(m):
        assert graph.step5_auto_capture_uses_ll(cfg,16*query)


def workspace_config(cfg):
    cfg.model_config.hf_text_config = NS(
        hidden_size=4096,moe_intermediate_size=1536,
        moe_num_experts=352,moe_top_k=8)
    cfg.scheduler_config.max_num_batched_tokens = 16384
    cfg.speculative_config = NS(num_speculative_tokens=3)
    return cfg


def test_workspace_bound_covers_all_assignments_and_static_ll(cfg):
    cfg = workspace_config(cfg)
    bound = graph.step5_auto_workspace_bound(cfg)
    assert bound["sender_rows"] == 4096
    assert bound["received_rows"] == 32768
    assert bound["assignments"] == 262144
    assert bound["local_experts"] == 44
    assert bound["aligned_rows"] == 273408
    assert bound["bytes"] == 4479516672
    assert bound["ll_bytes"] == 1232 * 1024**2
    assert bound["bf16_ht_bytes"] < bound["bytes"]
    # Independent adversarial expert-count distributions: all local, one
    # heavily loaded expert, uniform, and many counts just past tile edges.
    total=bound["assignments"];e=bound["local_experts"]
    counts_cases=[[total]+[0]*(e-1),
                  [total//e]*e,
                  [257]*(e-1)+[total-257*(e-1)]]
    for counts in counts_cases:
        aligned=sum(((v+255)//256)*256 for v in counts)
        assert aligned <= bound["aligned_rows"]


@pytest.mark.parametrize("field,value",[
    ("hidden_size",0),("moe_intermediate_size",0),
    ("moe_num_experts",0),("moe_top_k",0),("moe_top_k",353)])
def test_workspace_geometry_is_validated(cfg,field,value):
    workspace_config(cfg)
    setattr(cfg.model_config.hf_text_config,field,value)
    with pytest.raises(RuntimeError,match="geometry"):
        graph.step5_auto_workspace_bound(cfg)


def test_workspace_bound_rejects_elastic_experts(cfg):
    workspace_config(cfg);cfg.parallel_config.enable_eplb=True
    with pytest.raises(RuntimeError,match="geometry"):
        graph.step5_auto_workspace_bound(cfg)


def test_workspace_reservation_feature_off_is_noop(cfg,monkeypatch):
    monkeypatch.setenv("VLLM_STEP5_AUTO_DECODE_GRAPH","0")
    assert graph.reserve_step5_auto_workspace(cfg,"unused-device") is None

def test_real_runner_reserves_before_upstream_profile(monkeypatch):
    import ast
    path = ROOT / "vllm_hcu/v1/hcu_model_runner_v2.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "HcuGPUModelRunnerV2")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == "profile_run"]
    # A later same-name definition silently replaces an earlier reservation.
    assert len(methods) == 1
    calls = []
    class Base:
        def profile_run(self):
            calls.append(("upstream", self.max_num_tokens))
    namespace = {"GPUModelRunner": Base}
    unit = ast.Module(body=[ast.ClassDef(
        name=cls.name, bases=cls.bases, keywords=[], body=methods,
        decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    module = ModuleType("vllm_hcu.step5_auto_graph")
    module.reserve_step5_auto_workspace = lambda config, device: calls.append(
        ("reserve", config, device))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = namespace[cls.name]()
    runner.vllm_config = NS(parallel_config=NS(prefill_context_parallel_size=1))
    runner.device = "device"
    runner.max_num_tokens = 16384
    runner.profile_run()
    assert calls == [("reserve", runner.vllm_config, "device"),
                     ("upstream", 16384)]

@pytest.mark.parametrize("sequences,expected", [(1,64),(16,64),(32,64),(64,64),(65,128),(128,128)])
def test_auto_ll_bound_covers_every_padded_request_shape(cfg,monkeypatch,sequences,expected):
    import importlib.util
    path=ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_sp_workspace.py"
    spec=importlib.util.spec_from_file_location("sp_bound_under_test",path)
    sp=importlib.util.module_from_spec(spec);spec.loader.exec_module(sp)
    module=ModuleType("vllm_hcu.step5_auto_graph")
    module.step5_auto_decode_graph_enabled=graph.step5_auto_decode_graph_enabled
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setenv("VLLM_STEP5_AUTO_SP_LL_WORKSPACE","1")
    cfg.scheduler_config.max_num_seqs=sequences
    cfg.scheduler_config.max_num_batched_tokens=16384
    cfg.speculative_config=NS(num_speculative_tokens=3)
    moe=NS(max_num_tokens=16384,moe_parallel_config=NS(
        is_sequence_parallel=True,sp_size=4,ep_size=8))
    assert sp.step5_auto_ll_dispatch_bound(moe,cfg)==expected
    for reqs in range(1,sequences+1):
        for query in (1,2,3,4):
            assert (reqs*query+3)//4 <= expected
    assert moe.max_num_tokens==cfg.scheduler_config.max_num_batched_tokens==16384


def test_auto_ll_bound_is_noop_without_opt_in(cfg,monkeypatch):
    import importlib.util
    path=ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_sp_workspace.py"
    spec=importlib.util.spec_from_file_location("sp_disabled_under_test",path)
    sp=importlib.util.module_from_spec(spec);spec.loader.exec_module(sp)
    monkeypatch.setenv("VLLM_STEP5_AUTO_SP_LL_WORKSPACE","0")
    assert sp.step5_auto_ll_dispatch_bound(None,None) is None
