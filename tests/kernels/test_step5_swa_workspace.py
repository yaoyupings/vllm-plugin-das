"""Native FA math and Graph replay with a shared conversion workspace."""
from pathlib import Path
import importlib.util
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native HCU GPU required")


def load_runtime():
    path = Path(__file__).parents[2] / "vllm_hcu/v1/attention/backends/step5_swa_workspace.py"
    spec = importlib.util.spec_from_file_location("step5_swa_workspace_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(q, k, v, out, cuq, seq, table, mq, mk):
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    return flash_attn_varlen_func(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cuq, seqused_k=seq,
        block_table=table, max_seqlen_q=mq, max_seqlen_k=mk,
        causal=True, window_size=(511, 0), softmax_scale=192 ** -0.5,
        layout="bshd",
    )


@pytest.mark.parametrize("lens", [[16], [32,16,1,3], [4,1,2,3], [32]*16, [512,0]])
def test_native_output_and_graph_replay(monkeypatch, lens):
    from vllm.v1.worker import workspace
    rt = load_runtime()
    manager = workspace.WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(workspace, "current_workspace_manager", lambda: manager)
    rows=len(lens); mq=max(lens); mk=mq+511+63
    cols=(mk+63)//64
    torch.manual_seed(91)
    q=torch.randn(sum(lens),32,192,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(rows*cols+3,64,2,192,device="cuda",dtype=torch.bfloat16)
    v=torch.randn_like(k)
    table=torch.arange(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols)
    cuq=torch.tensor([0]+list(__import__("itertools").accumulate(lens)),device="cuda",dtype=torch.int32)
    # Lengths represent the page-aligned tail from long 64K contexts.
    seq=torch.tensor([max(1,n+511+((i*17)%64)) for i,n in enumerate(lens)],device="cuda",dtype=torch.int32)
    out=torch.empty_like(q); expected=torch.empty_like(q)
    reference(q,k,v,expected,cuq,seq,table,mq,mk)
    got=rt.native_swa_varlen_with_workspace(q,k,v,out,cuq,seq,table,mq,mk,192**-.5,(511,0))
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    arena=manager.get_simultaneous(((1,),torch.uint8))[0]
    address=arena.data_ptr()
    manager.lock()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt.native_swa_varlen_with_workspace(q,k,v,out,cuq,seq,table,mq,mk,192**-.5,(511,0))
    for shift in [1,3,0]:
        q.copy_(torch.randn_like(q))
        table.copy_((torch.arange(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols)+shift)%(rows*cols+3))
        seq.copy_(torch.tensor([max(1,n+511+((i*17+shift*11)%64)) for i,n in enumerate(lens)],device="cuda",dtype=torch.int32))
        reference(q,k,v,expected,cuq,seq,table,mq,mk)
        graph.replay()
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        assert manager.get_simultaneous(((1,),torch.uint8))[0].data_ptr()==address


def test_two_layers_reuse_arena_without_corrupting_outputs(monkeypatch):
    from vllm.v1.worker import workspace
    rt=load_runtime()
    manager=workspace.WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(workspace,"current_workspace_manager",lambda:manager)
    q=torch.randn(32,32,192,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(20,64,2,192,device="cuda",dtype=torch.bfloat16);v=torch.randn_like(k)
    table=torch.arange(20,device="cuda",dtype=torch.int32).view(2,10)
    seq=torch.tensor([533,546],device="cuda",dtype=torch.int32)
    cuq=torch.tensor([0,16,32],device="cuda",dtype=torch.int32)
    out1=torch.empty_like(q);out2=torch.empty_like(q);expected=torch.empty_like(q)
    reference(q,k,v,expected,cuq,seq,table,16,590)
    for out in [out1,out2]:
        rt.native_swa_varlen_with_workspace(q,k,v,out,cuq,seq,table,16,590,192**-.5,(511,0))
    manager.lock()
    arena=manager.get_simultaneous(((384*1024*1024,),torch.uint8))[0]
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt.native_swa_varlen_with_workspace(q,k,v,out1,cuq,seq,table,16,590,192**-.5,(511,0))
        arena.fill_(83)  # A following MoE may overwrite the same arena.
        rt.native_swa_varlen_with_workspace(q,k,v,out2,cuq,seq,table,16,590,192**-.5,(511,0))
    graph.replay()
    for out in [out1,out2]:torch.testing.assert_close(out,expected,rtol=0,atol=0)


@pytest.mark.parametrize("lens,maxq", [([1,1],1),([4,4],4),([2,2],2)])
def test_direct_paged_decode_is_not_rerouted(lens,maxq):
    rt=load_runtime()
    q=torch.empty(sum(lens),32,192,device="cuda",dtype=torch.bfloat16)
    k=torch.empty(4,64,2,192,device="cuda",dtype=torch.bfloat16)
    cuq=torch.empty(len(lens)+1,device="cuda",dtype=torch.int32)
    assert not rt.eligible(q,k,k,cuq,maxq,(511,0))
