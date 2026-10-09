# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
# Exercise production builder methods without importing vendor FA, which
# exposes no GPU functions when devices are hidden for this CPU-only suite.
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch
base_module=types.ModuleType("vllm_hcu.v1.attention.backends.flash_attn")
class FlashAttentionMetadataBuilder:
    def build(self,*args,**kwargs):
        raise AssertionError("test must provide native metadata")
base_module.FlashAttentionMetadataBuilder=FlashAttentionMetadataBuilder
source=Path(__file__).resolve().parents[2]/"vllm_hcu/model_executor/layers/step5_window_metadata.py"
spec=importlib.util.spec_from_file_location(
    "vllm_hcu.model_executor.layers._window_metadata_policy_under_test",source)
module=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{"vllm_hcu.v1.attention.backends.flash_attn":base_module}):
    spec.loader.exec_module(module)
Step5WindowMetadataBuilder=module.Step5WindowMetadataBuilder
truncate_window_metadata=module.truncate_window_metadata
from vllm_hcu.model_executor.layers import step5_window_metadata_kernel as kernel

def make_builder():
    b=object.__new__(Step5WindowMetadataBuilder)
    b.block_size=64
    b._step5_window=512
    b._step5_window_enabled=True
    b._step5_tail_table=torch.empty((32,17),dtype=torch.int32)
    b._step5_tail_lens=torch.empty(32,dtype=torch.int32)
    return b

def metadata(rows=4,maxq=4,cols=1088):
    seq=torch.tensor([65536,577,65,0][:rows],dtype=torch.int32)
    queries=[min(maxq,int(e)) for e in seq]
    cu=torch.tensor([0]+list(torch.tensor(queries).cumsum(0).tolist()),dtype=torch.int32)
    return NS(use_cascade=False,causal=True,seq_lens=seq,query_start_loc=cu,
              block_table=torch.arange(rows*cols,dtype=torch.int32).reshape(rows,cols),
              max_query_len=maxq,max_seq_len=65536)

def reference_into(seq,cu,table,out_seq,out_table,window,block_size):
    md=NS(use_cascade=False,causal=True,seq_lens=seq,query_start_loc=cu,
          block_table=table,max_query_len=4,max_seq_len=65536)
    result=truncate_window_metadata(md,block_size=block_size,window=window)
    out_seq.copy_(result.seq_lens);out_table.copy_(result.block_table)

@pytest.mark.parametrize("enabled",[False,True])
def test_build_uses_expected_path_and_keeps_graph_addresses(monkeypatch,enabled):
    monkeypatch.setenv("VLLM_STEP5_FUSED_WINDOW_METADATA","1" if enabled else "0")
    md=metadata()
    monkeypatch.setattr(FlashAttentionMetadataBuilder,"build",lambda *a,**kw:md)
    calls=[]
    def fused(*args):
        calls.append(True);reference_into(*args)
    monkeypatch.setattr(kernel,"fused_tail_into",fused)
    b=make_builder()
    expected=truncate_window_metadata(md,block_size=64,window=512)
    actual=b.build(0,None)
    torch.testing.assert_close(actual.seq_lens,expected.seq_lens,rtol=0,atol=0)
    torch.testing.assert_close(actual.block_table,expected.block_table,rtol=0,atol=0)
    assert actual.max_seq_len==expected.max_seq_len
    ptrs=(actual.seq_lens.data_ptr(),actual.block_table.data_ptr())
    md.seq_lens[0]=69632
    replay=b.build(0,None)
    assert (replay.seq_lens.data_ptr(),replay.block_table.data_ptr())==ptrs
    assert len(calls)==(2 if enabled else 0)
    # Input raw metadata remains raw, and active row slices retain addresses.
    assert md.max_seq_len==65536 and md.block_table.shape[1]==1088

@pytest.mark.parametrize("guard",["cascade","noncausal","empty","parallel_disabled","window_disabled"])
def test_unsupported_metadata_retains_native_guard(monkeypatch,guard):
    monkeypatch.setenv("VLLM_STEP5_FUSED_WINDOW_METADATA","1")
    md=metadata()
    b=make_builder()
    if guard=="cascade":md.use_cascade=True
    elif guard=="noncausal":md.causal=False
    elif guard=="empty":md.block_table=md.block_table[:,:0]
    elif guard=="parallel_disabled":b._step5_window_enabled=False
    elif guard=="window_disabled":b._step5_window=0
    monkeypatch.setattr(FlashAttentionMetadataBuilder,"build",lambda *a,**kw:md)
    def forbidden(*args):raise AssertionError("unsupported shape launched fused kernel")
    monkeypatch.setattr(kernel,"fused_tail_into",forbidden)
    assert b.build(0,None) is md

def test_every_width_has_contiguous_distinct_persistent_table():
    b=make_builder()
    tables=[]
    for cols in [9,10,17]:
        table,lengths=b._tail_buffers(4,cols)
        assert table.is_contiguous()
        assert table.stride(0)==cols
        again,_=b._tail_buffers(2,cols)
        assert again.data_ptr()==table.data_ptr()
        tables.append(table.data_ptr())
    assert len(set(tables))==3
