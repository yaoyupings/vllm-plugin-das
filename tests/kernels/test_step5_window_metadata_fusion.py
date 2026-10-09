# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
from vllm_hcu.model_executor.layers.step5_window_metadata import (
    Step5WindowMetadataBuilder, truncate_window_metadata,
)
from vllm_hcu.model_executor.layers.step5_window_metadata_kernel import fused_tail_into

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP unavailable")

def cu(queries):
    return torch.tensor([0] + list(torch.tensor(queries).cumsum(0).tolist()),
                        device="cuda", dtype=torch.int32)

def metadata(seq, query_start, table, maxq):
    return NS(use_cascade=False, causal=True, seq_lens=seq,
              query_start_loc=query_start, block_table=table,
              max_query_len=maxq, max_seq_len=69632)

def builder(rows, maxq, window):
    result = object.__new__(Step5WindowMetadataBuilder)
    result.block_size = 64
    result._step5_window = window
    cols = (maxq + window - 1 + 63 + 63) // 64
    result._step5_tail_table = torch.empty((rows, cols), dtype=torch.int32, device="cuda")
    result._step5_tail_lens = torch.empty(rows, dtype=torch.int32, device="cuda")
    return result

@pytest.mark.parametrize("maxq", [1,4,512])
@pytest.mark.parametrize("window", [1,64,512])
@pytest.mark.parametrize("strided", [False,True])
def test_fused_metadata_matches_native_and_graph_replay(maxq,window,strided):
    torch.manual_seed(217)
    rows, cols = 10,1088
    ends=[0,1,63,64,65,575,576,577,8193,65536]
    queries=[min(maxq,e) for e in ends]
    seq=torch.tensor(ends,device="cuda",dtype=torch.int32)
    starts=cu(queries)
    table=torch.randperm(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols)
    if strided:
        arena=torch.empty((rows,cols*2),device="cuda",dtype=torch.int32)
        arena[:,::2].copy_(table)
        table=arena[:,::2]
        seq_arena=torch.empty(rows*2,device="cuda",dtype=torch.int32)
        seq_arena[::2].copy_(seq);seq=seq_arena[::2]
        cu_arena=torch.empty((rows+1)*2,device="cuda",dtype=torch.int32)
        cu_arena[::2].copy_(starts);starts=cu_arena[::2]
    md=metadata(seq,starts,table,maxq)
    b=builder(rows,maxq,window)
    output=b._stage_fused_tail_metadata(md)
    table_ptr,length_ptr=output.block_table.data_ptr(),output.seq_lens.data_ptr()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused_tail_into(seq,starts,table,output.seq_lens,output.block_table,window,64)
    for replay in range(4):
        if replay:
            seq.copy_(torch.randint(0,69633,(rows,),device="cuda",dtype=torch.int32))
            q=torch.minimum(torch.randint(0,maxq+1,(rows,),device="cuda",dtype=torch.int32),seq)
            starts.copy_(torch.cat((torch.zeros(1,device="cuda",dtype=torch.int32),
                                   q.cumsum(0,dtype=torch.int32))))
            table.copy_(torch.randperm(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols))
        g.replay()
        expected=truncate_window_metadata(md,block_size=64,window=window)
        torch.testing.assert_close(output.seq_lens,expected.seq_lens,rtol=0,atol=0)
        torch.testing.assert_close(output.block_table,expected.block_table,rtol=0,atol=0)
        assert output.max_seq_len==expected.max_seq_len
        rebuilt=b._stage_fused_tail_metadata(md)
        assert rebuilt.seq_lens.data_ptr()==length_ptr
        assert rebuilt.block_table.data_ptr()==table_ptr

def test_fused_tail_attention_native_output_under_replay():
    from flash_attn import flash_attn_varlen_func
    torch.manual_seed(317)
    rows,cols=4,1088
    table=torch.randperm(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols)
    seq=torch.tensor([65536]*rows,device="cuda",dtype=torch.int32)
    starts=cu([4]*rows)
    q=torch.randn(16,32,192,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(rows*cols,64,2,192,device="cuda",dtype=torch.bfloat16)
    v=torch.randn_like(k)
    md=metadata(seq,starts,table,4)
    b=builder(rows,4,512)
    out=b._stage_fused_tail_metadata(md)
    def attention(lengths,pages):
        return flash_attn_varlen_func(q,k,v,cu_seqlens_q=starts,max_seqlen_q=4,
             seqused_k=lengths,max_seqlen_k=578,block_table=pages,
             causal=True,window_size=(511,0))
    attention(out.seq_lens,out.block_table)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused_tail_into(seq,starts,table,out.seq_lens,out.block_table)
        actual=attention(out.seq_lens,out.block_table)
    for ends in [[65536]*rows,[69632,65535,577,576],[65,64,63,575]]:
        seq.copy_(torch.tensor(ends,device="cuda",dtype=torch.int32))
        q.normal_()
        table.copy_(torch.randperm(rows*cols,device="cuda",dtype=torch.int32).view(rows,cols))
        g.replay()
        expected_md=truncate_window_metadata(md,block_size=64,window=512)
        expected=attention(expected_md.seq_lens,expected_md.block_table)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
