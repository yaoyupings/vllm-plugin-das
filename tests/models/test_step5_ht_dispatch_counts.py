# SPDX-License-Identifier: Apache-2.0
"""CPU metadata contract checks with an independent routing-count oracle."""
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import ast

import torch


def test_fresh_dispatch_counts_and_fallbacks():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/step5_ht_workspace.py"
    )
    @dataclass
    class ExpertTokensMetadata:
     expert_num_tokens:torch.Tensor
     expert_num_tokens_cpu:torch.Tensor|None
    class TorchProxy:
     def __getattr__(self,name):
      if name=="cuda":return SimpleNamespace(is_current_stream_capturing=lambda:False)
      if name=="compiler":return SimpleNamespace(is_compiling=lambda:False)
      return getattr(torch,name)
    calls=[]
    def oracle(ids,local,map):
     counts=[0]*local
     for row in ids.tolist():
      for value in row:
       if value<0:continue
       if map is None:e=value
       elif value>=map.numel():continue
       else:e=int(map[value])
       if 0<=e<local:counts[e]+=1
     return torch.tensor(counts,dtype=torch.int32)
    def recount(ids,local,map):
     calls.append(True);return oracle(ids,local,map)
    names={"Step5HTTokensMetadata","Step5HTDispatchTokensMetadata",
     "mark_fresh_dispatch_metadata","_dispatch_counts_match_local_map",
     "_tensor_version","prepare_compact_metadata","aligned_expert_rows",
     "compact_workspace_parallelism_supported"}
    tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    scope={"torch":TorchProxy(),"dataclass":dataclass,"ExpertTokensMetadata":ExpertTokensMetadata,"count_expert_num_tokens":recount}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),scope)
    mark=scope["mark_fresh_dispatch_metadata"];prepare=scope["prepare_compact_metadata"]
    rows=[]
    supported=scope["compact_workspace_parallelism_supported"]
    assert supported(1,8) and supported(2,4) and supported(4,2)
    assert not supported(8,1) and not supported(4,4)
    def setup(offset=4):
     ids=torch.tensor([[4,5,0,-1],[6,7,0,-1],[4,7,0,-1]],dtype=torch.int64)
     map=torch.full((12,),-1,dtype=torch.int32);map[offset:offset+4]=torch.arange(4,dtype=torch.int32)
     counts=oracle(ids,4,map)
     meta=mark(ExpertTokensMetadata(counts.clone(),counts.clone()),ids,offset,12)
     return ids,map,meta
    def check(name,ids,map,meta,cache,expected_recount):
     before=len(calls);out=prepare(ids,4,map,meta,cache)
     exact=oracle(ids,4,map);total=int(exact.sum())
     if total:
      assert torch.equal(out.expert_num_tokens,exact)
      assert torch.equal(out.expert_num_tokens_cpu,exact)
      expected=sum(((int(c)+255)//256)*256 for c in exact)
      assert out.aligned_rows==expected and out.routed_shape==tuple(ids.shape)
     else:assert out is meta
     assert len(calls)-before==expected_recount,name
     rows.append({"case":name,"recount_calls":expected_recount,"counts_and_workspace_match_independent_loop":True})
    ids,map,meta=setup();cache={}
    check("fresh_dispatch",ids,map,meta,cache,0)
    check("cached_same_map",ids,map,meta,cache,0)
    ids2,map2,meta2=setup()
    check("different_map_object_same_layout",ids2,map2,meta2,cache,0)
    ids2[0,0]=7
    check("routing_mutated",ids2,map2,meta2,cache,1)
    ids,map,meta=setup();copied=ids.clone()
    check("different_routing_tensor",copied,map,meta,{},1)
    ids,map,meta=setup();map[4]=2
    check("expert_map_mutated",ids,map,meta,{},1)
    ids,map,meta=setup()
    check("opt_in_disabled",ids,map,meta,None,1)
    ids,map,meta=setup();plain=ExpertTokensMetadata(meta.expert_num_tokens,meta.expert_num_tokens_cpu)
    check("missing_dispatch_provenance",ids,map,plain,{},1)
    ids,map,meta=setup();meta.expert_num_tokens_cpu=None
    check("missing_cpu_dispatch_counts",ids,map,meta,{},1)
    ids,map,meta=setup()
    with torch.inference_mode():unversioned=map.clone()
    check("map_without_mutation_counter",ids,unversioned,meta,{},1)
    ids,map,meta=setup();meta.rank_expert_offset=8
    check("wrong_dispatch_offset",ids,map,meta,{},1)
    ids,map,meta=setup();meta.global_num_experts=13
    check("wrong_global_expert_count",ids,map,meta,{},1)
    ids,map,meta=setup();meta.expert_num_tokens_cpu=torch.zeros(3,dtype=torch.int32)
    check("wrong_local_expert_count",ids,map,meta,{},1)
    ids,map,meta=setup();meta.expert_num_tokens_cpu[0]=-1
    try:prepare(ids,4,map,meta,{})
    except RuntimeError:rows.append({"case":"negative_fresh_counts_rejected","rejected":True})
    else:raise AssertionError("Negative counts accepted")
    ids,map,meta=setup();meta.expert_num_tokens_cpu[:]=100
    try:prepare(ids,4,map,meta,{})
    except RuntimeError:rows.append({"case":"impossible_fresh_assignment_total_rejected","rejected":True})
    else:raise AssertionError("Oversized counts accepted")
    assert len(rows) == 15
