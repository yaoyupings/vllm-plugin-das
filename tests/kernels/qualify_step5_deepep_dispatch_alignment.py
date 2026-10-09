# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=8; requires idle gfx938 GPUs."""
import os,json,time,datetime
from pathlib import Path
import torch
import torch.distributed as dist
from deep_ep import Buffer
rank=int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("gloo",timeout=datetime.timedelta(seconds=180))
world=dist.get_world_size();assert world==8
E,H,K=352,4096,8
hint=Buffer.get_low_latency_rdma_size_hint(192,H,world,E,0,K)
buffer=Buffer(dist.group.WORLD,num_rdma_bytes=hint,low_latency_mode=True,num_qps_per_rank=44,explicitly_destroy=True)
results=[]
try:
 for capacity in (192,):
  for qt in (0,2):
   m=capacity
   # Every source fills one expert on each of the eight destination ranks.
   # This forces worst-case per-expert occupancy = world*capacity.
   ids=(torch.arange(K,device="cuda",dtype=torch.int64)*44)[None,:].expand(m,K).contiguous()
   weights=torch.full((m,K),1/K,device="cuda",dtype=torch.float32)
   values=(2.**((torch.arange(m,device="cuda")+rank)%4)).to(torch.bfloat16)
   x=values[:,None].expand(m,H).contiguous()
   output=torch.empty_like(x)
   def run():
    received,counts,handle,_,hook=buffer.low_latency_dispatch(x,ids,weights,capacity,E,quant_type=qt,quant_group_size=0,return_recv_hook=True)
    hook()
    if qt:
     packed,scale=received
     if scale.ndim==2:scale=scale[...,None]
     expert=(packed.float()*scale).to(torch.bfloat16)
    else:expert=received
    _,_,hook=buffer.low_latency_combine(expert,ids,weights,handle,return_recv_hook=True,out=output)
    hook()
    return counts
   counts=run();torch.cuda.synchronize()
   torch.testing.assert_close(output,x,rtol=0,atol=0)
   assert counts[0].item()==world*m,counts
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):captured_counts=run()
   for iteration in range(3):
    x.mul_(2)
    if iteration==1:
     # Live GPU routing changes, preserving static shapes and graph addresses.
     ids.add_(1)
    elif iteration==2:
     ids[::3]=-1
    graph.replay();torch.cuda.synchronize()
    expected=x.clone()
    if iteration==2:expected[::3]=0
    torch.testing.assert_close(output,expected,rtol=0,atol=0)
   results.append({"capacity":capacity,"quant_type":qt,"requests":m,"worst_expert_rows":world*m,"eager_exact":True,"graph_replays_exact":3,"dynamic_routes_and_inactive_rows":True})
   del graph
finally:
 torch.cuda.synchronize();dist.barrier();buffer.destroy();dist.destroy_process_group()
if rank==0:
 result={"world":world,"experts":E,"hidden":H,"topk":K,"rdma_hint":hint,"cases":results,"time":time.time()}
 Path(os.getenv("STEP5_DEEPEP_ALIGNMENT_RESULT", "/tmp/step5-deepep-dispatch-alignment.json")).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
