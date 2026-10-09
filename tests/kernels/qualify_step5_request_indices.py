# SPDX-License-Identifier: Apache-2.0
"""Native fixed-bound request-map qualification, including changing-input graphs."""
import json, os
from pathlib import Path
import torch
from vllm_hcu.model_executor.layers.step5_dsa_shortquery import prefill_request_indices_device
cases=[
 ([0,4,8,12,16],0,16), ([0,4,8,8,8],0,16),
 ([0,0,4,4,8],0,16), ([0,0,0,0,0],0,16),
 ([0,1,2,6,10,10],2,12), ([0,1,1,5,5,9],2,8),
 ([0,4,8,8],0,8), ([0,4,8],0,0),
 ([0]+[4*i for i in range(1,65)],0,256),
 ([0]+[4*min(i,47) for i in range(1,65)],0,256),
]
def expected(starts,decode,tokens):
 counts=torch.tensor(starts[decode+1:])-torch.tensor(starts[decode:-1])
 real=torch.repeat_interleave(torch.arange(decode,len(starts)-1,dtype=torch.int32),counts)
 if real.numel()<tokens:
  real=torch.cat((real,torch.full((tokens-real.numel(),),len(starts)-2,dtype=torch.int32)))
 return real[:tokens]
rows=[]
for starts,decode,tokens in cases:
 x=torch.tensor(starts,dtype=torch.int32,device="cuda")
 want=expected(starts,decode,tokens)
 got=prefill_request_indices_device(x,decode,len(starts)-1-decode,tokens)
 assert torch.equal(got.cpu(),want),(starts,got.cpu(),want)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  out=prefill_request_indices_device(x,decode,len(starts)-1-decode,tokens)
 g.replay();torch.cuda.synchronize()
 assert torch.equal(out.cpu(),want)
 # The same captured launch must follow changed boundaries and more padding.
 changed=[min(v,starts[decode]+4) if i>decode else v for i,v in enumerate(starts)]
 x.copy_(torch.tensor(changed,dtype=torch.int32,device="cuda"))
 g.replay();torch.cuda.synchronize()
 assert torch.equal(out.cpu(),expected(changed,decode,tokens))
 rows.append(dict(starts=starts,decode=decode,tokens=tokens,graph_replays=2,passed=True))
report=dict(qualified=True,cases=rows,whole_model_validated=False)
Path(os.environ["STEP5_REQUEST_MAP_RESULT"]).write_text(json.dumps(report,indent=2))
print(json.dumps(report))
