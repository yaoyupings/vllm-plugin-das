
"""Execute the production metadata kernel via Triton's CPU interpreter.
Run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''.
No GPU or model work is performed.
"""
import os,importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
if os.environ.get("TRITON_INTERPRET")!="1":
    pytest.skip("requires explicit Triton CPU interpreter",allow_module_level=True)
source=Path(__file__).resolve().parents[2]/"vllm_hcu/ops/step5_dcp2.py"
spec=importlib.util.spec_from_file_location("step5_dcp2_pack_cpu",source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
@pytest.mark.parametrize("grain",[8,128])
@pytest.mark.parametrize("rank",[0,1])
@pytest.mark.parametrize("length",[1,2,7,8,9,127,128,129,257,65538])
def test_exact_device_packing(grain,rank,length):
    rng=np.random.default_rng(4707+length)
    pages=(length+127)//128
    region_count=(length+grain-1)//grain
    table=np.stack([rng.permutation(pages),pages+rng.permutation(pages)]).astype(np.int32)
    if pages>1:
        table[0,0]=-1
        table[1,-1]=2*pages+1
    positions=np.array([-1,0,length//2,length-1],dtype=np.int32)
    requests=np.array([0,-1,1,1],dtype=np.int32)
    nq=len(positions);groups=4;rows=nq*groups
    logical=np.broadcast_to(np.r_[np.arange(region_count),-1,region_count+16].astype(np.int32),
                            (rows,region_count+2)).copy()
    counts=np.full(rows,region_count+2,dtype=np.int32)
    counts[-1]=0
    tensors=[torch.from_numpy(v) for v in [logical,counts,positions,requests,table]]
    out=torch.empty_like(tensors[0])
    module._pack_striped_regions[((logical.size+255)//256,)](
        *tensors,out,ROWS=rows,TOPK=logical.shape[1],QUERIES=nq,REQUESTS=2,
        TABLE_WIDTH=pages,PHYSICAL_PAGES=pages*2,GLOBAL_REGION_SIZE=grain,RANK=rank,BLOCK=256)
    expected=np.full_like(logical,-1)
    per_page=128//grain
    for row in range(rows):
        qi=row%nq;request=int(requests[qi]);qpos=int(positions[qi])
        if request<0 or qpos<0:continue
        for slot in range(int(counts[row])):
            region=int(logical[row,slot]);page=region//per_page
            if region<0 or page>=pages:continue
            physical_page=int(table[request,page])
            if physical_page<0 or physical_page>=2*pages:continue
            valid=min(grain,max(0,qpos-region*grain+1))
            count=(valid+1-rank)//2
            if not count:continue
            physical_region=physical_page*per_page+region%per_page
            expected[row,slot]=physical_region|(count<<24)
    np.testing.assert_array_equal(out.numpy(),expected)
