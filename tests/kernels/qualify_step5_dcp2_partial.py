
"""Native two-GPU DCP2 attention prototype proof, not model support/performance."""
from pathlib import Path
import os,json,hashlib,time,datetime,psutil,gc
os.environ.pop("NCCL_TOPO_FILE",None)
for x in psutil.process_iter(["pid","name","cmdline"]):
    if (x.info["name"] or "").startswith(("VLLM::","vllm::")) or (
       x.info["cmdline"] and "serve" in x.info["cmdline"] and
       any(Path(a).name=="vllm" for a in x.info["cmdline"])):
        if x.status()!=psutil.STATUS_ZOMBIE:
            raise RuntimeError(f"Active model process {x.info['pid']}; no concurrent GPU qualification")
import torch,torch.distributed as dist,numpy as np
# Load the source beside this test, not an installed/editable HCU package.
# A small immutable snapshot need not duplicate the complete plugin runtime.
import importlib.util
_source=Path(__file__).resolve().parents[2]/"vllm_hcu/ops/step5_dcp2.py"
_spec=importlib.util.spec_from_file_location("step5_dcp2_qualified_source",_source)
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
partial,merge=_module.partial,_module.merge
root=Path(os.environ.get("STEP5_DCP2_RESULT_DIR","/tmp/step5-dcp2-qualification"))
root.mkdir(parents=True,exist_ok=True)
operator=Path(__file__).resolve().parents[2]/"vllm_hcu/ops/step5_dcp2.py"
rank=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(rank)
dist.init_process_group("nccl",timeout=datetime.timedelta(seconds=300))
assert dist.get_world_size()==2
props=torch.cuda.get_device_properties(rank)
assert props.gcnArchName.split(":")[0]=="gfx938"
record={"global_region_size":int(os.environ.get("STEP5_DCP2_REGION_SIZE","128")),"source_sha256":hashlib.sha256(operator.read_bytes()).hexdigest(),
        "world_size":2,"head_dim":192,"q_heads":64,"kv_heads":4,
        "native_distributed":True,"model_integrated":False,"cases":[]}
global_region_size=int(os.environ.get("STEP5_DCP2_REGION_SIZE","128"))
device=torch.device("cuda",rank)
generator=torch.Generator(device="cpu").manual_seed(7707)
def oracle(q,k,v,regions,counts,qpos,window,owner=None):
    q=q.float().numpy().astype(np.float64);k=k.float().numpy().astype(np.float64)
    v=v.float().numpy().astype(np.float64)
    nq,nh,hd=q.shape
    output=np.zeros(q.shape);lse=np.full((nq,nh),-np.inf)
    for qi in range(nq):
        for g in range(4):
            rr=regions[g*nq+qi,:counts[g*nq+qi]].numpy()
            live=rr[(rr>=0)&(rr<(len(k)+global_region_size-1)//global_region_size)]
            pos=np.arange(len(k));mask=np.isin(pos//global_region_size,live)&(pos<=int(qpos[qi]))
            if window:mask&=pos>=int(qpos[qi])-window+1
            if owner is not None:mask&=pos%2==owner
            pos=pos[mask]
            if not len(pos):continue
            scores=q[qi,g*16:(g+1)*16]@k[pos,g].T/np.sqrt(hd)
            maximum=scores.max(axis=-1);prob=np.exp(scores-maximum[:,None])
            denom=prob.sum(axis=-1)
            output[qi,g*16:(g+1)*16]=prob@v[pos,g]/denom[:,None]
            lse[qi,g*16:(g+1)*16]=maximum+np.log(denom)
    return torch.from_numpy(output),torch.from_numpy(lse)
def check(actual,expected,actual_lse,expected_lse):
    a=actual.cpu().double();e=expected.double()
    assert torch.isfinite(a).all()
    diff=(a-e).abs()
    l2=float(torch.linalg.vector_norm(a-e)/torch.linalg.vector_norm(e).clamp_min(1e-30))
    maxerr=float(diff.max());scale=float(e.abs().max())
    assert l2<=.002,(l2,maxerr,scale)
    assert maxerr<=.0045*scale+1e-7,(l2,maxerr,scale)
    al=actual_lse.cpu().double();el=expected_lse.double()
    assert torch.equal(torch.isneginf(al),torch.isneginf(el))
    live=torch.isfinite(el)
    lerr=float((al[live]-el[live]).abs().max()) if live.any() else 0.
    assert lerr<=.00012,lerr
    return dict(relative_l2=l2,max_abs=maxerr,lse_max_abs=lerr)
def inputs(length,qlen,sparse,mode=0):
    pages=(length+127)//128
    nregions=(length+global_region_size-1)//global_region_size
    table=torch.randperm(pages,generator=generator,dtype=torch.int64).to(torch.int32)[None]
    k=torch.randn((length,4,192),generator=generator).to(torch.bfloat16)
    v=torch.randn((length,4,192),generator=generator).to(torch.bfloat16)
    q=torch.randn((qlen,64,192),generator=generator).to(torch.bfloat16)
    if mode==1:q.zero_();k.zero_();v.zero_()
    if mode==2:q.mul_(1e-4);k.mul_(1e-4);v.mul_(1e-6)
    qpos=torch.arange(length-qlen,length,dtype=torch.int32)
    regions=torch.full((qlen*4,nregions+2),-1,dtype=torch.int32)
    counts=torch.empty((qlen*4,),dtype=torch.int32)
    for g in range(4):
        for qi in range(qlen):
            chosen=list(range(nregions)) if not sparse else sorted(set([0,nregions//2,nregions-1]))
            # Distinct valid regions plus explicit masked out-of-range/tail entries.
            regions[g*qlen+qi,:len(chosen)]=torch.tensor(chosen,dtype=torch.int32)
            regions[g*qlen+qi,len(chosen)]=nregions+3
            counts[g*qlen+qi]=len(chosen)+1
    if mode==1:
        qpos[-1]=-1
    requests=torch.zeros((qlen,),dtype=torch.int32)
    cache_k=torch.full((pages*64,4,192),float("nan"),dtype=torch.bfloat16)
    cache_v=torch.full_like(cache_k,float("nan"))
    pos=torch.arange(rank,length,2)
    slots=table[0,pos//128].long()*64+(pos%128)//2
    cache_k[slots]=k[pos];cache_v[slots]=v[pos]
    # Equal generators on both ranks yield identical global inputs/page permutations.
    return q,k,v,regions,counts,qpos,requests,table,cache_k,cache_v
def run(q,k,v,regions,counts,qpos,requests,table,window,splits):
    outputs,lses,packed=partial(q,k,v,regions,counts,qpos,requests,table,
         rank=rank,window=window,splits=splits,global_region_size=global_region_size)
    local,local_lse=merge(outputs,lses)
    message=torch.cat([local.reshape(-1),local_lse.reshape(-1)])
    gathered=torch.empty((2*message.numel(),),device=device,dtype=torch.float32)
    dist.all_gather_into_tensor(gathered,message)
    gathered=gathered.reshape(2,-1)
    nh=q.shape[1];nq=q.shape[0];size=nq*nh*192
    states=gathered[:,:size].contiguous().reshape(2,nq,nh,192)
    lse=gathered[:,size:].contiguous().reshape(2,nq,nh)
    combined,combined_lse=merge(states,lse)
    narrowed,_=merge(states,lse,dtype=torch.bfloat16)
    return combined,combined_lse,local,local_lse,packed,narrowed
cases=[(length,qlen,sparse,window,1 if length<129 else 4,0)
       for length in [1,2,127,128,129,257]
       for qlen in [1,2,3,4] if qlen<=length
       for sparse,window in [(False,0),(True,0),(False,128)]]
cases += [(65538,4,True,0,8,0),(65538,4,False,128,8,0)]
# Full 64K global comparison is a separate fixed-shape eager/Graph case.
cases += [(65538,4,False,0,8,0),(129,4,False,128,4,1),(129,4,True,0,4,2)]
if os.environ.get("STEP5_DCP2_CASE_SCOPE")=="graph_lifecycle":
    cases=[case for case in cases if (case[0],case[1],case[2],case[3],case[5])
           in [(257,4,True,0,0),(65538,4,False,0,0)]]
record["case_scope"]=os.environ.get("STEP5_DCP2_CASE_SCOPE","full")
for ci,(length,qlen,sparse,window,splits,mode) in enumerate(cases):
    cpu=inputs(length,qlen,sparse,mode)
    q,k,v,regions,counts,qpos,requests,table,ck,cv=cpu
    gpu=[x.to(device) for x in [q,ck,cv,regions,counts,qpos,requests,table]]
    result=run(*gpu,window,splits)
    expected,expected_lse=oracle(q,k,v,regions,counts,qpos,window)
    local,local_lse=oracle(q,k,v,regions,counts,qpos,window,owner=rank)
    full_error=check(result[0],expected,result[1],expected_lse)
    local_error=check(result[2],local,result[3],local_lse)
    assert result[2].dtype==torch.float32 and result[3].dtype==torch.float32
    assert torch.equal(result[5],result[0].to(torch.bfloat16))
    packed_expected=torch.full_like(regions,-1)
    for row in range(len(regions)):
        qi=row%qlen
        for slot in range(int(counts[row])):
            logical=int(regions[row,slot])
            per_page=128//global_region_size
            if logical<0 or logical//per_page>=table.shape[1] or qpos[qi]<0:continue
            valid=min(global_region_size,max(0,int(qpos[qi])-logical*global_region_size+1))
            local_count=(valid+1-rank)//2
            if local_count:packed_expected[row,slot]=(int(table[0,logical//per_page])*per_page+logical%per_page)|(local_count<<24)
    assert torch.equal(result[4].cpu(),packed_expected)
    item=dict(length=length,qlen=qlen,sparse=sparse,window=window,splits=splits,
              mode=mode,full=full_error,local=local_error,packed_exact=True)
    # Capture includes actual NCCL/RCCL all-gather and both merges.
    if (length,qlen,sparse,window,mode) in [(257,4,True,0,0),(65538,4,False,0,0)]:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run(*gpu,window,splits)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=run(*gpu,window,splits)
        changing=[]
        for multiplier in [0.,1e-4,-.7]:
            changed=(q.float()*multiplier).to(torch.bfloat16)
            gpu[0].copy_(changed)
            graph.replay();torch.cuda.synchronize()
            expected,expected_lse=oracle(changed,k,v,regions,counts,qpos,window)
            err=check(captured[0],expected,captured[1],expected_lse)
            changing.append(dict(multiplier=multiplier,**err))
        item["native_graph_replays"]=changing
        graph.reset()
        del graph,captured
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
    record["cases"].append(item)
    (root/f"gpu_qualification-rank{rank}.json").write_text(json.dumps(record,indent=2))
    if rank==0:print(json.dumps(dict(index=ci,**item)),flush=True)
record.update(completed=True,finished=time.time())
(root/f"gpu_qualification-rank{rank}.json").write_text(json.dumps(record,indent=2))
dist.barrier();dist.destroy_process_group()
if rank==0:print("DCP2_GPU_PROTOTYPE_QUALIFIED",len(cases),flush=True)
