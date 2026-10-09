# SPDX-License-Identifier: Apache-2.0
"""Opt-in startup memory attribution, excluded from performance measurements."""
import os

def trace_memory(worker, phase):
    if os.getenv("VLLM_STEP5_MEMORY_LIFECYCLE_TRACE", "0") != "1":
        return
    import torch,json,time
    from pathlib import Path
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    reserved=torch.cuda.memory_reserved()
    row=dict(phase=phase,time=time.time(),free=free,total=total,
             allocated=torch.cuda.memory_allocated(),reserved=reserved,
             outside_torch=total-free-reserved,
             profiled_total_consumed=getattr(worker,"total_consumed",None),
             peak_activation_memory=getattr(worker,"peak_activation_memory",None),
             requested_memory=getattr(worker,"requested_memory",None),
             dsa_budget=getattr(worker,"_step5_dsa_memory_budget",None))
    baseline=getattr(worker,"init_snapshot",None)
    if baseline is not None:
        row["init_snapshot"]={k:getattr(baseline,k,None) for k in
                              ("free_memory","total_memory","torch_memory","non_torch_memory")}
    handles=[]
    from vllm.distributed import get_ep_group
    comm=getattr(get_ep_group(),"device_communicator",None)
    manager=getattr(comm,"all2all_manager",None)
    cache=getattr(getattr(manager,"handle_cache",None),"_cache",{})
    for buffer in cache.values():
        handles.append(dict(type=type(buffer).__name__,
                            nvl_bytes=getattr(buffer,"num_nvl_bytes",None),
                            rdma_bytes=getattr(buffer,"num_rdma_bytes",None),
                            low_latency=getattr(buffer,"low_latency_mode",None)))
    row["deep_ep_handles"]=handles
    row["ht_buffer_budget"]=getattr(manager,"_step5_ht_buffer_budget",None)
    worker._step5_startup_memory_rows=getattr(worker,"_step5_startup_memory_rows",[])
    worker._step5_startup_memory_rows.append(row)
    label=os.environ["VLLM_STEP5_MEMORY_LIFECYCLE_LABEL"]
    dp=worker.vllm_config.parallel_config.data_parallel_rank
    path=Path("/workspace/step5-attention-optimization")/(label+f"-startup-dp{dp}-tp{worker.rank}.json")
    path.write_text(json.dumps(dict(rows=worker._step5_startup_memory_rows,
                                   pid=os.getpid(),rank=worker.rank,
                                   diagnostics_only=True),indent=2))
