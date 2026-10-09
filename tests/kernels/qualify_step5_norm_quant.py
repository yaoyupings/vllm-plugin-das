# SPDX-License-Identifier: Apache-2.0
"""Qualify the isolated Step5 norm/quant op against actual TP2 QKV weights.

Refuses active vLLM services. This does not qualify model accuracy or routing.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import os,sys,json,time,hashlib
import triton
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bf16-model",type=Path,required=True)
parser.add_argument("--runtime-model",type=Path,required=True)
parser.add_argument("--calibration",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--layer",type=int,default=0)
parser.add_argument("--shapes",type=int,nargs="+",default=[1,4,16,32,64,128,256,512,6144])
args=parser.parse_args()
if args.layer<0 or any(m<=0 for m in args.shapes):
    parser.error("Layer must be nonnegative and all token shapes positive")
args.output.parent.mkdir(parents=True,exist_ok=True)
os.environ["VLLM_HCU_USE_CUSTOM_OPS"]="1"
os.environ["VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM"]="1"
os.environ["VLLM_STEP5_QKV_FP8_SCALE_MODE"]="output_mse"
os.environ["VLLM_STEP5_QKV_FP8_CALIBRATION_FILE"]=str(args.calibration)
import psutil
active=[]
for process in psutil.process_iter(["pid","status","cmdline","name"]):
    try:
        cmd=process.info["cmdline"] or []
        if process.info["status"]==psutil.STATUS_ZOMBIE:continue
        if ("serve" in cmd and any(Path(part).name=="vllm" for part in cmd)) or (process.info["name"] or "").startswith("VLLM::Worker"):
            active.append({"pid":process.pid,"cmd":cmd[:6],"name":process.info["name"]})
    except (psutil.AccessDenied,psutil.NoSuchProcess):continue
if active:raise RuntimeError(f"Do not run GPU qualification beside active vLLM: {active}")
import torch
from torch import nn
from safetensors import safe_open
base=args.output.parent
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo))
from vllm_hcu.ops.step5_norm_quant import step5_gemma_norm_fp8_quant as source_run
from torch._inductor.codecache import PyCodeCache
from vllm.config import CompilationConfig,KernelConfig,set_current_vllm_config
from vllm_hcu.runtime_compat.scaled_mm import install_fp8_scaled_mm_compat
from vllm_hcu.patch.worker.op_opt.patch_input_quant_fp8 import apply_to_module
from vllm.model_executor.layers.quantization import input_quant_fp8
from vllm_hcu.model_executor.layers.step5_qkv_fp8 import Step5QKVFp8LinearMethod
from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
from vllm.utils.torch_utils import direct_register_custom_op
result_path=args.output
assert not result_path.exists()

record={"started":time.time(),"cases":[],"adopted":False,"model_accuracy_not_claimed":True,
 "candidate_sha256":hashlib.sha256((repo/"vllm_hcu/ops/step5_norm_quant.py").read_bytes()).hexdigest()}
def save():result_path.write_text(json.dumps(record,indent=2))
save()
try:
    ctx=set_current_vllm_config(NS(compilation_config=CompilationConfig(mode=0,custom_ops=["none"]),kernel_config=KernelConfig(ir_op_priority={"rms_norm":["native"]}),model_config=NS(dtype=torch.bfloat16,model=str(args.runtime_model))))
    ctx.__enter__();install_fp8_scaled_mm_compat();apply_to_module(input_quant_fp8)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(4707)
    torch.backends.cuda.matmul.allow_tf32=False
    layer=nn.Module();layer.input_size=4096;layer.output_partition_sizes=[6144,384,384]
    checkpoint=args.bf16_model
    index_path=checkpoint/"model.safetensors.index.json"
    mapping=json.loads(index_path.read_text())["weight_map"]
    shards=[]
    for projection,local_width in (("q_proj",6144),("k_proj",384),("v_proj",384)):
        key=f"model.layers.{args.layer}.self_attn.{projection}.weight"
        with safe_open(str(checkpoint/mapping[key]),framework="pt",device="cpu") as f:
            tensor=f.get_tensor(key)
            assert tensor.shape==(local_width*2,4096) and tensor.dtype==torch.bfloat16
            shards.append(tensor[:local_width].clone())
    weight_cpu=torch.cat(shards,dim=0)
    record["weight_fixture"]={"model_layer":args.layer,"tp_rank":0,"tp_size":2,
     "checkpoint_index_sha256":hashlib.sha256(index_path.read_bytes()).hexdigest(),
     "local_bf16_weight_sha256":hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}
    layer.weight=nn.Parameter(weight_cpu.t().to("cuda"),requires_grad=False)
    del shards,tensor,weight_cpu
    method=Step5QKVFp8LinearMethod(prefix=f"model.layers.{args.layer}.self_attn");method.process_weights_after_loading(layer)
    assert method.use_lightop_quant and method.scale_mode=="output_mse"
    assert getattr(type(method.kernel),"_hcu_fp8_backend",None)=="lightop"
    record.update(device=torch.cuda.get_device_name(0),baseline="current Step5QKVFp8LinearMethod.apply / LightOp",
     weight_scale_mode=method.scale_mode,calibration_sha256=hashlib.sha256(Path(os.environ["VLLM_STEP5_QKV_FP8_CALIBRATION_FILE"]).read_bytes()).hexdigest())
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    config=json.loads((checkpoint/"config.json").read_text())
    text=config.get("text_config",config)
    epsilon=float(text["rms_norm_eps"])
    norm=GemmaRMSNorm(4096,epsilon)
    key=f"model.layers.{args.layer}.input_layernorm.weight"
    with safe_open(str(checkpoint/mapping[key]),framework="pt",device="cpu") as f:
        norm_weight=f.get_tensor(key).cuda().contiguous()
    norm.weight=nn.Parameter(norm_weight,requires_grad=False)
    from vllm import ir
    native_impl=ir.ops.rms_norm.impls["native"].impl_fn
    def native_norm(x):
  # VllmIRLoweringPass inserts func_impl_fn of the selected native RMSNorm.
        return native_impl(x,norm.weight.float()+1.,epsilon).to(torch.bfloat16)
    torch._dynamo.config.recompile_limit=64
    compiled_norm=torch.compile(native_norm,backend="inductor",fullgraph=True,dynamic=False)
    record.update(epsilon=epsilon,compiled_native_norm=True,norm_weight_dtype=str(norm_weight.dtype),
     baseline="decomposed native Gemma norm/BF16 + current LightOp QKV method")
    def launch_config(m):
  # PyCodeCache contains only this process's actual compiled reference modules.
        selected=None
        for module in reversed(PyCodeCache.modules):
            for name,kernel in vars(module).items():
                if not name.startswith("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt"):continue
                if getattr(kernel,"size_hints",{}).get("x")!=triton.next_power_of_2(m):continue
                launchers=getattr(kernel,"launchers",None)
                if launchers: selected=(module,kernel,launchers[0].config);break
            if selected:break
        if selected is None:raise RuntimeError(f"No actual compiled native norm kernel found for M={m}")
        module,kernel,cfg=selected
        assert cfg.num_warps==8
        rows=cfg.kwargs["XBLOCK"];rblock=cfg.kwargs["R0_BLOCK"]
        if rows==1 and rblock in (2048,4096):
            spt=rblock//512;wrow=1
        elif rows in (2,4) and rblock==2048:
            spt=8;wrow=2
        else:raise RuntimeError(f"Native norm layout not yet qualified: {cfg}")
        return {"rows":rows,"rblock":rblock,"spt":spt,"wrow":wrow,
                "native_config":str(cfg),"native_cache_module":str(module.__file__)}
    current={}
    def run(x,weight,epsilon,*,warps):
        key=tuple(current[k] for k in ("rows","rblock","spt","wrow"))
        layout=((1,2048,4,1),(1,4096,8,1),(2,2048,8,2),(4,2048,8,2)).index(key)
        if layout in (2,3):layout=4
        return source_run(x,weight,epsilon,layout)
    def oracle(x,h,q,scales):
        reference=compiled_norm(x)
        mismatches=int((h!=reference).sum())
  # Report exact BF16 behavior before considering any adoption.
        if mismatches:
            torch.save({"residual":x.detach().cpu(),"norm_weight":norm_weight.cpu(),
             "reference":reference.cpu(),"candidate":h.cpu(),"epsilon":epsilon},
              args.output.with_suffix(".mismatch.pt"))
        torch.testing.assert_close(h,reference,rtol=0,atol=0)
        nq,ns=lightop_fp8_runtime.quantize(h,torch.float8_e4m3fn,direct_register_custom_op)
        torch.testing.assert_close(scales,ns,rtol=1e-6,atol=0)
        qmismatch=q.float()!=nq.float()
        count=int(qmismatch.sum())
        if count:
            av=q.float()[qmismatch].double();bv=nq.float()[qmismatch].double()
            levels=torch.cat([torch.zeros(1,device="cuda",dtype=torch.float64),
             torch.arange(1,8,device="cuda",dtype=torch.float64)*2**-9,
             torch.cat([2.**e*(1+torch.arange(8,device="cuda",dtype=torch.float64)/8) for e in range(-6,9)])])
            levels=levels[levels<=448]
            ai=torch.searchsorted(levels,av.abs());bi=torch.searchsorted(levels,bv.abs())
            assert ((ai-bi).abs()==1).all() and (av*bv>=0).all()
            am=h.double().abs().amax(1,keepdim=True)
            ideal=h.double()*448/torch.where(am>0,am,torch.ones_like(am))
            gap=(ideal[qmismatch]-(av+bv)*.5).abs()
            assert (gap<=8*torch.finfo(torch.float32).eps*ideal[qmismatch].abs().clamp_min(1)).all()
        return {"norm_bf16_mismatches":mismatches,"quant_midpoint_mismatches":count}
    def scaled_mm(q,s):
        return method.kernel.apply_scaled_mm(A=q,B=layer.weight,As=s,Bs=layer.weight_scale,
          out_dtype=torch.bfloat16,bias=None,output_shape=[q.shape[0],6912])
    def baseline(x):
        h=compiled_norm(x);return h,method.apply(layer,h)
    def candidate(x,warps):
        h,q,s=run(x,norm_weight,epsilon,warps=warps)
        return h,scaled_mm(q,s)
    def timed(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):ret=fn()
        for _ in range(3):g.replay()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(30):g.replay()
        b.record();b.synchronize();value=a.elapsed_time(b)/30
        del g,ret;return value
    for m in args.shapes:
        x=torch.randn((m,4096),device="cuda",dtype=torch.float32)
        row={"tokens":m,"variants":[]}
  # Explicitly warm the exact shape's compiled production norm before Graph.
        compiled_norm(x);torch.cuda.synchronize()
        current=launch_config(m);row["layout"]=current
        for warps in (8,):
            h,q,scales=run(x,norm_weight,epsilon,warps=warps)
            check=oracle(x,h,q,scales)
            out=scaled_mm(q,scales)
            ref=(q.float()@layer.weight.float())*scales*layer.weight_scale.t()
            torch.testing.assert_close(out.float(),ref,rtol=.016,atol=.008)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):gh,gq,gs=run(x,norm_weight,epsilon,warps=warps)
            for value in (0.,1.,-1.,1e-14,-1e-14):
                x.fill_(value);g.replay();torch.cuda.synchronize();oracle(x,gh,gq,gs)
            x.copy_(torch.randn_like(x));g.replay();torch.cuda.synchronize();oracle(x,gh,gq,gs)
            del g,gh,gq,gs
   # Both timings use the same final input, after correctness checks.
            row["baseline_graph_ms"]=timed(lambda:baseline(x))
            ms=timed(lambda:candidate(x,warps))
            row["variants"].append({"warps":4 if current["rows"]>=2 else 8,"rows":1,"graph_ms":ms,
              "speedup":row["baseline_graph_ms"]/ms,"changing_graph_replays":6,**check})
        record["cases"].append(row);save();print("NORM_QUANT",json.dumps(row),flush=True)
    record.update(completed=True,finished=time.time());save()
except Exception as e:
    record.update(error=repr(e),finished=time.time());save();raise
