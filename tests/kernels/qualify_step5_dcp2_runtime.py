# SPDX-License-Identifier: Apache-2.0
"""Native Step5 Attention bridge using actual vLLM TP2/DCP2 collectives.

This covers gathered heads/selections, SWA512, sparse mixed query batches,
FP32 ReduceScatter and changing-input Graphs. It is not whole-model proof.
"""
import gc
import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch

for process in psutil.process_iter(["pid", "cmdline"]):
    command = process.info["cmdline"] or []
    if any(Path(x).name.lower() == "vllm"
           or Path(x).name.lower().startswith("vllm::")
           or x.startswith("vllm.entrypoints.") for x in command):
        raise RuntimeError(f"refusing qualification alongside vLLM pid={process.pid}")
os.environ.pop("NCCL_TOPO_FILE", None)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == 2
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
source = Path(__file__).resolve().parents[2] / "vllm_hcu/ops"
package = types.ModuleType("_step5_runtime_gpu_snapshot")
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
runtime = importlib.import_module(package.__name__+".step5_dcp2_runtime")

from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment, initialize_model_parallel, get_dcp_group, get_tp_group,
)
from vllm.distributed.parallel_state import (
    destroy_model_parallel, destroy_distributed_environment,
)

qualify_adapter = os.environ.get("STEP5_DCP2_QUALIFY_ADAPTER", "0") == "1"
root = Path(os.environ.get("STEP5_DCP2_RESULT_DIR", "/tmp/step5-dcp2-runtime"))
root.mkdir(parents=True, exist_ok=True)
report = root/f"runtime-rank{rank}.json"
record = {
    "rank": rank, "pid": os.getpid(), "model_integrated": False, "qualified": False,
    "attention_adapter_qualified": False, "qualify_adapter": qualify_adapter,
    "source_sha256": {x: hashlib.sha256((source/x).read_bytes()).hexdigest()
                      for x in ["step5_dcp2_runtime.py", "step5_dcp2.py",
                                "step5_dcp2_exchange.py"]},
    "combine_cases": [], "attention_cases": [],
}


def save():
    report.write_text(json.dumps(record, indent=2))


def compare(actual, expected, tolerance=.006):
    actual = actual.cpu().double()
    assert not actual.isnan().any()
    difference = actual-expected.double()
    relative = float(difference.norm()/expected.double().norm().clamp_min(1.e-20))
    assert relative <= tolerance, relative
    return {"relative_l2": relative, "max_abs": float(difference.abs().max())}


# This is a communication/Attention qualification with no model loaded.
# Construct the default no-model config before selecting the tested groups;
# HCU's model-specific DCP validator requires an actual ModelConfig.
config = VllmConfig()
config.parallel_config = ParallelConfig(
    tensor_parallel_size=2, decode_context_parallel_size=2, disable_custom_all_reduce=True)
record["model_config_validated"] = False
with set_current_vllm_config(config):
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank,
                                distributed_init_method="env://", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=2,
                              decode_context_model_parallel_size=2)
    group = get_dcp_group()
    assert group.ranks == get_tp_group().ranks == [0, 1]
    record["communicator"] = str(type(group.device_communicator))
    record["group"] = group.ranks
    save()

    def check_lse(actual, expected):
        actual = actual.cpu().double()
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        assert not actual.isnan().any()
        finite = torch.isfinite(expected)
        error = float((actual[finite]-expected[finite]).abs().max()) if finite.any() else 0.
        assert error < 8.e-6, error
        return error

    def combine_suite(tokens):
        generator = torch.Generator().manual_seed(107+tokens)
        states = torch.randn(2, tokens, 64, 192, generator=generator)
        lses = torch.randn(2, tokens, 64, generator=generator)*8
        lses[0, 0, :16] = -float("inf")
        lses[1, 0, 16:32] = -float("inf")
        lses[:, 0, 32:48] = -float("inf")
        lses[0, 0, 48] = float("nan")
        lses[1, 0, 49] = float("inf")
        local_out, local_lse = states[rank].to(device), lses[rank].to(device)

        def oracle(scale):
            clean = torch.where(torch.isnan(lses) | torch.isposinf(lses),
                                -float("inf"), lses).double()
            peak = clean.max(0).values
            weights = torch.exp(clean-torch.where(torch.isneginf(peak), 0., peak))
            weights /= weights.sum(0).clamp_min(1.e-20)
            out = (states.double()*scale*weights.unsqueeze(-1)).sum(0)
            h = slice(rank*32, (rank+1)*32)
            return out[:, h], torch.logsumexp(clean, dim=0)[:, h]

        def pipeline():
            return runtime.combine_rank_states(local_out, local_lse, group, return_lse=True)

        actual, lse = pipeline()
        expected, expected_lse = oracle(1.)
        case = {"tokens": tokens, **compare(actual, expected, .003),
                "lse_abs": check_lse(lse, expected_lse), "graphs": []}

        # Exact native FP32->BF16 cast and TP-head transpose, independently.
        rs = states[rank, :, :32].permute(1, 0, 2).contiguous().to(device)
        final = torch.empty(tokens, 32, 192, device=device, dtype=torch.bfloat16)
        runtime._finish_local_heads_kernel[((final.numel()+255)//256,)](
            rs, final, TOKENS=tokens, HEADS=32, BLOCK=256,
        )
        torch.testing.assert_close(final.cpu(), rs.cpu().permute(1, 0, 2).bfloat16(),
                                   rtol=0, atol=0)
        case["native_bf16_cast_exact"] = True
        if tokens in (4, 64):
            for _ in range(3):
                pipeline()
            torch.cuda.synchronize()
            group.barrier()
            graph = torch.cuda.CUDAGraph()
            with group.graph_capture() as capture:
                with torch.cuda.graph(graph, stream=capture.stream):
                    captured = pipeline()
            torch.cuda.current_stream().wait_stream(capture.stream)
            for scale in (0., -.7, 1.e-4):
                local_out.copy_((states[rank]*scale).to(device))
                group.barrier()
                graph.replay()
                torch.cuda.synchronize()
                expected, expected_lse = oracle(scale)
                case["graphs"].append({
                    "scale": scale, **compare(captured[0], expected, .003),
                    "lse_abs": check_lse(captured[1], expected_lse),
                })
            graph.reset()
            del graph, captured
            gc.collect()
        record["combine_cases"].append(case)
        save()

    def attention_suite(mode, qlens, context, capture_graph, topk=9):
        nreq = len(qlens)
        tokens = sum(qlens)+2  # Exercise real padded query rows.
        width = (context+8+127)//128
        pages = width*nreq
        generator = torch.Generator().manual_seed(110+context+tokens)
        history_k = torch.randn(nreq, width*128, 4, 192, generator=generator).bfloat16()
        history_v = torch.randn(nreq, width*128, 4, 192, generator=generator).bfloat16()
        q_all = torch.randn(tokens, 64, 192, generator=generator).bfloat16()
        k_all = torch.randn(tokens, 4, 192, generator=generator).bfloat16()
        v_all = torch.randn(tokens, 4, 192, generator=generator).bfloat16()
        nd = sum(x == 1 for x in qlens) if mode != "window" else 0
        if mode != "window":
            assert all(x == 1 for x in qlens[:nd])
        nactual = sum(qlens)
        md = SimpleNamespace(num_decode_tokens=nd, num_actual_tokens=nactual, block_size=128)
        lengths_cpu = torch.full((nreq,), context, dtype=torch.int32)
        starts_cpu = torch.tensor([0]+list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32)
        starts = starts_cpu.to(device)
        lengths = lengths_cpu.to(device)
        positions = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        table = torch.zeros(nreq, width, device=device, dtype=torch.int32)
        query = q_all[:, rank*32:(rank+1)*32].contiguous().to(device)
        key = k_all[:, rank*2:(rank+1)*2].contiguous().to(device)
        value = v_all[:, rank*2:(rank+1)*2].contiguous().to(device)
        backing = torch.empty((pages*64, 4, 2, 192), device=device, dtype=torch.bfloat16)
        kc, vc = backing[:, :, 0], backing[:, :, 1]
        if nd:
            words = torch.empty(2*nd, topk, device=device, dtype=torch.int64)
            dc = torch.empty(2*nd, device=device, dtype=torch.int32)
        else:
            words = dc = None
        np = nactual-nd
        if np and mode != "window":
            logical = torch.empty(2*np, topk, device=device, dtype=torch.int32)
            pc = torch.empty(2*np, device=device, dtype=torch.int32)
        else:
            logical = pc = None
        plan = SimpleNamespace(md=md, decode=(words, dc) if nd else None,
                               prefill=(None, pc, logical, None) if logical is not None else None)

        def phase(shift, scale, reverse, inactive):
            table_cpu = torch.arange(pages, dtype=torch.int32).reshape(nreq, width)
            if reverse:
                table_cpu = (pages-1-table_cpu).contiguous()
            table.copy_(table_cpu)
            lens_cpu = lengths_cpu+shift
            if inactive:
                lens_cpu[-1] = 0
            lengths.copy_(lens_cpu)
            pos = torch.full((tokens,), -1, dtype=torch.int64)
            request_of = torch.full((tokens,), -1, dtype=torch.int32)
            slot = torch.full((tokens,), -1, dtype=torch.int64)
            for request, qlen in enumerate(qlens):
                a, b = int(starts_cpu[request]), int(starts_cpu[request+1])
                if int(lens_cpu[request]) == 0:
                    continue
                pos[a:b] = torch.arange(context+shift-qlen, context+shift)
                request_of[a:b] = request
                phys = table_cpu[request, pos[a:b]//128].long()
                slot[a:b] = phys*64+(pos[a:b]%128)//2
                slot[a:b][pos[a:b]%2 != rank] = -1
            positions.copy_(pos)
            slots.copy_(slot)
            qq, kk, vv = [(x*scale).bfloat16() for x in (q_all, k_all, v_all)]
            query.copy_(qq[:, rank*32:(rank+1)*32])
            key.copy_(kk[:, rank*2:(rank+1)*2])
            value.copy_(vv[:, rank*2:(rank+1)*2])
            # Rebuild physical backing from a logical-history oracle.
            physical_k = torch.empty(pages, 128, 4, 192, dtype=torch.bfloat16)
            physical_v = torch.empty_like(physical_k)
            for request in range(nreq):
                physical_k[table_cpu[request].long()] = history_k[request].view(width,128,4,192)
                physical_v[table_cpu[request].long()] = history_v[request].view(width,128,4,192)
            kc.copy_(physical_k[:, rank::2].reshape_as(kc))
            vc.copy_(physical_v[:, rank::2].reshape_as(vc))
            for token in range(tokens):
                req = int(request_of[token])
                if req >= 0:
                    p = int(pos[token])
                    page = int(table_cpu[req, p//128])
                    physical_k[page, p%128] = kk[token]
                    physical_v[page, p%128] = vv[token]
            selected = torch.full((4, nactual, topk), -1, dtype=torch.int32)
            counts = torch.zeros(4, nactual, dtype=torch.int32)
            if mode != "window":
                for g in range(4):
                    for token in range(nactual):
                        if request_of[token] < 0:
                            continue
                        current = int(pos[token])//8
                        candidates = list(range(g, current, 4))
                        chosen = candidates[-(topk-1):]+[current]
                        # Exercise empty selected rows with both ranks empty.
                        if g == 3 and token == 0:
                            chosen = []
                        counts[g, token] = len(chosen)
                        selected[g, token, :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
                local = selected[rank*2:(rank+1)*2]
                local_counts = counts[rank*2:(rank+1)*2]
                if nd:
                    encoded = torch.zeros(2, nd, topk, dtype=torch.int64)
                    for g in range(2):
                        for t in range(nd):
                            req = int(request_of[t])
                            for j in range(int(local_counts[g,t])):
                                r = int(local[g,t,j])
                                phys = int(table_cpu[req, r//16])*16+r%16
                                encoded[g,t,j] = (phys << 32) | (r*8)
                    words.copy_(encoded.flatten(0,1))
                    dc.copy_(local_counts[:, :nd].contiguous().flatten())
                if logical is not None:
                    logical.copy_(local[:, nd:].contiguous().flatten(0,1))
                    pc.copy_(local_counts[:, nd:].contiguous().flatten())

            expected = torch.zeros(tokens, 64, 192, dtype=torch.float64)
            expected_lse = torch.full((tokens, 64), -float("inf"), dtype=torch.float64)
            for token in range(nactual):
                req, pos_t = int(request_of[token]), int(pos[token])
                if req < 0:
                    continue
                for g in range(4):
                    if mode == "window":
                        logical_tokens = torch.arange(max(0,pos_t-511), pos_t+1)
                    else:
                        rs = selected[g, token, :int(counts[g,token])]
                        logical_tokens = (rs[:,None]*8+torch.arange(8)[None,:]).flatten().long()
                        logical_tokens = logical_tokens[logical_tokens <= pos_t]
                    if not logical_tokens.numel():
                        continue
                    phys = table_cpu[req, logical_tokens//128].long()
                    keys = physical_k[phys, logical_tokens%128, g].double()
                    vals = physical_v[phys, logical_tokens%128, g].double()
                    heads = slice(g*16,(g+1)*16)
                    scores = qq[token, heads].double() @ keys.T / (192**.5)
                    expected[token,heads] = scores.softmax(-1) @ vals
                    expected_lse[token,heads] = torch.logsumexp(scores, dim=-1)
            h = slice(rank*32,(rank+1)*32)
            return (expected[:,h], expected_lse[:,h],
                    physical_k[:,rank::2].reshape_as(kc),
                    physical_v[:,rank::2].reshape_as(vc))

        if qualify_adapter:
            from vllm.forward_context import set_forward_context
            from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAMetadata
            from vllm_hcu.model_executor.layers.step5_dcp2_attention import Step5DCP2Attention
            md = Step4DSAMetadata(
                num_decodes=nd, num_prefills=nreq-nd, num_decode_tokens=nd,
                num_prefill_tokens=nactual-nd, num_actual_tokens=nactual,
                block_table=table, slot_mapping=slots, seq_lens=lengths,
                query_start_loc=starts, query_start_loc_cpu=starts_cpu,
                seq_lens_cpu=lengths_cpu, positions=positions,
                max_seq_len=context, block_size=128)
            plan.md = md
            layer = Step5DCP2Attention.__new__(Step5DCP2Attention)
            torch.nn.Module.__init__(layer)
            layer.layer_name = f"qualification.{mode}"
            layer.sliding_window = 512 if mode == "window" else None
            layer.step4_sparse_plan = plan
            layer.bind_kv_cache(
                backing.view(pages,64,4,384).permute(0,2,1,3))
            config.compilation_config.static_forward_context[layer.layer_name] = layer
            dependency = torch.empty(1,device=device,dtype=torch.int32)
            record["adapter_source_sha256"] = hashlib.sha256(
                (source.parent/"model_executor/layers/step5_dcp2_attention.py").read_bytes()
            ).hexdigest()

        def pipeline():
            if qualify_adapter:
                with set_forward_context(
                    {layer.layer_name:md},config,num_tokens=tokens,
                    slot_mapping={layer.layer_name:slots}):
                    out = layer(query,key,value,positions=positions,
                                plan_token=dependency if mode != "window" else None)
                return out.view_as(query), None
            return runtime.dcp2_attention(
                query, key, value, kc, vc, slots, positions, starts, lengths, table,
                group, window=512 if mode == "window" else 0, plan=plan,
                metadata=md, splits=2, return_lse=True,
            )

        expected = phase(0, 1., False, False)
        actual = pipeline()
        case = {"mode":mode, "qlens":qlens, "context":context, "topk_slots":topk,
                "padded_tokens":tokens,
                **compare(actual[0],expected[0]),
                "lse_abs":check_lse(actual[1],expected[1]) if actual[1] is not None else None,
                "graphs":[]}
        torch.testing.assert_close(kc.cpu(), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(vc.cpu(), expected[3], rtol=0, atol=0)
        case["cache_exact"] = True
        if capture_graph:
            for _ in range(3):
                pipeline()
            torch.cuda.synchronize()
            group.barrier()
            graph = torch.cuda.CUDAGraph()
            with group.graph_capture() as capture:
                with torch.cuda.graph(graph, stream=capture.stream):
                    captured = pipeline()
            torch.cuda.current_stream().wait_stream(capture.stream)
            for shift, scale, reverse, inactive in [
                (1, -.7, True, False), (3, 1.e-4, False, True), (7, 1., True, False),
            ]:
                expected = phase(shift, scale, reverse, inactive)
                group.barrier()
                graph.replay()
                torch.cuda.synchronize()
                result = {"shift":shift,"scale":scale,"reverse_pages":reverse,
                          "inactive_last_request":inactive,
                          **compare(captured[0],expected[0]),
                          "lse_abs":check_lse(captured[1],expected[1]) if captured[1] is not None else None}
                torch.testing.assert_close(kc.cpu(), expected[2], rtol=0, atol=0)
                torch.testing.assert_close(vc.cpu(), expected[3], rtol=0, atol=0)
                result["cache_exact"] = True
                case["graphs"].append(result)
            graph.reset()
            del graph, captured
            gc.collect()
        record["attention_cases"].append(case)
        save()
        print(f"rank{rank} PASS {mode} q{qlens} ctx{context}", flush=True)

    try:
        selected_suite = os.environ.get("STEP5_DCP2_RUNTIME_SUITE", "all")
        if selected_suite not in ("all", "topk513"):
            raise ValueError("unknown Step5 DCP2 runtime qualification suite")
        record["suite"] = selected_suite
        if selected_suite == "all":
            for tokens in (1, 4, 64, 128):
                combine_suite(tokens)
            for mode, qlens, context, graph in [
                ("window", [1, 1, 1, 1], 129, True),
                ("window", [4]*16, 1025, True),
                ("window", [4], 65538, True),
                ("sparse", [1, 1, 1, 1], 129, True),
                ("sparse", [1, 1, 4, 4], 1025, True),
                ("sparse", [4]*16, 1025, True),
                ("sparse", [4], 65538, True),
                ("sparse", [64], 257, False),
            ]:
                attention_suite(mode, qlens, context, graph)

        attention_suite("sparse", [4], 65538, True, topk=513)
        record["qualified"] = True
        record["attention_adapter_qualified"] = qualify_adapter
        record["whole_model_dcp2_support"] = False
        save()
    except BaseException as error:
        record["error"] = repr(error)
        save()
        raise
    finally:
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()
