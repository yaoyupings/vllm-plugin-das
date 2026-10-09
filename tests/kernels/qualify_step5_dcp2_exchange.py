# SPDX-License-Identifier: Apache-2.0
"""Native two-rank QKV exchange, striped cache and changing-Graph qualification.

torchrun --standalone --nproc-per-node=2 tests/kernels/qualify_step5_dcp2_exchange.py
Refuses to run alongside vLLM. This proves the exchange primitive, not model
DCP2 support or a throughput gain.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import psutil
import torch
import torch.distributed as dist

for process in psutil.process_iter(["pid", "cmdline"]):
    command = process.info["cmdline"] or []
    if any(Path(x).name.lower() == "vllm"
           or Path(x).name.lower().startswith("vllm::")
           or x.startswith("vllm.entrypoints.") for x in command):
        raise RuntimeError(f"refusing GPU qualification alongside vLLM pid={process.pid}")
os.environ.pop("NCCL_TOPO_FILE", None)
torch.set_num_threads(1)
root = Path(os.environ.get("STEP5_DCP2_RESULT_DIR", "/tmp/step5-dcp2-exchange"))
root.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parents[2] / "vllm_hcu/ops/step5_dcp2_exchange.py"
spec = importlib.util.spec_from_file_location("step5_dcp2_exchange_gpu", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
assert world == 2 and rank in (0, 1)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
record = {"rank": rank, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "device": torch.cuda.get_device_name(), "cases": [], "model_integrated": False}
report = root / f"exchange_gpu-rank{rank}.json"


def gather_dim1(packed):
    # Same concat-gather / movedim / reshape semantics as vLLM's base
    # device communicator; the collective itself is real RCCL.
    output = torch.empty((2 * packed.shape[0], packed.shape[1]),
                         device=packed.device, dtype=packed.dtype)
    dist.all_gather_into_tensor(output, packed)
    return output.reshape(2, *packed.shape).movedim(0, 1).reshape(
        packed.shape[0], 2 * packed.shape[1])


def check(actual, expected):
    torch.testing.assert_close(actual.cpu().reshape(expected.shape), expected, rtol=0, atol=0)


try:

    record["decode_words"] = []
    for topk in [1, 2, 513]:
        logical = (torch.arange(8*topk, dtype=torch.int64).reshape(8, topk) % 8193)
        physical = logical.flip([0, 1]) + 4096
        words_cpu = (physical << 32) | (logical * 8)
        words_cpu[0, 0] = logical[0, 0] = 0
        words = words_cpu.to(device)
        counts = torch.zeros(8, device=device, dtype=torch.int32)
        def decode_pipeline():
            return module.decode_words_to_regions(words, counts)
        for count in [0, 1, topk]:
            counts.fill_(count)
            actual = decode_pipeline()
            oracle = logical.to(torch.int32)
            oracle[:, count:] = -1
            check(actual, oracle)
        case = {"topk": topk, "counts_eager": [0, 1, topk], "exact": True}
        if topk == 513:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = decode_pipeline()
            for count in [0, topk//2, topk]:
                counts.fill_(count)
                graph.replay()
                torch.cuda.synchronize()
                oracle = logical.to(torch.int32)
                oracle[:, count:] = -1
                check(captured, oracle)
            graph.reset()
            del graph, captured
            gc.collect()
            case["graph_counts"] = [0, topk//2, topk]
        record["decode_words"].append(case)
        report.write_text(json.dumps(record, indent=2))

    for heads in [4, 32]:
        for tokens in [1, 2, 3, 4, 16, 64, 128, 384]:
            for layout in ["sparse_v2", "flash_v2"]:
                generator = torch.Generator().manual_seed(106 + tokens + heads)
                bases = [torch.randn(tokens, heads + 4, 192, generator=generator).bfloat16()
                         for _ in range(2)]
                local = bases[rank].to(device)
                query = local[:, :heads].contiguous()
                key = local[:, heads:heads+2].contiguous()
                value = local[:, heads+2:]
                pages = (tokens + 127) // 128
                physical = torch.randperm(pages, generator=generator)
                positions = torch.arange(tokens)
                slots_cpu = physical[positions // 128]*64 + (positions % 128)//2
                slots_cpu[positions % 2 != rank] = -1
                slots_cpu[-1] = -1
                if tokens >= 3:
                    slots_cpu[-3] = pages*64
                slots = slots_cpu.to(device, dtype=torch.int64)
                if layout == "sparse_v2":
                    backing = torch.full((pages*64, 4, 2, 192), -123,
                                         device=device, dtype=torch.bfloat16)
                    kc, vc = backing[:, :, 0], backing[:, :, 1]
                else:
                    backing = torch.full((pages, 2, 64, 4, 192), -123,
                                         device=device, dtype=torch.bfloat16)
                    kc, vc = backing[:, 0], backing[:, 1]
                vc.fill_(123)

                def pipeline():
                    packed = module.pack_local_qkv(query, key, value)
                    gathered = gather_dim1(packed)
                    return module.unpack_qkv_and_write_cache(
                        gathered, slots, kc, vc, local_query_heads=heads)

                def expected(multiplier, row_slots):
                    q = torch.cat([b[:, :heads] for b in bases], dim=1)
                    k = torch.cat([b[:, heads:heads+2] for b in bases], dim=1)
                    v = torch.cat([b[:, heads+2:] for b in bases], dim=1)
                    q, k, v = [(x * multiplier).bfloat16() for x in [q, k, v]]
                    expected_kc = torch.full((pages*64, 4, 192), -123, dtype=torch.bfloat16)
                    expected_vc = torch.full_like(expected_kc, 123)
                    for row, slot in enumerate(row_slots.tolist()):
                        if 0 <= slot < pages*64:
                            expected_kc[slot], expected_vc[slot] = k[row], v[row]
                    return q, k, v, expected_kc, expected_vc

                result = pipeline()
                torch.cuda.synchronize()
                for actual, oracle in zip([*result, kc, vc], expected(1., slots_cpu)):
                    check(actual, oracle)
                case = {"heads": heads, "tokens": tokens, "layout": layout, "eager_exact": True}

                if tokens in [4, 64, 384]:
                    for _ in range(3):
                        pipeline()
                    torch.cuda.synchronize()
                    dist.barrier()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = pipeline()
                    torch.cuda.current_stream().wait_stream(stream)
                    replays = []
                    for multiplier, reverse_slots in [(0., False), (1e-4, True), (-.7, False)]:
                        changed = (bases[rank] * multiplier).bfloat16().to(device)
                        query.copy_(changed[:, :heads])
                        key.copy_(changed[:, heads:heads+2])
                        value.copy_(changed[:, heads+2:])
                        next_slots = slots_cpu.flip(0) if reverse_slots else slots_cpu
                        slots.copy_(next_slots)
                        kc.fill_(-123)
                        vc.fill_(123)
                        dist.barrier()
                        graph.replay()
                        torch.cuda.synchronize()
                        for actual, oracle in zip(
                            [*captured, kc, vc], expected(multiplier, next_slots)
                        ):
                            check(actual, oracle)
                        replays.append({"multiplier": multiplier, "changed_slots": reverse_slots,
                                        "exact": True})
                    case["graph_replays"] = replays
                    # Destroy collective Graphs before destroying the process group.
                    graph.reset()
                    del graph, captured
                    gc.collect()
                    torch.cuda.synchronize()
                    dist.barrier()
                record["cases"].append(case)
                report.write_text(json.dumps(record, indent=2))
    record["qualified"] = True
    report.write_text(json.dumps(record, indent=2))
    dist.barrier()
    if rank == 0:
        print("STEP5_DCP2_EXCHANGE_QUALIFIED", len(record["cases"]), flush=True)
finally:
    dist.destroy_process_group()
