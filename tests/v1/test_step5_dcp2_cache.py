# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed allocator with global128/local64 Step5 pages."""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager, get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode, SlidingWindowSpec, UniformTypeKVCacheSpecs,
)

# Load the exact checkout source even if another plugin snapshot is installed.
source = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/step5_dcp2_cache.py"
module_spec = importlib.util.spec_from_file_location("step5_dcp2_cache_under_test", source)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
Spec = module.Step5DCP2SlidingWindowSpec


def make_spec(extra=0):
    return Spec(block_size=64, num_kv_heads=4, head_size=192,
                dtype=torch.bfloat16, sliding_window=512,
                extra_retained_tokens=extra)


def config(dcp=2, pcp=1, prefix=False, inflight=384, max_len=65536):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp,
                                        prefill_context_parallel_size=pcp),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
        model_config=SimpleNamespace(max_model_len=max_len),
        max_in_flight_tokens=inflight,
    )


def make_manager(spec, pool, inflight):
    return get_manager_for_kv_cache_spec(
        spec, max_in_flight_tokens=inflight, max_model_len=65536,
        block_pool=pool, enable_caching=False, kv_cache_group_id=0,
        scheduler_block_size=128, dcp_world_size=2, pcp_world_size=1,
    )


@pytest.mark.parametrize("extra,inflight,max_len,expected", [
    (0, 1, 65536, 5), (0, 2, 65536, 6),
    (0, 4, 65536, 6), (3, 4, 65536, 6),
    (0, 384, 65536, 8), (3, 384, 65536, 9),
    (0, 384, 127, 2), (0, 384, 128, 2),
    (3, 384, 129, 3),
])
def test_admission_and_worker_table_use_same_logical_pages(
    extra, inflight, max_len, expected
):
    spec = make_spec(extra)
    cfg = config(inflight=inflight, max_len=max_len)
    assert spec.max_admission_blocks_per_request(inflight, max_len) == expected
    assert spec.max_memory_usage_bytes(cfg) == expected * 196608
    assert spec.max_num_blocks_per_req(cfg, max_len) == (max_len + 127) // 128


@pytest.mark.parametrize("kwargs", [
    {"dcp": 1}, {"dcp": 4}, {"pcp": 2}, {"prefix": True},
])
def test_unsupported_cache_configs_fail(kwargs):
    with pytest.raises(ValueError):
        make_spec().max_memory_usage_bytes(config(**kwargs))


@pytest.mark.parametrize("change", [
    {"block_size": 128}, {"num_kv_heads": 2}, {"head_size": 128},
    {"head_size_v": 128}, {"dtype": torch.float8_e4m3fn},
    {"sliding_window": 128}, {"extra_retained_tokens": -1},
    {"kv_quant_mode": KVQuantMode.FP8_PER_TOKEN_HEAD},
    {"tokens_per_state": 2}, {"num_head_slots": 2},
    {"state_content_bytes": 192},
])
def test_unsupported_storage_geometries_fail(change):
    with pytest.raises(ValueError):
        replace(make_spec(), **change)


def test_plain_spec_conversion_and_uniform_grouping():
    plain = SlidingWindowSpec(
        block_size=64, num_kv_heads=4, head_size=192,
        dtype=torch.bfloat16, sliding_window=512, extra_retained_tokens=3,
    )
    converted = Spec.from_spec(plain)
    assert converted.extra_retained_tokens == 3
    assert converted.head_size_v == 192
    assert not converted.prefix_cacheable
    assert not converted.is_uniform_with_collection(
        {"custom": converted, "stock": plain})
    uniform = UniformTypeKVCacheSpecs.from_specs({"a": converted, "b": converted})
    assert uniform is not None
    assert not uniform.prefix_cacheable
    assert uniform.max_memory_usage_bytes(config()) == (
        2 * converted.max_memory_usage_bytes(config()))
    assert uniform.max_num_blocks_per_req(config(), 65536) == 512


@pytest.mark.parametrize("batch,chunk,extra", [
    (1, 127, 0), (1, 129, 3), (32, 384, 0), (64, 384, 3),
])
def test_real_chunked_prefill_recycles_through_64k(batch, chunk, extra):
    spec = make_spec(extra)
    cap = spec.max_admission_blocks_per_request(chunk, 65536)
    # No spare capacity beyond the admission bound plus the permanent null page.
    pool = BlockPool(num_gpu_blocks=batch*cap+1, enable_caching=False,
                     hash_block_size=128)
    manager = make_manager(spec, pool, chunk)
    assert isinstance(manager, SlidingWindowManager)
    assert manager.block_size == 128
    assert manager._max_admission_blocks_per_request == cap
    request_ids = [f"req-{i}" for i in range(batch)]
    for req in request_ids:
        reservation = manager.get_num_blocks_to_allocate(
            req, 65536, (), 0, 0, 65536, apply_admission_cap=True)
        assert reservation == cap
        manager.add_local_computed_blocks(req, (), 0, 0)
    peak = 0
    for computed in range(0, 65536, chunk):
        end = min(computed+chunk, 65536)
        retained_start = max(0, computed-511-extra)
        first = retained_start // 128
        last = (end+127)//128
        for req in request_ids:
            manager.remove_skipped_blocks(req, computed)
            needed = manager.get_num_blocks_to_allocate(
                req, end, (), computed, computed, end)
            assert needed <= pool.get_num_free_blocks()
            new = manager.allocate_new_blocks(req, end, end)
            assert len(new) == needed
            blocks = manager.req_to_blocks[req]
            assert len(blocks) == last
            # Independent global-token window oracle; earlier pages must be freed.
            assert all(b is pool.null_block for b in blocks[:first])
            live = blocks[first:last]
            assert all(b is not pool.null_block and b.ref_cnt == 1 for b in live)
            assert len({b.block_id for b in live}) == len(live)
            assert len(live) <= cap
            peak = max(peak, len(live))
        held = sum(b is not pool.null_block
                   for req in request_ids for b in manager.req_to_blocks[req])
        assert pool.get_num_free_blocks() == batch*cap-held
    assert peak <= cap
    for req in request_ids:
        manager.free(req)
    assert pool.get_num_free_blocks() == batch*cap


def test_mtp3_rejection_keeps_replay_window_and_avoids_double_allocation():
    spec = make_spec(3)
    cap = spec.max_admission_blocks_per_request(4, 65536)
    pool = BlockPool(num_gpu_blocks=cap+1, enable_caching=False,
                     hash_block_size=128)
    manager = make_manager(spec, pool, 4)
    # Restore only the active suffix of a 64K request, as after preemption.
    req, computed = "mtp", 64001
    manager.add_local_computed_blocks(req, (), computed, 0)
    for step in range(192):
        manager.remove_skipped_blocks(req, computed)
        end = computed+4
        needed = manager.get_num_blocks_to_allocate(
            req, end, (), computed, computed, computed+1)
        assert needed <= pool.get_num_free_blocks()
        assert len(manager.allocate_new_blocks(req, end, computed+1)) == needed
        blocks = manager.req_to_blocks[req]
        first = max(0, computed-511-3)//128
        last = (end+127)//128
        assert all(b is pool.null_block for b in blocks[:first])
        assert all(b is not pool.null_block for b in blocks[first:last])
        assert sum(b is not pool.null_block for b in blocks) <= cap
        # 0/1/2/3 accepted draft tokens: cache allocation may retain rejected
        # speculative suffix, but the next verification must not allocate it twice.
        computed += 1 + step % 4
    manager.free(req)
    assert pool.get_num_free_blocks() == cap
