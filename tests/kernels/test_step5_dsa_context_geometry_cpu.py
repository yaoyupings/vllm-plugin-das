# SPDX-License-Identifier: Apache-2.0
"""CPU allocation and actual metadata-builder checks for logical DCP pages."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HIP_VISIBLE_DEVICES") != "":
    pytest.skip("requires explicitly hidden GPUs", allow_module_level=True)

from vllm_hcu.model_executor.layers.step4_dsa import (
    Step4DSAGeometry, Step4DSAMetadataBuilder, Step4DSAState,
)


@pytest.mark.parametrize("cp", [1, 2])
def test_summary_pool_size_reuse_and_replacement(cp):
    state = Step4DSAState(
        num_layers=2, geometry=Step4DSAGeometry(), max_pending=2,
        dtype=torch.bfloat16, context_parallel_size=cp,
    )
    cache = torch.zeros(2, 3, 64, 4, 192, dtype=torch.bfloat16)
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        state.ensure_allocated(cache, torch.device("cpu"))
        assert state.block_size == 64 * cp
        assert state.regions_per_page == 8 * cp
        assert state.summary.shape == (2, 24 * cp, 1, 256)
        previous = state.summary
        state.ensure_allocated(cache, torch.device("cpu"))
        assert state.summary is previous
        # V2 temporary profiling cache replacement cannot retain old pointers.
        cache = torch.zeros(2, 4, 64, 4, 192, dtype=torch.bfloat16)
        state.ensure_allocated(cache, torch.device("cpu"))
        assert state.summary is not previous
        assert state.summary.shape == (2, 32 * cp, 1, 256)
        assert state.pending_key.shape == (2, 3, 8, 256)


@pytest.mark.parametrize("cp", [1, 2])
@pytest.mark.parametrize("qlen", [1, 4, 24])
def test_builder_keeps_global_positions_and_local_slot_mapping(cp, qlen):
    state = Step4DSAState(
        num_layers=2, geometry=Step4DSAGeometry(), max_pending=2,
        dtype=torch.bfloat16, context_parallel_size=cp,
    )
    builder = Step4DSAMetadataBuilder.__new__(Step4DSAMetadataBuilder)
    builder.block_size, builder._state, builder.layer_names = 64, state, []
    starts = torch.tensor([0, qlen, 2*qlen], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2, num_actual_tokens=2*qlen,
        block_table_tensor=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(2*qlen, dtype=torch.int64),
        seq_lens=torch.tensor([129, 65536], dtype=torch.int32),
        query_start_loc=starts, query_start_loc_cpu=starts,
        seq_lens_cpu=torch.tensor([129, 65536], dtype=torch.int32),
        positions=torch.arange(65536-2*qlen, 65536, dtype=torch.int64),
        max_seq_len=65536, max_query_len=qlen,
    )
    common.slot_mapping[::2] = -1
    with patch.dict(os.environ, {"VLLM_STEP5_DEVICE_REQUEST_INDICES": "0"}):
        metadata = builder.build(0, common)
    assert metadata.block_size == 64*cp
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.seq_lens is common.seq_lens
    assert metadata.positions is common.positions
    assert metadata.num_actual_tokens == 2*qlen
    assert metadata.num_decodes == (2 if qlen == 1 else 0)


def test_default_geometry_remains_dcp1():
    state = Step4DSAState(
        num_layers=1, geometry=Step4DSAGeometry(), max_pending=1,
        dtype=torch.bfloat16,
    )
    assert state.context_parallel_size == 1
    for cp in [0, 3, 4]:
        with pytest.raises(ValueError):
            Step4DSAState(
                num_layers=1, geometry=Step4DSAGeometry(), max_pending=1,
                dtype=torch.bfloat16, context_parallel_size=cp,
            )

def test_dcp2_rejects_unimplemented_page_geometry():
    state = Step4DSAState(
        num_layers=1, geometry=Step4DSAGeometry(), max_pending=1,
        dtype=torch.bfloat16, context_parallel_size=2,
    )
    cache = torch.zeros(2, 2, 128, 4, 192, dtype=torch.bfloat16)
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        with pytest.raises(ValueError, match="pages64"):
            state.ensure_allocated(cache, torch.device("cpu"))
    builder = Step4DSAMetadataBuilder.__new__(Step4DSAMetadataBuilder)
    builder.block_size, builder._state, builder.layer_names = 128, state, []
    with pytest.raises(ValueError, match="pages64"):
        builder.build(0, SimpleNamespace())
