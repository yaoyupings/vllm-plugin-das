# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
from vllm_hcu.model_executor.layers.fused_moe import step5_bf16_tiles as tiles

BASE = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
            num_warps=8, num_stages=2)


@pytest.mark.parametrize("tokens,expected", [(0, 128), (1, 32), (128, 32),
                                             (512, 32), (513, 128), (2048, 128)])
def test_host_counts_select_live_batch_not_padded_capacity(tokens, expected):
    ctx = NS(batch_descriptor=None, dp_metadata=NS(
        num_tokens_across_dp_cpu=torch.tensor([tokens, 0, 0, 0])))
    actual = tiles.small_batch_tile_config(BASE, ctx, 4)
    assert actual['BLOCK_SIZE_M'] == expected
    assert actual['BLOCK_SIZE_K'] == 64
    assert BASE['BLOCK_SIZE_M'] == 128


def test_host_count_read_is_cached_once_per_forward():
    class Counts:
        device = NS(type='cpu')
        calls = 0
        def tolist(self):
            self.calls += 1
            return [64, 64, 64, 64]
    counts = Counts()
    ctx = NS(batch_descriptor=None, dp_metadata=NS(num_tokens_across_dp_cpu=counts))
    tiles.small_batch_tile_config(BASE, ctx, 4)
    tiles.small_batch_tile_config(BASE, ctx, 4)
    assert counts.calls == 1


def test_never_read_device_counts():
    class Counts:
        device = NS(type='cuda')
        def tolist(self):
            raise AssertionError('GPU count read would synchronize')
    ctx = NS(batch_descriptor=None, dp_metadata=NS(num_tokens_across_dp_cpu=Counts()))
    assert tiles.small_batch_tile_config(BASE, ctx, 4) is BASE
    ctx = NS(batch_descriptor=NS(num_tokens=64), dp_metadata=None)
    assert tiles.small_batch_tile_config(BASE, ctx, 4)['BLOCK_SIZE_M'] == 32


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_both_grid_paths_receive_small_tiles(monkeypatch, compact, positional):
    monkeypatch.setenv('VLLM_STEP5_BF16_TILE_TUNING', '1')
    monkeypatch.setenv('VLLM_STEP5_BF16_COMPACT_GRID', '1' if compact else '0')
    monkeypatch.setattr(tiles, '_is_gfx938', lambda device: True)
    cfg = NS(model_config=NS(hf_text_config=NS(model_type='step5')),
             parallel_config=NS(tensor_parallel_size=2, data_parallel_size=4,
                                pipeline_parallel_size=1, decode_context_parallel_size=1,
                                all2all_backend='deepep_low_latency'))
    monkeypatch.setattr(tiles, '_step5_model_config', cfg)
    ctx = NS(batch_descriptor=None, dp_metadata=NS(
        num_tokens_across_dp_cpu=torch.tensor([64,64,64,64])))
    import vllm.forward_context
    monkeypatch.setattr(vllm.forward_context, 'get_forward_context', lambda: ctx)
    recorded = []
    def native(*args, **kwargs):
        recorded.append(args[11] if len(args)>11 else kwargs['config'])
        return 'native'
    target = NS(invoke_moe_batched_triton_kernel=native)
    from vllm_hcu.model_executor.layers.fused_moe import step5_bf16_batched
    def live(a,b,c,counts,config):
        recorded.append(config)
        return 'compact'
    monkeypatch.setattr(step5_bf16_batched, 'invoke_step5_bf16_experts', live)
    tiles.install_step5_bf16_tiles(target)
    a = NS(ndim=3,dtype=torch.bfloat16,shape=(44,2048,4096),device='cuda')
    b = NS(dtype=torch.bfloat16,shape=(44,3072,4096))
    if positional:
        result=target.invoke_moe_batched_triton_kernel(
            a,b,None,None,None,None,None,None,False,False,False,BASE,False)
    else:
        result=target.invoke_moe_batched_triton_kernel(A=a,B=b,C=None,
            expert_num_tokens=None,config=BASE)
    assert result == ('compact' if compact else 'native')
    assert recorded[0]['BLOCK_SIZE_M'] == 32
    assert recorded[0]['BLOCK_SIZE_N'] == 64


def test_capture_uses_descriptor_when_dummy_host_counts_are_zero():
    ctx = NS(batch_descriptor=NS(num_tokens=32), dp_metadata=NS(
        num_tokens_across_dp_cpu=torch.zeros(4, dtype=torch.int32)))
    assert tiles.small_batch_tile_config(BASE, ctx, 4)['BLOCK_SIZE_M'] == 32
