#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
model_path="${STEP5_MODEL_PATH:-/data/model/step-5-preview-0913-fp8-per-channel}"
tokenizer_path="${STEP5_TOKENIZER_PATH:-/workspace/step5-adaptation/tokenizer-bytelevel}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HCU_DEEPEP_NUM_SMS=36
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0
export VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
export VLLM_KV_CACHE_LAYOUT=LBNHC
export VLLM_SERVER_DEV_MODE=0
export VLLM_STEP4_DISABLE_DSA=0
export VLLM_STEP4_DSA_FP8_MATH=off
export VLLM_STEP5_BF16_TILE_TUNING=0
export VLLM_STEP5_DEVICE_REQUEST_INDICES=1
export VLLM_STEP5_DSA_FUSED_SELECTOR=1
export VLLM_STEP5_DSA_SHORT_QUERY=1
export VLLM_STEP5_DSA_TORCH_SELECTOR=1
export VLLM_STEP5_FUSED_QK_NORM_ROPE=0
export VLLM_STEP5_HT_COMPACT_WORKSPACE=1
export VLLM_STEP5_HT_REUSE_DISPATCH_COUNTS=1
export VLLM_STEP5_LL_GEMM_TUNING=0
export VLLM_STEP5_MERGED_ATTENTION_PROJECTION=0
export VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=0
export VLLM_STEP5_QKV_FP8_W8A8=0
export VLLM_STEP5_SEQUENCE_PARALLEL_MOE=0
export VLLM_STEP5_SP_LL_WORKSPACE=0
export VLLM_STEP5_TP2_HT_BF16_CHUNKS=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_USE_V2_MODEL_RUNNER=1
unset VLLM_HCU_STEP5_DENSE_DIAGNOSTIC
if [[ "${NCCL_TOPO_FILE:-}" == "/usr/local/built-in-508-topo-input-tj-default.xml" ]]; then
    unset NCCL_TOPO_FILE
fi
if [[ ! -f "$tokenizer_path/tokenizer.json" ]]; then
    python3 tools/prepare_step5_tokenizer.py --model "$model_path" --output "$tokenizer_path"
fi
exec vllm serve "$model_path" \
    --tokenizer "$tokenizer_path" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --moe-backend deep_gemm \
    --all2all-backend deepep_high_throughput \
    --block-size 64 \
    --max-model-len 69632 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.89 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "NONE", "cudagraph_copy_inputs": false, "mode": 3}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --override-generation-config '{"eos_token_id": [1, 2, 128007]}' \
    --host "${STEP5_HOST:-127.0.0.1}" \
    --port "${STEP5_PORT:-18106}" \
    --served-model-name "${STEP5_SERVED_MODEL_NAME:-step5}" \
    --api-server-count 1
