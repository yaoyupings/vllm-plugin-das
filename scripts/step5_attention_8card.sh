#!/usr/bin/env bash
# Qualified Step5 attention optimizations with standard MTP3.
set -euo pipefail

usage() {
    echo "Usage: $0 {tp2-prefill|tp2-decode|dp8-prefill} CALIBRATION_FILE [vllm arguments...]" >&2
    echo "CALIBRATION_FILE may also be set with STEP5_QKV_FP8_CALIBRATION_FILE." >&2
}
profile="${1:-tp2-prefill}"
if [[ "$profile" == "--help" || "$profile" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
    tp2-prefill|tp2-decode|dp8-prefill) ;;
    *) usage; exit 2 ;;
esac
calibration_file="${STEP5_QKV_FP8_CALIBRATION_FILE:-}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    calibration_file="$1"
    shift
fi
if [[ "$calibration_file" != /* || ! -f "$calibration_file" ]]; then
    echo "Provide an absolute path to the Step5 QKV output-MSE calibration file." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"
model_path="${STEP5_MODEL_PATH:-/data/model/step-5-preview-0913-fp8-per-channel}"
tokenizer_path="${STEP5_TOKENIZER_PATH:-/workspace/step5-adaptation/tokenizer-bytelevel}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_HCU_USE_CUSTOM_OPS=1 VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_KV_CACHE_LAYOUT=LBNHC
export VLLM_STEP4_DISABLE_DSA=0 VLLM_STEP4_DSA_FP8_MATH=off
export VLLM_STEP5_DSA_TORCH_SELECTOR=1 VLLM_STEP5_DSA_SHORT_QUERY=1
export VLLM_STEP5_DEVICE_REQUEST_INDICES=1 VLLM_STEP5_BF16_TILE_TUNING=1
export VLLM_STEP5_DSA_FUSED_SELECTOR=1 VLLM_STEP5_SEQUENCE_PARALLEL_MOE=1
export VLLM_STEP5_TP2_HT_BF16_CHUNKS=0 VLLM_STEP5_LL_GEMM_TUNING=0
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
export VLLM_STEP5_FUSED_QK_NORM_ROPE=1 VLLM_STEP5_MERGED_ATTENTION_PROJECTION=1
export VLLM_STEP5_QKV_FP8_W8A8=1 VLLM_STEP5_QKV_FP8_SCALE_MODE=output_mse
export VLLM_STEP5_QKV_FP8_CALIBRATION_FILE="$calibration_file"
export VLLM_STEP5_QKV_FUSED_ACTIVATION_QUANT=0
export VLLM_STEP5_HT_REUSE_DISPATCH_COUNTS=0
export VLLM_USE_AOT_COMPILE=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_CACHE_ROOT="${STEP5_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/vllm/step5-attention}"
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE VLLM_HCU_DEEPEP_NUM_SMS VLLM_HCU_STEP5_DENSE_DIAGNOSTIC

tp_size=2
dp_size=4
if [[ "$profile" == "tp2-prefill" || "$profile" == "dp8-prefill" ]]; then
    backend=deepep_high_throughput
    batch_tokens=6144
    if [[ "$profile" == "dp8-prefill" ]]; then
        tp_size=1
        dp_size=8
        batch_tokens=4096
        export VLLM_HCU_DEEPEP_NUM_SMS=36
        export VLLM_STEP5_HT_REUSE_DISPATCH_COUNTS=1
    fi
    max_seqs=64
    memory=0.89
    compilation='{"cudagraph_mode":"NONE","cudagraph_copy_inputs":false,"mode":3}'
    export VLLM_STEP5_HT_COMPACT_WORKSPACE=1
    export VLLM_STEP5_SP_LL_WORKSPACE=0 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=0
else
    backend=deepep_low_latency
    batch_tokens=512
    max_seqs=32
    memory=0.92
    compilation='{"cudagraph_mode":"PIECEWISE","cudagraph_copy_inputs":true,"mode":3,"cudagraph_capture_sizes":[4,8,16,32,64,128,256,512]}'
    export VLLM_STEP5_HT_COMPACT_WORKSPACE=0
    export VLLM_STEP5_SP_LL_WORKSPACE=1 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=1
fi

python3 tools/prepare_step5_tokenizer.py --model "$model_path" --output "$tokenizer_path"
exec vllm serve "$model_path" \
    --tokenizer "$tokenizer_path" \
    --tensor-parallel-size "$tp_size" --data-parallel-size "$dp_size" --enable-expert-parallel \
    --dtype bfloat16 --moe-backend deep_gemm --all2all-backend "$backend" \
    --block-size 64 --max-model-len 69632 \
    --max-num-seqs "$max_seqs" --max-num-batched-tokens "$batch_tokens" \
    --gpu-memory-utilization "$memory" --no-enable-prefix-caching \
    --compilation-config "$compilation" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --override-generation-config '{"eos_token_id":[1,2,128007]}' \
    --host "${STEP5_HOST:-127.0.0.1}" --port "${STEP5_PORT:-18106}" \
    --served-model-name step5 --api-server-count 1 "$@"
