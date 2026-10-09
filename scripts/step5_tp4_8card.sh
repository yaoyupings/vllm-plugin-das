#!/usr/bin/env bash
# Step5 TP4/DP2/EP8 on one eight-card BW1101 host.
set -euo pipefail
sampling="${1:-standard}"
case "$sampling" in standard|synthetic) ;; *) echo "Usage: $0 [standard|synthetic]" >&2; exit 2;; esac
[[ "${STEP5_DCP_SIZE:-1}" == 1 ]] || { echo "This profile requires DCP=1" >&2; exit 2; }
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_TUNED_CONFIG_FOLDER="${repo_root}/configs/step5_moe"
export OMP_NUM_THREADS=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ROCBLAS_UNSELECT_CHECK=1
export STEP5_TARGET_GRAPH_MODE=FULL_DECODE_ONLY
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export VLLM_STEP4_DSA_FP8_MATH=off
export VLLM_STEP5_AUTO_DECODE_GRAPH=1
export VLLM_STEP5_AUTO_SHARED_FP8_WEIGHTS=1
export VLLM_STEP5_AUTO_SP_LL_WORKSPACE=1
export VLLM_STEP5_BF16_COMPACT_GRID=1
export VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512=1
export VLLM_STEP5_DSA_LIGHTOP_TOP512=1
export VLLM_STEP5_DSA_LIGHTOP_TOP512_DECODE=0
export VLLM_STEP5_DSA_MEMORY_BUDGET=1
export VLLM_STEP5_DSA_PACK_QKNORM=1
export VLLM_STEP5_EXACT_GRAPH_MEMORY=1
export VLLM_STEP5_FULL_MIXED_DECODE_ONLY=1
export VLLM_STEP5_FULL_MIXED_PROJECTION=1
export VLLM_STEP5_FUSED_WINDOW_METADATA=1
export VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES=1073741824
export VLLM_STEP5_GROUPED_HEAD_INDEXER=1
export VLLM_STEP5_HT_COMPACT_WORKSPACE=1
export VLLM_STEP5_HT_REUSE_DISPATCH_COUNTS=1
export VLLM_STEP5_HT_RIGHTSIZE_BUFFER=1
export VLLM_STEP5_INDEXER_ROUND_IN_NORM=1
export VLLM_STEP5_LL_COMPACT_WORKSPACE=1
export VLLM_STEP5_LL_DISPATCH_ALIGNMENT=64
export VLLM_STEP5_LL_MOE_CHUNK_TOKENS=0
export VLLM_STEP5_MEMORY_LIFECYCLE_TRACE=0
export VLLM_STEP5_MERGED_ATTENTION_PROJECTION=1
export VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=1
export VLLM_STEP5_QKV_FP8_SCALE_MODE=output_mse
export VLLM_STEP5_QKV_FP8_W8A8=1
export VLLM_STEP5_QKV_NORM_FP8_FUSION=0
export VLLM_STEP5_SP_LL_WORKSPACE=0
export VLLM_STEP5_SWA_PREFILL_CROP=1
export VLLM_STEP5_SWA_REUSE_WORKSPACE=1
export VLLM_STEP5_TP4_SMALL_QKV_FP8=1
export VLLM_STEP4_DISABLE_DSA=0 VLLM_STEP5_DSA_TORCH_SELECTOR=0
export VLLM_STEP5_DSA_RADIX_SELECTOR=radix_flash
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_KV_CACHE_LAYOUT=LBNHC VLLM_USE_AOT_COMPILE=0
export VLLM_STEP5_LL_GEMM_TUNING=1 VLLM_STEP5_SEQUENCE_PARALLEL_MOE=1
export VLLM_STEP5_DSA_SHORT_QUERY=1 VLLM_STEP5_DSA_TUNED_SHORT_SPLITS=1
export VLLM_STEP5_DEVICE_REQUEST_INDICES=1 VLLM_STEP5_BATCHED_SHORT_INDEXER=1
export VLLM_STEP5_BF16_TILE_TUNING=1 VLLM_STEP5_GROUPED_MTP_INDEXER=1
export VLLM_STEP5_TP2_HT_BF16_CHUNKS=0 VLLM_STEP5_DSA_FUSED_SELECTOR=1
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
export VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY=0 VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP=0
export VLLM_STEP5_FUSED_QK_NORM_ROPE=0
export VLLM_SERVER_DEV_MODE=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
unset NCCL_TOPO_FILE VLLM_STEP5_HOST_REQUEST_INDICES
export ROCBLAS_TENSILE_LIBPATH="${STEP5_TP4_ROCBLAS_LIBRARY:-/workspace/step5-tp4-optimization-20261008/rocblas-tuning-tp4-v1/auto_select_tools/optimization_configs/new/config/library_gpu6}"
export VLLM_STEP5_QKV_FP8_CALIBRATION_FILE="${STEP5_QKV_CALIBRATION:-/workspace/step5-attention-optimization/qkv-fp8-output-calibration-inputs-v1.pt}"
[[ -d "$ROCBLAS_TENSILE_LIBPATH" ]] || { echo "Missing qualified TP4 rocBLAS library" >&2; exit 2; }
[[ -f "$VLLM_STEP5_QKV_FP8_CALIBRATION_FILE" ]] || { echo "Missing QKV calibration file" >&2; exit 2; }
[[ -f "${VLLM_TUNED_CONFIG_FOLDER}/E=44,N=1536,device_name=BW1101.json" ]] || { echo "Missing Step5 MoE tuning config" >&2; exit 2; }
ar_args=()
case "${STEP5_CUSTOM_ALL_REDUCE:-on}" in
  off) ar_args=(--disable-custom-all-reduce) ;;
  on) ;;
  *) echo "STEP5_CUSTOM_ALL_REDUCE must be on or off" >&2; exit 2 ;;
esac
speculation='{"method":"mtp","num_speculative_tokens":3}'
if [[ "$sampling" == synthetic ]]; then
  speculation='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686]}'
fi
exec vllm serve "${STEP5_MODEL:-/data/model/step-5-preview-0913-fp8-per-channel}" \
  --tokenizer "${STEP5_TOKENIZER:-/workspace/step5-adaptation/tokenizer-bytelevel}" \
  --tensor-parallel-size 4 --data-parallel-size 2 --pipeline-parallel-size 1 --decode-context-parallel-size 1 --enable-expert-parallel \
  --dtype bfloat16 --moe-backend deep_gemm --all2all-backend deepep_auto \
  --block-size 64 --max-model-len 1048576 --max-num-seqs 64 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching "${ar_args[@]}" --max-cudagraph-capture-size 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_copy_inputs":true,"mode":3}' \
  --speculative-config "$speculation" --override-generation-config '{"eos_token_id":[1,2,128007]}' \
  --long-prefill-token-threshold 0 --cudagraph-metrics --host 127.0.0.1 --port "${STEP5_PORT:-18115}" --served-model-name step5 --api-server-count 1
