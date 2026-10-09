#!/usr/bin/env bash
# Launch Step5 TP2/DP4/EP8; DCP2 is an explicit experimental Decode opt-in.
set -euo pipefail
stage="${1:-decode}"
sampling="${2:-standard}"
case "$stage" in prefill|decode) ;; *) echo "Usage: $0 [prefill|decode] [standard|synthetic]" >&2; exit 2;; esac
case "$sampling" in standard|synthetic) ;; *) echo "MTP sampling must be standard or synthetic" >&2; exit 2;; esac
dcp_size="${STEP5_DCP_SIZE:-1}"
case "$dcp_size" in
  1) ;;
  2)
    [[ "$stage" == decode && "${VLLM_STEP5_DCP2_RUNTIME:-0}" == 1 ]] || {
      echo "Step5 DCP2 requires Decode and VLLM_STEP5_DCP2_RUNTIME=1" >&2
      exit 2
    }
    # Qualified on real MTP3/frozen100; BS32/64 long-context performance
    # remains experimental until the initialized capacity and run are checked.
    export VLLM_STEP5_DCP2_FULL_GRAPH="${VLLM_STEP5_DCP2_FULL_GRAPH:-1}"
    # Static shape tuning is qualified separately; fused exchange is opt-in
    # until the complete model path has passed its native/runtime checks.
    export VLLM_STEP5_DCP2_TUNED_SHORT_SPLITS="${VLLM_STEP5_DCP2_TUNED_SHORT_SPLITS:-1}"
    export VLLM_STEP5_DCP2_FUSED_EXCHANGE="${VLLM_STEP5_DCP2_FUSED_EXCHANGE:-0}"
    # Qualified on frozen100 and one BS64 Decode round with fixed splits.
    export VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE="${VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE:-0}"
    for dcp_flag in "$VLLM_STEP5_DCP2_TUNED_SHORT_SPLITS" "$VLLM_STEP5_DCP2_FUSED_EXCHANGE" "$VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE"; do
      [[ "$dcp_flag" == 0 || "$dcp_flag" == 1 ]] || {
        echo "Step5 DCP2 optimization flags must be 0 or 1" >&2; exit 2
      }
    done
    if [[ "$VLLM_STEP5_DCP2_FUSED_NORM_EXCHANGE" == 1 && "$VLLM_STEP5_DCP2_FUSED_EXCHANGE" != 1 ]]; then
      echo "Step5 norm exchange requires VLLM_STEP5_DCP2_FUSED_EXCHANGE=1" >&2; exit 2
    fi
    export VLLM_STEP5_EXACT_GRAPH_MEMORY="${VLLM_STEP5_EXACT_GRAPH_MEMORY:-1}"
    export VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES="${VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES:-1073741824}"
    export VLLM_STEP5_LL_DISPATCH_ALIGNMENT="${VLLM_STEP5_LL_DISPATCH_ALIGNMENT:-64}"
    case "$VLLM_STEP5_DCP2_FULL_GRAPH" in
      0|1) ;;
      *) echo "VLLM_STEP5_DCP2_FULL_GRAPH must be 0 or 1" >&2; exit 2;;
    esac
    ;;
  *) echo "STEP5_DCP_SIZE must be 1 or explicitly qualified 2" >&2; exit 2;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_TUNED_CONFIG_FOLDER="${repo_root}/configs/step5_moe"
[[ -f "${VLLM_TUNED_CONFIG_FOLDER}/E=44,N=1536,device_name=BW1101.json" ]] || {
  echo "Missing Step5 MoE tuning config" >&2
  exit 2
}
export OMP_NUM_THREADS=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_STEP4_DSA_FP8_MATH=off VLLM_STEP4_DISABLE_DSA=0
export VLLM_STEP5_DSA_TORCH_SELECTOR=0
export VLLM_STEP5_DSA_RADIX_SELECTOR="${VLLM_STEP5_DSA_RADIX_SELECTOR:-radix_flash}"
# Native exact Top512 with sorted causal packing; model candidate only.
export VLLM_STEP5_DSA_LIGHTOP_TOP512="${VLLM_STEP5_DSA_LIGHTOP_TOP512:-0}"
case "$VLLM_STEP5_DSA_LIGHTOP_TOP512" in
  0|1) ;;
  *) echo "VLLM_STEP5_DSA_LIGHTOP_TOP512 must be 0 or 1" >&2; exit 2;;
esac
export VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512="${VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512:-0}"
case "$VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512" in
  0|1) ;;
  *) echo "VLLM_STEP5_DSA_LIGHTOP_DECODE_TOP512 must be 0 or 1" >&2; exit 2;;
esac
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=LBNHC VLLM_USE_AOT_COMPILE=0
export VLLM_STEP5_LL_GEMM_TUNING=1
export VLLM_STEP5_SEQUENCE_PARALLEL_MOE=1 VLLM_STEP5_DSA_SHORT_QUERY=1
export VLLM_STEP5_DSA_TUNED_SHORT_SPLITS=1
export VLLM_STEP5_DEVICE_REQUEST_INDICES=1 VLLM_STEP5_BATCHED_SHORT_INDEXER=1 VLLM_STEP5_BF16_TILE_TUNING=1
# BF16 sparse Attention tile candidate; opt-in until model performance is measured.
export VLLM_STEP5_BF16_DSA_TUNED_TILES="${VLLM_STEP5_BF16_DSA_TUNED_TILES:-0}"
case "$VLLM_STEP5_BF16_DSA_TUNED_TILES" in
  0|1) ;;
  *) echo "VLLM_STEP5_BF16_DSA_TUNED_TILES must be 0 or 1" >&2; exit 2;;
esac
# Removes three norm-parameter casts in qualified short-query indexer shapes.
export VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS="${VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS:-0}"
case "$VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS" in
  0|1) ;;
  *) echo "VLLM_STEP5_INDEXER_FUSED_NORM_WEIGHT_CASTS must be 0 or 1" >&2; exit 2;;
esac
export VLLM_STEP5_GROUPED_MTP_INDEXER="${VLLM_STEP5_GROUPED_MTP_INDEXER:-1}"
export VLLM_STEP5_TP2_HT_BF16_CHUNKS=0 VLLM_STEP5_DSA_FUSED_SELECTOR=1
export VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP=0
case "${STEP5_SHARED_EXPERT_OVERLAP:-off}" in
  off)
    export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
    export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
    export VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY=0
    ;;
  decode)
    [[ "$stage" == decode ]] || { echo "Shared-expert overlap is limited to Decode" >&2; exit 2; }
    export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
    export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=1 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=1
    export VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY=1
    export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD="${VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD:-128}"
    ;;
  split)
    [[ "$stage" == decode ]] || { echo "Split shared-expert overlap is limited to Decode" >&2; exit 2; }
    export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
    export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
    export VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY=0
    export VLLM_STEP5_SHARED_EXPERT_SPLIT_OVERLAP=1
    ;;
  *) echo "STEP5_SHARED_EXPERT_OVERLAP must be off, decode or split" >&2; exit 2;;
esac
# These fused projection candidates regress the production TP2 shapes.
# QKV stays BF16 unless the optional calibrated FP8 path is explicitly enabled.
export VLLM_STEP5_FUSED_QK_NORM_ROPE=0
export VLLM_STEP5_MERGED_ATTENTION_PROJECTION=0
export VLLM_STEP5_FULL_MIXED_PROJECTION=0
# Candidate only; formal default remains disabled pending model qualification.
export VLLM_STEP5_QKV_NORM_FP8_FUSION="${VLLM_STEP5_QKV_NORM_FP8_FUSION:-0}"
case "$VLLM_STEP5_QKV_NORM_FP8_FUSION" in
  0) ;;
  1)
    [[ "${VLLM_STEP5_QKV_FP8_W8A8:-0}" == 1 ]] || {
      echo "Norm/FP8 fusion requires VLLM_STEP5_QKV_FP8_W8A8=1" >&2
      exit 2
    }
    [[ "${VLLM_HCU_USE_CUSTOM_OPS:-1}" == 1 && "${VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:-1}" == 1 ]] || {
      echo "Norm/FP8 fusion requires the LightOp quantization/GEMM path" >&2
      exit 2
    }
    ;;
  *) echo "VLLM_STEP5_QKV_NORM_FP8_FUSION must be 0 or 1" >&2; exit 2;;
esac
export VLLM_SERVER_DEV_MODE=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
unset NCCL_TOPO_FILE VLLM_STEP5_HOST_REQUEST_INDICES
decode_runtime_args=()
long_prefill_threshold="${STEP5_LONG_PREFILL_TOKEN_THRESHOLD:-0}"
if [[ "$stage" == prefill ]]; then
  backend=deepep_high_throughput; tokens=6144; sequences=64; memory=0.89
  export VLLM_STEP5_HT_COMPACT_WORKSPACE=1 VLLM_STEP5_SP_LL_WORKSPACE=0 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=0
  compilation='{"cudagraph_mode":"NONE","cudagraph_copy_inputs":false,"mode":3}'
else
  backend=deepep_low_latency; tokens=512; sequences=32; memory=0.92
  capture_sizes='[4,8,16,32,64,128,256,512]'
  default_graph_mode=FULL_AND_PIECEWISE
  if [[ "$dcp_size" == 2 ]]; then
    tokens=384; sequences=16; memory=0.95
    capture_sizes='[4,8,16,32,64]'
    default_graph_mode=PIECEWISE
    if [[ "$VLLM_STEP5_DCP2_FULL_GRAPH" == 1 ]]; then
      default_graph_mode=FULL_DECODE_ONLY
    fi
    # Use the qualified long-prefill setting and record actual graph modes.
    decode_runtime_args=(--long-prefill-token-threshold "$long_prefill_threshold" --cudagraph-metrics)
  fi
  if [[ "$dcp_size" == 1 && -n "${STEP5_LONG_PREFILL_TOKEN_THRESHOLD+x}" ]]; then
    decode_runtime_args=(--long-prefill-token-threshold "$long_prefill_threshold")
  fi
  export VLLM_STEP5_BF16_COMPACT_GRID="${VLLM_STEP5_BF16_COMPACT_GRID:-1}"
  export VLLM_STEP5_GROUPED_HEAD_INDEXER="${VLLM_STEP5_GROUPED_HEAD_INDEXER:-1}"
  # FULL_DECODE_ONLY keeps complete Decode graphs and runs mixed batches
  # without PIECEWISE graphs; useful when the long-context KV budget is tight.
  target_graph_mode="${STEP5_TARGET_GRAPH_MODE:-$default_graph_mode}"
  case "$target_graph_mode" in
    FULL_AND_PIECEWISE|FULL_DECODE_ONLY|PIECEWISE) ;;
    *) echo "STEP5_TARGET_GRAPH_MODE must be FULL_AND_PIECEWISE, FULL_DECODE_ONLY or PIECEWISE" >&2; exit 2;;
  esac
  if [[ "$dcp_size" == 2 ]]; then
    if [[ "$VLLM_STEP5_DCP2_FULL_GRAPH" == 1 ]]; then
      [[ "$target_graph_mode" == FULL_DECODE_ONLY ]] || {
        echo "Step5 DCP2 FULL graphs require FULL_DECODE_ONLY" >&2; exit 2
      }
    else
      [[ "$target_graph_mode" == PIECEWISE ]] || {
        echo "Step5 DCP2 without FULL opt-in requires PIECEWISE" >&2; exit 2
      }
    fi
  fi
  export VLLM_STEP5_HT_COMPACT_WORKSPACE=0 VLLM_STEP5_SP_LL_WORKSPACE=1 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=1
  compilation='{"cudagraph_mode":"'"$target_graph_mode"'","cudagraph_copy_inputs":true,"mode":3,"cudagraph_capture_sizes":'"$capture_sizes"'}'
fi
# Keep the measured stage defaults; memory headroom candidates opt in.
memory="${STEP5_GPU_MEMORY_UTILIZATION:-$memory}"
python - "$memory" "$long_prefill_threshold" <<'PY_CHECK_MEMORY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("STEP5_GPU_MEMORY_UTILIZATION must be a number in (0, 1]")
if not math.isfinite(value) or not 0 < value <= 1:
    raise SystemExit("STEP5_GPU_MEMORY_UTILIZATION must be a number in (0, 1]")
try:
    threshold = int(sys.argv[2])
except ValueError:
    raise SystemExit("STEP5_LONG_PREFILL_TOKEN_THRESHOLD must be an integer in [0, 69632]")
if not 0 <= threshold <= 69632:
    raise SystemExit("STEP5_LONG_PREFILL_TOKEN_THRESHOLD must be an integer in [0, 69632]")
PY_CHECK_MEMORY
speculation='{"method":"mtp","num_speculative_tokens":3}'
if [[ "$sampling" == synthetic ]]; then
  speculation='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686]}'
fi
exec vllm serve "${STEP5_MODEL:-/data/model/step-5-preview-0913-fp8-per-channel}" \
  --tokenizer "${STEP5_TOKENIZER:-/workspace/step5-adaptation/tokenizer-bytelevel}" \
  --tensor-parallel-size 2 --data-parallel-size 4 --pipeline-parallel-size 1 --decode-context-parallel-size "$dcp_size" --enable-expert-parallel \
  --dtype bfloat16 --moe-backend deep_gemm --all2all-backend "$backend" \
  --block-size 64 --max-model-len 69632 --max-num-seqs "$sequences" \
  --max-num-batched-tokens "$tokens" --gpu-memory-utilization "$memory" \
  --no-enable-prefix-caching --disable-custom-all-reduce --compilation-config "$compilation" \
  --speculative-config "$speculation" --override-generation-config '{"eos_token_id":[1,2,128007]}' \
  "${decode_runtime_args[@]}" --host 127.0.0.1 --port "${STEP5_PORT:-18106}" --served-model-name step5 --api-server-count 1
