#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
model_path="${STEP5_MODEL_PATH:-/data/model/step-5-preview-0913-fp8}"
tokenizer_path="${STEP5_TOKENIZER_PATH:-/workspace/step5-adaptation/tokenizer-bytelevel}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_HCU_STEP5_DENSE_DIAGNOSTIC=1
export VLLM_STEP4_DISABLE_DSA=1
export VLLM_STEP4_DSA_FP8_MATH=off
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=LBNHC
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
# The image embeds a gfx936 topology for a different server; use hardware discovery.
if [[ "${NCCL_TOPO_FILE:-}" == "/usr/local/built-in-508-topo-input-tj-default.xml" ]]; then
    unset NCCL_TOPO_FILE
fi
export NCCL_DEBUG=WARN
# Preserve the checkpoint's ByteLevel BPE backend. LlamaTokenizerFast overrides it.
python3 tools/prepare_step5_tokenizer.py --model "$model_path" --output "$tokenizer_path"
echo 'WARNING: dense text diagnostic; not equivalent to Step5 CSA/SSMax inference.' >&2
exec vllm serve "$model_path" \
    --tokenizer "$tokenizer_path" \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --dtype bfloat16 \
    --quantization fp8 \
    --moe-backend triton \
    --block-size 64 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-memory 8589934592 \
    --enforce-eager \
    --override-generation-config '{"eos_token_id":[1,2,128007]}' \
    --host 127.0.0.1 \
    --port 18106 \
    --served-model-name step5-dense-diagnostic
