# Step5 preview text inference on vLLM 0.28.1

This adaptation runs the text decoder of
`/data/model/step-5-preview-0913-fp8` on eight BW1101 (gfx938) GPUs. Vision and
MTP execution are not enabled.

## Source and implementation

The CSA/DSA driver and decoder originate from
`https://developer.sourcefind.cn/codes/OpenDAS/vllm-hcu.git`, branch
`v0.21.0-step37-dp`, commit
`0686260600ad725f3c594a6ad9f5dd0a16830091`.

- `vllm_hcu/step5_config.py` registers the composite and text configurations.
- `vllm_hcu/models/step5.py` implements the text decoder and checkpoint loading.
- `vllm_hcu/model_executor/layers/step4_dsa.py` adapts the reference CSA/DSA
  state and sparse attention plans to the V2 runner.
- `vllm_hcu/model_executor/layers/step5_dsa_compat.py` binds compatible cache
  views and supplies the Torch GPU region selector fallback.
- `step4_dsa_kernels.py` remains byte-identical to the reference branch. Its
  SHA256 is `8dc94365211d002c8506447f54aa0e82ea7ce6c640d3ffbee299d353e39947a6`.

The current DTK compiler rejects the reference selector's register allocation.
The fallback preserves IEEE score ordering, logical-ID tie breaks, causal
visibility, padding, and physical-region packing. It changes only selection;
the reference compression and sparse attention kernels remain in use.

## Correctness requirements

The checkpoint uses block-FP8 scales for routed experts in layers 3 through 87.
The loader preserves `weight_scale_inv` names and values and checks complete
decoder parameter coverage after consuming the entire weight stream. Expert
layers 88 through 90 retain BF16 weights. Normalization, residual accumulation,
and expert routing follow the reference FP32/Gemma/sigmoid behavior.

The 23 full-attention layers use CSA/DSA. The 69 sliding layers explicitly use
HCU FlashAttention with separate K/V sections inside each page. Cache views
alias the V2 allocation and do not allocate a second KV cache.

Keep `--disable-custom-all-reduce`: the HCU custom collective produced
different intermediate values for repeated identical requests. NCCL restored
bit-identical results across all 92 layers and matching prefill/decode tokens.

The tokenizer preparation tool copies tokenizer files to a writable sidecar
and selects `PreTrainedTokenizerFast`. `LlamaTokenizerFast` replaces the
checkpoint's ByteLevel backend and breaks Chinese round trips. Checkpoint
files are not modified.

## Launch

Inside container `xuwq_vllm_0281`:

```bash
cd /workspace/vllm-plugin-das
bash scripts/step5_8card.sh
```

The script locates the source checkout relative to itself and prepends it to
`PYTHONPATH`. Optional path overrides are `STEP5_MODEL_PATH` and
`STEP5_TOKENIZER_PATH`. The default sidecar is
`/workspace/step5-adaptation/tokenizer-bytelevel`.

| Setting | Validated value |
| --- | --- |
| Tensor/expert parallelism | TP8 + EP8; DP1, PP1 |
| Routed experts per rank | 44 of 352 |
| Maximum context / sequences | 8192 / 1 |
| Prefill chunk | 512 tokens |
| Runner / execution | V2 / eager |
| KV layout / page size | LBNHC / 64 tokens |
| KV allocation per GPU | 8 GiB |
| Expert quantization / backend | checkpoint FP8 / Triton |
| Collective | NCCL |
| Endpoint / model name | `127.0.0.1:18106` / `step5` |

The launcher retains checkpoint EOS IDs 1 and 2 and adds the tokenizer's
`<|im_end|>` ID 128007. Loaded weights consumed approximately 76.9 GiB per GPU
in the validated run.

`scripts/step5_dense_diagnostic.sh` disables the indexer. It is a diagnostic
configuration: dense attention matches the selected region set only while
`topk * region_block_size = 4096` tokens cover the entire causal history.

## Validation

Run the focused regression suite from the source checkout:

```bash
PYTHONPATH="$PWD" VLLM_STEP4_DSA_FP8_MATH=off \
VLLM_STEP5_DSA_TORCH_SELECTOR=1 python3 -m pytest -q \
    tests/models/test_step5_preview.py \
    tests/models/test_step5_dsa_compat.py \
    tests/patch/test_platform_dispatcher.py \
    tests/patch/test_plugin_lifecycle.py
```

The suite covers configuration round trips, FP8 scale loading and missing
parameter detection, expert metadata ownership, normalization, region
selection/packing, cache aliasing, actual HCU attention kernels, and patch
registration/lifecycle. GPU tests require an HCU runtime; CPU-only runs skip
those checks.

On 2026-09-30 the eight-GPU run passed short arithmetic (`2+2 -> 4`), a Chinese
greeting (`你好！`), and arithmetic in a 5246-token prompt (`3+5 -> 8`). Eight
consecutive generated tokens matched separate complete-prefill evaluations.
Repeated identical input produced bit-identical hidden states at all 92 layers.
Raw results and logs are stored outside the repository under
`/workspace/step5-adaptation`.

These checks establish correctness for the exercised text path; they are not
an accuracy benchmark. CUDA graphs, multiple concurrent requests, vision,
MTP, and EPLB/elastic EP serving have not been validated. The reference loads
`ssmax_s` but does not use it in forward; this port preserves that behavior.
