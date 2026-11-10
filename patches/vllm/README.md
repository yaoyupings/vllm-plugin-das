# Step5 runtime integration

Step5 runs through `vllm_hcu`; installation and startup do not rewrite the
installed vLLM package. The former five optional upstream source patches are
implemented by the plugin and no longer need to be applied to vLLM.

| Historical vLLM commit | Plugin implementation |
| --- | --- |
| `684a73f` QKV/indexer projection | `vllm_hcu/models/step5_text.py`, `vllm_hcu/model_executor/layers/step4_dsa.py` |
| `7c779c1` causal FP8 prefill scoring and resident selection | `vllm_hcu/model_executor/layers/step4_dsa.py`, `step4_dsa_kernels.py` |
| `6a9ad96` pending-row retirement | `vllm_hcu/model_executor/layers/step4_dsa.py` |
| `f10f778` static expert placement | `vllm_hcu/models/step5_text.py` checkpoint loader |
| `12907f8` interpolated top-k threshold search | `vllm_hcu/model_executor/layers/step4_dsa_kernels.py` |
| `7485fa9` fused MTP draft vocab-parallel argmax | `vllm_hcu/model_executor/layers/skinny_lm_head.py`, `vllm_hcu/patch/worker/framework_opt/patch_step5_skinny_lm_head.py` |
| `1cedd95` two-tile resident DSA top-k, packed router keys, multi-KV-group prefill | `vllm_hcu/model_executor/layers/step4_dsa_kernels.py`, `vllm_hcu/ops/step5/router_bias.py` |

Model registration and the existing worker/backend patch chain select these
plugin-owned implementations. The historical patch texts remain available in
Git history at `c8b17ba`; they are not a deployment prerequisite.

See `docs/step5_runtime_integration.md` for switches, compatibility checks and
the distinction between kernel validation and full-model performance acceptance.
