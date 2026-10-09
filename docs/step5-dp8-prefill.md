# Step5 DP8 prefill profile

Run from the container with eight BW1101 cards:

```bash
bash scripts/step5_dp8_prefill.sh
```

The profile uses TP1, DP8, EP8, DeepGEMM, DeepEP high-throughput communication
with 36 SMs, and three standard MTP draft tokens. It allows 69,632 context tokens,
4,096 batched tokens and 64 sequences. Attention and KV cache remain BF16.
The default checkpoint is `/data/model/step-5-preview-0913-fp8-per-channel`.
Override `STEP5_MODEL_PATH`, `STEP5_TOKENIZER_PATH`, `STEP5_HOST`, `STEP5_PORT`
or `STEP5_SERVED_MODEL_NAME` when needed.

On the validated channel-FP8 checkpoint, one 64K-input, concurrency-eight
prefill round increased input throughput from 8,279.99 to 8,869.85 tok/s
(+7.12%). Output length was eight tokens, with a separate untimed warmup.
These performance measurements used fixed synthetic MTP acceptance rates;
they are not measurements of the model's real MTP acceptance rate.

A separate paired check used standard MTP, the same frozen GSM8K100 questions,
prompts, temperature zero, seed 42, 8,192 maximum output tokens and scorer.
SMS24 scored 96/100 and SMS36 scored 98/100. Both passed four boundary probes
and four 64K probes without truncation. This is a smoke check, not a full
EvalScope score or evidence that the communication setting improves accuracy.

The DeepEP SM setting is opt-in through `VLLM_HCU_DEEPEP_NUM_SMS`. The runtime
validates the supported six-SM granularity and communication budget before
constructing the buffer. This profile is for single-node DP8 prefill.

The profile also reuses the fresh DeepEP dispatch expert counts for channel-FP8
compact workspace sizing. The receiver records routing provenance, and the
workspace helper verifies the same routing tensor and contiguous local expert
map. Changed versioned routing or maps, missing provenance and maps without a
mutation counter use the original recount path.

With SMS36 held fixed, one additional 64K-input, concurrency-eight prefill round
increased input throughput from 8,869.85 to 8,967.62 tok/s (+1.10%). Output length,
warmup, prompts and synthetic MTP parameters were unchanged. All eight native
model traces removed the repeated GPU expert-count kernel. Forty native
rank/case checks covered dispatch, compact permutation and combine.

The corresponding standard-MTP GSM8K100 check remained 98/100 with identical
per-question grading. Four short boundary probes and four 64K probes passed
without truncation. This is the same smoke-check scope described above.
