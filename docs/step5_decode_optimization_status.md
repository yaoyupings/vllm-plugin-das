# Step5 decode optimization status

This note records the production decisions for the single-node BW1101
`TP2 + DP4 + EP8 + MTP3` profile. Measurements use the per-channel FP8 model
and the same 64K decode topology unless a row explicitly says operator-only.

## Adopted

- The Triton MoE helper loads the checked-in `E=44,N=1536,device_name=BW1101.json`
  tuning file through `VLLM_TUNED_CONFIG_FOLDER`. It is not
  the tile template for masked FP8 DeepGEMM; that path separately chooses
  measured MODE=1101 for up to 512 global tokens. The measured improvement is small
  but repeatable: about 0.17% on short decode and 0.24% on the matching prefill
  comparison.
- Short Step5 sparse attention uses the FlashAttention-style sparse path for
  one to 128 target-verification tokens. At the production geometry,
  single-token decode improved from 0.0892 ms to 0.0391 ms (2.28x), and the
  16-token MTP shape improved from 0.1680 ms to 0.0910 ms (1.85x). A BS32
  service run improved decode from 689.32 to 696.27 token/s (about 1.0%). Long
  prefill remains on the tiled Triton path.
- The HD192 short-query kernel now uses one software-pipeline stage on BW1101.
  The previous two-stage launch reduced occupancy because a second K/V tile
  exceeded the useful LDS working set. At 64K/top-513 the per-layer Attention
  time improved from 0.3086 to 0.2910 ms at M=32 (1.06x), 0.6106 to 0.3395 ms
  at M=64 (1.80x), and 1.2131 to 0.4000 ms at M=128 (3.03x). M=1/4/16 and
  effective region counts 64/256/513 also improved, with bit-identical output
  and LSE in every comparison. The warmed total-BS32 synthetic service check
  was neutral within run variance (1596.26/528.64 before versus
  1586.73/525.49 total/output token/s after), as DP4 maps it to the smallest
  M=32 target shape; total BS64 maps to the larger M=64 shape.
- The stage-1 short-query kernel uses a BW1101-specific split schedule instead
  of stopping as soon as its grid fills the 64 CUs. At top-513 this reduces
  M=32 from 0.2910 to 0.1344 ms (2.17x), M=64 from 0.3394 to 0.2179 ms
  (1.56x), and M=128 from 0.4002 to 0.3693 ms (1.08x). Against the immediately
  preceding stage-1 service run, with identical synthetic MTP acceptance and
  per-position accepted-token counts, total/output throughput improved from
  1586.73/525.49 to 1598.64/529.43 token/s (+0.75%), mean TPOT fell from
  49.56 to 49.17 ms, and mean ITL fell from 135.35 to 134.26 ms.
- Split-state merge now writes directly into the Attention output buffer and
  omits the unused final LSE, eliminating one output copy per sparse layer.
  Including that copy, the isolated M=32/64/128 paths improve from
  0.1377/0.2218/0.3744 ms to 0.1342/0.2178/0.3681 ms, with bit-identical
  output. The split-state merge also selects one wave for two splits and eight
  waves for four splits, reducing the complete direct-output path further to
  0.1313/0.2134/0.3620 ms while preserving the reference reduction result.
- The formal `step5_tp2_dp4_ep8_64k.sh` profile currently keeps QKV in BF16:
  it does not set `VLLM_STEP5_QKV_FP8_W8A8`, whose model default is 0. The
  optional `step5_attention_8card.sh` profile enables output-MSE-calibrated
  QKV FP8 and requires a calibration file. That path uses LightOP's
  channel-wise FP8 hipBLASLt kernel, which was faster than the installed FP8
  alternatives. Those optional-kernel results are not evidence that the
  formal profile was running FP8 QKV. The 698.48 -> 806.07 BS64 comparison
  below retained BF16 QKV on both sides.
- MTP3 short-prefill indexer scoring is batched across requests and reads the
  summary pool directly through the block table. At 64K context, the per-layer
  score stage improved from 1.2004 ms to 0.3430 ms at total BS32 (3.50x), and
  from 2.3946 ms to 0.6778 ms at total BS64 (3.53x), with bit-identical fp32
  scores.
- The MTP3 scorer now puts the four consecutive queries of each request into
  one tensor-core M tile and reuses its summary K tile. The per-layer scorer at
  total BS32 improved from 0.3723 ms to 0.1145 ms (3.25x); at BS64 it improved
  from 0.7241 ms to 0.2096 ms (3.45x), with bit-identical scores including
  groups that cross a region boundary. With synthetic acceptance fixed at
  61.21%, the BS32 1024-to-512 service comparison improved total/output
  throughput from 1557.44/515.79 to 1596.26/528.64 token/s (2.49%), reduced
  mean TPOT from 50.24 to 48.74 ms, and reduced mean ITL from 141.51 to
  137.27 ms.
- Q/K indexer normalization and partial RoPE share one launch for up to 128
  target-verification tokens. The original joint kernel used four warps and was
  slower at 64 tokens; using one warp keeps the 256-wide reduction in one wave.
  It improves M=32 from 0.01314 to 0.00840 ms, M=64 from 0.01362 to 0.00909 ms
  (1.50x), and M=128 from 0.01686 to 0.01148 ms (1.47x), bit-identically.
  Larger prefill shapes retain the reference two-launch path because the fused
  reduction begins to diverge there.
- Prefill radix selection now packs physical regions and appends the current
  causal region in the same kernel. At 64K/top-512 this reduces the selection
  metadata stage from 0.1184 ms to 0.1128 ms at BS32 (5.0%), and from 0.1537 ms
  to 0.1470 ms at BS64 (4.5%). Short verification no longer allocates or writes
  logical IDs that its untiled Attention kernel does not consume; at 8192
  regions this reduces 32/64/128-row selection from
  0.1130/0.1135/0.1481 ms to 0.1058/0.1060/0.1393 ms, with identical packed
  slots and counts. Long tiled prefill still retains logical IDs.
  A warmed BS32 check after combining the later exact indexer, direct-output,
  merge-wave, and metadata changes kept identical MTP acceptance but measured
  1593.50/527.73 total/output token/s versus 1598.64/529.43 immediately before
  them (-0.32%), so no additional BS32 service-level gain is claimed.

### Live BF16 expert tiles and ragged MTP batches

The three routed-expert layers that remain BF16 now use a device-counted
expert grid for the TP2/DP4/EP8 low-latency profile. The previous native grid
launched M tiles over the full 2048-row dispatch capacity for all 44 local
experts, even though most decode experts only received a few live rows.
The replacement launches expert/N tiles and loops over live M tiles using
GPU counts. It reuses the native BF16 tile calculation and the selected
BLOCK_SIZE_M/N/K, including its K-reduction order. It also works inside
PIECEWISE capture when the batch descriptor is absent.

With the production M128/N128/K64 configuration and 44 experts, the two
isolated GEMMs measured 7.7919 -> 1.9930 ms and 3.8295 -> 1.0122 ms.
Results and untouched padding were bit-identical, including graph replays
with zero, partial, and multiple M tiles. With M32/N64/K64 the same grid
change measured 6.8317 -> 1.2504 ms and 3.6132 -> 0.6479 ms; that tile
configuration is not newly selected by this change.

The warmed BS64, input1024/output512 synthetic-MTP service comparison
completed 64/64 requests. Output throughput improved 698.4773 -> 806.0666
token/s (+15.40%), mean TPOT 67.7000 -> 57.8990 ms (-14.48%), and mean
ITL 189.4302 -> 162.0062 ms (-14.48%). Both runs had exactly 11624 drafts,
34872 draft tokens, 21156 accepted tokens, and 60.6675843% acceptance.
Baseline result: `/tmp/step5_norm_baseline_bench/openai-chat-infqps-concurrency64-step5-20261006-223408.json`.
Candidate result: `/tmp/step5_bf16_compact_bench/openai-chat-infqps-concurrency64-step5-20261007-001232.json`.
This is a short-context service screening result; it does not establish a
new 64K/4K throughput figure. The formal Decode launcher enables
`VLLM_STEP5_BF16_COMPACT_GRID=1`; an explicit value of 0 restores the prior
path. The FP8 experts and extra shared-expert streams are unchanged.

Ragged MTP batches of 65--128 tokens now also use short-query DSA when
host metadata says every request has at most four queries. They no longer
need to satisfy the uniform four-query graph schedule to avoid CPU tiles.
Ordinary 65--128-token prefills keep the tiled path. Selector and Attention
routing share one predicate, so selection omits logical IDs exactly when
Attention will not consume them. Boundary/selection checks and 65/96/128
query numerical comparisons passed. The DSA checks passed 33 tests
(excluding an existing direct-import duplicate-registry test); the BF16
expert checks passed two tests with dynamic graph counts and strided output.

### Decode profiling scope

The original `/tmp/step5_profile` trace was collected from the beginning of
a benchmark and predominantly covered 512-token prefill chunks. The run
also failed during profiler teardown, so its per-layer scalar-read counts
must not be interpreted as steady Decode synchronization.

A replacement `/tmp/step5_decode_profile` trace delays capture by 100 worker
iterations and records three steady iterations on all eight ranks. Rank0
Attention inputs are [64,32,192]; sliding layers use direct paged Attention,
not a paged-to-contiguous prefill conversion. No scalar reads occur inside
Attention or indexer scopes. The 45 CPU scalar operations and 12 DtoH events
are outside those scopes. BF16 expert GEMMs took 86.869 ms of 442.601 ms
GPU execution across the three sampled steps (~19.6%); DeepEP combine took
53.893 ms (~12.2%). This evidence directed the live-tile optimization above.
The profiled run was stopped after traces were collected and is not used
for service throughput or correctness claims.

## Implemented candidates pending long-context adoption

### Full target-model graph

The target model can now capture all six uniform MTP3 graph sizes. Its
four-query DSA schedule keeps fixed addresses for compression, rollback
prefixes, request indices, and score storage, and is bit-identical to the
general scheduler. A 32-request drain test with staggered 16--78 token outputs
completed 32/32 requests without a VM fault or worker failure.

An earlier short-drain service A/B did not justify enabling it: the same synthetic workload
took 9.56 seconds with `FULL_AND_PIECEWISE` and 6.67 seconds with `PIECEWISE`, a
43.3% regression. The completion-drain interval was essentially unchanged
(2.92 versus 2.94 seconds), so that workload did not establish a steady
decode benefit and showed additional initial latency. At that point the formal launcher retained target
`PIECEWISE`; later qualification below adopted FULL_AND_PIECEWISE.
The MTP draft keeps its verified full decode graph.

After the live BF16 grid and ragged-query fixes, a matching warmed BS64,
input1024/output512 synthetic run measured 806.0666 -> 817.3194 output token/s
(+1.40%) and 57.8990 -> 56.9690 ms mean TPOT (-1.61%) when only the target
Graph mode changed. Both runs completed 64 requests with exactly 11624 drafts,
34872 draft tokens, and 21156 accepted tokens (60.6675843%). Real dispatcher
metrics recorded FULL target execution, including padded shrinking batches;
a fresh 32-request 16--78-token drain also completed every requested length.
Result: `/tmp/step5_compact_full_graph_bench/openai-chat-infqps-concurrency64-step5-20261007-002440.json`.
This single screening run shows a modest gain, not statistical significance or
64K/4K performance. FULL at memory utilization 0.92 initialized 924376 tokens
per DP (13.28 requests at 69632 tokens); BS64/DP4 requires 1114112 per DP.
The candidate is therefore pending long-context capacity/performance checks.

## Rejected after measurement

### Small-M QKV alternatives

For `K=4096,N=6912`, including per-token activation quantization:

| M | LightOP end-to-end | AITER Triton stock best | Result |
|---:|---:|---:|---|
| 1 | 0.0680 ms | 0.1033 ms | AITER 52% slower |
| 4 | 0.0676 ms | 0.0998 ms | AITER 48% slower |
| 16 | 0.0679 ms | 0.0998 ms | AITER 47% slower |
| 32 | 0.0707 ms | 0.0998 ms | AITER 41% slower |

A second M=1 search tried 24 small-M AITER tile variants. Its best GEMM was
0.0815 ms versus LightOP's 0.0610 ms, still 34% slower before activation
quantization. AITER has no `BW200B/BW1101-GEMM-A8W8` tuning file in the
installed package. LightOP W8A16 was also slower, so retaining an extra BF16
weight copy for a shape switch would add memory without a decode gain.

A Step5-specific FP32 Gemma norm + BF16 store + per-token FP8 quantization
prototype was also measured. At the production BS64 per-DP shape (M=64), it
was bit-identical to native Inductor norm plus LightOP quantization across 100
random inputs and reduced that isolated pair from 0.01377 to 0.00997 ms. The
end-to-end concurrency-64 1024-to-512 synthetic-MTP comparison nevertheless
regressed: baseline/fused output throughput was 698.48/695.17 token/s and mean
TPOT was 67.70/68.10 ms. Acceptance statistics were identical (60.6676%,
21,156 accepted of 34,872 drafted tokens). The extra custom-op boundary erased
the isolated gain, so the candidate was removed and is not enabled.

### Projection and shared-expert fusion

- The TP2 merged projection service comparison regressed decode from the
  225.46 token/s QKV-FP8 reference to 220.26 token/s. The mixed FP8/BF16
  projection kernel was also 15.5% slower than the two native GEMMs at the
  TP2 four-token sparse shape. These paths remain disabled.
- Merging only the four indexer projections reduced their M=32 operator time
  from 0.0407 ms to 0.0244 ms, but changed BF16 GEMM rounding from M=8 upward.
  In the identical 1024-to-512 BS32 service check, total throughput fell from
  1053.79 to 793.94 token/s and MTP acceptance fell from 29.82% to 23.38%.
  LightOP grouped BF16 GEMM cannot replace it because the installed gfx938
  wheel lacks its grouped-GEMM HSACO. AITER grouped GEMM requires N divisible
  by 128 (the gate projection has N=8); grouping the other three projections
  was about 0.66--0.77 ms versus 0.036--0.040 ms for native separate GEMMs.
- Fusing the attention head sigmoid gate into the output projection was also
  rejected. The prototype took 0.100--0.117 ms for M=4--64, more than twice
  the 0.046--0.062 ms native gate-plus-GEMM path, and was not bit-exact.
- Folding the indexer Q e4m3 rounding into fused QK-norm/RoPE was
  bit-identical and reduced the isolated norm-plus-scorer pipeline by 9.5% at
  BS32. It did not improve the fixed-acceptance service run: total/output
  throughput changed from 1596.26/528.64 to 1571.71/520.52 token/s and mean
  TPOT increased from 48.74 to 49.71 ms, so the standalone rounding pass stays.
- The fused main QK-norm/RoPE candidate was compared with the deployed
  Inductor path rather than only with eager PyTorch. Inductor carries FP32
  normalized values through RoPE before the BF16 store, so eager PyTorch
  differs by as much as 0.0625. The custom kernel matched compiled output
  exactly at M=1 and M=32; across the other tested M=4/16/64 shapes only one to
  three Q elements differed, by at most 0.0078125, and K was exact. It still
  adds a custom-op boundary around work Inductor already fuses. The earlier
  service run fell from 1053.79/348.99 to 926.54/306.85 total/output token/s,
  so the production profile retains the compiled native QK-norm/RoPE sequence.
- Early shared-expert launch on an auxiliary stream increased the 64K warm
  prefill maximum from 153.7 s to 324.4 s and reduced the observed active
  generation rate to 22.96 token/s. The existing token threshold does not
  remove contention at the production decode shape, so the formal profile
  keeps the shared-expert stream disabled. vLLM also rejects dual-batch
  overlap when speculative decoding or CUDA graphs are enabled, so DBO cannot
  overlap DeepEP in the required MTP3 graph-enabled profile.

## Constraints that cannot be enabled by a flag

### TP2 + DP4 + DCP2

This topology is invalid for Step5's GQA geometry. The model has four total KV
heads, and vLLM requires `tensor_parallel_size > num_kv_heads` before DCP may
split repeated KV heads. With TP2, startup rejects DCP2 before workers launch.
On an eight-card node the smallest valid DCP topology is TP8 + DP1 + DCP2,
which loses DP4 and is a different performance target. The formal TP2 launcher
fails early with this explanation when `STEP5_DCP_SIZE` is not 1.

## LightOP and AITER kernel audit

The installed LightOP and AITER packages were checked against the actual TP2
Step5 geometry: 32 local Q heads, two local KV heads, head dimension 192, and
`K=4096,N=6912` for QKV. AITER `fused_qk_rmsnorm_rope` and LightOP
`split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant` both reject this
geometry because their reviewed paths require head dimension 128. LightOP's
sparse-attention package exports index transforms and top-k helpers, but no
sparse attention compute kernel. AITER's sparse MLA kernel requires its MLA
cache layout and power-of-two dimensions, which do not match Step5's separate
GQA K/V cache and 192-dimensional heads.

For QKV, LightOP's small-M smooth-GEMM table has no tuned
`N=6912,K=4096` entry. Its public channel-wise hipBLASLt path is the optional
FP8-QKV backend and remains faster than the installed AITER alternatives in
the measurements above. Therefore none of the installed library kernels can
replace the remaining Step5-specific decode stages without changing semantics
or regressing the measured shape.

## Remaining kernel work

Short MTP queries now batch scoring across requests, and the BS32 Q/K
normalization shares one launch. Projection, summary update, radix selection,
and attention are still separate stages. LightOP and AITER only provide
MLA-specific fused kernels for incompatible head dimensions and cache
semantics. Fusing the remaining stages therefore requires a new Step5-specific
kernel rather than wiring an existing library operator into this backend.


### Grouped provider/head scoring candidate

`VLLM_STEP5_GROUPED_HEAD_INDEXER=1` folds provider groups and all index heads
into one short-query GEMM, then preserves the original sequential FP32
head accumulation. It retains the existing e4m3 activation rounding and
native causal/region masks. The actual TP2 geometry is G=2,H=4,D=256;
the initial G=1,H=8 prototype is not a production-geometry measurement.
Twenty-four native-reference tests passed with BF16 and production FP8 summary pools for queries/request 1--4, group-major
strided outputs, page-shuffled summaries, and graph replay with changing
positions, Q and weights. The feature remains opt-in pending a service check.
Operator measurements including activation rounding are in
`/tmp/step5_grouped_head_actual_geometry_bench.log` (BF16 pool) and
`/tmp/step5_grouped_head_fp8_summary_bench.log` (production FP8 pool).
The first service startup caught an FP8 masked-load fill type that BF16-only
tests missed. The fill is now floating-point zero; the added FP8-pool graph
tests cover it. That failed startup produced no benchmark result.

The 64K full-graph capacity candidate at memory utilization 0.97 was rejected
by startup's free-memory check. Lowering dispatch capacity from 2048 to 1024
by using a 256-token scheduler and smaller capture sizes worked, but 0.955
still ran out of memory during Graph warmup. Neither candidate reached serving
state or produced a performance result. Formal memory remains 0.92.

Production FP8-pool scoring (including activation rounding) selected BK128,
four warps and two stages. At 16 requests per DP, queries/request 1/2/3/4
measured 0.20251/0.20279/0.20233/0.20023 ms on native and
0.09926/0.10124/0.12289/0.12306 ms on grouped-head scoring. Every tested tile
variant produced bit-identical scores. BK64 was slower than native for q=1/2,
so the BF16-pool tile choice is not used for the production FP8 summary pool.
These are operator results; service gains are still pending.

The warmed BS64 input1024/output512 service screen with grouped-head scoring
completed 64/64 requests, output throughput 817.3194 -> 818.8987 token/s
(+0.19%), mean TPOT 56.9690 -> 56.8968 ms (-0.13%). Synthetic acceptance was
identical: 11624 drafts, 34872 draft tokens, 21156 accepted tokens. Both sides
used the full target Graph and BF16 QKV. The operator's long-context gain did
not become a material short-context service gain. The selected BK128 tests
passed 24 cases again after service startup.
Result: `/tmp/step5_grouped_head_v2_bench/openai-chat-infqps-concurrency64-step5-20261007-010820.json`.
The completed BS32 65536-input/4096-output screen is recorded below.

A native KV allocator cost audit also rejected treating the 20 padding-layer
warning as an automatic 27.78% capacity gain. Full and sliding groups share a
physical page pool. With 23 full and 72 sliding layers, regrouping from size23
to size24 reduces groups from five to four but increases the 69632-token
per-request bound from 2622750720 to 2694316032 bytes in the native calculator
(+2.73%). Smaller exact groups add substantial scheduler/group overhead.
No speculative KV regrouping change was adopted. Audit:
`/tmp/step5_kv_group_cost_audit.py`, `/tmp/step5_kv_group_cost_audit.log`.

The first BS32 long-context wrapper incorrectly imported the metadata adapter
before setting its DP argument, so the adapter retained DP=1 and capacity
preflight skipped the case. No measured 64K iteration ran in that attempt.
The corrected wrapper explicitly sets DP=4, retains the 0.9 guard, reads all
four startup capacities, and completed one BS32 iteration:
`/tmp/step5_grouped_head_graph64k_bs32_v2_benchmark.py`.
Result and per-iteration sidecar:
`/tmp/step5_grouped_head_graph64k_bs32_v2_result.json` and
`/tmp/step5_grouped_head_graph64k_bs32_v2_result.iterations.json`.
The manifest also records preemption counters, so queued/preempted runs cannot
be passed off as an unpreempted Decode comparison.


### Completed 64K full-target-graph screen

The grouped-head FP8-summary scorer and full target Decode Graph service
completed 32/32 requests with 65536 input tokens and 4096 output tokens each.
The valid common Decode window measured 859.36049 output token/s and mean
ITL 101.11884 ms (p50 100.98841, p99 104.18955). Window coverage was 35.7349%,
with 46827 output tokens over 54.49052 seconds. All four DP preemption
counter deltas were zero. The request token hashes match the original
random.Random(42 + request_index) input generator. QKV remained BF16.
This is one measured iteration using the fixed synthetic MTP acceptance
rates [0.838992, 0.598322, 0.362686], not an accuracy or natural-acceptance run.

The historical matching BS32 64K/4K screen measured 575.90817 token/s and
mean ITL 154.90923 ms, so cumulative throughput improved about 49.2% and ITL
fell about 34.7%. Several kernels, target Graph, and all-reduce configuration
changed; the historical run also had a separate long-context warmup while
this run had a short-context warmup. This is not a sole-variable indexer A/B.
The mixed prefill stage's 3214.38 token/s must not be reported as the
standalone input64K/output1 Prefill benchmark.

Evidence:
- /tmp/step5_grouped_head_graph64k_bs32_v2_result.json
- /tmp/step5_grouped_head_graph64k_bs32_v2_result.iterations.json
- /tmp/step5_grouped_head_graph64k_bs32_v2_manifest.json
- /workspace/step5-tp2-performance/tp2dp4-final-decode-synthetic-v4-final-single-round-v4-summary.json

### Small live-batch tile selection

The compact BF16 grid used to return before the existing small-batch tile
selection, leaving M128/N128 even for small Decode batches. It now selects
M32/N64/K64 for at most 512 global scheduler tokens, using already-present
CPU DP counts cached once per forward. Capture falls back to its padded
batch descriptor when dummy CPU counts are zero. It never copies expert
counts or DP device counts back to the CPU.

At E44/capacity2048, representative Decode counts changed compact up/down
GEMM from 1.97433/0.99295 ms to 1.16736/0.58911 ms. Extreme one-expert
skew prefers M128; this host-token policy does not claim to optimize skew.
All five tested tile variants preserved reference output and padding
bit-exactly. Six native-reference GPU tests include dynamic-count Graph
replays; thirteen CPU tests cover host-count caching and both grid paths.
The scorer also skips wholly invisible score tiles on-device, avoiding
full-width summary work in short-context FULL captures. Its 24 native
reference tests cover changing positions and empty regions under replay.
Combined kernel/policy tests passed 42 cases, followed by the added zero
dummy-count capture regression passing in the 13-case CPU policy run.

Evidence: /tmp/step5_live_bf16_tiles_bench.json,
 /tmp/step5_small_tiles_and_indexer_tests.log,
 /tmp/step5_bf16_tile_policy_capture_tests.log.
The running service predates these tile and causal-skip changes; the next
short service screen is needed before attributing an end-to-end gain.

The formal Decode launcher now defaults to FULL_AND_PIECEWISE for the
target model and enables grouped-head indexer scoring after the successful
64K BS32 screen. STEP5_TARGET_GRAPH_MODE=PIECEWISE and
VLLM_STEP5_GROUPED_HEAD_INDEXER=0 remain explicit fallback overrides.
Prefill retains Graph NONE. This does not establish 64K BS64 capacity.

The single warmed BS64 input1024/output512 service screen completed 64/64:
output throughput 818.89871 -> 827.83147 token/s (+1.09%), mean TPOT
56.89679 -> 55.90127 ms (-1.75%). Total input/output tokens were identical
(66176/32768), as were synthetic MTP counts (11624 drafts, 34872 draft
tokens, 21156 accepted tokens). No DP preemption counter increased.
Both sides retained full target Graph, grouped-head indexer and BF16 QKV;
the new side adds small live-batch tiles and invisible-score-tile skipping.
This is a one-round short-context screen, not a new 64K/4K measurement.
Result: /tmp/step5_small_tiles_v1_bench/openai-chat-infqps-concurrency64-step5-20261007-014602.json
Manifest: /tmp/step5_small_tiles_v1_manifest.json
Comparison: /tmp/step5_small_tiles_v1_comparison.json

### Rounding-fusion candidates rejected for the 64K target

Fusing BF16-to-e4m3-to-BF16 Q rounding into the grouped scorer removes one
launch and a temporary tensor. It passed the sampled bit-exact score and
Graph tests but repeated conversion in every key CTA regressed width8192
from 0.1012/0.1023/0.1221/0.1179 ms to
0.1088/0.1108/0.1483/0.1438 ms for queries/request 1/2/3/4.
The width128 benefit is not a reason to adopt it for the 64K workload.

Direct FP8 MMA reduced the width8192 q1/2/3 screen to
0.0548/0.0563/0.0947 ms but q4 differed from the native score by
3.814697265625e-6. It is not qualified for replacing the native scorer.
An interleaved-zero FP8-K prototype preserved the sampled native rounding
at every tested q count and Graph replay but slowed width8192 to
0.1354/0.1343/0.2142/0.1726 ms. Neither FP8-MMA prototype is in
production code or enabled in the service.

That completed BF16 service screen used PID3162510, source dd54506. Its Graph dispatch table explicitly shows FULL at real token
counts 12--64, including padded batches. The MoE log confirms loading
E=44,N=1536,device_name=BW1101.json and masked DeepGEMM mode1101.
The kernel and service evidence is preserved outside /tmp at
/workspace/step5-tp2-performance/live-tile-and-index-rounding-20261007.
The previous long-context evidence is preserved at
/workspace/step5-tp2-performance/grouped-head-full-graph-64k-bs32-v2.

### Small-M QKV and calibrated full-Graph FP8 screen

A new fused token-amax / activation-quant / GEMM / scale prototype matched
LightOp quantized bytes and per-token scales, including the zero-row scale
contract. All tested GEMM tiles were slower than the current hipBLASLt path
and differed in a small number of BF16 output elements. At M=1/4/16 the best
candidate was about 0.0488/0.0487/0.0494 ms versus 0.0349/0.0349/0.0345 ms
on native quant-plus-GEMM. It was not added to production code.

The installed LightOp small-M smooth config has no N6912/K4096 entry.
Six neighboring tile configs were tested through its raw smooth operator
at M=1/4/16; all failed the FP8 output reference and were slower than the
qualified channel-wise hipBLASLt path. They are not an FP8 replacement.

Graph-level precision microbenchmarks used actual TP2 K4096/N6912 and
included activation quantization. Native BF16 vs FP8 at M=1/4/16/32/64/128
measured 0.0498/0.0431/0.0432/0.0439/0.0417/0.0538 ms versus
0.0286/0.0286/0.0283/0.0315/0.0306/0.0413 ms. FP8 also won M256--4096.
These random-weight operator results do not justify switching small M
to BF16, and are not a model-accuracy evaluation.

The current profile plus output-MSE-calibrated QKV FP8 completed a single
warmed BS64 input1024/output512 screen: output throughput
827.83147 -> 838.88914 token/s (+1.34%), mean TPOT
55.90127 -> 55.18102 ms (-1.29%). All 64 requests completed with zero failures,
input/output token totals 66176/32768 and identical synthetic MTP counts
(11624 drafts, 34872 draft tokens, 21156 accepted tokens). No DP preemption
counter increased. Main target FULL Graph remained active.
Per-DP KV capacity rose from about 924201 to 994709 tokens (+7.63%).
This still does not fit 16 requests/DP at input65536/output4096.

The calibration file has all 92 main and three MTP groups and the matching
model-index hash. It uses independent document activations from DP0/1;
heldout DP2/3 and evaluation answers were excluded. That completed candidate
service used PID3217802, source1138074, with QKV FP8. Formal QKV stays BF16:
ordinary MTP accuracy and a 64K/4K FP8 comparison remain unverified.
Result: /tmp/step5_qkv_fp8_graph_v1_bench/openai-chat-infqps-concurrency64-step5-20261007-021335.json
Evidence: /workspace/step5-tp2-performance/qkv-small-m-and-decode-overlap-20261007.

### Decode-only shared-expert overlap candidate

STEP5_SHARED_EXPERT_OVERLAP=decode now enables an opt-in profile whose
shared-expert aux stream is gated by the scheduler's host BatchDescriptor:
uniform batches with 1--4 queries/request are eligible. The initial
candidate made missing, mixed, nonuniform or large chunks serial, including
the MK-internal fallback. The follow-up repair below preserves native MK
overlap while restricting only the added aux-stream schedule. The decision is cached once per ForwardContext without reading
a GPU count. This avoids relying only on the padded hidden-state shape,
which allowed the previous overlap candidate to compete with Prefill.
The formal default stays off and the standalone Prefill profile rejects
this option.

Fifteen tests passed, including the existing input-alias preservation
policy and native auxiliary-stream/Graph replays with changing BF16 inputs.
The tests cover both serial and overlapped branches and their output
dependencies. CLI checks verify default-off, Decode opt-in and rejection
of standalone Prefill/invalid modes. A full-model service performance
screen is still required before adopting this overlap profile.

The native MRv2 FULL capture caller was inspected before accepting the
overlap performance run: it sets runtime NONE with no BatchDescriptor.
The initial descriptor-only guard would therefore capture serial execution.
That v1 startup was explicitly superseded before a measured benchmark.
It was not restarted because of an observation timeout.

The guard now falls back to existing CPU query-start mirrors during FULL
capture, caches the result once per ForwardContext, and rejects large,
ragged, inconsistent or nonuniform query groups. No device tensor is copied
or reduced on the host. Twenty-eight tests passed after this repair,
including auxiliary-stream Graph replays without a descriptor, GPU-count
poison tests and mixed attention groups. The successful earlier 15-test
descriptor-only screen did not cover the actual FULL-capture call shape.
The repaired v2 service screen completed 64/64 at input1024/output512,
BS64, with the same input/output tokens and synthetic acceptance counts as
the preceding calibrated-FP8, no-overlap screen. Output throughput fell
838.88914 -> 706.43915 token/s (-15.79%), while mean TPOT rose
55.18102 -> 67.65255 ms (+22.60%). No DP preemption counter increased.
The opt-in profile is rejected; the formal default remains off.

The service env flags requested aux-stream overlap, but the captured log
contains no mode-selection diagnostic. Isolated stream/Graph tests passed;
the actual full-model aux-stream activation is not independently proven.
Therefore this result rejects the configured candidate, without attributing
the entire regression to verified communication/computation overlap.
Results and comparison are preserved in the evidence directory above.

Checkpoint shape inspection confirms Q/K/V rows 12288/768/768 at both SWA
and full Attention, including MTP: TP2 QKV width is N6912 in either case.
A separate N3840 SWA hypothesis was incorrect and is not used as a basis
for precision selection. The formal QKV precision remains BF16.


### Fold indexer E4M3 query rounding into short Q/K norm + RoPE

The indexer requires BF16 -> E4M3 -> BF16 activation rounding before its
score GEMM. Moving that stage into the grouped scorer regressed long
contexts because every key CTA repeated the conversion. The new candidate
instead rounds Q exactly once while the joint norm/RoPE kernel stores it.
It preserves the BF16 store boundary before E4M3 conversion and keeps K
unrounded for CSA compression. Both short-score entry points explicitly
skip their standalone rounding when projection has already performed it.
The opt-in VLLM_STEP5_INDEXER_ROUND_IN_NORM=1 applies only to 1--128 tokens,
proxy_dim256 and BF16; longer Prefill and other dtypes keep the old path.
The formal default is still off pending a full-model screen.

The isolated reference combines the existing fused norm/RoPE and deployed
Torch casts; both timings include CUDA Graph launches. At M32/64/128,
rotary_pairs16, the complete operator improved from
0.01161/0.01334/0.01700 to 0.00412/0.00523/0.00748 ms. Rotary_pairs64
showed the same trend. Thirty-six random, zero-row and overflow-weight
cases preserved Q bits/NaN masks, K bits and signed zero. Graph replays
with changing inputs/positions also matched. Ten production tests passed,
including actual Step4Indexer projection -> verification-score integration
for queries/request1--4, partial/full rotation and long8192-region scores.
A poison cast verifies that the optimized scorer omits its redundant stage.
No full-service throughput improvement is inferred from this microbenchmark.

The formal BF16-QKV service, full target Decode Graph and ordinary MTP3,
completed the frozen 100-question zero-shot GSM8K screen at 96/100, with
no request errors or output truncations (max_tokens8192). Temperature1,
top_p0.95, seed42 and EvalScope1.12.0 native rule scoring were fixed.
This provides ordinary inference evidence for the current Graph/tile
profile; it is not a full-dataset accuracy guarantee. The calibrated-QKV
FP8 pair subsequently completed as recorded below. Evidence is preserved at
/workspace/step5-tp2-performance/fullgraph-qkv-ordinary-20261007.


### Ordinary full-Graph QKV pair and native MK overlap repair

The calibrated-QKV FP8 service completed the same frozen100 zero-shot
GSM8K screen at 97/100 versus BF16's 96/100. Both sides used ordinary
MTP3, full target Decode Graph, identical CLI, question IDs, prompt
messages, input token counts, sampling parameters and EvalScope rules.
Both completed without request errors or truncation. The new norm/round
fusion was disabled on both sides to isolate QKV precision. Custom-op
and quantization-GEMM settings were explicit true for FP8 and default true
for BF16, as checked against the installed env defaults. Worker model
inspection samples one DP's two TP workers; it is not an all-eight-worker
audit. The per-question comparison records all grade transitions.
This fixed100 result does not prove better FP8 accuracy or full-dataset
noninferiority, and the matching64K/4K FP8 service performance is unverified.
Formal QKV remains BF16 pending the performance decision.

The rejected v2 aux-stream guard also disabled the baseline MK-internal
shared/combine schedule on non-short batches. Native DeepEP-LL reports
supports_async=True, and the modular finalizer starts combine, runs shared
experts, then executes the receive hook. The revised guard limits only the
new aux stream: non-short batches retain MK-internal overlap when supported,
or remain serial when unsupported/native restrictions apply. EPLB and
two-sided-backend restrictions are preserved. A native-finalizer ordering
test confirms shared compute remains before receive completion. The repaired candidate was subsequently service-benchmarked; its result
and actual mode-selection evidence are recorded below. Twenty-eight CPU tests and four native
aux-stream/Graph GPU tests passed after the repair.

The original vllm_hcu logger inherits WARNING and drops the candidate's
INFO mode diagnostics. It now uses vllm.hcu.shared_experts, whose effective
level is INFO, and the v3 full-model screen verified both selected modes.
Evidence and test logs are preserved in the fullgraph-qkv-ordinary directory.


### Repaired auxiliary-stream result and long-context capacity candidate

The repaired v3 short-context screen (BS64, input1024/output512) completed
64/64 requests but measured 708.01256 output token/s versus 838.88914 with
the auxiliary stream off (-15.60%). Mean TPOT increased from55.18102 to
67.56472 ms (+22.44%). Input/output totals, synthetic draft/acceptance
counts and zero DP preemption deltas match the baseline. INFO logs prove
that short Decode selected the auxiliary stream and non-short batches
retained MK-internal overlap. The candidate remains disabled.
Evidence: /workspace/step5-tp2-performance/shared-overlap-v3-20261007.

In this sequence-parallel profile shared MLP gate/up/down projections set
disable_tp=True. The routed-plus-shared result uses the same final TP
all-gather in both modes; an added shared-MLP TP collective is therefore
not the cause. Auxiliary event/stream scheduling or compute contention
remain hypotheses without a full trace. The default still retains native
MK shared/combine overlap where supported.

The optional SWA Prefill crop passed ten native GPU tests, including exact
FlashAttention output equality for actual head_dim192, KVheads2 and64K
contexts. It removes only KV pages invisible to every query and reads no
device count on the CPU. It is disabled during full Graph capture.
Test evidence: /workspace/step5-tp2-performance/bs64-capacity-20261007.

The launcher accepts STEP5_GPU_MEMORY_UTILIZATION as an explicit candidate
override and rejects non-finite or out-of-range values. Defaults remain0.92
Decode and0.89 Prefill. Bash syntax and nine stub CLI cases passed. This
adds no claim of successful higher-memory startup. Sixteen requests/DP at
65536 input plus4096 output and four draft slots need1115136 whole-page
tokens/DP; the0.92 FP8 profile's994709 tokens do not meet that bound.
Higher-memory, cropped-workspace capacity must be verified by all four
initialized DP capacities, successful Graph warmup and no runtime preemption.


The0.963 capacity v1 candidate failed the startup free-memory check after
communication initialization:137.72GiB free versus138.66GiB requested.
It did not load the model or produce a performance measurement. The revised
v2 candidate uses0.955,256 scheduler tokens and capture sizes through256,
plus calibrated QKV FP8, norm/round fusion, SWA crop and explicit DSA reserve.
It passed model startup and Graph warmup. All four DP engines initialized
1205894 effective KV tokens (17.32 requests at69632), meeting1115136
page/draft tokens and the explicit1% capacity guard. Each worker reserves
1583996576 bytes for DSA side pools and fixed workspaces before KV allocation.
A single BS64 input65536/output4096 benchmark is in progress; no new TPS
result is claimed from the capacity proof.

The benchmark retains the supplied common Decode window formula:
max(first-token timestamps) to min(completed last-token timestamps).
A window observer will terminate this owned benchmark only if a measured
4096-token request finishes while another request is still queued, which
proves that common window is empty. This avoids continuing a known invalid
performance run; observation delays alone do not trigger a restart.
Candidate source and runtime evidence are under the bs64-capacity directory.


### Fused fixed-address SWA window metadata

VLLM_STEP5_FUSED_WINDOW_METADATA=1 replaces the builder's separate length,
offset, gather, mask and staging-copy operations with one Triton kernel.
It writes directly into the existing contiguous tables, one persistent table
per width, and persistent length buffers. It preserves every table element
including invisible trailing cells, accepts strided input metadata, reads
no device count on the CPU and keeps Graph replay addresses stable.
Cascade, noncausal and unsupported parallel/window cases retain native guards.
The default remains off pending a full-model performance decision.

Eight CPU policy tests passed (vendor FA dependency isolated for CPU testing),
and nineteen native GPU tests passed: queries1/4/512, window1/64/512, strided
inputs, changing sequences/pages/query offsets, persistent builder addresses
and exact production BF16 GQA Attention under Graph replay. The independent
qualification adds fifteen metadata geometries, three replays each and three
native Attention replays with long/ragged/padded requests. All outputs match
the old path exactly.
At the measured eight-request shape, old/new metadata time including staging
was0.06372/0.00819ms eager and0.03792/0.00190ms in an isolated Graph.
This removes builder overhead outside the model's FULL Graph; it is not yet
an end-to-end throughput result. Evidence: bs64-capacity-20261007.

The256-token BS64 capacity candidate did not produce a valid Decode window.
A4096-token main request completed on every DP while nine requests/DP were
still queued. Thus min(completed last token) precedes max(first token) and
the supplied benchmark's common64-request window is necessarily empty.
The owned benchmark was terminated after this proof, clients disconnected,
all running/waiting counters returned to zero, and the service was stopped.
There was no preemption. Capacity remains proven; BS64 pure-Decode TPS does
not. Preserve the original output4096 and measurement formula while improving
scheduler/workspace capacity rather than relabeling this as valid BS64 TPS.


### Bound masked FP8 gate/up workspace to its actual width

VLLM_STEP5_LL_COMPACT_WORKSPACE=1 reduces workspace2 from [E,M,max(N,K)]
to [E,M,N] only for the qualified Step5 gfx938 FP8 E44/N3072/K4096 LL path.
The gate/up kernel and LightOP SiLU/clamp/FP8 quantization consume N3072;
the down result still needs K4096. The native allocator aliases workspace13
with down output, so that arena is retained rather than incorrectly claiming
it as an unused allocation. BF16 workspace2 requires only activation N1536,
so it does not erase this FP8 workspace2 saving in the shared pool.

The saving is88MiB at expert capacity1024 and352MiB at capacity4096.
The default is off. Twelve GPU tests passed: ten shape/precision guards,
plus native packed masked DeepGEMM gate/up -> LightOP activation -> down
application at capacities128/512. Changing counts under Graph replay covers
empty experts, all-empty batches, full-capacity and skewed experts. Live
output matches the original allocation exactly and output/workspace storage
canaries remain intact. This is a storage-bound optimization; successful
higher-scheduler startup and service throughput remain to be established.


### Measure all FULL Graphs during Step5 startup

The1024-token capacity v3 service started and captured successfully, but
initialized only895624--895791 effective KV tokens/DP, below1115136 required
for16 requests at65536+4096 and four draft slots. No BS64 benchmark was run.
The native two-sample FULL extrapolation reserved6.24GiB while real Graph
capture consumed3.08--3.09GiB. This is observed estimation overhead, not
proof that all of it can safely be reclaimed.

VLLM_STEP5_EXACT_GRAPH_MEMORY=1 opts Step5 MRv2 startup into capturing all
FULL descriptors in the native temporary profiling pool. Complete measured
FULL deltas replace extrapolation; PIECEWISE/MTP contributions, native
profiling cleanup and real capture/replay remain unchanged. Incomplete
samples retain the original estimate. ContextVar limits the override to the
owning startup context; normal captures and other threads keep native behavior.
The formal default remains off until actual startup/capacity evidence.
Eleven CPU policy tests passed, covering complete/partial samples, the native
pool-accounting equation, exception cleanup, idempotent installation, real
capture passthrough, model/parallel guards and thread isolation.
Evidence: bs64-capacity-20261007/exact_graph_memory_policy_tests.log.


### Reuse native SWA conversion storage from the MoE arena

The512-token/maxseq16 v4 candidate captured all five target FULL profiling
descriptors and retained the native whole-pool measurement2.75GiB. All eight
workers reported1132513 effective KV tokens/DP. It then failed during real
PIECEWISE capture while native FA allocated a192MiB temporary K tensor;
there was no benchmark. Capacity alone is not sufficient startup evidence.
Signed per-FULL deltas can be negative when eager storage is released:
the exact-mode accounting cancels their sum and retains the positive
whole-capture measurement. A regression test covers the observed signed
samples; twelve CPU Graph-memory tests now pass.

VLLM_STEP5_SWA_REUSE_WORKSPACE=1 opts Step5 BF16/HD192/NHD causal sliding
prefill and ragged multi-query batches into native pagedkv_to_contiguouskv
and native varlen_fwd, using the existing current_workspace_manager arena
for K/V and cumulative lengths. It retains the vendor262144-token buffer
floor. The builder's static tail bound replaces the device-length sum.
Uniform1--4-query paged Decode, FP8 KV, sinks, softcap, PCP/DCP and other
models retain the original path. Main and draft workspace lanes remain
independent. Formal default is off pending whole-service qualification.

Nine native GPU tests pass: one/four/sixteen requests, short ragged queries,
32-query batches and512-query prefill with an empty request; output is exact
against original FA. Graph replay changes queries, lengths and physical
pages. Two layers reuse the arena with an intervening overwrite while
preserving both outputs. Direct paged Decode routing is unchanged.

At16 requests/512 queries, eager time changed0.38026 ->0.30769ms; isolated
Graph time0.30459 ->0.30485ms is neutral. With an already allocated1.5GiB
shared arena, additional capture-reserved memory changed406847488 ->
4194304 bytes (388 ->4MiB); physical free-memory delta changed400 ->10MiB.
Current allocated-byte deltas were only1KiB on both sides and cannot measure
the private pool reservation. This is an operator/storage qualification,
not an end-to-end Decode throughput result.
Evidence: bs64-capacity-20261007/swa_workspace_gpu_tests.log and
qualify_swa_shared_arena.json.


The v5 native shared-arena path was confirmed active in worker startup.
All-FULL profiling measured2.58GiB, down from v4's2.75GiB, and effective
capacity reached1137127 tokens/DP. Real PIECEWISE capture nevertheless
failed on a352MiB DeepEP receive allocation with less than1GiB physical
free memory. This is not a service result or proof of a sufficient reserve.
The owned failed service/controller were stopped; no BS64 benchmark ran.

The exact-profile option now also accepts explicit nonnegative integer
VLLM_STEP5_GRAPH_MEMORY_MARGIN_BYTES (default0). This adds headroom to the
native whole-pool profile rather than bypassing profiling. Invalid values
fail before capture, and other contexts retain native accounting. Sixteen
CPU tests pass, including complete/partial/signed samples and margin guards.
A v6 candidate uses512MiB extra Graph headroom and memory utilization0.956;
all four actual capacities, successful real capture and native free-memory
checks must still pass. The one-percent/page/draft benchmark guard is retained.


### 2026-10-07: target Decode-only graph candidate

The launcher now accepts `STEP5_TARGET_GRAPH_MODE=FULL_DECODE_ONLY` as an
explicit opt-in. Native vLLM keeps FULL graphs for Decode and uses eager
execution for prefill/mixed batches. The default remains FULL_AND_PIECEWISE.
This does not disable the target model's complete Decode graph.

The BS64 v6 candidate passed initialized KV capacity (1,127,489 tokens/DP;
required page/draft bound 1,115,136 and 1% guard 1,125,366), but actual
PIECEWISE capture failed in native TP all_gather allocation. No benchmark
was started. Its service/controller were stopped after preserving the OOM
stack and shutdown manifest. A Decode-only candidate will test whether
removing the mixed-batch graph pool leaves sufficient real capture headroom;
it is not yet a verified BS64 throughput result.


### 2026-10-07: native GEMM/combine overlap interface audit

The installed HCU DeepEP exposes SBO combine arguments (`packed_recv_count`,
`comp_signal`, `block_m`, `threshold`, `num_sms`). Its native validation
requires a contiguous 1-D int32 signal of size
`local_experts * ceil(num_ranks * max_dispatch_tokens / 64)`, receive counts,
explicit launch settings and `return_recv_hook=True`. This alone does not
prove compatibility with a GEMM producer or its release ordering.

The exact installed Step5 FP8 down-GEMM path (`deepgemm.m_grouped_fp8_gemm_nt_masked`
-> `low_latency_grouped_gemm`, E44/M64/K1536/N4096, valid six-dimensional
packed weights, float8_e4m3fn) successfully executed with overlap disabled.
With the same valid inputs and `enable_overlap=True`, it raised
`RuntimeError: low_latency_grouped_gemm does not support overlap`. The API
parameter cannot currently enable routed-expert GEMM/combine overlap.
The installed ASM wrapper separately rejects its removed overlap mode.

Evidence: `bs64-capacity-20261007/native_deepgemm_overlap_rejection.json`.
A native producer implementation or a separately verified ready-tile
kernel is still required. Existing MK combine/shared-expert overlap remains
available; the additional shared-expert stream remains disabled after its
measured regression. This item is incomplete, not counted as adopted.

The v7 Decode-only graph candidate reduced profiled graph memory from
2.58 to 2.22 GiB and reported 1,137,011 KV tokens/DP. Actual FULL warmup
still failed on a 704 MiB native BF16 MTP DeepEP receive allocation. No
benchmark ran. v8 tests a smaller scheduler bound (384 scheduled tokens; the existing
128-token dispatch alignment still rounds 192 SP rows to 256 per rank), 1 GiB explicit graph headroom, and a native
24-token per-request long-prefill limit. The limit aims to admit and advance
all 16 prompts/DP together, preserving a common BS64 Decode measurement
window. All original request lengths, sampling and measurement gates remain
unchanged. v8 is a candidate until actual startup, capacity, Graph replay
and the single measured iteration pass.


### 2026-10-07: qualified 64-token DeepEP dispatch alignment

V8 completed all actual target/draft Graph captures and `/health` became
ready. Initialized capacity was 1,123,585 tokens/DP, below the unchanged
1% guard of 1,125,366. No BS64 benchmark was started. The idle service was
stopped before native distributed qualification.

`VLLM_STEP5_LL_DISPATCH_ALIGNMENT=64` is now an opt-in for qualified Step5
SP LL topologies on gfx938. Default alignment remains 128. Every source of
static capacity (MoE input, scheduler tokens, target Graph capture sizes,
MTP request padding) remains covered. With scheduler384/capture256/seq16,
this reduces the native per-rank bound from 256 to 192; TP2/DP4/EP8 and
all request counts/lengths are unchanged. Invalid alignments and unqualified
architectures fail before constructing a different buffer.

Native EP8 validation used 352 experts, hidden4096, TopK8, BF16 and FP8
per-token dispatch, worst-case full occupancy (1536 rows for one expert on
every destination), live routing changes and inactive rows. Both eager and
three Graph replays per quantization case matched expected outputs exactly.
The exact 192-sized RDMA pool was validated, not just a larger pool: its
size hint is 1,124,603,520 bytes vs 1,499,469,440 for256 (357.5 MiB less).
BF16/FP8 receive allocations also fall by 176/88 MiB respectively for E44.
These are allocation reductions, not a measured full-model TPS increase.

26 CPU policy tests passed, covering capacity/padding, model/parallel/SP
guards, config ownership, validation and architecture guards. Native
qualification is reproducible via
`torchrun --standalone --nproc-per-node=8 tests/kernels/qualify_step5_deepep_dispatch_alignment.py`
on idle gfx938 GPUs; CPU metadata uses Gloo, while dispatch/combine uses
the real native GPU DeepEP buffer and Graph. Results are written to
`STEP5_DEEPEP_ALIGNMENT_RESULT` (default `/tmp/step5-deepep-dispatch-alignment.json`).


### 2026-10-07: BS64 target Graph and capacity gate passed

The v9 candidate (source905666e, scheduler384, per-request prefill24,
64-token LL alignment, target FULL_DECODE_ONLY, calibrated QKV FP8,
1 GiB graph margin, memory utilization0.956) completed real target/draft
Graph capture and became healthy. All four initialized capacities passed
the unchanged page/draft and1% gates: DP0/1=1,168,645, DP2=1,168,586,
DP3=1,168,349 tokens; the1% requirement is1,125,366 per DP.

The single BS64/input65536/output4096 benchmark is running with all16
requests/DP admitted and no waiting requests or preemptions in the live
snapshot. No completed TPS result is claimed yet. Original prompts,
sampling, timestamp window formula and completion checks are retained.

An isolated small-M QKV candidate now fuses FP32 row scale reduction,
E4M3FN activation conversion and per-channel scaled GEMM in one kernel.
CPU-only gfx938 compilation found zero-scratch configurations for M4/64
after reducing K tile and stages. It remains outside production pending
native quantization/Graph correctness and timing against the actual
LightOp QKV method. No GPU microtest was run during the long benchmark.


### Small-M QKV offline qualification follow-up

The isolated quantization/GEMM candidate now compiles both debug and
production specializations for M1/4/16/32/64 and two tiles per shape.
All ten production gfx938 variants have zero scratch spill. Explicit
HIP/gfx938 compilation runs with both CUDA_VISIBLE_DEVICES and
HIP_VISIBLE_DEVICES empty; compilation is not a GPU timing result.

The pending GPU qualifier uses the actual layer0 TP2 rank0 BF16 Q/K/V
checkpoint shards, then the existing output-MSE weight calibration and
LightOp baseline method. Quantization scales use a relative-only check,
so a large absolute tolerance cannot hide an incorrect tiny-input scale.
It checks production/debug equivalence in eager and six input-changing
Graph replays (zero, signed constant, signed tiny, random), adjacent FP8
midpoint differences, FP32 scaled-GEMM outputs and bias. It refuses to run
alongside an active vLLM service/worker. No GPU results or adoption yet.
Evidence: bs64-capacity-20261007/qkv_smallm_fused_all_shapes_compile.json.

A fresh AST inventory includes methods as well as top-level functions
from installed AITER, LightOp, DeepGEMM and DeepEP. Native AITER sources
confirm HD128-only and BF16-weight constraints in fused_qk_rmsnorm_rope;
cache-quant variants dispatch 64/128/256 or require weights in the input
dtype. The more general pts variant also reads scale.item<float>() on
the host and takes weights via data_ptr<scalar_t>(). These APIs do not
directly preserve Step5 HD192/FP32 Gemma norm weights and full-Graph
semantics. AITER/LightOp exposed Python producer APIs did not contain
signal/overlap arguments; DeepGEMM's exposed arguments still lead to
the previously verified native overlap rejection. This is interface
evidence, not proof that every conceivable vendor implementation is
unavailable.
Evidence: library_fusion_inventory.json and library_native_overlap_inventory.json.


### BS64 long Decode result and rejected small-M fused GEMM

The single v9 BS64/input65536/output4096 iteration completed64/64 requests,
262144 output tokens and4194304 input tokens with zero native preemptions.
The independently checked common window lasts173.26135s, contains256636
tokens and covers98.0979% of decode candidates. Decode throughput is
1481.2074 token/s; client chunk mean/P50/P90/P99 ITL is
121.5150/121.4509/123.0527/125.0942ms. Target dispatcher logs explicitly
record steady64-token FULL replay on all four DP engines. Ragged drain
batches can still use NONE/eager. This proves uniform target Decode Graph
execution, not every ragged Decode shape.

Balanced24-token per-request prefill completed with maximum TTFT1661.1613s
and test-internal Prefill throughput2524.9227 token/s. The capacity-safe
profile has a material Prefill cost and is not promoted as the best combined
Prefill/Decode profile. This remains synthetic-MTP performance evidence,
not a real acceptance-rate or accuracy result and not a controlled BS32 A/B.
The service was stopped only after independent result validation and all
eight running/waiting request counters were zero.

Evidence: bs64-capacity-20261007/fp8-crop-reserve-v9_verified_result.json,
fp8-crop-reserve-v9_runtime_graph_log.txt and fp8-crop-reserve-v9_shutdown.json.

The isolated small-M fused QKV candidate completed native qualification
against actual layer0 TP2 rank0 weights and output-MSE-calibrated LightOp.
The first strict zero-input check discovered that LightOp uses amax/448
without the exploratory1e-12 floor, including scale0 for an all-zero row.
The candidate was corrected, retaining the strict oracle and prior failure.
All18 shape/tile variants then passed FP8 midpoint/scales, FP32 GEMM, bias
and six changing Graph replay checks. No model-accuracy claim is made.

Every candidate was slower than the current native method. M1/4/16/32/64
native versus best fused times (microseconds, captured Graph) were
33.59/55.57,33.62/55.83,33.23/56.49,35.49/87.11,35.11/106.93.
All current fused-GEMM tiles are rejected; the production path is unchanged.
Evidence: qkv_native_zero_tiny_scale.json, qkv_smallm_fused_qualification.json
and qkv_smallm_fused_decision.json in bs64-capacity-20261007.

LightOp also exposes gemma_rms_norm_per_token_fp8_quant(input,weight,
epsilon,fp8type,residual,update_input). A next candidate must preserve
Step5's FP32 Gemma weight, FP32 residual and explicit BF16 narrowing before
activation FP8 conversion, while retaining the BF16 hidden tensor used by
the indexer/gate. The exposed API alone does not prove these semantics.


Native LightOp Gemma-FP8 probing with the actual FP32 layer0 norm weight
executed on FP32 residual input, but differed from Step5's BF16-narrowed
reference in403/16384 FP8 values (reconstruction RMS difference0.01077).
BF16 input with the required FP32 weight was rejected for dtype mismatch.
No norm-weight cast or rounding-stage removal is adopted. A custom fused
norm/quant candidate must explicitly retain the BF16 stage and be checked
against the compiled production norm before any model-level adoption.
Evidence: bs64-capacity-20261007/lightop_gemma_fp8_semantics_probe.json.


### Explicit-layout Gemma norm and FP8 activation quantization (8a51dd7)

A new isolated source operator fuses FP32 Gemma normalization, the required
BF16 narrowing, token amax/scale and E4M3FN conversion. It returns both BF16
hidden and FP8 hidden, retains FP32 norm weights/residuals, and reuses the
existing LightOp scaled GEMM. At measured revision 8a51dd7 it was not wired
into the model or enabled by the formal launcher. The later opt-in integration
is described below; the timings here remain historical operator evidence.

The initial ordinary Triton variants failed exact BF16 checks because
their reduction layouts differed from the selected Inductor native norm.
Changing warp count alone did not fix all batches. Explicit Gluon layouts
matched the native profiles at all nine tested shapes, with five random
norm trials per shape. A single-row/four-warp variant retains the per-row
reduction order of the native two-row-warp profiles while increasing the
small-batch grid, removing the first candidate's M4/M32 regressions.

Actual layer0 TP2 rank0 QKV weights and output-MSE calibration then passed
strict BF16 norm/scales, FP32 GEMM and six changing-input Graph replay
checks at M1/4/16/32/64/128/256/512/6144. FP8 values are not universally
byte-identical: differing values passed the independent FP64 oracle for
adjacent E4M3FN midpoint rounding. No model-accuracy equivalence is claimed.
The source custom-op tests passed11 cases, including input preservation,
changing weights, partial row CTAs, dynamic fullgraph compilation, empty
batches and dtype/configuration rejection. The operator is gfx938-only.

The final operator pipeline screen uses identical input for both sides and
includes norm + activation quantization + actual LightOp QKV GEMM. Each
entry is one captured-Graph screen, not a model throughput measurement:

| M | Native pipeline (us) | Fused pipeline (us) | Time reduction |
| --- | --- | --- | --- |
| 1 | 36.98 | 33.95 | 8.18% |
| 4 | 37.76 | 33.77 | 10.56% |
| 16 | 36.60 | 33.47 | 8.54% |
| 32 | 39.77 | 36.22 | 8.92% |
| 64 | 38.38 | 35.47 | 7.57% |
| 128 | 48.67 | 46.78 | 3.89% |
| 256 | 68.04 | 66.73 | 1.93% |
| 512 | 86.92 | 87.83 | -1.06% |
| 6144 | 861.04 | 845.98 | 1.75% |
M512 is slightly slower, so global adoption is not justified. An earlier
M4 baseline timing outlier is preserved but excluded from the final table.
The qualifier recorded the measured-source SHA at revision 8a51dd7. Later
BF16-input and automatic-layout changes do not inherit these timings as a
new same-source measurement.

Reproduce on idle gfx938 GPUs:
```bash
python3 tests/kernels/qualify_step5_norm_quant.py \
  --bf16-model /data/model/step-5-preview-0913-bf16 \
  --runtime-model /data/model/step-5-preview-0913-fp8-per-channel \
  --calibration /workspace/step5-attention-optimization/qkv-fp8-output-calibration-inputs-v1.pt \
  --output /tmp/step5-norm-quant-qualification.json
```

The qualifier selects the actual compiled native norm profile and refuses
unsupported layouts or active vLLM services. It does not infer model routing.

Evidence: bs64-capacity-20261007/qkv_norm_gluon_native_configs_probe.json,
qkv_norm_quant_source_rowwise_qualification.json and
qkv_norm_quant_custom_op_final_tests.log. All six original optimization
requirements remain in scope; this is progress on the small-M QKV item.


### 2026-10-07: opt-in norm/quant integration and DCP2 CPU reference

Local commit e9f9098 connects the norm/FP8 custom op to Step5 decoder input
normalization and passes prequantized activations/scales to the existing
LightOp QKV GEMM without a second quantization. BF16 hidden remains available
to the indexer and gate; norm weights remain FP32. FP32 and BF16 inputs are
supported because the first MTP decoder receives BF16 eh_proj output.
The formal launcher leaves VLLM_STEP5_QKV_NORM_FP8_FUSION=0 by default.
Its opt-in guards require FP8 QKV, custom ops and the native custom GEMM.

34 kernel/interface checks passed, including independent FP64 rounding
oracles, input preservation, changing Graph inputs/weights, fullgraph
compilation and the no-requantization contract. Six CPU launcher checks
passed. These are not model-accuracy or throughput results.

The ordinary-MTP3 candidate at norm-quant-model-20261007-v2 loaded all eight
workers, completed actual Graph captures and passed health. A fixed100
GSM8K evaluation uses the prior frozen inputs, prompts, sampling and native
EvalScope scoring. The candidate completed97/100, zero request errors and
zero truncations, equal to the previous FP8 score. Independent per-item
audit found no changed correctness labels and three changed extracted
answers (all already incorrect). Same input bytes, prompts, targets and
generation config were checked. This is fixed100 accuracy evidence, not
full-dataset equivalence, a single-variable source A/B or throughput. The first controller audit failed because worker
/proc environ was empty (all flags unavailable). Its failure was preserved.
The resumed controller records that limitation instead of treating missing
/proc data as proof the feature is disabled. Root launch environment, rank0
feature log and DP0/TP2 model inspection provide narrower activation evidence;
this is not an all-eight runtime environment audit.

An isolated DCP2 CPU FP64 reference completed query lengths1-4, causal/SWA128,
dense and sparse regions, striped cache slots, global replicated compressed
summary slots, empty local keys and output/LSE combination. Maximum absolute
error was 6.661338147750939e-16. This is CPU mapping proof only, with no
production GPU DCP2 implementation or benchmark.

TP2/DCP2 needs replicated full Q/K/V heads with striped contexts. Physical
page64 corresponds to global logical page128; compression8 needs16 globally
replicated summary slots per logical page. Preserve FP32 partial output/LSE
until distributed combination, then narrow to BF16. Prefill parity, final
KV clipping, SWA/causal offsets and output head sharding still need GPU
implementation and Graph qualification. With doubled heads and halved
contexts, per-GPU KV storage/attention work is not automatically halved;
do not claim a BS64 capacity improvement from DCP2 before measurement.

Evidence: bs64-capacity-20261007/qkv_norm_quant_integration_tests.log,
qkv_norm_quant_launcher_checks.json; norm-quant-model-20261007-v2 runtime
manifest/logs; dcp2-prototype-20261007/cpu_qualification.json.
All six original optimization requirements remain active.


### Native DCP2 partial Attention qualification

The isolated prototype is now source-owned in vllm_hcu/ops/step5_dcp2.py.
It is not enabled in the model and does not remove the launcher DCP1 guard.
At b97dc34 the source file was byte-identical to the native GPU-qualified
prototype. The later fine-region geometry extension is pending new GPU
qualification and does not inherit that exact source claim.

Two gfx938 GPUs completed62 cases per rank against independent CPU FP64
references, with64 Q heads/4 KV heads, HD192, query lengths1-4, lengths1,
2,127,128,129,257 and65538, sparse/dense selection, SWA128, empty/padded
queries, zero/tiny inputs and shuffled physical pages. Device packing was
exact. Partial output and LSE remained FP32 through a real RCCL all-gather
and global LSE merge; BF16 narrowing occurred afterward. Q and probability
arithmetic retains the existing BF16 path. This is bounded-error proof,
not byte equivalence to unstriped Attention.

Actual communication Graphs at257 and65538 keys passed three changing-Q
replays each. Retaining graphs caused native process-group destruction to
remain pending after all checks; resetting and releasing the communication
graphs before destruction fixed it. Both Graph cases passed again and
both process groups destroyed cleanly. Initial evidence and exact owned
test-tree cleanup were retained.

The portable test is tests/kernels/qualify_step5_dcp2_partial.py, launched
with torchrun --standalone --nproc-per-node=2 on idle GPUs. Result directory
uses STEP5_DCP2_RESULT_DIR (default /tmp/step5-dcp2-qualification).
FP8-cache variants have offline compilation only; their GPU qualification,
full Step5 DCP2 integration, Prefill parity/cache write adaptation and
model performance remain pending. All six optimization requirements remain
active.

Evidence: dcp2-prototype-20261007/gpu_qualification_summary.json,
gpu_qualification-rank0/1.json.before-graph-reset, gpu_graph_lifecycle.log.


### Native E44 LL mode/CU screen

A single native operator screen completed240 cases: gate/up E44/N3072/K4096
and down E44/N4096/K1536, the actual packed GEMM dimensions for intermediate
width1536; global token buckets32/64/128/256/512, uniform/skewed routing,
native modes1100/1101/1102 and launchCU32/64/128/256. Physical deviceCU=64,
native default launchCU=256; static per-expert capacity1536 matches the
selected BS64 profile. All variants passed active-output checks and Graph
replay after device routing counts changed, against mode1101/defaultCU.
Selected expert rows also passed an independent unpacked FP32 GEMM oracle.

Mode1101 remains the best family. CU128 showed small changes, generally
0-3%, so no new CU override is adopted from this single ordered screen.
The native low_latency_grouped_gemm wrapper exposes MODE and cu; arbitrary
tile/warps/stages JSON entries are ignored on that path. Dedicated native
tile-family work remains pending. This is native operator evidence, not a
model TPS gain, and does not turn the legacy E44/N1536 MoE helper config into
native LL tile tuning.

Evidence: bs64-capacity-20261007/native_step5_ll_modes_cu_screen.json,
native_step5_ll_modes_cu_decision.json and screen_native_step5_ll_modes_cu.py.


### Fine sparse regions and isolated fusion performance run

The DCP2 operator now distinguishes global sparse regions8 from logical
cache pages128. A physical page64 contains16 local regions4. The packer
maps region -> logical page -> physical page, preserves the in-page region
offset and rank parity, and clips every query's visible local prefix.
This geometry is required for Step5's region8 selector; the earlier native
GPU prototype used region128 and is not direct proof of the fine variant.

40 tests executed the actual production metadata kernel through Triton's
CPU interpreter, covering both granularities, both ranks, lengths1/2/7/8/9/
127/128/129/257/65538, shuffled two-request tables, invalid page/request/
region IDs, padded queries and zero counts. Packing matched the independent
integer oracle exactly.22 fine-region gfx938 variants compiled offline
with no visible GPUs. BF16 attention variants had zero scratch; two FP8
cache/BR8 variants had68 bytes of scratch per thread. Fine-region native
GPU accuracy/Graph and FP8-cache qualification remain pending.

The opt-in norm/FP8 fusion passed the fixed10097/100 check with unchanged
per-item correctness labels. A single BS64/input65536/output4096 performance
iteration now runs from frozen worktree /workspace/step5-norm-quant-perf-repo
at0190ec8, retaining the v9 prompts, sampling, graph, capacity guards and
measurement formula. The initial v10 startup omitted the ignored native
hcu_ops extension and terminated before any benchmark. Its exact existing
binary was copied and hash-checked into the worktree, then CPU-imported
after torch. V11 loaded, captured actual Graphs, passed the unchanged
capacity gate and began the one measured iteration. No TPS result yet.

Evidence: dcp2-prototype-20261007/fine_regions_cpu_interpreter.log,
fine_regions_gpu_compile.json; bs64-capacity-20261007/
fp8-norm-fusion-v11_state.json and service/controller logs.

### TP2/DCP2 QKV/cache and selection bridge

Source-owned ops/step5_dcp2_exchange.py adds a local normalized/rotated
Q/K/V pack, rank-ordered head unpack plus owned BF16 KV scatter in one
kernel, and conversion of existing int64 Decode words to region8 logical
IDs. QKV projection weights remain TP2-sharded: each rank contributes its
local Q heads and two KV heads, while the context-striped cache carries
all four KV heads. The final globally merged Attention output must return
only that TP rank's Q heads before the existing per-head gate/O projection.

The existing indexer Q/W weights are provider-group sharded, not replicated:
each TP2 rank scores two distinct groups. Its K/Z providers are replicated,
so each rank can retain the exact eight-token compression arithmetic over
the complete token stream, rather than compressing alternate tokens and
changing the model. Exchange group-major selected logical region IDs and
counts in TP rank order; do not rescale indexer weights or choose TopK over
only the rank-local half context.

vLLM V2's existing CP_SIZE2/CP_INTERLEAVE1 slot mapping already masks remote
tokens with -1 and maps global page128 into local page64. The new cache
writer consumes that mapping unchanged, preserving padding. It accepts
only BF16 cache and HD192. Summary pools and schedules must use logical
page128 with sixteen region8 summaries per page, independently of physical
cache page64. Step5 now declares the context size to Step4DSAState; its
summary allocation and metadata builder use this logical page size, with
an explicit local64 guard for DCP2. Ten CPU checks exercised the actual
state allocator/builder (DCP1/default and DCP2, reuse/replacement, queries
1/4/24, unchanged global lengths/positions and local slot mapping). This
is source integration of metadata geometry, not whole-model DCP2 proof.

Decode's source pack stores logical token offsets in low32 and physical
region IDs in high32. The new conversion masks the tail using counts,
including the legitimate zero word for region0, retains the forced current
region/order, and avoids attempting to invert shuffled physical block IDs.
Three CPU-interpreter Decode-word cases passed (TopK slots1/2/513).
Seven current gfx938 offline variants compiled with zero scratch/shared
memory. The initial contiguous-cache variant passed32 CPU-interpreter cases
(tokens1/2/3/4/16/64/128/384, local Q heads4/32, int32/int64 slot mappings).
The actual V2 cache layouts then required explicit head/token/page strides:
sparse K/V are interleaved per head; SWA K/V are separate within each page,
with non-affine flattened token addresses across pages. The current writer
supports both 3D affine and 4D paged views without materializing cache copies.
Nine current-source CPU cases passed, including both Q geometries, sparse
and SWA layouts, the page63/64 boundary, -1 and out-of-range int64 slots,
and Decode-word conversion. The32 earlier cases establish the contiguous
variant's head-order behavior, not complete current-source GPU proof. Native exchange and changing collective-Graph qualification is
prepared in tests/kernels/qualify_step5_dcp2_exchange.py and refuses live
vLLM processes; it has not run while the isolated BS64 fusion measurement
uses the GPUs. Native whole-model qualification remains pending.

No model activation or DCP launcher guard has changed. Still required:
native exchange proof; GPU qualification of logical summary/builder
integration; gathered selection plans; full-query/local-cache Prefill and SWA paths; FP32 partial
state exchange/global merge and TP-head slicing; stock DCP validation
adaptation; whole-model correctness and one performance comparison.
Doubling cached KV heads while halving context does not by itself reduce
per-GPU KV storage or Attention FLOPs relative to TP2/DCP1; communication
overhead and the measured end-to-end result must decide adoption.

Evidence: dcp2-prototype-20261007/exchange_offline_compile.json,
decode_word_cpu_interpreter.log, exchange_cpu_interpreter.log,
exchange_strided_cpu.log, context_geometry_cpu.log. The initial
CPU interpreter used255 threads; its exact owned process was stopped and
the same matrix relaunched with one thread, retaining the partial log and
exchange_cpu_thread_correction.json. GPU kernels were not run.

### Completed BS64 norm/FP8 fusion measurement

The one V11 measured iteration completed64/64 requests, input4194304 and
output262144 tokens, with zero preemptions and a valid common Decode window:
254635 tokens /174.8611541800s =1456.212508685 tok/s,97.2910% coverage.
Mean client-chunk ITL121.86794ms; observed Prefill2522.4329tok/s and
latency1662.8010s. MTP3 acceptance was synthetic, as in V9.

The original controller rejected prompt_sha256 while comparing ALL
iterations. Independent audit found that all64 measured prompts match V9
exactly, but all16 random warmup prompts differ. Adapter hash and measured
shape/protocol match. Preserve that rejection: this is a measured-only
paired comparison, not the originally required all-prompt comparison.
CPU-only validation also ran during Prefill and is logged separately.
The observation is1.6875% below V9's1481.2074tok/s. There is no measured
throughput gain to justify default adoption, and one iteration with these
qualifications cannot establish a statistically significant regression.
Keep fusion opt-in/default off; do not repeat this expensive case.

The supplied benchmark generated warmup random tokens before applying
per-request seeds in its measured generator. Future HCU adapter calls now
require explicit --seed and seed the global RNG before the original main.
The previous exact adapter is retained as
benchmark_hcu_metadata.before-warmup-seed-v1.py; completed report hashes
have not been changed. No new model measurement has been started.

After two eight-counter idle snapshots, the exact owned server tree was
stopped; benchmark/controller handles are terminal. Current DCP2 fine8
and QKV/cache exchange native qualifications have now started sequentially
from a four-file immutable snapshot of88800bd, on idle GPUs. This is not
model DCP2 activation. Results remain pending.

Evidence: bs64-capacity-20261007/
fp8-norm-fusion-v11_measured_only_audit.json, result.iterations.json,
manifest.json, runtime_graph_log.txt, shutdown.json and
cpu_validation_activity_v11.json; dcp2-prototype-20261007/
current_dcp2_native_qualification_handle.json.

### Native fine8 and real cache-layout exchange completed

The current fine8/HD192 operator passed62 cases per rank against the FP64
oracle, including real RCCL FP32 state exchange/global LSE merge, shuffled
pages, q1-4, long65538 contexts, SWA128, empty/tiny cases and changing-Q
communication Graphs. Packed metadata was exact. Source SHA:
f1c4f2e72e89e88c8010b3327f65580adc3310b9aadbb55a01614caf936b3e33.

Current QKV/cache exchange passed32 cases per rank (local Q heads4/32,
tokens1/2/3/4/16/64/128/384, actual sparse/SWA V2 cache strides).
All eager QKV/cache outputs matched exactly. Twelve real collective Graph
cases per rank passed36 changing-input/slot-mapping replays; Decode-word
conversion also passed changing-count Graphs. Source SHA:
a935d869211aef72f040091f89b8a86a421732bad78be0db7bf71e37b3d69a30.

The first thin snapshot loaded the installed HCU package and failed before
numerical cases because that package lacked the snapshot's DCP2 module.
The portable qualifier now imports the explicit adjacent source file.
That attempt and its terminal handle are retained. The corrected sequential
run completed from immutable8a63073 source, and all four native process
groups destroyed cleanly; jobs are terminal. These are operator proofs,
not model DCP2 support or performance. Model activation remains disabled
pending actual Attention/cache/head and gathered-plan/state integration.

Evidence: dcp2-prototype-20261007/current_dcp2_native_qualification_summary.json,
fine-native-v3/gpu_qualification-rank0/1.json,
exchange-native-v2/exchange_gpu-rank0/1.json and
current_dcp2_native_qualifications-v2.log.

### DCP2 sliding-window allocation qualification

The new experimental Step5DCP2SlidingWindowSpec preserves the installed
SlidingWindowManager recycling behavior while aligning startup memory and
admission sizing to global page128 / physical page64. It retains all four
BF16 KV heads, HD192, window512, and MTP extra trailing tokens. Prefix
caching is disabled, and unsupported storage/config geometries fail.

Thirty CPU tests passed against the installed vLLM allocator in5.67s.
They include chunked prefill through65536 tokens at BS32/64 with no spare
pool capacity beyond the admission bound, logical page127/128/129 edges,
uniform group sizing/table width, preempted-request suffix restoration,
and MTP3 accept/reject schedules without double allocation or premature
window release. GPU/whole-model behavior is not established by these tests.

This spec is not yet returned by model Attention and does not enable DCP2.
Integration must convert every SWA layer, including MTP, before grouping;
stock-first mixed grouping would otherwise retain stock DCP1-only sizing.
The upstream hybrid-cache coordinator still rejects DCP + sliding-window
groups, and Step5 TP2/GQA validation, FP32 LSE/output communication, and the
full Attention call chain still need scoped integration and qualification.
No launch guard, production cache behavior or performance result changed.

Evidence: tests/v1/test_step5_dcp2_cache.py;
dcp2-prototype-20261007/window-cache/allocator-v1.log.

### DCP2 Attention bridge and gated model adapter

The bridge exchanges TP-sharded, normalized/rotated QKV and two local
provider-group selection plans, evaluates fine8 sparse or window512
Attention over striped BF16 cache, merges local splits in FP32, gathers
LSE, then ReduceScatters FP32 weighted output to the original32 TP heads.
The head-major reduction layout avoids the generic dim1 transpose/copy;
BF16 conversion occurs only after inter-rank reduction. HD192 uses a
masked256-wide kernel because the stock DCP correction kernel passes192
directly to tl.arange. Empty shards produce zero output and -inf LSE.

Actual vLLM TP2/DCP2 GroupCoordinator/CudaCommunicator qualification passed
per rank: four standalone combine shapes and nine complete Attention cases.
Each rank passed30 changing-input Graph replays. Cases cover q1/q4,
ragged mixed q1/q4,64-query Prefill, padded/inactive requests, shuffled and
changed physical tables, contexts129/1025/65538, SWA512, and real513-slot
TopK512 plus current-region selections. Cache writes were exact. Maximum
Attention relativeL2 against the FP64 oracle was0.00227701; maximum LSE
absolute error1.08212e-6. Native FP32-to-BF16 conversion was exact. These
are numerical/operator gates, not model accuracy or throughput results.

Current runtime SHA:
e3aa8e3b98bb2a7cc9e14b2450401fc3fa899a99203c2966b99f607e44fa38db.
Seven CPU-interpreter layout checks and five actual build_plan selection
contract checks passed. The CPU interpreter truncates BF16 casts, so its
layout gate uses FP32; the native GPU gate tests exact BF16 conversion.
All qualification handles/processes are terminal, with clean group teardown.

Step5 now selects the new Attention/cache/metadata adapter only for opt-in
VLLM_STEP5_DCP2_RUNTIME=1 with supported TP2/DCP2 head geometry. It preserves
sharded QKV weights, declares four cached KV heads, uses affine sparse-style
SWA cache views with the DCP2 recycling spec, and carries the indexer's
dependency token into the Attention op. DCP2 retains logical selections
even when short-query DCP1 suppresses them. Early config registration and
a constructor guard keep the new op outside piecewise graph segments.
Seven adapter declaration/alias/fake-op/split tests and the existing DCP1
Flash cache declaration test passed. The complete model adapter has not
yet been exercised with model weights.

The first model-free communication setup exposed Hy4 validation reading a
None ModelConfig. The scoped guard now returns outside Hy4's scope for
model-free configs; all35 Hy4 config tests passed. No Hy4 model checks
were relaxed. Qualification explicitly records model_config_validated=false.

Whole-model DCP2 still requires scoped TP2/GQA validation, hybrid SWA
coordinator integration, workspace/memory/full-Graph policy adaptation,
model correctness and one end-to-end measurement. The production launcher
still rejects DCP2; no official result/default has changed.

Evidence: dcp2-prototype-20261007/runtime-bridge/
current_qualification_summary.json, native-v2/runtime-rank0/1.json,
native-topk513-v1/runtime-rank0/1.json and respective logs.
The native-v2 qualifier source is retained next to that run's reports.
Initial setup and CPU test failures remain in their original logs.
