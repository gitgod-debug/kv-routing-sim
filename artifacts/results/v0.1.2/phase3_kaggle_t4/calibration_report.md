# Phase 3 — Real-Engine Calibration Report

**Engine:** vLLM 0.30.0 · `Qwen/Qwen3-4B` · fp16 · Tesla T4 · block size 16 · `--num-gpu-blocks-override 512` · max_num_seqs 256 · async scheduling: true
**Pre-registration:** trace `54472296d0f8549e…`, predictions `055f57c9051ad096…` — committed before the run.

## 1 · Sequential workloads — exact predictions

| workload | model | exact runs | hit-rate error, pts (measured − predicted) | computed-token error | requests matching |
|---|---|---|---|---|---|
| W1_long | vllm | 3/3 | +0.00, +0.00, +0.00 | +0, +0, +0 | 120/120, 120/120, 120/120 |
| W1_long | lru | 0/3 | -11.90, -11.90, -11.90 | +6,896, +6,896, +6,896 | 92/120, 92/120, 92/120 |
| W1_long | vllm, +1 token ¹ | 3/3 | +0.00, +0.00, +0.00 | +0, +0, +0 | 120/120, 120/120, 120/120 |
| W1_long | *all runs identical* | True | | | |
| W2_short | vllm | 3/3 | +0.00, +0.00, +0.00 | +0, +0, +0 | 120/120, 120/120, 120/120 |
| W2_short | lru | 0/3 | +0.17, +0.17, +0.17 | -96, -96, -96 | 118/120, 118/120, 118/120 |
| W2_short | vllm, +1 token ¹ | 0/3 | +0.63, +0.63, +0.63 | -368, -368, -368 | 117/120, 117/120, 117/120 |
| W2_short | *all runs identical* | True | | | |

¹ The pre-registered alternative: the engine commits one token more per request than its source says. On W1 it predicts exactly what the vllm model does (no request ends on a block boundary), so only W2 can tell them apart.

## 2 · Timing model — fitted, Tesla T4 figures

- decode step = **52.91 ms + 1.232 ms × batch**  (R² 0.577, 6 batch sizes)
- prefill = **0.00 ms + 0.7014 ms × uncached token**  (R² 0.932, 720 requests)
- ⚠️ fitted prefill_base_s was -20.82 ms (< 0, not physical); clamped to 0 — the decode fit absorbs the whole fixed cost

## 3 · W3_concurrent — predicted with the fitted profile, never tuned on it

| metric | engine mean [min, max] | vllm model (vs engine) | lru model (vs engine) |
|---|---|---|---|
| hit_rate | 0.5076 [0.5068, 0.5079] | 0.509 (+0.3%) | 0.6073 (+19.7%) |
| local_compute | 28,541 [28,520, 28,584] | 28,456 (-0.3%) | 22,760 (-20.3%) |
| preemptions | 27 [27, 27] | 24 (-11.1%) | 0 (-100.0%) |
| p50_ms | 15,416 [15,298, 15,574] | 14,837 (-3.8%) | 8,464 (-45.1%) |
| p95_ms | 41,017 [40,674, 41,241] | 38,255 (-6.7%) | 11,248 (-72.6%) |
| p99_ms | 41,492 [41,149, 41,716] | 38,730 (-6.7%) | 11,568 (-72.1%) |

✓ = inside the engine's run-to-run range.

## 4 · H4 — eviction histograms (engine vs vllm model)

| run | family | engine count | model count | max CDF distance |
|---|---|---|---|---|
| W2_short_r1 | lifetime | 963 | 963 | 0.000 |
| W2_short_r1 | idle | 963 | 963 | 0.000 |
| W2_short_r1 | reuse_gap | 99 | 99 | 0.000 |
| W2_short_r2 | lifetime | 963 | 963 | 0.000 |
| W2_short_r2 | idle | 963 | 963 | 0.000 |
| W2_short_r2 | reuse_gap | 99 | 99 | 0.000 |
| W2_short_r3 | lifetime | 963 | 963 | 0.000 |
| W2_short_r3 | idle | 963 | 963 | 0.000 |
| W2_short_r3 | reuse_gap | 99 | 99 | 0.000 |
| W1_long_r1 | lifetime | 1,880 | 1,880 | 0.042 |
| W1_long_r1 | idle | 1,880 | 1,880 | 0.042 |
| W1_long_r1 | reuse_gap | 3 | 3 | 0.000 |
| W1_long_r2 | lifetime | 1,880 | 1,880 | 0.042 |
| W1_long_r2 | idle | 1,880 | 1,880 | 0.042 |
| W1_long_r2 | reuse_gap | 3 | 3 | 0.000 |
| W1_long_r3 | lifetime | 1,880 | 1,880 | 0.042 |
| W1_long_r3 | idle | 1,880 | 1,880 | 0.042 |
| W1_long_r3 | reuse_gap | 3 | 3 | 0.000 |
