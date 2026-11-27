# Phase 4 — the first calibrated sweep

Profile `t4-qwen3-4b-vllm0.30.0` (1,537 blocks, fitted T4 timings) · cost `g4dn.xlarge` at $0.526/h · SLO p95 TTFT ≤ 2,000 ms · 5 seeds · generated `2026-11-27T18:26:43+00:00` at `482c722`.

⚠️ **T4 figures.** Milliseconds and rupees are measured-profile numbers for one Tesla T4 per replica; whether the *ranking* transfers is H6's question.

## Replicas each policy needs to hold the SLO

### low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 3 | 4 | 5 | 7 | 8 |
| least-loaded | 1 | 3 | 4 | 5 | 7 | 8 |
| power-of-two | 1 | 3 | 4 | 5 | 7 | 8 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | 2 | 3 | 3 | 4 | 5 |
| session-sticky | 1 | 2 | 3 | 3 | 4 | 5 |

— = not met at 8 replicas.

### high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 3 | 4 | 5 | 7 | — |
| least-loaded | 1 | 3 | 4 | 5 | 7 | 8 |
| power-of-two | 1 | 3 | 4 | 5 | 7 | 8 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | — | — | — | — | — |
| session-sticky | 1 | 2 | 3 | 3 | 4 | 5 |

— = not met at 8 replicas.

## Why — how evenly each policy spreads load, and how often it preempts

*Descriptive, not a registered test.* Each cell: load imbalance (max ÷ mean requests per replica) · preemptions per run, at the replica count round-robin needs at that load (8 where it never meets the SLO).

### low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.01 · 0 | 1.00 · 14 |
| least-loaded | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.03 · 0 | 1.03 · 3 |
| power-of-two | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.05 · 0 | 1.04 · 3 |
| consistent-hash-prefix | 1.00 · 0 | 3.00 · 206 | 4.00 · 182 | 5.00 · 110 | 7.00 · 126 | 8.00 · 121 |
| radix-affinity | 1.00 · 0 | 3.00 · 206 | 4.00 · 182 | 5.00 · 110 | 7.00 · 126 | 8.00 · 121 |
| radix-affinity-balanced | 1.00 · 0 | 1.05 · 0 | 1.05 · 0 | 1.11 · 0 | 1.16 · 0 | 1.12 · 0 |
| session-sticky | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.05 · 0 | 1.06 · 0 |

### high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.00 · 1 | 1.01 · 0 | 1.00 · 19 |
| least-loaded | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.02 · 0 | 1.02 · 0 | 1.03 · 5 |
| power-of-two | 1.00 · 0 | 1.01 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.04 · 7 |
| consistent-hash-prefix | 1.00 · 0 | 3.00 · 212 | 4.00 · 139 | 5.00 · 120 | 7.00 · 125 | 8.00 · 123 |
| radix-affinity | 1.00 · 0 | 3.00 · 212 | 4.00 · 139 | 5.00 · 120 | 7.00 · 125 | 8.00 · 123 |
| radix-affinity-balanced | 1.00 · 0 | 1.73 · 0 | 1.65 · 75 | 1.60 · 122 | 1.85 · 79 | 1.82 · 78 |
| session-sticky | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.04 · 0 | 1.10 · 0 |

## Recommendation at each load (cheapest configuration meeting the SLO)

| trace @ load | recommended | ₹ / 1k requests | vs round-robin |
|---|---|---|---|
| low_sharing@1 | round-robin × 1 | 4.01 | +0.0% · +0 replicas |
| low_sharing@2 | radix-affinity-balanced × 2 | 4.40 | -33.2% · -1 replicas |
| low_sharing@3 | radix-affinity-balanced × 3 | 4.78 | -25.2% · -1 replicas |
| low_sharing@4 | session-sticky × 3 | 3.91 | -39.7% · -2 replicas |
| low_sharing@6 | session-sticky × 4 | 4.00 | -42.4% · -3 replicas |
| low_sharing@8 | session-sticky × 5 | 4.23 | -41.6% · -3 replicas |
| high_sharing@1 | round-robin × 1 | 4.01 | +0.0% · +0 replicas |
| high_sharing@2 | session-sticky × 2 | 4.40 | -33.2% · -1 replicas |
| high_sharing@3 | session-sticky × 3 | 4.79 | -25.1% · -1 replicas |
| high_sharing@4 | session-sticky × 3 | 3.97 | -38.8% · -2 replicas |
| high_sharing@6 | session-sticky × 4 | 4.07 | -41.4% · -3 replicas |
| high_sharing@8 | session-sticky × 5 | 4.26 | -42.5% · -3 replicas (round-robin misses the SLO even at 8 — vs its best attempt) |

## H1 — prefix-affinity vs least-loaded on hit rate, high sharing

**radix-affinity: FALSIFIED**

| load | 2 rep | 3 rep | 4 rep | 5 rep | 6 rep | 7 rep | 8 rep | strictly higher | widens |
|---|---|---|---|---|---|---|---|---|---|
| 1/s | +0.040 [+0.036, +0.043] | +0.063 [+0.058, +0.068] | +0.084 [+0.080, +0.090] | +0.091 [+0.086, +0.097] | +0.099 [+0.094, +0.105] | +0.104 [+0.099, +0.110] | +0.108 [+0.102, +0.114] | True | True |
| 2/s | +0.041 [+0.038, +0.044] | +0.064 [+0.060, +0.069] | +0.078 [+0.073, +0.083] | +0.084 [+0.079, +0.090] | +0.090 [+0.084, +0.096] | +0.100 [+0.095, +0.106] | +0.112 [+0.107, +0.118] | True | True |
| 3/s | -0.019 [-0.026, -0.013] | -0.002 [-0.010, +0.006] | +0.014 [+0.006, +0.023] | +0.020 [+0.012, +0.028] | +0.029 [+0.021, +0.038] | +0.031 [+0.023, +0.041] | +0.035 [+0.026, +0.045] | False | True |
| 4/s | -0.046 [-0.052, -0.040] | -0.040 [-0.047, -0.032] | -0.029 [-0.035, -0.022] | -0.016 [-0.023, -0.008] | -0.011 [-0.019, -0.001] | -0.008 [-0.016, +0.001] | -0.001 [-0.009, +0.008] | False | True |
| 6/s | -0.029 [-0.034, -0.024] | -0.041 [-0.047, -0.035] | -0.030 [-0.036, -0.024] | -0.024 [-0.030, -0.018] | -0.023 [-0.029, -0.015] | -0.014 [-0.021, -0.006] | -0.009 [-0.017, -0.001] | False | False |
| 8/s | -0.016 [-0.020, -0.012] | -0.036 [-0.042, -0.030] | -0.032 [-0.038, -0.026] | -0.025 [-0.030, -0.018] | -0.018 [-0.024, -0.012] | -0.017 [-0.023, -0.010] | -0.014 [-0.021, -0.007] | False | False |

**radix-affinity-balanced: FALSIFIED**

| load | 2 rep | 3 rep | 4 rep | 5 rep | 6 rep | 7 rep | 8 rep | strictly higher | widens |
|---|---|---|---|---|---|---|---|---|---|
| 1/s | +0.034 [+0.030, +0.039] | +0.057 [+0.051, +0.063] | +0.079 [+0.073, +0.084] | +0.085 [+0.079, +0.092] | +0.094 [+0.088, +0.100] | +0.098 [+0.092, +0.105] | +0.102 [+0.095, +0.109] | True | True |
| 2/s | +0.014 [+0.009, +0.019] | +0.038 [+0.032, +0.044] | +0.053 [+0.046, +0.059] | +0.058 [+0.051, +0.065] | +0.063 [+0.056, +0.071] | +0.073 [+0.067, +0.081] | +0.087 [+0.080, +0.094] | True | True |
| 3/s | +0.018 [+0.013, +0.024] | +0.028 [+0.021, +0.035] | +0.035 [+0.028, +0.042] | +0.041 [+0.034, +0.049] | +0.051 [+0.042, +0.058] | +0.053 [+0.045, +0.061] | +0.057 [+0.048, +0.066] | True | True |
| 4/s | +0.041 [+0.034, +0.047] | +0.034 [+0.027, +0.042] | +0.021 [+0.014, +0.028] | +0.025 [+0.017, +0.033] | +0.030 [+0.020, +0.039] | +0.031 [+0.022, +0.040] | +0.039 [+0.030, +0.047] | True | False |
| 6/s | +0.018 [+0.011, +0.024] | +0.021 [+0.014, +0.029] | +0.023 [+0.014, +0.032] | +0.012 [+0.004, +0.021] | -0.002 [-0.011, +0.007] | +0.006 [-0.003, +0.016] | +0.011 [+0.002, +0.020] | False | False |
| 8/s | -0.007 [-0.012, -0.000] | -0.003 [-0.013, +0.006] | +0.012 [+0.004, +0.020] | +0.010 [+0.001, +0.018] | +0.005 [-0.004, +0.014] | -0.001 [-0.010, +0.008] | +0.002 [-0.008, +0.012] | False | False |

Δ hit rate (policy − least-loaded), paired session bootstrap, 95% CI.

## H5 — does prefix-affinity hold the SLO with fewer replicas?

- **low_sharing · radix-affinity: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6', '8']
- **low_sharing · radix-affinity-balanced: HOLDS** — fewer at ['2', '3', '4', '6', '8']; more at no load
- **high_sharing · radix-affinity: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6']
- **high_sharing · radix-affinity-balanced: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6']
