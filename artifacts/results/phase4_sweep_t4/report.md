# Phase 4 — the first calibrated sweep

Profile `t4-qwen3-4b-vllm0.30.0` (1,537 blocks, fitted T4 timings) · cost `g4dn.xlarge` at $0.526/h · SLO p95 TTFT ≤ 2,000 ms · 5 seeds · generated `2026-09-22T14:23:16+00:00` at `d3d527a`.

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
| least-loaded | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.01 · 0 | 1.03 · 0 | 1.03 · 3 |
| power-of-two | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.02 · 0 | 1.03 · 0 | 1.03 · 6 |
| consistent-hash-prefix | 1.00 · 0 | 3.00 · 206 | 4.00 · 182 | 5.00 · 110 | 7.00 · 126 | 8.00 · 121 |
| radix-affinity | 1.00 · 0 | 3.00 · 206 | 4.00 · 182 | 5.00 · 110 | 7.00 · 126 | 8.00 · 121 |
| radix-affinity-balanced | 1.00 · 0 | 1.03 · 0 | 1.05 · 0 | 1.10 · 0 | 1.18 · 0 | 1.16 · 0 |
| session-sticky | 1.00 · 0 | 1.02 · 0 | 1.02 · 0 | 1.04 · 0 | 1.04 · 0 | 1.09 · 0 |

### high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1.00 · 0 | 1.00 · 0 | 1.00 · 0 | 1.00 · 1 | 1.01 · 0 | 1.00 · 19 |
| least-loaded | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.02 · 0 | 1.04 · 6 |
| power-of-two | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.02 · 0 | 1.03 · 0 | 1.05 · 9 |
| consistent-hash-prefix | 1.00 · 0 | 3.00 · 212 | 4.00 · 139 | 5.00 · 120 | 7.00 · 125 | 8.00 · 123 |
| radix-affinity | 1.00 · 0 | 3.00 · 212 | 4.00 · 139 | 5.00 · 120 | 7.00 · 125 | 8.00 · 123 |
| radix-affinity-balanced | 1.00 · 0 | 1.71 · 0 | 1.62 · 67 | 1.63 · 121 | 1.84 · 87 | 1.81 · 85 |
| session-sticky | 1.00 · 0 | 1.01 · 0 | 1.02 · 0 | 1.03 · 0 | 1.06 · 0 | 1.08 · 0 |

## Recommendation at each load (cheapest configuration meeting the SLO)

| trace @ load | recommended | ₹ / 1k requests | vs round-robin |
|---|---|---|---|
| low_sharing@1 | round-robin × 1 | 4.01 | +0.0% · +0 replicas |
| low_sharing@2 | radix-affinity-balanced × 2 | 4.40 | -33.2% · -1 replicas |
| low_sharing@3 | radix-affinity-balanced × 3 | 4.78 | -25.2% · -1 replicas |
| low_sharing@4 | session-sticky × 3 | 3.89 | -39.9% · -2 replicas |
| low_sharing@6 | session-sticky × 4 | 4.00 | -42.4% · -3 replicas |
| low_sharing@8 | session-sticky × 5 | 4.24 | -41.4% · -3 replicas |
| high_sharing@1 | round-robin × 1 | 4.01 | +0.0% · +0 replicas |
| high_sharing@2 | session-sticky × 2 | 4.40 | -33.2% · -1 replicas |
| high_sharing@3 | session-sticky × 3 | 4.79 | -25.1% · -1 replicas |
| high_sharing@4 | session-sticky × 3 | 3.90 | -39.8% · -2 replicas |
| high_sharing@6 | session-sticky × 4 | 4.01 | -42.2% · -3 replicas |
| high_sharing@8 | session-sticky × 5 | 4.23 | -42.9% · -3 replicas (round-robin misses the SLO even at 8 — vs its best attempt) |

## H1 — prefix-affinity vs least-loaded on hit rate, high sharing

**radix-affinity: FALSIFIED**

| load | 2 rep | 3 rep | 4 rep | 5 rep | 6 rep | 7 rep | 8 rep | strictly higher | widens |
|---|---|---|---|---|---|---|---|---|---|
| 1/s | +0.043 [+0.040, +0.046] | +0.069 [+0.065, +0.074] | +0.081 [+0.076, +0.086] | +0.092 [+0.087, +0.098] | +0.104 [+0.099, +0.110] | +0.106 [+0.101, +0.112] | +0.110 [+0.104, +0.117] | True | True |
| 2/s | +0.043 [+0.040, +0.047] | +0.065 [+0.061, +0.070] | +0.077 [+0.073, +0.082] | +0.083 [+0.078, +0.090] | +0.092 [+0.086, +0.099] | +0.104 [+0.098, +0.111] | +0.105 [+0.100, +0.111] | True | True |
| 3/s | -0.017 [-0.024, -0.010] | +0.001 [-0.007, +0.009] | +0.013 [+0.005, +0.022] | +0.018 [+0.010, +0.027] | +0.031 [+0.023, +0.040] | +0.028 [+0.019, +0.038] | +0.035 [+0.026, +0.045] | False | True |
| 4/s | -0.044 [-0.050, -0.038] | -0.039 [-0.046, -0.032] | -0.027 [-0.034, -0.020] | -0.020 [-0.027, -0.012] | -0.012 [-0.019, -0.004] | -0.008 [-0.015, +0.001] | +0.000 [-0.008, +0.010] | False | True |
| 6/s | -0.026 [-0.030, -0.021] | -0.036 [-0.042, -0.030] | -0.031 [-0.037, -0.024] | -0.023 [-0.029, -0.016] | -0.018 [-0.024, -0.009] | -0.015 [-0.021, -0.006] | -0.011 [-0.018, -0.002] | False | False |
| 8/s | -0.019 [-0.024, -0.015] | -0.033 [-0.038, -0.027] | -0.028 [-0.034, -0.022] | -0.024 [-0.030, -0.018] | -0.020 [-0.026, -0.013] | -0.019 [-0.025, -0.011] | -0.010 [-0.017, -0.002] | False | False |

**radix-affinity-balanced: FALSIFIED**

| load | 2 rep | 3 rep | 4 rep | 5 rep | 6 rep | 7 rep | 8 rep | strictly higher | widens |
|---|---|---|---|---|---|---|---|---|---|
| 1/s | +0.037 [+0.033, +0.041] | +0.063 [+0.058, +0.069] | +0.075 [+0.070, +0.081] | +0.087 [+0.081, +0.092] | +0.098 [+0.092, +0.104] | +0.100 [+0.094, +0.108] | +0.104 [+0.098, +0.111] | True | True |
| 2/s | +0.022 [+0.017, +0.027] | +0.044 [+0.038, +0.050] | +0.057 [+0.051, +0.064] | +0.062 [+0.055, +0.070] | +0.072 [+0.064, +0.079] | +0.084 [+0.076, +0.091] | +0.085 [+0.078, +0.092] | True | True |
| 3/s | +0.019 [+0.012, +0.025] | +0.029 [+0.022, +0.036] | +0.031 [+0.024, +0.039] | +0.038 [+0.030, +0.046] | +0.051 [+0.043, +0.059] | +0.046 [+0.038, +0.055] | +0.055 [+0.046, +0.063] | True | True |
| 4/s | +0.038 [+0.030, +0.045] | +0.024 [+0.017, +0.030] | +0.014 [+0.006, +0.021] | +0.020 [+0.012, +0.028] | +0.026 [+0.018, +0.035] | +0.032 [+0.024, +0.041] | +0.040 [+0.031, +0.049] | True | False |
| 6/s | +0.011 [+0.004, +0.018] | +0.016 [+0.008, +0.024] | +0.008 [-0.000, +0.017] | +0.003 [-0.006, +0.012] | -0.002 [-0.011, +0.006] | +0.002 [-0.007, +0.012] | +0.006 [-0.003, +0.015] | False | False |
| 8/s | -0.006 [-0.012, +0.000] | -0.005 [-0.013, +0.004] | +0.002 [-0.006, +0.010] | -0.003 [-0.012, +0.005] | +0.002 [-0.008, +0.010] | -0.002 [-0.011, +0.008] | +0.007 [-0.003, +0.017] | False | False |

Δ hit rate (policy − least-loaded), paired session bootstrap, 95% CI.

## H5 — does prefix-affinity hold the SLO with fewer replicas?

- **low_sharing · radix-affinity: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6', '8']
- **low_sharing · radix-affinity-balanced: HOLDS** — fewer at ['2', '3', '4', '6', '8']; more at no load
- **high_sharing · radix-affinity: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6']
- **high_sharing · radix-affinity-balanced: FALSIFIED** — fewer at no load; more at ['2', '3', '4', '6']
