# R1 — real traffic: the Mooncake FAST'25 traces

T4 profile · token counts × 0.03125 · first 60 min, the first 5 min warm-up · sessions inferred (adapter) · SLO p95 TTFT ≤ 2,000 ms · 1,152 runs in 219 s · generated `2026-11-27T19:22:08+00:00` at `eac9db9`.

**R1: FALSIFIED**

| trace | prediction | winner at each speed | verdict |
|---|---|---|---|
| toolagent | affinity_wins | 1×: tie · 2×: sticky · 4×: tie | FALSIFIED |
| conversation | affinity_never_wins | 1×: tie · 2×: tie · 4×: tie | HOLDS |

## Structure of the window (reported)

| trace | requests | sessions (inferred) | cross-session sharing | total reuse |
|---|---:|---:|---:|---:|
| toolagent | 23,608 | 19,544 | 0.397 | 0.586 |
| conversation | 12,031 | 8,057 | 0.100 | 0.382 |

## toolagent — replicas needed to hold the SLO, speeds 1×, 2×, 4× (— = not met at 8)

| policy | replicas | hit rate at 4 replicas |
|---|---|---|
| round-robin | 4 5 8 | 0.371 0.371 0.365 |
| least-loaded | 4 5 7 | 0.370 0.369 0.370 |
| power-of-two | 4 5 7 | 0.371 0.371 0.370 |
| consistent-hash-prefix | 5 6 — | 0.430 0.430 0.425 |
| radix-affinity | — — — | 0.357 0.356 0.356 |
| session-sticky | 4 4 6 | 0.423 0.424 0.422 |
| radix-affinity-balanced@16 | 4 5 6 | 0.420 0.414 0.400 |
| radix-affinity-balanced@1 | 4 5 6 | 0.422 0.424 0.424 |

## conversation — replicas needed to hold the SLO, speeds 1×, 2×, 4× (— = not met at 8)

| policy | replicas | hit rate at 4 replicas |
|---|---|---|
| round-robin | 5 5 8 | 0.071 0.071 0.069 |
| least-loaded | 5 6 8 | 0.070 0.069 0.067 |
| power-of-two | 5 6 8 | 0.071 0.072 0.068 |
| consistent-hash-prefix | 6 7 — | 0.142 0.143 0.141 |
| radix-affinity | — — — | 0.050 0.049 0.049 |
| session-sticky | 5 5 7 | 0.138 0.137 0.136 |
| radix-affinity-balanced@16 | 5 5 7 | 0.134 0.127 0.110 |
| radix-affinity-balanced@1 | 5 5 7 | 0.144 0.144 0.140 |
