# R1 — real traffic: the Mooncake FAST25 traces — on the validated timing model, calibration s0

T4 profile · token counts × 0.03125 · first 60 min, the first 5 min warm-up · sessions inferred (adapter) · SLO p95 TTFT ≤ 2,000 ms · 1,152 runs in 243 s · generated `2026-10-06T09:01:55+00:00` at `3fd95df`.

**R1: FALSIFIED**

| trace | prediction | winner at each speed | verdict |
|---|---|---|---|
| toolagent | affinity_wins | 0.25×: tie · 0.5×: tie · 1×: tie | FALSIFIED |
| conversation | affinity_never_wins | 0.25×: tie · 0.5×: tie · 1×: tie | HOLDS |

## Structure of the window (reported)

| trace | requests | sessions (inferred) | cross-session sharing | total reuse |
|---|---:|---:|---:|---:|
| toolagent | 23,608 | 19,544 | 0.397 | 0.586 |
| conversation | 12,031 | 8,057 | 0.100 | 0.382 |

## toolagent — replicas needed to hold the SLO, speeds 0.25×, 0.5×, 1× (— = not met at 8)

| policy | replicas | hit rate at 4 replicas |
|---|---|---|
| round-robin | 5 5 6 | 0.371 0.371 0.371 |
| least-loaded | 5 5 6 | 0.371 0.370 0.369 |
| power-of-two | 5 5 6 | 0.370 0.372 0.370 |
| consistent-hash-prefix | 6 6 7 | 0.430 0.430 0.430 |
| radix-affinity | — — — | 0.358 0.358 0.357 |
| session-sticky | 5 5 5 | 0.423 0.423 0.422 |
| radix-affinity-balanced@16 | 5 5 5 | 0.419 0.420 0.420 |
| radix-affinity-balanced@1 | 5 5 5 | 0.419 0.422 0.421 |

## conversation — replicas needed to hold the SLO, speeds 0.25×, 0.5×, 1× (— = not met at 8)

| policy | replicas | hit rate at 4 replicas |
|---|---|---|
| round-robin | 6 6 7 | 0.071 0.071 0.071 |
| least-loaded | 6 6 7 | 0.070 0.072 0.070 |
| power-of-two | 7 7 8 | 0.068 0.069 0.071 |
| consistent-hash-prefix | 8 8 — | 0.142 0.142 0.143 |
| radix-affinity | — — — | 0.050 0.050 0.049 |
| session-sticky | 6 6 6 | 0.138 0.137 0.139 |
| radix-affinity-balanced@16 | 6 6 6 | 0.136 0.132 0.132 |
| radix-affinity-balanced@1 | 6 6 6 | 0.144 0.143 0.145 |
