# R2 — real traffic: SemiAnalysis's AgentX traces (coding-agent sessions)

T4 profile · token counts × 0.015625 · first 180 min, the first 30 min warm-up · real sessions (one per trace), hashes session-local · SLO p95 TTFT ≤ 2,000 ms · 576 runs in 176 s · generated `2026-09-28T05:09:44+00:00` at `cf74aec`.

**R2: HOLDS**

| trace | prediction | winner at each speed | verdict |
|---|---|---|---|
| agentx | tie_within_one + pure_affinity_ties_sticky | 0.5×: tie · 1×: tie · 2×: tie | HOLDS |

## Structure of the window (reported)

| trace | requests | sessions | cross-session sharing | total reuse |
|---|---:|---:|---:|---:|
| agentx | 6,063 | 136 | 0.000 | 0.959 |

## agentx — replicas needed to hold the SLO, speeds 0.5×, 1×, 2× (— = not met at 8)

| policy | replicas | hit rate at 4 replicas |
|---|---|---|
| round-robin | 2 2 5 | 0.591 0.590 0.590 |
| least-loaded | 2 2 3 | 0.588 0.588 0.589 |
| power-of-two | 2 2 3 | 0.589 0.589 0.586 |
| consistent-hash-prefix | 2 2 3 | 0.933 0.933 0.933 |
| radix-affinity | 2 2 2 | 0.936 0.934 0.933 |
| session-sticky | 2 2 2 | 0.935 0.934 0.933 |
| radix-affinity-balanced@16 | 2 2 2 | 0.937 0.937 0.936 |
| radix-affinity-balanced@1 | 2 2 2 | 0.936 0.934 0.933 |
