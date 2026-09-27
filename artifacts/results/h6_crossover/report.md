# H6, crossover half — does H2b's flip survive a change of hardware?

Phase 5's H2b grid under 5 cost regimes · 14,400 runs in 570 s · generated `2026-09-27T19:34:42+00:00` at `16fcc53`.

⚠️ **Only `t4` is a measured profile.** The other regimes rescale its timings and describe no real GPU; only which policy needs fewer replicas is under test.

## Verdict

| regime | affinity wins at K = 1 | sticky wins at K = 64 | reversals along K | flip still visible | verdict |
|---|---|---|---|---|---|
| t4 | none | none | none | True | (baseline) |
| faster_2x | none | none | none | True | HOLDS |
| faster_4x | none | none | none | True | HOLDS |
| cheap_prefill | none | none | none | True | HOLDS |
| dear_prefill | none | none | none | False | HOLDS |

**H6, crossover half: HOLDS.**

## t4

Replicas needed to hold the SLO: radix-affinity-balanced@16 vs session-sticky → the winner.

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie | 3 vs 3 → tie |
| 2/s | 3 vs 2 → sticky | 3 vs 2 → sticky | 3 vs 3 → tie | 5 vs — → affinity |
| 3/s | 3 vs 3 → tie | 5 vs 3 → sticky | 6 vs 5 → sticky | 8 vs — → affinity |
| 4/s | 4 vs 3 → sticky | 6 vs 4 → sticky | 8 vs 6 → sticky | — vs — → tie |
| 6/s | 6 vs 4 → sticky | — vs 5 → sticky | — vs — → tie | — vs — → tie |
| 8/s | 7 vs 5 → sticky | — vs 6 → sticky | — vs — → tie | — vs — → tie |

Every policy (replicas needed at each load, left to right):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 1 2 3 3 4 5 | 1 2 3 4 5 6 | 2 3 5 6 — — | 3 — — — — — |
| radix-affinity-balanced@16 | 1 3 3 4 6 7 | 1 3 5 6 — — | 2 3 6 8 — — | 3 5 8 — — — |
| round-robin | 1 3 4 5 7 — | 1 3 4 5 7 — | 3 6 8 — — — | 5 — — — — — |
| least-loaded | 1 3 4 5 7 8 | 1 3 4 5 8 — | 3 8 — — — — | 8 — — — — — |
| radix-affinity | 1 — — — — — | 1 2 3 — — — | 2 3 3 4 — — | 3 4 5 5 — — |

## faster_2x

Replicas needed to hold the SLO: radix-affinity-balanced@16 vs session-sticky → the winner.

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie | 2 vs 2 → tie |
| 2/s | 1 vs 1 → tie | 2 vs 2 → tie | 3 vs 3 → tie | 3 vs 3 → tie |
| 3/s | 2 vs 2 → tie | 2 vs 2 → tie | 3 vs 4 → affinity | 4 vs 5 → affinity |
| 4/s | 2 vs 2 → tie | 3 vs 3 → tie | 4 vs 5 → affinity | 5 vs 6 → affinity |
| 6/s | 3 vs 3 → tie | 3 vs 3 → tie | 5 vs 6 → affinity | 7 vs — → affinity |
| 8/s | 4 vs 3 → sticky | 4 vs 4 → tie | 5 vs 8 → affinity | 7 vs — → affinity |

Every policy (replicas needed at each load, left to right):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 1 1 2 2 3 3 | 1 2 2 3 3 4 | 2 3 4 5 6 8 | 2 3 5 6 — — |
| radix-affinity-balanced@16 | 1 1 2 2 3 4 | 1 2 2 3 3 4 | 2 3 3 4 5 5 | 2 3 4 5 7 7 |
| round-robin | 1 1 2 3 4 5 | 1 2 2 3 4 5 | 2 3 4 5 7 8 | 2 4 6 8 — — |
| least-loaded | 1 1 2 3 4 5 | 1 2 2 3 4 5 | 2 3 4 5 7 8 | 2 5 7 — — — |
| radix-affinity | 1 1 — — — — | 1 2 2 3 — — | 2 2 3 3 4 — | 2 4 4 5 5 7 |

## faster_4x

Replicas needed to hold the SLO: radix-affinity-balanced@16 vs session-sticky → the winner.

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie |
| 2/s | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie | 3 vs 3 → tie |
| 3/s | 2 vs 2 → tie | 2 vs 2 → tie | 2 vs 3 → affinity | 3 vs 4 → affinity |
| 4/s | 2 vs 2 → tie | 2 vs 2 → tie | 2 vs 3 → affinity | 4 vs 4 → tie |
| 6/s | 2 vs 2 → tie | 2 vs 2 → tie | 3 vs 4 → affinity | 5 vs 6 → affinity |
| 8/s | 3 vs 3 → tie | 3 vs 3 → tie | 4 vs 5 → affinity | 5 vs 7 → affinity |

Every policy (replicas needed at each load, left to right):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 1 1 2 2 2 3 | 1 1 2 2 2 3 | 1 2 3 3 4 5 | 2 3 4 4 6 7 |
| radix-affinity-balanced@16 | 1 1 2 2 2 3 | 1 1 2 2 2 3 | 1 2 2 2 3 4 | 2 3 3 4 5 5 |
| round-robin | 1 1 2 2 2 3 | 1 1 2 2 2 3 | 1 2 2 3 4 4 | 2 3 3 4 6 7 |
| least-loaded | 1 1 2 2 2 3 | 1 1 2 2 2 3 | 1 2 2 3 4 4 | 2 3 4 4 6 7 |
| radix-affinity | 1 1 — — — — | 1 1 2 2 3 — | 1 2 2 2 3 3 | 2 2 3 4 4 4 |

## cheap_prefill

Replicas needed to hold the SLO: radix-affinity-balanced@16 vs session-sticky → the winner.

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie | 2 vs 2 → tie |
| 2/s | 2 vs 2 → tie | 2 vs 2 → tie | 3 vs 3 → tie | 4 vs 4 → tie |
| 3/s | 3 vs 2 → sticky | 3 vs 3 → tie | 4 vs 4 → tie | 5 vs 6 → affinity |
| 4/s | 3 vs 3 → tie | 3 vs 3 → tie | 4 vs 5 → affinity | 6 vs 6 → tie |
| 6/s | 4 vs 4 → tie | 5 vs 4 → sticky | 6 vs 6 → tie | 7 vs — → affinity |
| 8/s | 5 vs 4 → sticky | 5 vs 5 → tie | 6 vs 8 → affinity | — vs — → tie |

Every policy (replicas needed at each load, left to right):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 1 2 2 3 4 4 | 1 2 3 3 4 5 | 2 3 4 5 6 8 | 2 4 6 6 — — |
| radix-affinity-balanced@16 | 1 2 3 3 4 5 | 1 2 3 3 5 5 | 2 3 4 4 6 6 | 2 4 5 6 7 — |
| round-robin | 1 2 3 4 5 6 | 1 2 3 4 5 7 | 2 4 5 7 — — | 3 5 8 — — — |
| least-loaded | 1 2 3 4 5 6 | 1 2 3 4 5 7 | 2 4 5 7 — — | 3 6 8 — — — |
| radix-affinity | 1 — — — — — | 1 2 3 3 — — | 2 2 3 4 — — | 2 3 4 5 6 7 |

## dear_prefill

Replicas needed to hold the SLO: radix-affinity-balanced@16 vs session-sticky → the winner.

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 1/s | 3 vs 2 → sticky | 4 vs 2 → sticky | 5 vs — → affinity | — vs — → tie |
| 2/s | — vs 3 → sticky | — vs 3 → sticky | — vs — → tie | — vs — → tie |
| 3/s | — vs 4 → sticky | — vs — → tie | — vs — → tie | — vs — → tie |
| 4/s | — vs 5 → sticky | — vs — → tie | — vs — → tie | — vs — → tie |
| 6/s | — vs 6 → sticky | — vs — → tie | — vs — → tie | — vs — → tie |
| 8/s | — vs 7 → sticky | — vs — → tie | — vs — → tie | — vs — → tie |

Every policy (replicas needed at each load, left to right):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 2 3 4 5 6 7 | 2 3 — — — — | — — — — — — | — — — — — — |
| radix-affinity-balanced@16 | 3 — — — — — | 4 — — — — — | 5 — — — — — | — — — — — — |
| round-robin | 2 4 7 — — — | 2 5 7 — — — | — — — — — — | — — — — — — |
| least-loaded | 3 — — — — — | 2 — — — — — | — — — — — — | — — — — — — |
| radix-affinity | — — — — — — | 2 3 — — — — | 3 4 — — — — | — — — — — — |

