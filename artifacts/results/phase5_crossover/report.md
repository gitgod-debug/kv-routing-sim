# Phase 5 — the crossover

Profile `t4-qwen3-4b-vllm0.30.0` (T4 figures) · SLO p95 TTFT ≤ 2,000 ms · 14,592 runs in 540 s · generated `2026-11-27T18:26:49+00:00` at `901195f`. Generated traffic, one model.

## A1 — the load weight α of radix-affinity-balanced (Phase 4's two traces)

Replicas needed to hold the SLO at loads 1/s, 2/s, 3/s, 4/s, 6/s, 8/s (— = not met at 8).

| α | low_sharing | high_sharing | total |
|---:|---|---|---:|
| 1 | 1 2 3 3 4 5 | 1 — — — — — | 64 |
| 2 | 1 2 3 3 4 5 | 1 2 — — — — | 57 |
| 4 | 1 2 3 4 4 5 | 1 2 3 — — — | 52 |
| 8 | 1 2 3 4 5 5 | 1 2 3 5 — — | 49 |
| 16 | 1 2 3 4 5 6 | 1 2 3 4 5 6 | 42 |
| 32 | 1 3 3 4 6 7 | 1 3 4 4 6 7 | 49 |
| 64 | 1 3 3 4 6 7 | 1 3 4 5 6 7 | 50 |
| *Phase 4 round-robin, low_sharing* | 1 3 4 5 7 8 | | |
| *Phase 4 session-sticky, low_sharing* | 1 2 3 3 4 5 | | |
| *Phase 4 round-robin, high_sharing* | 1 3 4 5 7 — | | |
| *Phase 4 session-sticky, high_sharing* | 1 2 3 3 4 5 | | |

**α\* = 16** — the fewest replicas in total; ties go to the smaller α.

## H2 — where does least-loaded beat prefix-affinity on p95 TTFT?

Each cell: the winner against least-loaded, from the paired session-bootstrap 95% CI on p95(least-loaded) − p95(P) (seed 0). **P** = P faster, **LL** = least-loaded faster, · = inside the CI. Rows: load (sessions/s); columns: requested sharing (achieved cross-session sharing in brackets).

### P = radix-affinity — **FALSIFIED**

4 replicas — crossover: True · violations along sharing: [[1, 0.0, 0.2], [1, 0.0, 0.4], [1, 0.0, 0.6], [1, 0.0, 0.8], [2, 0.0, 0.2], [2, 0.0, 0.4], [2, 0.0, 0.6], [2, 0.0, 0.8], [3, 0.0, 0.2], [3, 0.0, 0.4], [3, 0.0, 0.6], [3, 0.0, 0.8], [4, 0.0, 0.2], [4, 0.0, 0.4], [4, 0.0, 0.6], [4, 0.0, 0.8], [6, 0.0, 0.2], [6, 0.0, 0.4], [6, 0.0, 0.6], [6, 0.0, 0.8], [8, 0.0, 0.2], [8, 0.0, 0.4], [8, 0.0, 0.6], [8, 0.0, 0.8]] · along load: none · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +218 ms | **LL** -114 ms | **LL** -111 ms | **LL** -117 ms | **LL** -119 ms |
| 2/s | **P** +224 ms | **LL** -63442 ms | **LL** -60975 ms | **LL** -70228 ms | **LL** -71362 ms |
| 3/s | **P** +674 ms | **LL** -160769 ms | **LL** -160405 ms | **LL** -181516 ms | **LL** -178017 ms |
| 4/s | **P** +4581 ms | **LL** -219753 ms | **LL** -218429 ms | **LL** -223466 ms | **LL** -221196 ms |
| 6/s | **P** +19221 ms | **LL** -228406 ms | **LL** -224815 ms | **LL** -226750 ms | **LL** -223318 ms |
| 8/s | **P** +20346 ms | **LL** -232849 ms | **LL** -231473 ms | **LL** -228932 ms | **LL** -225133 ms |

8 replicas — crossover: True · violations along sharing: [[1, 0.0, 0.2], [1, 0.0, 0.4], [1, 0.0, 0.6], [1, 0.0, 0.8], [2, 0.0, 0.2], [2, 0.0, 0.4], [2, 0.0, 0.6], [2, 0.0, 0.8], [3, 0.0, 0.2], [3, 0.0, 0.4], [3, 0.0, 0.6], [3, 0.0, 0.8], [4, 0.0, 0.2], [4, 0.0, 0.4], [4, 0.0, 0.6], [4, 0.0, 0.8], [6, 0.0, 0.2], [6, 0.0, 0.4], [6, 0.0, 0.6], [6, 0.0, 0.8], [8, 0.0, 0.2], [8, 0.0, 0.4], [8, 0.0, 0.6], [8, 0.0, 0.8]] · along load: none · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +211 ms | **LL** -168 ms | **LL** -152 ms | **LL** -198 ms | **LL** -170 ms |
| 2/s | **P** +310 ms | **LL** -63649 ms | **LL** -61041 ms | **LL** -70500 ms | **LL** -71654 ms |
| 3/s | **P** +307 ms | **LL** -161239 ms | **LL** -160766 ms | **LL** -182027 ms | **LL** -178448 ms |
| 4/s | **P** +318 ms | **LL** -224993 ms | **LL** -223362 ms | **LL** -229106 ms | **LL** -225854 ms |
| 6/s | **P** +668 ms | **LL** -247520 ms | **LL** -244527 ms | **LL** -247695 ms | **LL** -242887 ms |
| 8/s | **P** +930 ms | **LL** -259975 ms | **LL** -257243 ms | **LL** -254609 ms | **LL** -249838 ms |

### P = radix-affinity-balanced@16 — **FALSIFIED**

4 replicas — crossover: False · violations along sharing: none · along load: none · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +83 ms | **P** +40 ms | · +21 ms | · +77 ms | · +23 ms |
| 2/s | · -9 ms | **P** +88 ms | **P** +137 ms | **P** +77 ms | **P** +215 ms |
| 3/s | **P** +194 ms | **P** +350 ms | · -55 ms | **P** +191 ms | **P** +186 ms |
| 4/s | **P** +4196 ms | **P** +4821 ms | **P** +4774 ms | **P** +5427 ms | **P** +3864 ms |
| 6/s | **P** +12832 ms | **P** +12849 ms | **P** +13916 ms | **P** +13844 ms | **P** +12647 ms |
| 8/s | **P** +16100 ms | **P** +17401 ms | **P** +16685 ms | **P** +15865 ms | **P** +14126 ms |

8 replicas — crossover: True · violations along sharing: [[2, 0.0, 0.8], [2, 0.2, 0.8], [2, 0.4, 0.8], [2, 0.6, 0.8], [3, 0.2, 0.4], [3, 0.2, 0.8], [4, 0.6, 0.8], [6, 0.0, 0.8], [6, 0.4, 0.8], [6, 0.6, 0.8], [8, 0.0, 0.8], [8, 0.2, 0.8], [8, 0.4, 0.8], [8, 0.6, 0.8]] · along load: [[0.4, 3, 6], [0.4, 3, 8]] · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +48 ms | **P** +50 ms | **P** +48 ms | **P** +55 ms | · -1 ms |
| 2/s | **P** +148 ms | **P** +102 ms | **P** +166 ms | **P** +54 ms | **LL** -217 ms |
| 3/s | · -12 ms | **P** +148 ms | **LL** -82 ms | · -9 ms | **LL** -475 ms |
| 4/s | · -6 ms | · -127 ms | · +61 ms | **P** +220 ms | **LL** -281 ms |
| 6/s | **P** +313 ms | · +138 ms | **P** +320 ms | **P** +230 ms | **LL** -534 ms |
| 8/s | **P** +444 ms | **P** +455 ms | **P** +420 ms | **P** +588 ms | **LL** -362 ms |

## H2b — does affinity pay once the prompts outgrow one replica's cache?

Sharing 0.8, K distinct system prompts with Zipf popularity (s = 1.0). Each cell: replicas needed to hold the SLO — radix-affinity-balanced@16 vs session-sticky.

**H2b: HOLDS** — (a) no affinity win at K = 1: True · (b) an affinity win at K = 64: True · (c) monotone in K: yes

| load | K=1 (working set 0.06× a replica) | K=4 (working set 0.22× a replica) | K=16 (working set 0.89× a replica) | K=64 (working set 2.66× a replica) |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 3 vs 2 → sticky | 4 vs 3 → sticky |
| 2/s | 2 vs 2 → tie | 3 vs 2 → sticky | 4 vs 4 → tie | 6 vs — → affinity |
| 3/s | 3 vs 3 → tie | 4 vs 3 → sticky | 6 vs 5 → sticky | — vs — → tie |
| 4/s | 4 vs 3 → sticky | 6 vs 4 → sticky | 8 vs 6 → sticky | — vs — → tie |
| 6/s | 5 vs 4 → sticky | — vs 5 → sticky | — vs — → tie | — vs — → tie |
| 8/s | 6 vs 5 → sticky | — vs 6 → sticky | — vs — → tie | — vs — → tie |

Every policy (replicas needed, loads left to right; reported, not tested):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| round-robin | 1 3 4 5 7 — | 1 3 4 5 7 — | 3 6 8 — — — | 5 — — — — — |
| least-loaded | 1 3 4 5 7 8 | 1 3 4 5 8 — | 3 8 — — — — | 8 — — — — — |
| power-of-two | 1 3 4 5 7 8 | 1 3 4 5 8 — | 4 — — — — — | — — — — — — |
| radix-affinity | 1 — — — — — | 1 2 3 — — — | 2 3 3 4 — — | 2 4 5 5 — — |
| session-sticky | 1 2 3 3 4 5 | 1 2 3 4 5 6 | 2 4 5 6 — — | 3 — — — — — |
| radix-affinity-balanced@16 | 1 2 3 4 5 6 | 1 3 4 6 — — | 3 4 6 8 — — | 4 6 — — — — |

| *best α per cell (oracle)* | 1 2 3 4 5 6 | 1 2 3 5 7 7 | 2 3 3 4 7 — | 2 4 5 5 — — |

