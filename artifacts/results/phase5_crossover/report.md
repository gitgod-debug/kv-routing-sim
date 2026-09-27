# Phase 5 — the crossover

Profile `t4-qwen3-4b-vllm0.30.0` (T4 figures) · SLO p95 TTFT ≤ 2,000 ms · 14,592 runs in 570 s · generated `2026-09-27T19:17:17+00:00` at `3967a56`. Generated traffic, one model.

## A1 — the load weight α of radix-affinity-balanced (Phase 4's two traces)

Replicas needed to hold the SLO at loads 1/s, 2/s, 3/s, 4/s, 6/s, 8/s (— = not met at 8).

| α | low_sharing | high_sharing | total |
|---:|---|---|---:|
| 1 | 1 2 3 3 4 5 | 1 — — — — — | 64 |
| 2 | 1 2 3 3 4 5 | 1 2 — — — — | 57 |
| 4 | 1 2 3 4 4 5 | 1 2 — — — — | 58 |
| 8 | 1 2 3 4 5 5 | 1 2 3 4 — — | 48 |
| 16 | 1 2 3 4 5 6 | 1 3 3 4 6 7 | 45 |
| 32 | 1 3 4 4 6 6 | 1 3 3 4 6 7 | 48 |
| 64 | 1 3 4 4 6 6 | 1 3 4 4 6 7 | 49 |
| *Phase 4 round-robin, low_sharing* | 1 3 4 5 7 8 | | |
| *Phase 4 session-sticky, low_sharing* | 1 2 3 3 4 5 | | |
| *Phase 4 round-robin, high_sharing* | 1 3 4 5 7 — | | |
| *Phase 4 session-sticky, high_sharing* | 1 2 3 3 4 5 | | |

**α\* = 16** — the fewest replicas in total; ties go to the smaller α.

## H2 — where does least-loaded beat prefix-affinity on p95 TTFT?

Each cell: the winner against least-loaded, from the paired session-bootstrap 95% CI on p95(least-loaded) − p95(P) (seed 0). **P** = P faster, **LL** = least-loaded faster, · = inside the CI. Rows: load (sessions/s); columns: requested sharing (achieved cross-session sharing in brackets).

### P = radix-affinity — **FALSIFIED**

4 replicas — crossover: True · violations along sharing: [[1, 0.0, 0.2], [1, 0.0, 0.6], [1, 0.0, 0.8], [2, 0.0, 0.2], [2, 0.0, 0.4], [2, 0.0, 0.6], [2, 0.0, 0.8], [3, 0.0, 0.2], [3, 0.0, 0.4], [3, 0.0, 0.6], [3, 0.0, 0.8], [4, 0.0, 0.2], [4, 0.0, 0.4], [4, 0.0, 0.6], [4, 0.0, 0.8], [6, 0.0, 0.2], [6, 0.0, 0.4], [6, 0.0, 0.6], [6, 0.0, 0.8], [8, 0.0, 0.2], [8, 0.0, 0.4], [8, 0.0, 0.6], [8, 0.0, 0.8]] · along load: none · power-of-two order: [[0.4, 2, 1]]

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +225 ms | **LL** -155 ms | · -49 ms | **LL** -97 ms | **LL** -105 ms |
| 2/s | **P** +163 ms | **LL** -63492 ms | **LL** -60918 ms | **LL** -70347 ms | **LL** -71415 ms |
| 3/s | **P** +760 ms | **LL** -160835 ms | **LL** -160512 ms | **LL** -181442 ms | **LL** -178100 ms |
| 4/s | **P** +5147 ms | **LL** -220283 ms | **LL** -218997 ms | **LL** -223123 ms | **LL** -220951 ms |
| 6/s | **P** +20217 ms | **LL** -227495 ms | **LL** -224515 ms | **LL** -225121 ms | **LL** -222567 ms |
| 8/s | **P** +21138 ms | **LL** -234164 ms | **LL** -231414 ms | **LL** -227633 ms | **LL** -224126 ms |

8 replicas — crossover: True · violations along sharing: [[1, 0.0, 0.2], [1, 0.0, 0.4], [1, 0.0, 0.6], [1, 0.0, 0.8], [2, 0.0, 0.2], [2, 0.0, 0.4], [2, 0.0, 0.6], [2, 0.0, 0.8], [3, 0.0, 0.2], [3, 0.0, 0.4], [3, 0.0, 0.6], [3, 0.0, 0.8], [4, 0.0, 0.2], [4, 0.0, 0.4], [4, 0.0, 0.6], [4, 0.0, 0.8], [6, 0.0, 0.2], [6, 0.0, 0.4], [6, 0.0, 0.6], [6, 0.0, 0.8], [8, 0.0, 0.2], [8, 0.0, 0.4], [8, 0.0, 0.6], [8, 0.0, 0.8]] · along load: none · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +209 ms | **LL** -171 ms | **LL** -151 ms | **LL** -166 ms | **LL** -172 ms |
| 2/s | **P** +303 ms | **LL** -63612 ms | **LL** -61048 ms | **LL** -70525 ms | **LL** -71684 ms |
| 3/s | **P** +321 ms | **LL** -161291 ms | **LL** -160811 ms | **LL** -182031 ms | **LL** -178459 ms |
| 4/s | **P** +298 ms | **LL** -224784 ms | **LL** -223302 ms | **LL** -229119 ms | **LL** -225775 ms |
| 6/s | **P** +665 ms | **LL** -247459 ms | **LL** -244600 ms | **LL** -247551 ms | **LL** -242888 ms |
| 8/s | **P** +899 ms | **LL** -259944 ms | **LL** -257216 ms | **LL** -254770 ms | **LL** -249675 ms |

### P = radix-affinity-balanced@16 — **FALSIFIED**

4 replicas — crossover: True · violations along sharing: [[2, 0.4, 0.6]] · along load: [[0.6, 2, 3], [0.6, 2, 4], [0.6, 2, 6], [0.6, 2, 8]] · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | · +66 ms | · +10 ms | **P** +108 ms | **P** +54 ms | **P** +67 ms |
| 2/s | · -80 ms | · +58 ms | **P** +139 ms | **LL** -93 ms | · +139 ms |
| 3/s | **P** +157 ms | **P** +209 ms | · +83 ms | **P** +489 ms | **P** +85 ms |
| 4/s | **P** +4733 ms | **P** +4464 ms | **P** +4132 ms | **P** +5665 ms | **P** +4250 ms |
| 6/s | **P** +14622 ms | **P** +14053 ms | **P** +13881 ms | **P** +16854 ms | **P** +12585 ms |
| 8/s | **P** +17338 ms | **P** +15679 ms | **P** +16600 ms | **P** +16906 ms | **P** +14705 ms |

8 replicas — crossover: True · violations along sharing: [[2, 0.0, 0.8], [2, 0.2, 0.8], [2, 0.4, 0.8], [2, 0.6, 0.8], [3, 0.2, 0.8], [6, 0.0, 0.8], [6, 0.2, 0.8], [6, 0.4, 0.8], [6, 0.6, 0.8]] · along load: [[0.0, 4, 6], [0.0, 4, 8]] · power-of-two order: none

| load | 0.0 (0.00) | 0.2 (0.19) | 0.4 (0.40) | 0.6 (0.59) | 0.8 (0.80) |
|---|---|---|---|---|---|
| 1/s | **P** +47 ms | **P** +47 ms | **P** +48 ms | **P** +87 ms | · -79 ms |
| 2/s | **P** +151 ms | **P** +113 ms | **P** +150 ms | **P** +59 ms | **LL** -264 ms |
| 3/s | · +19 ms | **P** +83 ms | · -44 ms | · -94 ms | **LL** -368 ms |
| 4/s | **LL** -78 ms | **P** +294 ms | · +54 ms | **P** +159 ms | · -203 ms |
| 6/s | **P** +300 ms | **P** +222 ms | **P** +260 ms | **P** +369 ms | **LL** -617 ms |
| 8/s | **P** +551 ms | **P** +455 ms | **P** +567 ms | **P** +564 ms | · -143 ms |

## H2b — does affinity pay once the prompts outgrow one replica's cache?

Sharing 0.8, K distinct system prompts with Zipf popularity (s = 1.0). Each cell: replicas needed to hold the SLO — radix-affinity-balanced@16 vs session-sticky.

**H2b: HOLDS** — (a) no affinity win at K = 1: True · (b) an affinity win at K = 64: True · (c) monotone in K: yes

| load | K=1 (working set 0.06× a replica) | K=4 (working set 0.22× a replica) | K=16 (working set 0.89× a replica) | K=64 (working set 2.66× a replica) |
|---|---|---|---|---|
| 1/s | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie | 3 vs 3 → tie |
| 2/s | 3 vs 2 → sticky | 3 vs 2 → sticky | 3 vs 3 → tie | 5 vs — → affinity |
| 3/s | 3 vs 3 → tie | 5 vs 3 → sticky | 6 vs 5 → sticky | 8 vs — → affinity |
| 4/s | 4 vs 3 → sticky | 6 vs 4 → sticky | 8 vs 6 → sticky | — vs — → tie |
| 6/s | 6 vs 4 → sticky | — vs 5 → sticky | — vs — → tie | — vs — → tie |
| 8/s | 7 vs 5 → sticky | — vs 6 → sticky | — vs — → tie | — vs — → tie |

Every policy (replicas needed, loads left to right; reported, not tested):

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| round-robin | 1 3 4 5 7 — | 1 3 4 5 7 — | 3 6 8 — — — | 5 — — — — — |
| least-loaded | 1 3 4 5 7 8 | 1 3 4 5 8 — | 3 8 — — — — | 8 — — — — — |
| power-of-two | 1 3 4 5 7 8 | 1 3 4 5 8 — | 4 — — — — — | — — — — — — |
| radix-affinity | 1 — — — — — | 1 2 3 — — — | 2 3 3 4 — — | 3 4 5 5 — — |
| session-sticky | 1 2 3 3 4 5 | 1 2 3 4 5 6 | 2 3 5 6 — — | 3 — — — — — |
| radix-affinity-balanced@16 | 1 3 3 4 6 7 | 1 3 5 6 — — | 2 3 6 8 — — | 3 5 8 — — — |

| *best α per cell (oracle)* | 1 2 3 4 6 7 | 1 2 4 5 7 — | 2 3 3 4 8 — | 3 4 5 5 — — |

