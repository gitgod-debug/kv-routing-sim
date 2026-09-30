# H2b under C2's timing model

Phase 5's H2b traffic on the context-aware T4 profile `t4-qwen3-4b-vllm0.30.0-ctx` · loads 0.25, 0.5, 1, 2, 3, 4 sessions/s · replicas 1–16 · 3 seeds · 6,912 runs in 319 s · generated `2026-09-30T06:30:47+00:00` at `c0c8b18` (dirty tree).

**H2b under C2's timing: FALSIFIED** (registered prediction: holds).

| rule | result |
|---|---|
| (a) at K = 1, balanced affinity wins at no load | ✅ |
| (b) at K = 64, balanced affinity wins at one load or more | ⛔ nowhere |
| (c) no reversal along K | ✅ |

## radix-affinity-balanced@16 against session-sticky — replicas needed, and the winner (— = not held on 16)

| load | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| 0.25/s | 1 vs 1 → tie | 1 vs 1 → tie | 1 vs 1 → tie | 2 vs 2 → tie |
| 0.5/s | 2 vs 2 → tie | 2 vs 2 → tie | 3 vs 2 → sticky | 3 vs 3 → tie |
| 1/s | 4 vs 3 → sticky | 4 vs 3 → sticky | 5 vs 4 → sticky | 5 vs 5 → tie |
| 2/s | 10 vs 5 → sticky | 9 vs 5 → sticky | 9 vs 7 → sticky | 11 vs 10 → sticky |
| 3/s | — vs 6 → sticky | 16 vs 6 → sticky | 16 vs 12 → sticky | — vs — → tie |
| 4/s | — vs 7 → sticky | — vs 9 → sticky | — vs 16 → sticky | — vs — → tie |

## Every policy — replicas needed at each load, left to right

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 1 2 3 5 6 7 | 1 2 3 5 6 9 | 1 2 4 7 12 16 | 2 3 5 10 — — |
| radix-affinity-balanced@16 | 1 2 4 10 — — | 1 2 4 9 16 — | 1 3 5 9 16 — | 2 3 5 11 — — |
| radix-affinity-balanced@1 | 1 2 — — — — | 1 2 3 — — — | 1 2 3 — — — | 2 3 5 8 — — |
| round-robin | 1 2 3 7 9 12 | 1 2 4 7 9 12 | 1 3 6 11 16 — | 2 4 9 16 — — |
| least-loaded | 1 2 4 11 14 — | 1 2 5 10 15 — | 1 4 9 — — — | 3 7 13 — — — |
| radix-affinity | 1 — — — — — | 1 2 3 — — — | 1 2 3 — — — | 2 3 5 — — — |

## Hit rate on 8 replicas (loads left to right)

| policy | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| session-sticky | 0.92 0.92 0.92 0.92 0.92 0.92 | 0.89 0.89 0.89 0.89 0.89 0.89 | 0.84 0.84 0.84 0.84 0.84 0.85 | 0.80 0.80 0.80 0.80 0.80 0.79 |
| radix-affinity-balanced@16 | 0.91 0.90 0.88 0.86 0.86 0.88 | 0.92 0.91 0.90 0.87 0.86 0.87 | 0.91 0.90 0.89 0.86 0.84 0.86 | 0.88 0.87 0.86 0.83 0.80 0.82 |
| radix-affinity-balanced@1 | 0.92 0.92 0.90 0.87 0.83 0.82 | 0.92 0.92 0.92 0.91 0.90 0.89 | 0.91 0.91 0.91 0.90 0.89 0.89 | 0.88 0.88 0.88 0.87 0.87 0.87 |
| round-robin | 0.82 0.82 0.81 0.81 0.82 0.81 | 0.79 0.80 0.79 0.79 0.79 0.79 | 0.66 0.67 0.65 0.65 0.66 0.65 | 0.47 0.49 0.47 0.47 0.48 0.46 |
| least-loaded | 0.82 0.82 0.82 0.82 0.82 0.82 | 0.79 0.79 0.79 0.79 0.79 0.79 | 0.66 0.66 0.66 0.66 0.66 0.66 | 0.49 0.49 0.48 0.49 0.47 0.47 |
| radix-affinity | 0.92 0.92 0.92 0.92 0.86 0.82 | 0.92 0.92 0.92 0.92 0.92 0.92 | 0.91 0.91 0.91 0.91 0.91 0.91 | 0.88 0.88 0.88 0.88 0.88 0.88 |
