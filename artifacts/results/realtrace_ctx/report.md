# R1 and R2 again — on the validated timing model

Registered `configs/realtrace_ctx.yaml`. The same traces, adapters, windows, policies, seeds and rules as R1 and R2; only the timing profile changed, and the speeds, re-chosen by the registered rule from a round-robin-only probe. Each cell below was decided by `scripts/run_realtrace.py`, R1's and R2's own procedure.

| test | registered prediction | result | on each calibration | on the first model |
|---|---|---|---|---|
| **R1** — its own rules, unchanged | falsified | **FALSIFIED** | A: falsified · s0: falsified | falsified |
| **R2** — its own rules, unchanged | holds | **UNTESTABLE** | not run | held |
| **R1w** — on Mooncake, balanced affinity and session-sticky within one replica | holds | **HOLDS** | A: holds · s0: holds | not a registered rule then |

## R1 — replicas needed at speeds 0.25×, 0.5×, 1× (— = not met at 8)

The first model's speeds were 1×, 2×, 4×: the two tables are not at the same load.

### toolagent

| policy | A | s0 | first model | hit rate at 4, A |
|---|---|---|---|---|
| round-robin | 5 5 6 | 5 5 6 | 4 5 8 | 0.371 0.371 0.371 |
| least-loaded | 5 5 6 | 5 5 6 | 4 5 7 | 0.371 0.370 0.370 |
| power-of-two | 6 6 6 | 5 5 6 | 4 5 7 | 0.370 0.371 0.369 |
| consistent-hash-prefix | 7 7 7 | 6 6 7 | 5 6 — | 0.430 0.430 0.430 |
| radix-affinity | — — — | — — — | — — — | 0.358 0.358 0.356 |
| session-sticky | 5 5 5 | 5 5 5 | 4 4 6 | 0.423 0.423 0.422 |
| radix-affinity-balanced@16 | 5 5 5 | 5 5 5 | 4 5 6 | 0.419 0.420 0.420 |
| radix-affinity-balanced@1 | 5 5 5 | 5 5 5 | 4 5 6 | 0.419 0.422 0.420 |

Balanced affinity (α 16) against session-sticky — A: 0.25× tie, 0.5× tie, 1× tie · s0: 0.25× tie, 0.5× tie, 1× tie.

### conversation

| policy | A | s0 | first model | hit rate at 4, A |
|---|---|---|---|---|
| round-robin | 6 6 7 | 6 6 7 | 5 5 8 | 0.071 0.071 0.071 |
| least-loaded | 7 6 7 | 6 6 7 | 5 5 8 | 0.070 0.071 0.069 |
| power-of-two | 7 7 8 | 7 7 8 | 5 6 8 | 0.068 0.068 0.069 |
| consistent-hash-prefix | — — — | 8 8 — | 6 7 — | 0.142 0.142 0.143 |
| radix-affinity | — — — | — — — | — — — | 0.050 0.050 0.049 |
| session-sticky | 6 6 7 | 6 6 6 | 5 5 7 | 0.138 0.137 0.138 |
| radix-affinity-balanced@16 | 6 6 6 | 6 6 6 | 4 5 7 | 0.136 0.133 0.133 |
| radix-affinity-balanced@1 | 6 6 7 | 6 6 6 | 5 5 7 | 0.144 0.142 0.144 |

Balanced affinity (α 16) against session-sticky — A: 0.25× tie, 0.5× tie, 1× affinity · s0: 0.25× tie, 0.5× tie, 1× tie.

## R2 — untestable as designed on this model

