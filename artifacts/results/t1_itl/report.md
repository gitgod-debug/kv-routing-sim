# T1 — the simulator's per-token latency against the engine's

Registered `configs/tpot_slo.yaml` · five held-out runs on one live T4 each (C2 part D, 2026-09-30; C3 part 2, 2026-10-05), replayed at the times their requests were sent · generated `2026-10-05T11:12:53+00:00` at `0be7bec` by `scripts/t1_itl_check.py`.

**T1: DOES NOT HOLD** (registered prediction: holds). It holds only if every factor in the table is ≤ 1.5; the worst is 1.71×.

*Mean inter-token latency* = Σ time between a request's consecutive tokens ÷ the number of such gaps, over every request of the run. Engine: Δsum ÷ Δcount of `vllm:inter_token_latency_seconds`. Simulator: the same sum and count.

| session · workload | engine | simulator, profile A | factor | preemptions engine · simulator | first-token factor p50 · p95 |
|---|---:|---:|---:|---|---|
| c2 · K1_half | 470 ms | 804 ms | 1.71× | 0 · 6 | 1.10× · 1.41× |
| c2 · K64_quarter | 161 ms | 173 ms | 1.07× | 0 · 0 | 1.31× · 1.11× |
| c3 · K1_half | 521 ms | 804 ms | 1.54× | 0 · 6 | 1.01× · 1.25× |
| c3 · K64_quarter | 174 ms | 173 ms | 1.01× | 0 · 0 | 1.35× · 1.21× |
| c3 · T6_K8 | 134 ms | 137 ms | 1.02× | 0 · 0 | 1.38× · 1.18× |

⚠️ **What this cannot show.** A per-token limit of 100 ms is the target T2 sizes on, and 5 of the 5 engine numbers above are beyond it. So this checks the model where the engine was loaded more heavily than a configuration T2 calls acceptable; below the limit it is untested on traffic.

## Reported, not tested

The simulator's per-request time per output token on the same replays (profile A). No engine measurement of it exists yet — the notebooks kept only the mean above:

| session · workload | p50 | p95 |
|---|---:|---:|
| c2 · K1_half | 898 ms | 1,176 ms |
| c2 · K64_quarter | 152 ms | 326 ms |
| c3 · K1_half | 898 ms | 1,176 ms |
| c3 · K64_quarter | 152 ms | 326 ms |
| c3 · T6_K8 | 126 ms | 274 ms |

The same check with the two profiles `kvsim calibrate` wrote in C3 (B: its first run; C: its second), on that session's traffic:

| workload | engine | B | factor | C | factor |
|---|---:|---:|---:|---:|---:|
| K1_half | 521 ms | 785 ms | 1.51× | 891 ms | 1.71× |
| K64_quarter | 174 ms | 179 ms | 1.03× | 188 ms | 1.08× |
| T6_K8 | 134 ms | 135 ms | 1.01× | 140 ms | 1.04× |
