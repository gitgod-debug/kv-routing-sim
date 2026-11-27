# Post hoc — does the edge check see the knee?

⚠️ **Written after T1's result was known; the check's threshold (1.5×) was chosen after seeing these numbers.** Nothing here was registered. Five runs of three workloads on one GPU: an indication, not a validation.

T1's five held-out runs, replayed with T1's own function on profile A and again with every timing coefficient 5% larger · generated `2026-10-08T07:30:00+00:00` at `903ecdd` by `scripts/edge_posthoc_t1.py`.

| session · workload | simulator ÷ engine, per token (T1) | p95 TTFT, base → 5% slower | p95 time per token, base → 5% slower | preemptions | flagged |
|---|---:|---|---|---|:---:|
| c2 · K1_half | 1.71× | 3,277 → 17,923 ms (×5.47) | 1,176 → 1,334 ms (×1.13) | 6 → 36 | ⛔ knee |
| c2 · K64_quarter | 1.07× | 1,892 → 2,143 ms (×1.13) | 326 → 393 ms (×1.21) | 0 → 0 | — |
| c3 · K1_half | 1.54× | 3,278 → 17,475 ms (×5.33) | 1,176 → 1,328 ms (×1.13) | 6 → 33 | ⛔ knee |
| c3 · K64_quarter | 1.01× | 1,892 → 2,143 ms (×1.13) | 326 → 393 ms (×1.21) | 0 → 0 | — |
| c3 · T6_K8 | 1.02× | 659 → 751 ms (×1.14) | 274 → 318 ms (×1.16) | 0 → 0 | — |

The check flags a configuration as *at the knee* when a p95 moves by more than 1.5× under the 5% slowdown. It flags both runs on which the simulator was more than 1.5× off, and none of the three on which it was within 1.07×. Any threshold between 1.21× and 5.33× would do the same on these five.
