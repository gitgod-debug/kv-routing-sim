# What one agent session costs under each routing policy (S1)

Registered before it ran: docs/09-hypotheses.md, 2026-11-24, S1. Made by `scripts/session_cost.py`; the numbers
are in `session_cost.json` beside this file. A derived table: its cells are cells of
H2b on the validated timing model (calibration A), run again and checked against the
stored grids. One GPU type (T4), one model, generated traffic.

A session sends 6,800 prompt tokens in 4 requests. 519 of them are new in this replay: text that no earlier request sent, which no routing can avoid computing. The replay is open-loop, so the model's own earlier replies count as new text; a live engine could still hold them.
Prefilled is what the GPUs computed. Re-prefilled is prefilled minus new: work a shared
cache that never forgot would have saved. Each policy is shown at the fewest T4s that
meet the target. Session-sticky's re-prefill is a one-time cost, the shared prompt
computed once on each further GPU, so its figure per session falls as a trace gets
longer.

## One shared prompt, 2 sessions per second

First-token target (p95 time to first token within 2 s):

| Policy | T4s | Hit rate | Prefilled | Re-prefilled (95% interval) | ₹ per session |
|---|---:|---:|---:|---:|---:|
| session-sticky | 5 | 92.0% | 546 | 27 (7 to 54) | 0.0474 |
| round-robin | 7 | 82.5% | 1,188 | 669 (627 to 718) | 0.0635 |
| radix-affinity-balanced@16 | 10 | 85.9% | 957 | 438 (398 to 483) | 0.0912 |
| least-loaded | 11 | 81.0% | 1,296 | 777 (736 to 824) | 0.0977 |

Two-part target (also p95 time per output token within 100 ms):

| Policy | T4s | Hit rate | Prefilled | Re-prefilled (95% interval) | ₹ per session |
|---|---:|---:|---:|---:|---:|
| session-sticky | 10 | 91.5% | 580 | 61 (27 to 102) | 0.0906 |
| least-loaded | 13 | 80.5% | 1,324 | 805 (760 to 857) | 0.1175 |
| round-robin | 14 | 80.1% | 1,353 | 834 (785 to 889) | 0.1265 |

A policy with no row met the target at no replica count of the grid. Intervals are
percentile bootstrap intervals over sessions. Rupees are counted over a trace's whole
run, which includes the time after the last arrival.

## The three predictions

- **S1a held.** New tokens per session: 518.8 (predicted 500 to 540).
- **S1b held.** Re-prefilled per session, first-token target: session-sticky 27.2 (predicted below 100), round-robin 669.4 (predicted above 500).
- **S1c held.** Ratio of rupees divided by ratio of replicas, two-part against first-token (predicted within 10% of 1): round-robin 0.997, least-loaded 1.018, session-sticky 0.955.

All 195 rows, for every prompt count and load, are in `session_cost.json`.
