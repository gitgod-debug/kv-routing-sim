# T2 — the finding's grid under a target with both halves

Registered in [`configs/tpot_slo.yaml`](../../../configs/tpot_slo.yaml) before it ran; the story and
what it changes are in `docs/09-hypotheses.md`, entries of 2026-10-05.

| File | What it is |
|---|---|
| `grid_A.json`, `grid_B.json`, `grid_C.json` | Written by `scripts/run_t2_grid.py`: for one calibration of the T4, every (prompts, load, policy, replicas, seed) run's p95 time to first token and p95 time per output token. No target is applied in them. |
| `analysis.json`, `report.md` | Written by `scripts/t2_analyze.py`, committed before the grid ran: the replica counts under the first-token target and under the two-part one, the three registered rules, and the reported items. |
| `verification.json` | Written by `scripts/t2_verify_grid.py`: 240 random cells of each grid, run again from a clean checkout. |

**The three grids here were made by release 0.2.0, at a clean tree.** Their headers say so, and
`verification.json` adds a second check: 720 of 720 sampled cells, run again from a clean
checkout, are identical to the last stored digit. The grids of release 0.1.2 are kept in
[`../v0.1.2/t2_two_part_slo/`](../v0.1.2/README.md), with this note as it was then. Their grids B
and C were computed while other files were being edited, so their headers say "dirty", and their
own `verification.json` was the evidence for them.

⚠️ **These are the model's counts, not sizing guidance.** T1 — the check of the simulator's
per-token latency against a live engine — did not hold
([`../t1_itl/report.md`](../t1_itl/report.md)), and the registration said beforehand what follows
from that.
