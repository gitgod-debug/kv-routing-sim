# T2 — the finding's grid under a target with both halves

Registered in [`configs/tpot_slo.yaml`](../../../configs/tpot_slo.yaml) before it ran; the story and
what it changes are in `docs/09-hypotheses.md`, entries of 2026-10-05.

| File | What it is |
|---|---|
| `grid_A.json`, `grid_B.json`, `grid_C.json` | Written by `scripts/run_t2_grid.py`: for one calibration of the T4, every (prompts, load, policy, replicas, seed) run's p95 time to first token and p95 time per output token. No target is applied in them. |
| `analysis.json`, `report.md` | Written by `scripts/t2_analyze.py`, committed before the grid ran: the replica counts under the first-token target and under the two-part one, the three registered rules, and the reported items. |
| `verification.json` | Written by `scripts/t2_verify_grid.py`: 240 random cells of each grid, run again from a clean checkout. |

⚠️ **The report's first line says "a dirty tree", and that is true of grids B and C.** The three
calibrations ran one after another in one process that started on a clean tree. While B and C were
computing, other files in the working tree were being edited; each grid's header lists the paths.
The process had loaded the simulator before those edits and none of the files is imported by a
grid run — but a header cannot show that. `verification.json` is the evidence instead: 720 of 720
sampled cells, run again from a clean checkout whose `src/` is identical to the registration
commit's, are identical to the last stored digit.

⚠️ **These are the model's counts, not sizing guidance.** T1 — the check of the simulator's
per-token latency against a live engine — did not hold
([`../t1_itl/report.md`](../t1_itl/report.md)), and the registration said beforehand what follows
from that.
