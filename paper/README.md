# The preprint — draft

**When Does Prefix-Aware Routing Pay? A vLLM-Calibrated, Pre-Registered Study of KV-Cache Routing
Across Replicas.** Five pages, two columns. Drafted 2026-09-28; due on arXiv by ~Nov 9 (cs.DC).

## Build

```bash
tectonic main.tex          # a single binary that fetches the packages it needs
```

Or upload `main.tex` and `figures/flip.pdf` to Overleaf. The source is ASCII-only and uses standard
packages (`newtx`, `microtype`, `booktabs`, `enumitem`, `caption`, `hyperref`), so arXiv's pdfLaTeX
builds it as well. The figure comes from `python scripts/make_figures.py`, which draws it from the
committed grid; do not edit it by hand.

## Where every number comes from

Nothing in the paper is typed from memory. If a result changes, change it here and in the paper.

| In the paper | Source |
|---|---|
| Abstract, §1 | the sections below |
| §3, cost and service level | [`configs/cost_model.yaml`](../configs/cost_model.yaml) — each constant with its source and date |
| Table 1, §4 | [`artifacts/results/phase3_kaggle_t4/calibration_report.md`](../artifacts/results/phase3_kaggle_t4/calibration_report.md) |
| §4, capacity 1,537 blocks | [`configs/profiles/t4-qwen3-4b-vllm0.30.0.json`](../configs/profiles/t4-qwen3-4b-vllm0.30.0.json); the 56% over-estimate: `CHECKPOINTS.md`, Phase 4 |
| §4, two-replica 1,088 tokens | [`artifacts/results/phase0_v2_kaggle_t4/`](../artifacts/results/phase0_v2_kaggle_t4/) |
| §4 "Two live replicas", Table 2 | [`phase6_kaggle_t4/report.md`](../artifacts/results/phase6_kaggle_t4/report.md) (`scripts/phase6_analyze.py`; two runs) |
| §4, the packaged calibration command | [`c1_kaggle_t4/report.md`](../artifacts/results/c1_kaggle_t4/report.md) (`scripts/c1_analyze.py`) |
| §4 "A context-aware timing model" | [`c2_kaggle_t4/report.md`](../artifacts/results/c2_kaggle_t4/report.md) (`scripts/c2_analyze.py`, committed before the run); the post-hoc Phase 6 replay: [`posthoc_phase6.md`](../artifacts/results/c2_kaggle_t4/posthoc_phase6.md) |
| §5.1, first sweep | [`artifacts/results/phase4_sweep_t4/report.md`](../artifacts/results/phase4_sweep_t4/report.md) |
| §5.1, H2 and the α sweep | [`artifacts/results/phase5_crossover/report.md`](../artifacts/results/phase5_crossover/report.md) |
| §5.2, Table 3 (first timing model) | `h2b` in [`phase5.json`](../artifacts/results/phase5_crossover/phase5.json), reproduced exactly by [`demo_grid.json`](../artifacts/results/demo_grid.json) |
| §5.3, Table 4, Figure 1 (validated model) | [`h2b_c2/report.md`](../artifacts/results/h2b_c2/report.md) (`scripts/run_h2b_c2.py`, registered in `configs/h2b_c2.yaml`), reproduced exactly by [`demo_grid_ctx.json`](../artifacts/results/demo_grid_ctx.json) |
| §5.4 (post hoc) | the two grids; `docs/09-hypotheses.md`, amendment of 2026-09-28 and entry of 2026-09-30 |
| §5.5 | [`h6_ranking/report.md`](../artifacts/results/h6_ranking/report.md), [`h6_crossover/report.md`](../artifacts/results/h6_crossover/report.md) |
| §5.6, Table 5 — Mooncake (R1) | [`realtrace_mooncake/report.md`](../artifacts/results/realtrace_mooncake/report.md); the registration in `docs/09-hypotheses.md` |
| §5.6, Table 5 — AgentX (R2) | [`realtrace_agentx/report.md`](../artifacts/results/realtrace_agentx/report.md); the probes and rules in [`configs/realtrace_agentx.yaml`](../configs/realtrace_agentx.yaml) |
| §6, scale | [`artifacts/results/scale/report.md`](../artifacts/results/scale/report.md) |

The 19 references were each checked on 2026-09-28: the eight arXiv papers against their arXiv pages
(title, authors, and venue where arXiv lists one), the rest against their repositories, the AgentX
dataset card, or the cost model's cited pages.

## Before it goes to arXiv

- [ ] Author affiliation and e-mail (title footnote), and the repository URL once it is public.
- [x] **Phase 6:** ran twice (2026-09-28); its result is in §4, Table 2, the abstract and §6.
- [x] **C2 and the re-test** (2026-09-30): §4's timing paragraph and §5.3; the abstract and conclusion now lead with the reversal.
- [ ] **Re-audit the prior art** (`docs/04-prior-art.md` §7, due 2026-10-20): Frontier's router list,
      Vidur's `canary` branch, anything new. Update the "As of" footnote.
- [ ] Check every number against the table above — the author signs for them.
- [ ] Replace "Draft of \today" with the posting date.
- [ ] arXiv endorsement for cs.DC (first submission), then upload `main.tex` + `figures/flip.pdf`.
