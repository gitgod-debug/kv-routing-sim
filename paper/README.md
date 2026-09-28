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
| §5.1, first sweep | [`artifacts/results/phase4_sweep_t4/report.md`](../artifacts/results/phase4_sweep_t4/report.md) |
| §5.1, H2 and the α sweep | [`artifacts/results/phase5_crossover/report.md`](../artifacts/results/phase5_crossover/report.md) |
| §5.2, Table 2, Figure 1 | `h2b` in [`phase5.json`](../artifacts/results/phase5_crossover/phase5.json), reproduced exactly by [`demo_grid.json`](../artifacts/results/demo_grid.json) |
| §5.3 (post hoc) | the same grid; `docs/09-hypotheses.md`, amendment of 2026-09-28 |
| §5.4 | [`h6_ranking/report.md`](../artifacts/results/h6_ranking/report.md), [`h6_crossover/report.md`](../artifacts/results/h6_crossover/report.md) |
| §5.5, Table 3 | [`realtrace_mooncake/report.md`](../artifacts/results/realtrace_mooncake/report.md); the registration in `docs/09-hypotheses.md` |
| §6, scale | [`artifacts/results/scale/report.md`](../artifacts/results/scale/report.md) |
| §6, the live two-replica test | [`phase6_prereg/predictions.json`](../artifacts/results/phase6_prereg/predictions.json) — **not yet run** |

The 17 references were each checked on 2026-09-28: the eight arXiv papers against their arXiv pages
(title, authors, and venue where arXiv lists one), the rest against their repositories or the
cost model's cited pages.

## Before it goes to arXiv

- [ ] Author affiliation and e-mail (title footnote), and the repository URL once it is public.
- [ ] **Phase 6:** run the notebook, then add its result to §6 — or delete that sentence.
- [ ] **Re-audit the prior art** (`docs/04-prior-art.md` §7, due 2026-10-20): Frontier's router list,
      Vidur's `canary` branch, anything new. Update the "As of" footnote.
- [ ] Check every number against the table above — the author signs for them.
- [ ] Replace "Draft of \today" with the posting date.
- [ ] arXiv endorsement for cs.DC (first submission), then upload `main.tex` + `figures/flip.pdf`.
