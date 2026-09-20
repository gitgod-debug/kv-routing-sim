# kv-routing-sim — Checkpoints

> **What this file is.** The living to-do list for the whole build. Every task from the approved plan is
> here as a tickable box. **Tick a box only when the thing is done AND verified** — for this project
> "verified" usually means *a number was measured and committed*, not *the code ran*.
>
> **Companion files:** [`docs/00-START-HERE.md`](docs/00-START-HERE.md) explains what everything is and
> why. This file only tracks *where we are*.
>
> Legend: `[ ]` not started · `[~]` in progress · `[x]` done & verified · `[!]` blocked · `[-]` cut

**Last updated:** 2026-09-20 (repo scaffolded: docs 00–09, ADR-001…009, H1–H5 pre-registered, pinned
deps, Phase-0 spike scripts written but **not yet run**)
**Current phase:** Phase 0 — Spike & de-risk · **gate NOT passed; no simulator code may start**

> ## 🆕 2026-09-20 — two decisions recorded, both of which the spec said were owed
>
> **1. Build order.** `kv-routing-sim` is now the **sole critical path**; `colpali-lite` **pauses at its
> passed Phase-0 gate**. This is what the spec's ⛔ box demanded be recorded with a date, and the reason is
> that this project gates the entire application ramp while the other one does not.
> *Accepted cost:* `colpali-lite`'s Indic query generation is wall-clock-bound and its own checkpoints say
> *"do not leave until Phase 4."* Pausing it means that clock is not running. Accepted deliberately.
>
> **2. Calibration hardware.** An **A100/H100 replaces Kaggle dual-T4** (ADR-005). Two of the three rows in
> the spec's 2026-09-20 feasibility risk table are **superseded**; the third (do the `vllm:kv_block_*`
> histograms exist on the installed version?) survives as a Phase-0 checkbox below.

> ### ⭐ Read [`docs/09-hypotheses.md`](docs/09-hypotheses.md) before running any phase that produces a
> number. H1–H5 were committed **before any code existed**. That ordering is the claim — without it this
> project sweeps a grid and reports a table.

---

## Progress at a glance

| Phase | Name | Status | Gate to pass before moving on |
|---|---|---|---|
| **0** | **Spike & de-risk** | `[~]` | Committed raw `/metrics` scrape where `vllm:prefix_cache_hits` **and** `_queries` are present **and moving**, + a two-server coexistence result. ⛔ **blocks all simulator code** |
| 1 | Trace schema + ingestion + generator | `[ ]` | A real log **and** a generated agentic trace both load into one versioned schema; malformed lines log-and-continue with a count |
| 2 | Block-level KV cache model | `[ ]` | Known prefix sequence → known hit/miss pattern, in tests; partial-block rule holds; memory ceiling measured |
| 3 | **Real-Engine Calibration Report** | `[ ]` | Predicted-vs-measured on **≥2 shapes**, ≥3 live runs each, error bars, **residual explained with the `kv_block_*` histograms** — tests **H3 + H4** |
| 4 | Six policies + Pareto sweep | `[ ]` | A curve naming a recommended policy + replica count with a quantified ₹ delta, multi-seed bootstrap CIs — tests **H1 + H5** |
| 5 | **The crossover finding** | `[ ]` | The sharing-ratio × load boundary published as a curve with CIs — tests **H2**. ⛔ never cut |
| 6 | Sim-to-real 2-replica validation | `[ ]` | Predicted vs measured across two MIG instances, or the documented fallback and why |
| 7 | Package · CLI · Space · hardening | `[ ]` | `pip install` + `kvsim replay` + HF Space + "what breaks at 10×/100×" answered with measured numbers |
| 8 | Write-up & launch | `[ ]` | README numbers-first, Show HN, **the `guidellm` PR open** |

### ⛔ Two ordering rules, written down now so a compressed week does not decide them under pressure

1. **The protection order: the live-engine calibration outranks extra policies.** Three policies
   calibrated against a real engine beats six validated only against themselves — the first is an
   infrastructure credential, the second is a spreadsheet. **The simulator is the contribution; the
   calibration is the credential. If only one survives a collapsed week, it is the calibration.**
   MVP under compression = trace ingestion + cache model + **3** policies (round-robin, session-sticky,
   prefix-affinity) + the calibration report.
2. **The upstream PR runs alongside Phases 1–4, not after them.** Its old slot was too late.
   the upstream log starts **today**, seeded by Phase-0/3 calibration
   friction — that friction *is* the spike, it is already on the critical path, and it lands inside
   `vllm-project/guidellm`'s own surface (benchmark harness, metrics parsing, report output, scenario
   config).

---

## Phase 0 — Spike & de-risk

**Goal:** prove the engine behaves as documented and the counters are real, **before** any design depends
on them. The sibling project caught two hard blockers cheaply for exactly this reason; this project had no
Phase-0 equivalent until now.

### Scaffold — done 2026-09-20

- [x] `git init`; `.gitignore` excludes the three planning notes kept beside the build (they are private and
      this repo ships public)
- [x] `pyproject.toml` with pinned deps, CPU-only core, `vllm` isolated in `[calib]` (ADR-009)
- [x] Docs system `docs/00`–`09`, written from `FLAGSHIP_PROJECTS.md` Project 1
- [x] **ADR-001…ADR-009** recorded in [`docs/02-architecture-decisions.md`](docs/02-architecture-decisions.md)
- [x] **H1–H5 pre-registered** in [`docs/09-hypotheses.md`](docs/09-hypotheses.md), before any code
- [x] Record the local hardware baseline — GTX 1650 4 GB / **compute_cap 7.5** / driver 595.71.05 /
      12 cores / 14 GB RAM / 726 GB free / Python 3.13.12 ([`docs/07-environment.md`](docs/07-environment.md))
- [x] Port the prior-art audit into the repo, including the ⛔ **sentences that must never be written**
      ([`docs/04-prior-art.md`](docs/04-prior-art.md))
- [x] Create the upstream log with the target decided (`vllm-project/guidellm`)
- [x] Write `scripts/spike_vllm_metrics.py` + `scripts/spike_two_servers.py` — **written, not run**
- [x] Record the two owed decisions with a date, here and in the project plan

### The spike itself — on the A100/H100 box (see the runbook in `docs/07-environment.md`)

- [ ] Create the local `.venv`, `pip install -e ".[dev]"`, and **record what actually installed** in
      `docs/07-environment.md` (pins are intentions; only an install produces facts)
- [ ] Install `.[calib]` **on the GPU box** and record the **vLLM version that actually installed**
      ⚠️ its resolution has never been attempted — the sibling shipped an extra that turned out not to
      install at all, so this is a real checkbox, not a formality
- [ ] ⛔ **Re-run the metric-name audit against the installed version's own `v1/metrics/loggers.py`**
      (ADR-008). If it is not v0.29.0, do not skip this
- [ ] **Stand up one vLLM replica** with `--enable-prefix-caching`; scrape `/metrics`; confirm
      **`vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` are present AND MOVING**
      (⛔ not `vllm:prefix_cache_hit_rate`, which does not exist)
- [ ] **Commit the raw scrape text**, not just the parsed JSON
- [ ] Run the **prefix-caching off vs on** pair, so the hit-rate delta is visible rather than asserted
- [ ] **Confirm the three `vllm:kv_block_*` histograms appear** under `--kv-cache-metrics-sample`
      ⚠️ **H4 is gated on this.** If they are absent, amend H4 with a date — do not keep the narrative
- [ ] **Confirm two servers can coexist**: MIG if available (record the profile), else the memory-split
      fallback. **Record which path**, because it decides which numbers may be reported (ADR-005)
- [ ] Record GPU, driver, CUDA, MIG availability into `docs/07-environment.md`
- [ ] Log every friction point hit along the way into the upstream log — **this is
      the upstream PR's seed and it is free, because the friction happens anyway**

**Phase 0 exit gate:** a committed `/metrics` scrape from a live engine in which the two prefix-cache
counters are present and increasing, plus a recorded answer to "can two servers coexist, and by which
path."

> ⛔ **NOT PASSED as of 2026-09-20.** Nothing in `src/` may be written until it is. Every assumption in the
> calibration plan — prefix caching behaving as documented, the counters existing and moving, two servers
> coexisting — has **never been executed once**.

---

## Phase 1 — Trace schema, ingestion, and the agentic generator

- [ ] `NormalizedRequest` + `SCHEMA_VERSION` (pydantic). Fields: request id, arrival ts, session id,
      block hashes, prompt/output token counts
- [ ] `hashing.py` — mirror the engine's **hash-chain** rule (a block's hash depends on its predecessors),
      verified against the spike's observed block behaviour, not against the docs alone
- [ ] Adapters: vLLM JSONL · OpenAI-style request log · (SGLang if a sample is obtainable)
- [ ] **Stream, never load.** A trace must not be read into RAM; state the tested ceiling
- [ ] **The agentic generator** — shared system prompt + tool definitions + N turns, with
      **prefix-sharing ratio as an explicit parameter** (this is the H2 axis, so it is built in from day one)
- [ ] Malformed line → log-and-continue, with `skipped_lines` in the results header
- [ ] Bad config → **fail loudly at startup**, never silently mid-sweep
- [ ] Commit generator configs (a generated trace is reproduced from seed + config, never shipped as data)

**Gate:** one real serving log and one generated agentic trace both load into the same versioned schema,
and a corrupted line is survived and counted.

---

## Phase 2 — The block-level KV cache model

- [ ] `block_trie.py` — trie of resident block hashes on plain dicts (ADR-002)
- [ ] `eviction.py` — LRU order + optional TTL + a hard capacity bound derived from a GPU memory config
- [ ] `replica_cache.py` — counters **named to mirror vLLM's own**, so predicted and measured are literally
      the same quantity
- [ ] **Tests that matter most:** known prefix sequence → known hit/miss pattern; the **partial-block rule**
      (sharing 47 of 48 tokens at block size 16 shares **two** blocks, not three)
- [ ] Measure and state the **memory ceiling** at 1M requests on the 14 GB dev box

> ⚠️ **This component is table stakes, not the core IP** — Vidur-canary and Frontier both ship one. **The
> contribution is its *fidelity*** (Phase 3). Do not describe it as the contribution anywhere.

---

## Phase 3 — The Real-Engine Calibration Report *(the credential · tests H3 + H4)*

- [ ] `calib/metrics_scrape.py` — parse the exposition format; **derive** the hit rate from two counters
- [ ] Calibrate on **≥2 workload shapes**, **≥3 live runs each** (a single run has no error bar)
- [ ] Predicted-vs-measured table with error bars — ⛔ **published whatever it says** (H3)
- [ ] **Explain the residual with a measurement**, not a narrative: check the `kv_block_*` histograms
      against H4's directional predictions, and try the capacity-matching intervention
- [ ] The **limits section** — what the simulator gets wrong, one failure mode owned publicly
- [ ] Commit raw scrapes + the notebook/script + pinned deps
- [ ] README section names: engine, **version**, flags, block size, `max_num_seqs`, model, quantization, the
      exact CLI, hardware, and **every `/metrics` field consumed by name**

**Gate:** predicted vs measured on ≥2 shapes with the residual accounted for. ⛔ If H3 fails, downgrade the
sweep's claim from *predictive* to *comparative* — and say so — rather than re-tuning until it matches.

---

## Phase 4 — The six policies and the Pareto sweep *(tests H1 + H5)*

- [ ] Six policies, one file each, behind one interface (ADR-003), **every tie broken from a seeded RNG**
- [ ] `sim/` — replica with `max_num_seqs` admission, fitted prefill/decode cost, fleet event loop
- [ ] Sweep policy × replicas 1–8 on ≥2 traces (one real if obtainable, one generated agentic)
- [ ] **Multi-seed (≥5) + bootstrap CIs, resampled by session/segment** — ⚠️ not by request
      (requests are not independent; see methodology §1)
- [ ] `eval/cost.py` from `configs/cost_model.yaml` — every constant cited and dated, **₹ per 1k
      successful requests**
- [ ] `eval/pareto.py` + `eval/recommend.py` → **one** policy, **one** replica count, a ₹ delta vs
      round-robin
- [ ] Property test: prefix-affinity never scores below round-robin on perfectly-shared prefixes
- [ ] Golden-file test on a small sweep; CI green gate

---

## Phase 5 — The crossover finding *(tests H2 · ⛔ never cut)*

- [ ] Sweep **prefix-sharing ratio as its own axis**, crossed with offered load
- [ ] Locate the boundary where least-loaded overtakes prefix-affinity on **p95 TTFT**; publish it as a
      **curve with CIs**
- [ ] Check H2's three directional sub-predictions (monotonicity in each axis; power-of-two in between)
- [ ] If the effect sits inside the CIs, report **that** — *"no detectable crossover in this region, here is
      the region and the intervals"* is a real result
- [ ] Make it the README headline, in one sentence a reader can repeat without opening the repo

---

## Phase 6 — Sim-to-real, two replicas

- [ ] Enable MIG, create two instances, record the profile (`docs/07-environment.md` has the commands)
- [ ] Two vLLM servers, one arrival stream, the top recommended policy actually deployed
- [ ] Predicted vs measured hit rate across the pair
- [ ] ⛔ **TTFT percentiles only from true MIG isolation** — never from the memory-split fallback (ADR-005)
- [ ] **Fallback, decided in advance:** if 2 replicas prove infeasible, ship a **single-replica calibration
      plus a documented, reproducible account of why**. That beats no calibration, and the failure write-up
      is itself an OSS-PR premise

---

## Phase 7 — Package, CLI, Space, hardening

- [ ] `pip install kv-routing-sim`; `kvsim replay trace.jsonl --policies all --replicas 1-8
      --format json|md|html`; semver + changelog
- [ ] HF Space (free CPU) + docs on Render free
- [ ] ⭐ **"What breaks at 10× and 100×"** as a named README section, answered with **measured** numbers:
      RSS ceiling and sweep wall-clock at 10× trace length; at 100×, which gives out first — block-hash
      memory, the policy inner loop, or the report renderer; at 10× replicas, does cache-model fidelity
      hold or does eviction skew compound? *(These are the questions a design review asks.)*
- [ ] 👥 `CONTRIBUTING.md` + **three genuinely good first issues** (a new trace adapter, a new policy class —
      both are single-file by design), then **review the PRs properly**: real comments, changes requested,
      then merged.

---

## Phase 8 — Write-up and launch

- [ ] README **numbers-first**, with **the credential and the finding in separate sections**
- [ ] Scope discipline in the README from day one: vs `cachetrace` (multi-replica policy choice, not
      single-path linting) · vs **Vidur** (cache-aware *policy* choice, not parallelism/capacity planning) ·
      vs **Frontier** (reads *your* log and *recommends*, rather than being configured by hand)
- [ ] Show HN + X thread, with the engine and version
      **in** the title
- [ ] ⭐ **The `guidellm` PR merged** — see the upstream log
- [ ] Cross-link with `colpali-lite`: two instantiations of one method

---

## 🚧 Honesty gates — the things that make this repo trustworthy

Check every one before anything is published.

- [ ] **No number in the README that was not measured.** Placeholders are marked `TBD — not yet measured`
      and are never quietly replaced by a plausible figure
- [ ] **The predicted-vs-measured table is published even if it is unflattering.** A large explained error
      is a result; a quietly re-tuned model that matches is fraud
- [ ] **No sentence from the ⛔ list** in [`docs/04-prior-art.md`](docs/04-prior-art.md) — especially
      *"0 results"*, *"Vidur has no prefix-cache model"*, *"nothing else validates against a live engine"*,
      and the nonexistent metric name
- [ ] **The claim is stated as an intersection**, with the four competitors named
- [ ] **Every artifact in `artifacts/results/` is regenerable** by a script in `scripts/`
- [ ] **Every cost constant is cited and dated** in a committed config
- [ ] **No TTFT percentile from the memory-split hardware path**
- [ ] **Bootstrap CIs resample sessions/segments, not requests** — and every caption says which
- [ ] **A hypothesis that failed is reported as a finding**, not deleted
- [ ] **No trace containing user prompt text is committed**, ever — hashes only
- [ ] **The upstream PR:** every line reviewed; full test suite run
