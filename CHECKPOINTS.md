# kv-routing-sim — Checkpoints

> **What this file is.** The living to-do list for the whole build. Every task from the approved plan is
> here as a tickable box. **Tick a box only when the thing is done AND verified** — for this project
> "verified" usually means *a number was measured and committed*, not *the code ran*.
>
> **Companion files:** [`docs/00-START-HERE.md`](docs/00-START-HERE.md) explains what everything is and
> why. This file only tracks *where we are*.
>
> Legend: `[ ]` not started · `[~]` in progress · `[x]` done & verified · `[!]` blocked · `[-]` cut

**Last updated:** 2026-10-06 (**C4 RAN on a live T4 — seven of nine rules hold.** The per-token
model is within 7% of the engine around the 100 ms limit, and T1's failure at the knee has its
cause: the fitted line is 12–15% too slow at very large batches. ⛔ Two predictions failed: one
calibration's own spread covers the next one's GPU count in 82% of cases, not 90; and with both
halves of the target prefix routing is one GPU below session-sticky at more loads than the rule
allowed. **And a prior-art check on 2026-10-05 found what three earlier ones had missed:** the
routing result was published first (SMetric, LMetric), and NVIDIA's Dynamo ships a simulator for
the same question. The "intersection nobody occupies" is withdrawn; `docs/04-prior-art.md` says
what can still be claimed. Before that, on 2026-10-05: **T1 and T2 RAN — the per-token half of the
SLO.** T1 ⛔ did not
hold: the simulator's per-token latency is within 7% of the engine off the knee and 1.5–1.7× too
slow at it. T2 ✅ all three rules on all three calibrations: under a target with both halves
session-sticky is still best or tied with 1 and 4 prompts, never worse than round-robin or
least-loaded, and needs 1.93–2.00× the first-token count — the model's answer, unchecked where it
lives. `kvsim calibrate` now repeats every cell and `kvsim replay` shows how firm a count is.
Earlier the same day: **C3 RAN — all three rules hold, as predicted** — `kvsim calibrate`
works on a real T4 and its profile predicts unseen traffic within 1.41×. **And the same run narrowed
the finding:** three calibrations of the same T4 agree on a GPU count exactly at 102 of 144 cells and
within one at 138 — and H2b's verdict of 2026-09-30, decided by one replica at one point, reverses on
the other two. Earlier the same day **C3's first attempt did not start** — Kaggle's software image
changed and vLLM could not be imported; nothing was measured. The notebook is amended with a
pre-flight that repairs it, checked on Kaggle's image with a probe; nothing registered changed. GPU
runs now go through Kaggle's API, `scripts/kaggle_run.py`. 2026-10-01: **C3 registered** — the
context-aware `kvsim calibrate`'s first run on a
real GPU; before it, a pre-check on C2's data added one measurement to the command. 2026-09-30: C2
ran — all three predictions true; the finding re-tested on the validated timing model and its "route
by prefix" half FALSIFIED. Before that: H6, R1 and R2, Phase 6 and C1, the demo, the preprint draft,
`kvsim calibrate`, a wheel that works outside a clone. 269 tests green. Phase 3 PASSED on 2026-09-22 — the block-pool model reproduced a live
vLLM under eviction pressure to the token, against predictions committed before the run)
**Current phase:** ✅ **0** · ✅ **2b** · ✅ **3** · ✅ **5, the finding — changed 2026-09-30, narrowed
2026-10-05:** *keep each user on one GPU* holds on three calibrations of the T4 (fewest T4s or tied at
every load with 1 and 4 prompts; never more than round-robin or least-loaded); the second half
(*then route by prefix*) was falsified on calibration A and holds on B and C — **not settled**; every
count ± one · 🔨 **4** — H6 and
R1/R2 ran on the first, context-free timing model (not yet re-run) · ✅ **6** — hit rates on two live
T4s within 1.4% (run twice); latency explained by C2 · ✅ **C2** — context, not CPU; the context-aware
model predicts held-out traffic within 1.41× · 🔨 **7** — demo on the validated grid, 10×/100× done,
`kvsim calibrate` proven on a real T4 in both its forms (C1, C3); PyPI left · 🔨 **8** — the preprint
(optional; ⚠️ it predates C3 and the prior-art check of 5 Oct: do not publish as it is) ·
✅ **C4** (5 Oct, a live T4) — 7 of 9 rules hold: the per-token model is within 7% around the
limit; ⛔ one calibration's spread covers the next one's count in 82% of cases, not 90 ·
⛔ **prior art** (5 Oct) — the routing result was published first (SMetric, LMetric) and NVIDIA
ships a simulator for the question; the "intersection" claim is withdrawn.
▶ **Next:** R3, registered first — the two published policies and real sessions of unequal size,
on a public production trace. Then H6 and R1/R2 on the validated model; the two-part target as
the default; a warning when a workload is at the knee.

> ## ⚠️ 2026-09-21 — the deployment plan was based on a stale fact. Corrected; see **ADR-011**.
> Every deployment note here assumed a **free Gradio Space on CPU Basic**. Hugging Face now requires a
> **paid plan (PRO, $9/month personal) to create a Gradio or Docker Space** — only Static Spaces are
> free. Phase 7's demo was therefore **undeliverable as written**, and nobody had checked.
>
> ✅ **The demo is now a STATIC page driven by a precomputed grid**, on GitHub Pages and a free Static
> Space. It is the better answer regardless of price: it **never sleeps** (the cold-start defect this
> plan had already flagged and could only hedge around — and which PRO would not fix either, since a
> paid Space still sleeps after 48 h), and a precomputed grid is **reproducible from a committed
> script** where a live app is not.
>
> 📏 **Method lesson, and it is the same one that killed the Aug-25 OSS-PR target:** platform terms and
> free tiers change, and a plan that names a specific free tier has a **shelf life**. Re-check every
> hosting assumption the week before it is used — not the month it was written.

> ## 🔨 2026-09-20 — the simulator is built. What the gate actually blocked, and what it did not.
>
> The Phase-0 gate exists to stop **unverified engine behaviour** being baked into the model. That is
> three numbers, not the whole build:
>
> | Blocked by the gate | Built now |
> |---|---|
> | the prefill/decode cost coefficients | the trace schema, hashing, adapters, the agentic generator |
> | the KV-cache block capacity | the block cache, LRU eviction, stop-at-first-miss |
> | whether our hash-chain rule matches the engine's | 7 routing policies, the event loop, sweeps, bootstrap CIs, the cost model, the CLI |
>
> The three live in `sim/cost_model.py` behind `CostProfile.calibrated = False`, and the reporting
> layer **refuses in code** to present an uncalibrated run as a prediction — `RunResult.summary()`,
> `Recommendation.as_dict()` and the CLI all carry the tier warning (ADR-010). So the gate is
> enforced by the software rather than by remembering to be careful.
>
> **73 tests pass. `ruff` clean with `E501` enforced.** Three real bugs were caught by writing them:
> a router that modelled block ownership as exclusive and therefore thrashed (scoring a *lower* hit
> rate than round-robin); an `or`-chain that treated timestamp `0` as missing and silently dropped
> the first request of every relative-timestamped trace; and a trace writer that emitted unsorted
> records, so the adapter clamped early arrivals to zero.

> ## 🆕 2026-09-20 — two decisions recorded, both of which the spec said were owed
>
> **1. Build order.** `kv-routing-sim` is now the **sole critical path**; `colpali-lite` **pauses at its
> passed Phase-0 gate**. This is what the spec's ⛔ box demanded be recorded with a date, and the reason is
> that this project gates the entire application ramp while the other one does not.
> *Accepted cost:* `colpali-lite`'s Indic query generation is wall-clock-bound and its own checkpoints say
> *"do not leave until Phase 4."* Pausing it means that clock is not running. Accepted deliberately.
>
> **2. Calibration hardware.** ~~An A100/H100 replaces Kaggle dual-T4~~ — ⛔ **REVERSED 2026-09-22:
> Kaggle 2x T4 is the hardware for this project; A100/H100 is not available and is not in the plan.**
> Sufficient, and measured as such: cache accounting and routing are hardware-independent; latency and ₹
> figures are published as T4 figures. Phase 3 uses **Qwen3-4B** (8B does not fit one T4). ADR-005.

> ### ⭐ Read [`docs/09-hypotheses.md`](docs/09-hypotheses.md) before running any phase that produces a
> number. H1–H5 were committed **before any code existed**. That ordering is the claim — without it this
> project sweeps a grid and reports a table.

---

## Progress at a glance

| Phase | Name | Status | Gate to pass before moving on |
|---|---|---|---|
| **0** | **Spike & de-risk** | `[x]` | Committed raw `/metrics` scrape where `vllm:prefix_cache_hits` **and** `_queries` are present **and moving**, + a two-server coexistence result. ⛔ **blocks all simulator code** |
| 1 | Trace schema + ingestion + generator | `[x]` | A real log **and** a generated agentic trace both load into one versioned schema; malformed lines log-and-continue with a count |
| 2 | Block-level KV cache model | `[x]` | Known prefix sequence → known hit/miss pattern, in tests; partial-block rule holds; memory ceiling measured |
| **2b** | **Cache model = vLLM's block pool** 🆕 | `[x]` | Every eviction behaviour cites a line of vLLM's source; naive LRU kept as the ablation; both still reproduce v1/v2 exactly |
| 3 | **Real-Engine Calibration Report** | `[x]` ✅ exact on W1 + W2, +0.3% on W3 | Predicted-vs-measured on **≥2 shapes**, ≥3 live runs each, error bars, **residual explained with the `kv_block_*` histograms** — tests **H3 + H4** |
| 4 | **Seven** policies + Pareto sweep | `[~]` H1 ⛔ · H5 1 of 4 · H6 ranking ⛔, crossover ✅ · α swept · R1 (Mooncake) ⛔ · R2 (AgentX) ✅ · property + golden tests ✅ — per-cell CIs left | Tests **H1 + H5 + H6** |
| 5 | **The crossover finding** | `[x]` H2 ⛔ · H2b ✅ on the first timing model, ⛔ on the validated one (2026-09-30) — the headline changed | The sharing-ratio × load boundary published as a curve with CIs — tests **H2**. ⛔ never cut |
| 6 | Sim-to-real 2-replica validation | `[x]` ✅ HOLDS, run twice (2026-09-28): hit rates within 1.4%, S2 in order · ⚠️ timing 5–222× optimistic | Predicted vs measured across two MIG instances, or the documented fallback and why |
| 7 | Package · CLI · static demo · hardening | `[~]` demo page + grid + 10×/100× ✅ · `kvsim calibrate` ✅ on a real T4 (C1) · PyPI left | `pip install` + `kvsim replay` + a **static** demo page (ADR-011) + "what breaks at 10×/100×" answered with measured numbers |
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

### The spike itself — on the Kaggle 2x T4 box (see the runbook in `docs/07-environment.md`)

> ✅ **Done — see the PASSED record below.** The spike ran as the notebook (v1 2026-09-21, v2
> 2026-09-22), not as the shell scripts. The boxes below are that original plan, kept as written.

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

> ### ⭐ HOW to run it — the box is a Jupyter environment, so use the notebook
> **[`notebooks/phase0_spike_vllm.ipynb`](notebooks/phase0_spike_vllm.ipynb)** — one self-contained
> file: no repo checkout, no dataset, no HF token, only `vllm` + the standard library. Run it top to
> bottom, then copy `phase0_results/` into `artifacts/results/` and commit it **including the raw
> `.txt` scrapes**.
>
> **Model: `Qwen/Qwen3-0.6B`** for the spike (ungated — verified 2026-09-21; ~1.5 GB; 112 KiB of KV per
> token) and **`Qwen/Qwen3-8B`** for the Phase-3 calibration (also ungated; 144 KiB/token).
> ⛔ **Not `meta-llama/*` — verified `gated=manual`**, i.e. a licence click and a token for no benefit.
>
> The notebook's logic was **dry-run against a mock engine across five failure scenarios** before
> handover, including the two that matter: **flat counters correctly FAIL the gate**, and the
> **token-unit trap is detected** without failing it. See [`notebooks/README.md`](notebooks/README.md).
>
> *`scripts/spike_vllm_metrics.py` and `scripts/spike_two_servers.py` remain for a box with a shell —
> same checks, also stdlib-only.*

**Phase 0 exit gate:** a committed `/metrics` scrape from a live engine in which the two prefix-cache
counters are present and increasing, plus a recorded answer to "can two servers coexist, and by which
path."

> ### ✅ PASSED 2026-09-21 — vLLM 0.29.0 · Qwen/Qwen3-0.6B · Kaggle 2x Tesla T4
> Evidence: [`artifacts/results/phase0_kaggle_t4/`](artifacts/results/phase0_kaggle_t4/) — the verdict
> JSON, **both raw `/metrics` scrapes**, both server logs, and the executed notebook with its outputs.
>
> | Question | Answer | How we know |
> |---|---|---|
> | Counters present? | ✅ yes | metric-name audit of 0.29.0's own source; both in the scrape |
> | Counters **move**? | ✅ yes | +33,344 queried, +32,768 hit |
> | `prefix_cache_hit_rate` absent? | ✅ absent | ADR-008 confirmed |
> | ⭐ **Blocks or tokens?** | ⛔ **TOKENS** | vLLM's own HELP text **and** 512 sent → +512, not +32 |
> | Partial trailing block? | ⭐ in `queries`, never in `hits` | each 513-token request hit exactly 512 |
> | Does the mechanism work? | ✅ **513x less prefill** | `prompt_tokens_by_source{local_compute}`: 64 vs 32,832 |
> | Our KV arithmetic right? | ✅ to 0.11% | engine: 114,819 B/token · `capacity.py`: 114,688 |
> | `kv_block_*` histograms? | ⚠️ **not tested** | wrong flag — `--kv-cache-metrics` enables them (ADR-008 amendment) |
> | Two servers coexist? | ⚠️ **not tested properly** | both died on `max_model_len` 40,960 — and the box had **two GPUs**, which v1 never used |
>
> ### ✅ v2 — 2026-09-22 — vLLM **0.30.0**, Qwen3-0.6B, Kaggle 2x T4 — every open item closed
> Evidence: [`artifacts/results/phase0_v2_kaggle_t4/`](artifacts/results/phase0_v2_kaggle_t4/).
>
> | v1 left open | v2 answer |
> |---|---|
> | Reproducible? | ✅ **33,344 / 32,768 / 576 again, identically** — on a different vLLM version, a different day, and a **different physical machine** (other GPU UUIDs) |
> | ⭐ Does the sim predict a real 2-replica run? | ✅ **EXACT.** Round-robin over two T4s: **1,088** computed, hit rate **0.966862** — both predicted and committed before the run. Each replica paid for exactly one cold prefix (544 each) |
> | Two replicas coexist? | ✅ one server per physical T4 |
> | `kv_block_*` histograms? | ✅ present with `--kv-cache-metrics` — ⚠️ but they record *recycled* blocks, not only evictions (below) |
> | Caching on vs off | ✅ **513x** less prefill · TTFT **41.7 → 75.1 ms** mean (T4) |
>
> **Three findings v2 surfaced that change Phase 3:**
> 1. ⭐ **vLLM's block pool is not an LRU trie** — it frees tail-first, pins in-use blocks, reuses uncached
>    blocks LIFO, and caches decode blocks. Invisible on v1/v2 (the cache never filled), decisive under
>    pressure. Read from vLLM's source, not guessed. **ADR-002 amendment.**
> 2. ⚠️ **The eviction histograms over-count.** `on_block_evicted` fires before the hash check, so the 64
>    samples v2 recorded are recycled tail blocks, not evictions. **H4 amended**; logged upstream as
>    **upstream finding #1** — our first real friction entry, with a repro.
> 3. **Capacity must be read per run.** `capacity.py` overestimates by 7–12.5%, and two launches with
>    identical flags differed by 4.8% (94,496 vs 99,024 tokens).
>
> Also: 513x less prefill bought only **1.8x** lower TTFT — a fixed floor of ~40 ms dominates a 0.6B model
> on T4. That is a real input to the crossover: the cheaper the prefill, the less locality is worth. Another
> reason Phase 3 uses the 4B.
>
> **What it changed in the code:** the cache model counted blocks over full blocks only; vLLM counts
> tokens over *all* prompt tokens. Fixed, and **the model now reproduces the engine's counters to the
> token** — 33,344 / 32,768 / 576 — pinned by four regression tests (77 tests green).
>
> ⚠️ **Three of v1's secondary verdicts were wrong, and all three were notebook bugs, not findings.**
> `caching_on_beats_off: false` compared an undefined hit rate (the real answer is 513x);
> `kv_block_histograms: false` used the wrong flag; `two_servers_coexist: false` was a
> `max_model_len` config error on a box with a spare GPU. **Notebook v2 fixes all three** — and was
> dry-run across three hardware scenarios, which surfaced a fourth bug (a stale server on the port
> could mask a failed start) before it could cost a run.
>
> **Hardware: Kaggle 2x T4 — and that is the project's hardware** (decided 2026-09-22). The log
> confirms what T4 means: no bf16 (fp16), no FlashAttention-2, attention on TRITON_ATTN. Everything
> above is hardware-independent. TTFT and cost figures from this box are **T4 figures**, labelled so.

---

## Phase 1 — Trace schema, ingestion, and the agentic generator

- [x] `NormalizedRequest` + `SCHEMA_VERSION` (pydantic). Fields: request id, arrival ts, session id,
      block hashes, prompt/output token counts
- [~] `hashing.py` — hash-chain rule implemented and unit-tested (blake2b for cross-process
      determinism; partial trailing blocks never shared). ⛔ **Still owed: verification against the
      engine's OBSERVED behaviour.** Right now it matches the documentation, which is exactly the
      thing `Mike-Ew` did
- [x] Adapters: vLLM JSONL · OpenAI-style request log · (SGLang if a sample is obtainable)
- [x] **Stream, never load.** A trace must not be read into RAM; state the tested ceiling
- [x] **The agentic generator** — with **prefix-sharing ratio as an explicit parameter**. ⭐ It reports
      **cross-session sharing** and **total block reuse** SEPARATELY: within-session history reuse is
      captured by session-stickiness for free, so using total reuse as the H2 axis would hide the
      entire effect. It also reports what it ACHIEVED, not what was requested
- [x] Malformed line → log-and-continue, with `skipped_lines` in the results header
- [x] Bad config → **fail loudly at startup**, never silently mid-sweep
- [ ] Commit generator configs (a generated trace is reproduced from seed + config, never shipped as data)

**Gate:** one real serving log and one generated agentic trace both load into the same versioned schema,
and a corrupted line is survived and counted.

---

## Phase 2 — The block-level KV cache model

- [x] `block_trie.py` (`BlockCache`) — resident block hashes on plain dicts (ADR-002), LRU order, a hard
      capacity bound derived from a GPU memory config, and counters **named to mirror vLLM's own** (in
      tokens), so predicted and measured are literally the same quantity. *(Planned as three files;
      `eviction.py` and `replica_cache.py` were never separate modules. No TTL was built.)*
- [x] **Tests that matter most:** known prefix sequence → known hit/miss pattern ✅; the
      **partial-block rule** ✅; **stop-at-first-miss** with stranded blocks still occupying capacity ✅
- [ ] Measure and state the **memory ceiling** at 1M requests on the 14 GB dev box
      *(partial: `NormalizedRequest` measured at 159 B/instance vs 1,215 B for a pydantic model —
      152 MiB vs 1,159 MiB at 1M. The cache's own ceiling is still unmeasured)*

> ⚠️ **This component is table stakes, not the core IP** — Vidur-canary and Frontier both ship one. **The
> contribution is its *fidelity*** (Phase 3). Do not describe it as the contribution anywhere.

---

## Phase 2b — Rebuild the cache model as vLLM's block pool *(CPU-only · added 2026-09-22)*

**Why this comes before Phase 3:** v2 proved the accounting exact, but vLLM's source shows our eviction
model is structurally different from the engine's (ADR-002 amendment). Calibrating a model already known
to be wrong wastes GPU time and produces a residual nobody can interpret.

- [x] ✅ **`cache/block_pool.py`** — `cache_model="vllm"`, now the **default**. Every behaviour cites
      vLLM 0.30.0: reference counts · tail-first free · uncached LIFO / cached FIFO · partial and decode
      blocks in capacity · decode blocks hashed · the lookup cap (a fully cached prompt recomputes its
      last block) · first-lookup-only counters. **Plus the scheduler rules it needs to be usable**:
      memory-based admission (a request that does not fit stops admission), FCFS preemption of
      `running[-1]`, no admission in a step that preempted — each cited to `scheduler.py`
- [x] ✅ `cache_model="lru"` kept unchanged as the **ablation baseline** — except the lookup cap, which
      is a lookup rule, so both apply it and the ablation isolates eviction
- [x] ✅ Eviction events by vLLM's **own** rule, recycles of uncached blocks included, last 4 accesses
- [x] ✅ The replica **releases** a request's blocks when it finishes; preempted ones keep their chain
- [x] ✅ **Regression, through the full simulator, both models:** v1/v2's **33,344 / 32,768 / 576** and
      **1,088 / 0.966862** exactly. ⭐ And the vLLM model reproduces the engine's **eviction histograms
      exactly — 64, and 31 + 31 per replica**, idle = lifetime, zero reuse gaps — from source-derived
      rules alone, nothing fitted. The lru model cannot produce them at all
- [x] ✅ Invariants checked after **every step** under pressure: no block double-booked, **none ever
      stranded** (a property the source implies; plain LRU does strand, briefly)
- [x] ⭐ **Pre-register the Phase-3 predictions** — done in the Phase-3 build (2026-09-22). Capacity turned
      out not to need reading per run: `--num-gpu-blocks-override` makes it a controlled input, which
      the notebook verifies from the engine's own log. See Phase 3

> ### 📏 What building it MEASURED — sim vs sim, uncalibrated profile, 2026-09-22
> 1. ⛔ **The H4 amendment's prediction was wrong.** Eviction ORDER is worth ≤ 0.2 points of hit rate;
>    CAPACITY accounting (decode + partial blocks) is worth up to ~2 points the other way and dominates.
>    Plain LRU **over**-predicts — H4's original direction. See the second H4 amendment.
> 2. ⭐ **Hit rate barely moves; tail latency moves enormously.** On the T4/Qwen3-4B-shaped profile
>    (2,736 blocks, 900 agentic requests, 4 replicas): hit rate differs by ≤ 1.6 points, but p95 TTFT
>    is **1.8× (round-robin) to 8.5× (session-sticky)** higher under the vLLM model, with **147–190
>    preemptions per run** that the lru model never has. The lru model admits everything and never
>    waits for memory, so **its TTFT under memory pressure is unrealistically optimistic.** Absolute
>    milliseconds mean nothing until Phase 3 — the ratio is the finding.
> 3. **H2 therefore needs the vLLM model.** The crossover is defined on p95 TTFT, and TTFT under
>    pressure is dominated by admission control and preemption, which only this model has.
> 4. Cost: **~6× slower** per run (165–236 ms vs 28–82 ms on 900 requests). Fine for Phase 3; the
>    demo grid goes from ~5 to ~30 minutes.
> 5. The early observation survives: session-sticky still has the best hit rate and p95 under both.

---

## Phase 3 — The Real-Engine Calibration Report *(the credential · tests H3 + H4)*

> ### ✅ 2026-09-22 (evening) — IT RAN. Exact on both sequential shapes; +0.3% on the concurrent one.
> Kaggle T4, vLLM 0.30.0, Qwen3-4B, 511 usable blocks. **The vllm model matched every pre-registered
> count in all 6 sequential runs** — hits, computed tokens, all three `kv_block_*` histogram counts, and
> all 120 requests' `cached_tokens` — and the three runs of each workload were identical. Plain LRU
> over-predicted W1 by exactly the 11.90 points it was predicted to. The engine refuted the +1-token
> alternative. W3, predicted by a timing profile fitted without it: hit rate **+0.3%**, p95 TTFT
> **−6.7%**, 24 vs 27 preemptions (LRU: +19.7% and −73%, zero preemptions).
> [`calibration_report.md`](artifacts/results/phase3_kaggle_t4/calibration_report.md) ·
> [`posthoc.md`](artifacts/results/phase3_kaggle_t4/posthoc.md) (labelled post-hoc) · H3 ✅ · H4 ✅
>
> ⚠️ **The weak part is timing, and it is owned, not tuned:** decode fit R² 0.58 (one sequence decodes
> *slower* than two to eight on this T4), and the step model double-charges the fixed per-pass cost when
> a step prefills (fitted prefill base −21 ms, clamped). Neither touches a cache count.
>
> *The build log below is kept as written before the run.*
>
> ### 🔨 2026-09-22 — the experiment is built, dry-run and pre-registered. It has NOT run yet.
> One config ([`configs/phase3_workloads.yaml`](configs/phase3_workloads.yaml)) → one script
> (`scripts/phase3_prepare.py`) → the trace, **both models' predictions** (committed:
> [`artifacts/results/phase3_prereg/predictions.json`](artifacts/results/phase3_prereg/predictions.json)) and
> the notebook that tests them — all stamped with one fingerprint, so none can drift from the others.
> **Designed to discriminate:** an experiment where both models predict the same number cannot say
> which one is right.
>
> | Workload (120 requests, Qwen3-4B, 511 usable blocks) | `vllm` model | `lru` model | Role |
> |---|---|---|---|
> | **W2** — sequential, 1-token outputs | hit **0.6090** · 22,664 computed | **0.6073** · 22,760 | control: the accounting alone |
> | **W1** — sequential, 64-token outputs | hit **0.4883** · 29,656 computed | **0.6073** · 22,760 | **11.9 points apart** — decode blocks crowd the cache, or they don't |
> | **W3** — concurrent, at arrival times | preemptions **> 0** (15 / 20 / 25 across a 4× timing bracket) | **0** preemptions | admission control and preemption |
>
> Sequential counts are **exact integers**, checked timing-independent (identical under a 10× slower
> profile), and include the three `kv_block_*` histogram counts and every request's `cached_tokens`.
> W2 also carries a pre-registered **alternative** — the engine committing one token more per request
> than its source says (hit 0.6026) — so an off-by-one is diagnosed by the run, not argued afterwards.
>
> **Three facts read from vLLM 0.30.0's source while building it** — each now an audit line the notebook
> checks against the *installed* engine before sending anything:
> 1. ⚠️ **Async scheduling is ON by default** (`config/vllm.py` resolves `None` to `True`); our simulator
>    schedules synchronously. For one-at-a-time requests both end each request with *prompt + output − 1*
>    tokens committed (`AsyncScheduler._update_request_with_output` catches up before the free), so W1/W2
>    do not depend on it. **W3 may** — named in the prereg as a simulator assumption.
> 2. **A cache query is recorded at admission, not at lookup** (`scheduler.py`: *"Record at admission so
>    unscheduled lookups are not counted"*): a request that waits for memory counts once. The model
>    already did this.
> 3. **`/reset_prefix_cache` also clears the block-metrics collector** (`block_pool.py`), so a reset engine
>    is, for every predicted count, a fresh one — nine runs need one engine start, not nine.
>
> **Dry-run** (`scripts/phase3_dryrun.py`, ~6 min): the notebook, unmodified, against a mock engine — a real
> HTTP server backed by our own vLLM model — in four scenarios. **A** the real setup: the vllm model exact
> on all 6 sequential runs, the lru model visibly off (−11.90 pts on W1), W3 evaluated, zip written.
> **B** no reset endpoint: falls back to a restart per run. **C** the override ignored: stops at the
> capacity check. **D** a stale server on the port: refuses to launch. ⛔ **D found a real bug** —
> checking "our process is alive" before `/health` narrows the stale-server race but does not close it
> (the new process has not died yet on the first check). The notebook now refuses to launch while
> anything listens on the port.
>
> **The analysis is written too** (`scripts/phase3_analyze.py`) and was run end to end on the dry-run's
> output — so the report's shape, the timing fit and the W3 procedure were fixed before any real number
> existed. On the mock it recovered the mock's decode cost (0.67 + 2.51 ms × batch, true 0.5 + 2.5) and
> predicted W3 within 1–3% on hit rate and TTFT.

- [x] ⭐ **Model: `Qwen/Qwen3-4B`** on one T4 — not 8B, whose 15.26 GiB of fp16 weights exceed the card.
      4B's KV cache is architecturally identical to 8B's (144 KiB/token), which is what matters here
- [x] ~~Read the true KV block count from each run's log~~ — **superseded:** capacity is now a
      *controlled* input (`--num-gpu-blocks-override 512`, one of them vLLM's null block), and the
      notebook stops unless the engine's own log reports exactly 8,192 tokens
- [x] Profile identity recorded — `gpu`, `model`, `engine_version="0.30.0"` (was 0.29.0: the pin moved),
      written into `fitted_profile.json` by the analysis
- [x] Calibrate against **`prompt_tokens_by_source{local_compute}`** as well as the hit rate — compared
      counter by counter, with `prefix_cache_queries`, `_hits`, `num_preemptions` and the histogram counts
- [x] `/metrics` parsing — in the notebook (stdlib only: it runs where the repo is not) and in
      `phase3_analyze.py` (histogram buckets). The hit rate is **derived from two counters** everywhere.
      A `calib/metrics_scrape.py` package module waits for `kvsim calibrate` — ✅ built 2026-09-28 as
      `calib/calibrate.py`, the same stdlib parsing
- [x] **≥2 shapes, ≥3 live runs each, one overflowing the cache** — W1 + W2 × 3, plus W3 × 3; the trace
      needs ~1,000 blocks against 511
- [x] Traces generated **locally by our generator**, embedded in the notebook and fingerprinted — never
      re-implemented inside it
- [x] Open-loop and deterministic (`temperature=0`, `ignore_eos`, exact output lengths checked per
      request). Closed-loop decode-block reuse stays a known, unmeasured limitation (ADR-002, gap 4)
- [x] ⭐ **Predictions pre-registered and committed before the run** — with the analysis procedure and
      the decision rules for every outcome (`predictions.json`, `analysis_procedure`, `decision_rules`)
- [x] ✅ **Ran it** (Kaggle T4, 2026-09-22, ~40 min) — `artifacts/results/phase3_kaggle_t4/`, raw scrapes,
      server log, `server_info.json` and the executed notebook committed
- [x] ✅ Predicted-vs-measured table, three runs per workload, **published** — `calibration_report.md`
- [x] ✅ **The residual explained with measurements**: histogram counts exact; W2 distributions identical
      bucket for bucket; W1's distribution residual traced to the timing fit (`posthoc.md`, labelled
      post-hoc); naive-LRU and block-pool error side by side
- [x] ✅ **Limits owned publicly** in the README: the batch-1 decode non-linearity, the prefill
      double-charge, open-loop only
- [x] ✅ README calibration section: engine, version, the exact CLI, model, quantization, capacity,
      hardware, and every `/metrics` field consumed by name
- [x] ✅ Pinned by tests: `test_live_engine_regression.py` now recomputes the model and demands the
      engine's Phase-3 counts, request by request
- [ ] → moved to Phase 4: load `fitted_profile.json` in the CLI (`--profile`) — the curves need it

**Gate:** predicted vs measured on ≥2 shapes with the residual accounted for. ⛔ If H3 fails, downgrade the
sweep's claim from *predictive* to *comparative* — and say so — rather than re-tuning until it matches.

---

## Phase 4 — The six policies and the Pareto sweep *(tests H1 + H5)*

> ### 🔨 2026-09-22 (night) — the first CALIBRATED sweep ran, pre-registered
> 2 shapes × 6 loads × 7 policies × 1–8 T4s × 5 seeds = 3,360 runs, on the Phase-3 profile at the
> engine's own capacity. [`report.md`](artifacts/results/phase4_sweep_t4/report.md)
> - ⛔ **H1 falsified by load.** Holds exactly at 1–2 sessions/s; from 3/s, pure radix-affinity puts
>   every request on one replica, preempts 110–212 times per run, and its hit-rate lead turns negative.
> - **H5 holds in one case of four:** balanced affinity at LOW sharing needs 1–3 fewer T4s than
>   round-robin at every load. Pure affinity never holds the SLO above 1/s; balanced affinity at high
>   sharing neither — ⚠️ with its load weight (alpha = 1) never swept.
> - ⭐ **Session-sticky** (untested, reported) needs the fewest T4s everywhere — 2, 3, 3, 4, 5 against
>   round-robin's 3, 4, 5, 7, 8 — **25–42% cheaper** wherever round-robin holds the SLO at all.
> - Scope: generated traffic, T4 timings, one model. **No README claim until H6.**

- [x] ✅ **Loaded the fitted T4 profile** — `configs/profiles/t4-qwen3-4b-vllm0.30.0.json`
      (`scripts/phase4_profile.py`): fitted timings + the engine's own capacity, **1,537 blocks** (the
      arithmetic said 2,394). `--profile` on `replay` and `crossover` — which also fixed `replay`
      silently ignoring any profile
- [x] ✅ Decided on the step-cost model's two measured limits — **ADR-012**: kept as fitted; a fix needs
      a new pre-registered GPU run, never a re-fit to W3
- [x] Seven policies, one file each, behind one interface (ADR-003), every tie from a seeded RNG
- [x] `sim/` — replica with memory-based admission and preemption, fitted cost, fleet event loop
- [~] Sweep policy × replicas 1–8 on ≥2 traces — ✅ two generated shapes × six loads. **Real traces found
      2026-09-23** (`FLAGSHIP_PROJECTS.md`): SemiAnalysis **AgentX** (64-token block hashes, scoped to the
      session) and **Mooncake** FAST'25 (512-token block hashes, global). ✅ **Mooncake done 2026-09-28**:
      the `mooncake` adapter (sessions inferred; Apache-2.0), R1 registered and run — ⛔ FALSIFIED: on
      real traffic sticky and balanced affinity tie within a replica, both beat round-robin, pure
      affinity fails ([`report.md`](artifacts/results/realtrace_mooncake/report.md)). **AgentX:
      R2 registered and ✅ HOLDS 2026-09-28** — the `agentx` adapter (sessions real, hashes
      session-local), 48 sessions at a time at 1/64, speeds set by round-robin-only probes. As
      predicted, sticky, balanced AND pure affinity all need 2 T4s at every speed; round-robin needs
      5 at 2× ([`report.md`](artifacts/results/realtrace_agentx/report.md)). Pure affinity's collapse
      on Mooncake is caused by shared prompts
- [~] **Multi-seed (5) + bootstrap CIs by session** — ✅ used for every H1 decision; per-cell p95 CIs
      for the README tables are still to add
- [x] ✅ `eval/cost.py` from `configs/cost_model.yaml` — every constant **cited and dated**: g4dn.xlarge
      $0.526/h, ₹95.87/$, SLO p95 TTFT ≤ 2,000 ms (MLPerf). **₹ per 1k successful requests**
- [x] ✅ `eval/pareto.py` + `eval/recommend.py` → one policy, one replica count, a ₹ delta vs
      round-robin — at every (shape, load)
- [x] ⭐ **Sweep alpha** for radix-affinity-balanced (its load weight), registered as a NEW experiment
      before it runs — the one open question in H5's balanced result. ✅ Done as Phase 5's **A1**
      (2026-09-28): α\* = 16 on Phase 4's traces (45 replicas in total against 64 at α = 1)
- [x] Property test: prefix-affinity never scores below round-robin on perfectly-shared prefixes —
      ⚠️ **restate it first: only without memory pressure.** Phase 4 measured it failing under load
      (pure affinity 0.815 vs round-robin 0.832 at high sharing, 4/s, 5 replicas), as H1 now explains.
      ✅ 2026-09-28, `tests/test_sim.py`: restated and checked on **24 generated workloads** (sharing
      0–0.95, 1–16 tenant prompts, 1–5 turns, 2–6 replicas) with a cache the test proves cannot fill;
      and the Phase 4 counterexample is **pinned** (0.815 < 0.832), so the precondition stays visible
- [x] Golden-file test on a small sweep; CI green gate — ✅ 2026-09-28, `tests/test_golden.py` +
      `tests/golden/small_sweep.json`: 4 policies × 1–3 replicas × 2 seeds on the calibrated profile,
      every number and the recommendation pinned; regenerate with `KVSIM_UPDATE_GOLDEN=1`. CI runs it
      on Python 3.11–3.13 (checked locally on 3.13 only, under three hash seeds)
- [x] ⭐ **Test H6 — re-run the sweep under a rescaled cost function** (a "faster GPU", and a different
      prefill/decode ratio) and compute Kendall's τ between the policy orderings. **Pure CPU, no extra GPU
      time, the sweep already exists.** ⛔ Do this *before* the README makes any tier-1 claim: H6 is what
      says whether a user without a GPU can trust the recommendation at all (ADR-010).
      **Ranking half registered, run and ⛔ FALSIFIED 2026-09-28** — τ-b down to 0.51 and one flip in
      48 points ([`report.md`](artifacts/results/h6_ranking/report.md)); session-sticky stayed in the best
      set at all 48 (post-hoc). The README's free-ranking claim is withdrawn; the crossover half waits
      for Phase 5's grid

---

> ### 📌 An early observation from the first full sweep — ⚠️ NOT a result
> Uncalibrated profile, no CIs, one workload shape, one seed set. Recorded because it is a
> prediction worth testing properly, **not** because it is a finding.
>
> At ~0.6 cross-session sharing with 6-turn sessions, **session-sticky dominated every prefix-aware
> policy** — it held a ~0.89 hit rate at every replica count while staying balanced, where
> round-robin decayed 0.89 → 0.66 as replicas diluted the cache. Pure `radix-affinity` collapsed
> onto a single replica (imbalance = N) and paid ~450x in tail latency.
>
> ⚠️ **If this survives calibration and CIs, H1 as written is FALSE in this regime** — which is
> exactly what pre-registration is for. It also sharpens where prefix-affinity should win: SHORT
> sessions with HIGH cross-session sharing, where stickiness has little history to exploit. That is
> a sweep axis Phase 5 must cover, and it was not in the plan before this run.

## Phase 5 — The crossover finding *(tests H2 · ⛔ never cut)*

- [x] ⭐ **Registered and RAN 2026-09-28** ([`configs/phase5_crossover.yaml`](configs/phase5_crossover.yaml),
      `scripts/run_phase5.py`, dated entry in `docs/09`): **A1** — sweep radix-affinity-balanced's α on
      Phase 4's traces and fix α\* by a written rule; **H2**'s procedure — sharing × load at 4 and 8
      replicas, paired session-bootstrap CIs; **H2b** (new) — K tenant prompts (the generator's new
      `n_shared_prefixes`), affinity at α\* vs session-sticky on replicas needed
- [x] Sweep **prefix-sharing ratio as its own axis**, crossed with offered load — the H2 grid
      ([`report.md`](artifacts/results/phase5_crossover/report.md)): 14,592 runs, clean tree `3967a56`
- [-] ~~Add turns-per-session as a third axis~~ — **deferred** (2026-09-28): the reason sticky won in
      Phase 4 is more likely that ONE small shared prompt fits in every replica's cache. H2b tests that
      mechanism directly (the number of tenant prompts); turns per session comes back if H2b fails
- [x] Locate the boundary where least-loaded overtakes prefix-affinity on **p95 TTFT** — published as a
      win map with paired session-bootstrap CIs. ⛔ **It is not the boundary H2 predicted:** pure
      affinity loses at every sharing > 0 (the prompt's existence decides, not its length), and the
      balanced policy shows no monotone boundary. ✅ *The README figure* — `docs/figures/flip-*.svg`,
      drawn by `scripts/make_figures.py` from the committed grid (2026-09-28)
- [x] Check H2's three directional sub-predictions — ⛔ **H2 FALSIFIED** for both policies (docs/09)
- [-] ~~If the effect sits inside the CIs, report that~~ — not needed: the effects were far outside them
- [x] ⭐ **H2b HOLDS** — the README headline, scoped (2026-09-28): *keep each user on one GPU until your
      tenants' prompts outgrow one GPU's cache; then route by prefix, and keep its load weight small.*
      (Reworded 2026-09-28 from "with a small load weight": see docs/09's amendment.) ✅ The
      flip's direction survived all four hardware rescales — **H6's crossover half HOLDS**
      ([`report.md`](artifacts/results/h6_crossover/report.md)); replica counts stay T4 figures.
      ⛔ **2026-09-30: on the validated, context-aware timing model the re-test FALSIFIED this**
      (below, Phase 6) — the headline is now *keep each user on one GPU*

---

## Phase 6 — Sim-to-real, two replicas

> ### ✅ 2026-09-28 — RAN TWICE: HOLDS. The cache model predicts two live replicas; the timing model does not.
> [`report.md`](artifacts/results/phase6_kaggle_t4/report.md) (`scripts/phase6_analyze.py`; S1/S2
> recomputed from the raw counters). **S1:** largest hit-rate error 1.40% (run 1) and 0.22% (run 2);
> K1 exact to four decimals. **S2:** all five tested pairs in order in both runs, including K64
> affinity above sticky. **Reported, not tested:** p95 TTFT 5–222× the prediction, and 15–83
> preemptions in four of the five runs predicted to have none — decode steps ran 2.7–4.0× the profile, and the engines
> overloaded. The latency *order* still matched the prediction (post hoc). Cause — context length or
> CPU contention — not yet tested. Every replica count is now marked a simulator figure.
>
> ### (was) 🔨 2026-09-28 — built, dry-run and REGISTERED. ⏳ Waiting for one Kaggle run.
> [`notebooks/phase6_sim2real.ipynb`](notebooks/phase6_sim2real.ipynb) — self-contained (vLLM + the
> standard library): two vLLM 0.30.0 servers, Qwen3-4B, **one per physical T4**, 1,537 blocks each; H2b's
> traffic with one shared prompt (K1) and 64 tenant prompts (K64); routed by round-robin, session-sticky
> and radix-affinity through ports of the simulator's own policies. Predictions committed first
> ([`predictions.json`](artifacts/results/phase6_prereg/predictions.json), from a clean tree at `8951a1e`);
> decided by **S1** (each run's hit rate within 10%) and **S2** (policies predicted ≥ 0.02 apart keep
> their order — the headline pair is K64 radix-affinity 0.8436 above session-sticky 0.8214).
> `scripts/phase6_dryrun.py` runs the whole notebook against two mock engines and passes.
> **To run:** Kaggle → GPU T4 ×2, Internet on → Run All (~40 min) → bring back `phase6_results.zip`.

- [x] **One vLLM server per physical T4** (`CUDA_VISIBLE_DEVICES=0` / `=1`) — no MIG needed; two real
      GPUs are complete isolation
- [x] Two vLLM servers, one arrival stream, routed by three policies — ✅ ran twice on Kaggle 2× T4
- [x] Predicted vs measured hit rate across the pair — ✅ all 12 runs within 1.4%; S2 in order *(v2's
      round-robin arm was the first data point — predicted 1,088 tokens computed, hit rate 0.966862,
      committed before it ran)*
- [x] ⭐ **A context-aware timing model**, fitted on data Phase 6 did not produce and registered before
      it is checked: a single-engine decode sweep across context lengths (~100 to ~2,000 tokens) ×
      batch sizes, which also tells context cost from CPU contention. Phase 6's measured latencies
      are then the held-out test — seen already, so labelled post hoc — and a fresh two-replica run
      the real one. 🔨 **C2 registered 2026-09-28** ([`configs/c2_timing.yaml`](configs/c2_timing.yaml),
      [`c2_timing_t4.ipynb`](notebooks/c2_timing_t4.ipynb), from the clean commit `adfada5`): the
      model has two context terms (zero by default — every earlier result unchanged); parts A–C
      measure decode, prefill and contention, part D is held-out traffic the fit never sees. C2a
      (context real) / C2b (contention real) / C2c (D within 2×) — predicted holds / fails / holds.
      Dry-run against mocks with known terms: recovered, C2c within 1.05×.
      ✅ **C2 RAN 2026-09-30 — all three predictions came true** ([`report.md`](artifacts/results/c2_kaggle_t4/report.md)):
      C2a holds (context 1.92×), C2b fails (contention 1.00× — context, not CPU), C2c holds
      (held-out traffic within 1.10–1.41×; the old profile was up to 7.1× optimistic). Shipped as
      the profile `t4-qwen3-4b-vllm0.30.0-ctx`. Post hoc, labelled so: replayed at Phase 6's
      measured placement it lands within 1.02–1.79× of all six live two-replica runs
- [x] ⭐ **Re-test the finding on the new timing model** — H2b's rules unchanged, on
      `t4-qwen3-4b-vllm0.30.0-ctx`, loads 0.25–4/s, replicas 1–16. Registered 2026-09-30
      ([`configs/h2b_c2.yaml`](configs/h2b_c2.yaml)); predicted: holds. ⛔ **FALSIFIED**
      ([`report.md`](artifacts/results/h2b_c2/report.md)): at 64 prompts balanced affinity ties
      sticky at 0.25–1/s and loses at 2 (11 against 10). Post hoc: sticky best or tied at 22 of 24
      points; the exceptions are prefix routing with a small weight (8 against 10 at K=64, 2/s)
- [x] Re-run H6 and R1/R2 on the validated model — they ran on the context-free profile.
      *(Closed 2026-10-06: R1 re-run; R2 untestable as designed; H6 not re-run, by decision.)*
      🔨 **R1 and R2: registered 2026-10-06** ([`configs/realtrace_ctx.yaml`](configs/realtrace_ctx.yaml)):
      the same traces, adapters, policies, seeds and rules, on two calibrations (A and C4's
      s0); speeds re-chosen by R2's rule from a round-robin-only probe. Predicted: R1
      falsified again, R1w (within one replica) holds, R2 holds.
      ✅ **RAN 2026-10-06** ([`report.md`](artifacts/results/realtrace_ctx/report.md)): ⛔ R1
      falsified again on both calibrations and ✅ R1w holds on both — on Mooncake, session-sticky
      and balanced affinity tie at 5 · 5 · 5 (tool-agent) and are within one on conversation; no
      policy with a load term is more than one replica from round-robin. ⚠️ **R2 is untestable as
      designed** on the validated model: round-robin holds the target on 8 replicas at no probe
      speed (28% of AgentX's prompts at that scale are longer than the engine's limit; a miss on
      the median prompt costs 1.6 s). Its first-model result stays a first-model result.
      ✅ **H6: decided 2026-10-06 — not re-run** (decided on this log's
      recommendation). It stays a first-model result and is labelled so: what followed from its
      failed half — a ranking is a result for one calibration, so calibrate your own GPU — was
      built (`kvsim calibrate`) and then measured more directly than a rescaling can (C3, C4);
      its other half was about a flip that is unsettled on the validated model
- [x] **Restore the four registered notebooks** deleted from disk on 2026-09-30 (Phase 0, 3, 6,
      C1) — restored from git on 2026-10-01; their tests pass again (189 green)
- [x] ⛔ **TTFT percentiles only from separate GPUs** — the notebook runs one server per physical GPU.
      They are **T4** percentiles, and reported, not tested
- [-] **Fallback, decided in advance** — not needed: two replicas ran. (It was: if 2 replicas prove
      infeasible, ship a single-replica calibration plus a documented, reproducible account of why.)

---

## Phase 7 — Package, CLI, the static demo, hardening

- [~] `pip install kv-routing-sim`; `kvsim replay trace.jsonl --policies all --replicas 1-8
      --format json|md|html`; semver + changelog.
      ✅ 2026-09-28: the wheel builds, installs into a clean venv with PyYAML as its only dependency,
      and runs `kvsim replay` from outside the repo — it now ships the measured T4 profile
      (`--profile t4-qwen3-4b-vllm0.30.0`) and the cited cost model (`--cost-model default`),
      byte-identical to `configs/` (`tests/test_packaging.py`); [`CHANGELOG.md`](CHANGELOG.md) for 0.1.0.
      Left: `--format html`; project URLs in `pyproject.toml`, and absolute image links in the README
      (PyPI shows neither relative links nor `<picture>`), once the repo is public; the PyPI upload
- [~] ⭐ **`kvsim calibrate`** — expose the Phase-3 harness so a USER can fit a profile to their own GPU
      and model (~30 min). The ADR-010 mechanism, and the answer to *"why trust your numbers on my
      hardware?"* — *"you shouldn't; run this."* Needs the `[calib]` extra.
      🔨 **Built 2026-09-28** ([`calib/calibrate.py`](src/kv_routing_sim/calib/calibrate.py)) — and
      lighter than planned: stdlib only, no extra, the engine reached over HTTP; minutes, not 30. It
      reads the KV capacity (never estimates it), fits prefill and decode as Phase 3 did, and checks
      the cache model against the engine's counters **request by request**. Against the mock engine
      it recovers the known timings (decode 2.53 ms/seq against 2.5; prefill 0.022 ms/token against
      0.020), its check is exact (48 of 48 requests), and a wrong block size is caught — which the
      first version of the check missed. **C1 registered 2026-09-28:** [`c1_calibrate_t4.ipynb`](notebooks/c1_calibrate_t4.ipynb)
      carries the wheel of the clean commit `5a34c6d`; predicted exact (48 of 48; hit rate 0.714286).
      ✅ **C1 RAN on a real T4 and HOLDS** ([`report.md`](artifacts/results/c1_kaggle_t4/report.md)):
      the check exact (48 of 48; 9,840 of 13,776 tokens), 1,537 blocks read = the engine log, 48 s.
      ⚠️ Its straight-line timing fit is not good enough to size a fleet: T4 prefill is 0.71 ms/token
      to 1,024 tokens (Phase 3's value) and 1.12 ms at 2,040; the line clamps a −212 ms intercept.
      Still `[~]`: the timing fit needs the context-aware model (Phase 6). 🔨 **2026-09-30:** it now
      fits C2's context-aware model (decode cells over context × batch; `calib/fit.py`, the same
      NNLS as C2's analysis) and recovers a mock's known context terms (3.97e-6 against 4e-6).
      🔨 **2026-10-01, before spending a GPU:** a pre-check on C2's recorded data
      ([`precheck.md`](artifacts/results/c3_prereg/precheck.md), post hoc) — fitted on the cells
      the command measured, cold prompts only, the model put a held-out p95 off by 5.7–7.2×; with
      C2's cached-prefix cells, within 1.36×. The command now times those cells too
      (`--prefill-cached`, `--prefill-new`).
      🔨 **C3 registered 2026-10-01** ([`configs/c3_calibrate.yaml`](configs/c3_calibrate.yaml),
      [`c3_calibrate_t4.ipynb`](notebooks/c3_calibrate_t4.ipynb) with the wheel of the clean commit
      `016d167`): C3a the command works (check exact, capacity read) / C3b its profile predicts
      three held-out workloads within 2× / C3c C2's profile predicts them within 2×, from numbers
      registered now — predicted holds / holds / holds. The third workload has a shape no design
      decision saw. Dry-run against a mock with known terms: recovered, all three within 1.07×.
      ⚠️ **2026-10-05: the first attempt did not start** — Kaggle's image now ships a
      `torchaudio` built for another CUDA than the torch vLLM installs, under the version vLLM
      pins, so pip kept it and vLLM could not be imported. Nothing was measured
      ([record](artifacts/results/c3_env_failure/)). The notebook is amended, not re-registered: a
      pre-flight cell repairs it, refuses a GPU that is not a T4 and checks `vllm serve --help` in
      the first minute; every early stop leaves a zip. Checked on Kaggle's own image with a probe
      notebook that starts no engine. The same wheel, traffic, predictions and rules.
      ✅ **C3 RAN 2026-10-05 — all three hold, as predicted**
      ([`report.md`](artifacts/results/c3_kaggle_t4/report.md)), through Kaggle's API
      (`scripts/kaggle_run.py`), 2,320 s: C3a exact (exit 0 in 160 s; 9,840 of 13,776 tokens; 48 of
      48; 1,537 blocks); C3b within 1.41× on three unseen workloads; C3c within 1.38× from numbers
      registered four days before. ⚠️ Reported: the command's second run is 4.17× off on `K1_half`'s
      p95 (3–5% different coefficients, at the knee); and H2b-C2's grid on the command's profile
      matches the committed table at 122 of 144 cells, 141 within one — **but reverses H2b's
      verdict**. Post hoc, three calibrations
      ([`posthoc_three_calibrations.md`](artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.md)):
      102 of 144 identical, 138 within one; sticky best or tied on all three at 19 of 24 points — 12
      of 12 with 1 and 4 prompts. Also found in the data: the SLO has no per-token half.
      🔨 **2026-10-05, built:** every timing cell is now measured `--reps` times (3) and the
      profile keeps the fit of each repeat; `kvsim replay` sizes on each and prints the range
      beside its recommendation ("how firm: 3 to 4 replicas across the calibration's 3 repeats");
      a profile without repeats says its spread is unknown (commit `5db6a9a`).
      ✅ **2026-10-05, on a real T4 (C4a holds):** three runs, three repeats each, exact on the
      cache check every time, 359 s a run. ⛔ **And the spread it prints is too narrow (C4's rule E
      does not hold):** another run's count is inside a run's own range in 178 of 216 cases
      (82.4%; 90% was the rule; 94.9% with one replica more each way). The two runs taken back to
      back agree (97%); the run an hour earlier does not agree with them (75%). ✅ **2026-10-06:**
      the line now says so — "size for the larger, plus one", and when the repeats agree, "still
      allow one more" — and the report gives C4's two percentages
- [~] ⭐ **An SLO with its per-token half.** The first-token target was the only one: C3's live
      engine took 134–521 ms per token on traffic that met, or nearly met, it, and the MLPerf row
      the 2 s come from also limits the time per output token (100 ms). 🔨 **2026-10-05:** the
      simulator measures every request's time per output token, `kvsim replay` prints its p95, and
      a cost model can make it part of the SLO with `slo.tpot_p95_ms` (off by default; no earlier
      number moved — commit `534f5b9`). **T1 and T2 registered 2026-10-05**
      ([`configs/tpot_slo.yaml`](configs/tpot_slo.yaml)): T1 — the model's mean inter-token latency
      within 1.5× of the engine's on the five held-out runs on disk; T2 — the finding's grid under
      both halves on three calibrations (sticky still best with 1 and 4 prompts / never worse than
      round-robin or least-loaded / first-token counts optimistic by ≥ 1.5×). Predicted: all hold.
      ⛔ **T1 RAN — DOES NOT HOLD** ([`report.md`](artifacts/results/t1_itl/report.md)): within
      1.07× on the three runs off the knee, **1.54× and 1.71× too slow** on the two at it (the
      model has 6 preemptions there, the engine none). Post hoc
      ([`posthoc_range.md`](artifacts/results/t1_itl/posthoc_range.md)): at the knee 97% of the
      model's tokens are decoded in batches whose summed context is 8× past the largest any
      calibration cell measured (20,200 tokens — a cell's prompts are unique, so it cannot exceed
      the cache; a shared prompt can).
      ✅ **T2 RAN — all three hold on A, B and C**
      ([`report.md`](artifacts/results/t2_two_part_slo/report.md)): T2a 12 of 12 points · T2b no
      exception in 24 · T2c **1.93 · 2.00 · 2.00×**. Per the registration, since T1 failed these
      are the model's counts, not sizing guidance. Exploratory: the load weight of 1 — the
      first-token hint at 64 prompts — holds nothing under the two-part target; sticky and
      balanced affinity are 1–2 replicas apart there, with no steady winner. ⚠️ Grids B and C
      carry a dirty header (other files were edited while they ran): 720 of 720 sampled cells
      reproduce from a clean checkout ([`verification.json`](artifacts/results/t2_two_part_slo/verification.json)).
      🔨 **C4 registered 2026-10-05** ([`configs/c4_per_token.yaml`](configs/c4_per_token.yaml)):
      one GPU session — `kvsim calibrate` with its repeats, three times (C4a); decode of batches
      that share a prefix, up to a summed context 9.2× the largest ever fitted (D1a, D1b); what an
      arriving request costs those already decoding (D2); three held-out workloads under, on and
      over the 100 ms limit, every request's token times kept and the engine's own histogram read
      (T3a, T3b); then, in the simulator, whether one run's spread covers another's count (E) and
      the 64-prompt question on the fresh fits under both targets (F1, F2). Predicted: all nine
      hold. Dry-run against the mock: every timing rule within 7%.
      ✅ **C4 RAN 2026-10-05 — seven of nine hold**
      ([`report.md`](artifacts/results/c4_kaggle_t4/report.md)): D1a −3%, −4% · D1b +3% to +15%
      (allowed ±25%) · D2 1.03–1.19× · **T3a 1.00×, 1.00×, 1.07×** on p95 time per token under, on
      and over the limit · T3b the model's verdict is the engine's own · F1 8 · 7 · 8 against
      9 · 10 · 10. ⛔ **E** 82.4% (above). ⛔ **F2**: under both halves balanced affinity (α 16) is
      one replica below session-sticky at 3 and 4 of 6 loads on two of the three fresh fits (the
      rule allowed 2). So: the factor of two rests on a checked per-token model — on one replica;
      T1's failure is the knee multiplying a 12–15% error at very large batches. **Left:** the
      two-part target as the default; a "this workload is at the knee" warning; shared-prefix
      cells in `kvsim calibrate`
- [ ] **`configs/profiles/`** — committed calibration profiles keyed by
      `(gpu, model, block_size, max_num_seqs, engine_version)`; the CLI reports **how far** the user is
      from the nearest one. ⛔ Never interpolate between profiles — an interpolated error bar has no
      measured basis
- [x] ⭐ **`scripts/build_grid.py`** — sweep sharing × load × replicas × policy and emit ONE JSON file.
      This is the demo's whole backend (ADR-011). Parallel via `joblib` (`[sweep]` extra); commit the
      grid plus its wall-clock and byte size, since both are currently estimates
      ✅ **Built 2026-09-28** on the finding's axes instead of sharing: K tenant prompts (1, 4, 16, 64)
      × loads 1–8 × replicas 1–8 × 6 policies × 3 seeds = **3,456 runs in 149 s**, stdlib
      `ProcessPoolExecutor` like the other scripts (no joblib). **34 KiB raw, 12 KiB gzipped.** A guard
      stops the build unless it reproduces Phase 5's H2b table exactly — it does
- [x] **The static demo page** — [`site/index.html`](site/index.html), 2026-09-28 — the one-screen design in
      the deployment notes: finding + crossover curve +
      calibration table above the fold, then three sliders that **look up** the grid (no server, no
      wait) and whose payoff is watching the recommendation **flip**.
      ⛔ **Bundled + generated traces only, no upload box** — a serving log is someone's users' prompts
      ✅ Finding + calibration table on top; three selectors (prompts, load, replicas) that look the
      answer up; the flip happens on the prompts selector (H2b), not on sharing (H2 was falsified).
      One HTML file, no libraries, light and dark, checked in Chrome at 1280 px and at a true 390 px
      (no sideways scroll), and the K = 1 → 64 flip checked by clicking
- [x] ⚠️ **Rebuild `demo_grid.json` from the CALIBRATED T4 profile.** The committed grid was built on an
      A100-shaped guess (2026-09-21) that no longer describes any hardware in the plan. It is a
      placeholder that proved the pipeline, nothing more — ✅ **rebuilt 2026-09-28** from the calibrated
      profile, from a clean tree (`6bdee6c`)
- [~] Publish that page to **GitHub Pages** AND to a **free Static Space** — same build, two URLs.
      `.github/workflows/pages.yml` publishes `site/` + the grid on every push to `master` that touches
      them. **Manual step:** push the repo to GitHub, then Settings → Pages → Source: *GitHub
      Actions*; the Static Space needs a Hugging Face account (upload the same two files)
- [x] ⭐ **The same curve images and calibration table go IN THE README**, for the reader who never
      leaves GitHub. *(No longer a workaround for cold starts — a static page does not sleep — just
      belt-and-braces.)*
- [x] ✅ **Render: CUT.** Verified 2026-09-21: free web services **spin down after 15 minutes** idle,
      take **~1 minute** to wake, run on 512 MB / 0.1 CPU, and the 750 hours are a **monthly cap**, not
      always-on. Returns only if we build the thin HTTP recommender (real production evidence)
- [x] ✅ **Demo platform decided: STATIC** (2026-09-21, **ADR-011**). ⛔ The free Gradio Space this plan
      assumed **no longer exists** — HF now requires a paid plan (PRO, $9/mo personal) to create a
      Gradio or Docker Space; only Static is free. The one free exception is 2 Gradio Spaces on
      **ZeroGPU** — rationed GPU, absurd for a CPU-only simulator. Static is also *better*: it never
      sleeps, and a precomputed grid is reproducible from a committed script where a live app is not
- [ ] Docs published to GitHub Pages
- [x] ⭐ **"What breaks at 10× and 100×"** as a named README section, answered with **measured** numbers:
      RSS ceiling and sweep wall-clock at 10× trace length; at 100×, which gives out first — block-hash
      memory, the policy inner loop, or the report renderer; at 10× replicas, does cache-model fidelity
      hold or does eviction skew compound? *(These are the questions a design review asks.)*
      ✅ **Measured 2026-09-28** ([`scale/report.md`](artifacts/results/scale/report.md)): 10× is 81 MiB
      and seconds; at 100× **block-hash memory** gives out first (537 MiB of trace before the simulation
      starts; the policy loop stays linear); at 80 replicas affinity routing costs 2.4× per request.
      Fidelity at 80 replicas is not answerable on a CPU, and the README says so
- [~] 👥 `CONTRIBUTING.md` + **three genuinely good first issues** (a new trace adapter, a new policy class —
      both are single-file by design), then **review the PRs properly**: real comments, changes requested,
      then merged.

---

## Phase 8 — Write-up and launch

      ✅ 2026-09-28: [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, the six habits, how to add a policy,
      an adapter or a profile) and [`docs/good-first-issues.md`](docs/good-first-issues.md) — a
      `least-tokens` policy that tests the open least-loaded question, `--format html`, and per-cell
      bootstrap CIs by session. The adapter issue waits until AgentX's format is checked. Open them
      once the repo is public; then review the PRs properly
- [x] README **numbers-first**, with **the credential and the finding in separate sections** —
      2026-10-06: the first screen is one table (engine and version, the flag, the two counters, the
      GPU; cache error, timing errors; the finding and that it is not new; the upstream fix), then a
      short status box. The finding and the Real-Engine Calibration Report stay separate sections
- [~] *(2026-10-06: the 30-second reader's words and the 2-minute reader's four lines are on the first screen —
      the fourth, the merged PR, reads "not yet submitted" until it is; the engineer's path is
      the ADRs, the residual, 10×/100× and `04-prior-art.md`.)*
      Write the README against the **three readers** in
      the deployment notes: the 30-second reader (needs the
      grep targets *visible* — engine, version, metric names, GPU), the 2-minute reader (needs four
      lines), and the 30–60-minute engineer who asks *"isn't this just Vidur?"* (needs the ADRs and the
      residual)
- [x] *(2026-10-06: the scope section now starts with what is closest — NVIDIA's simulator, and
      LMetric, SMetric and Preble — before Vidur, Frontier and `cachetrace`.)*
      Scope discipline in the README from day one: vs `cachetrace` (multi-replica policy choice, not
      single-path linting) · vs **Vidur** (cache-aware *policy* choice, not parallelism/capacity planning) ·
      vs **Frontier** (reads *your* log and *recommends*, rather than being configured by hand)
- [ ] Show HN + X thread, with the engine and version
      **in** the title
- [ ] ⭐ **The `guidellm` PR merged** — see the upstream log
- [~] 📄 **The preprint — OPTIONAL** (the project is complete without it; the plan calls arXiv
      "optional upside", useful only for research-heavy teams; if wanted, due ~Nov 9, cs.DC) —
      [`paper/main.tex`](paper/main.tex), drafted 2026-09-28, rewritten 2026-09-30 around the
      reversal: seven pages that compile; every number mapped to its results file in
      [`paper/README.md`](paper/README.md); 19 references, each checked on arXiv or at its source; no
      "first" claims. If it is published: author details and the repository URL, the October
      prior-art re-audit, C3's result, and the arXiv endorsement. If not: it stays a draft, or its
      text becomes a blog post
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
      and the nonexistent metric name. **Six sentences were added on 2026-10-05** — among them *"the
      recommendation is the output nobody else produces"* and *"we found that sessions should stay on
      one GPU"*. *(Searched on 2026-10-06 in the README, the demo page, CONTRIBUTING, the changelog
      and `src/`: none left. Search again on the day of publishing.)*
- [ ] ~~**The claim is stated as an intersection**, with the four competitors named~~ — the
      intersection claim was **withdrawn 2026-10-05** (NVIDIA's simulator occupies it; the routing
      result was already published). The gate is now: **the README names AISimulate/DynoSim, LMetric
      and SMetric where it states the finding, and claims only the three things in
      [`docs/04-prior-art.md`](docs/04-prior-art.md) §5**
- [ ] **Every artifact in `artifacts/results/` is regenerable** by a script in `scripts/`
- [ ] **Every cost constant is cited and dated** in a committed config
- [ ] **No TTFT percentile from the memory-split hardware path**
- [ ] ⛔ **No sentence blends a calibrated error bar with an uncalibrated recommendation.** Our error bar
      belongs to our GPU and our model. H6 (2026-09-28) found that even the full ranking does not
      transfer across hardware — only the direction of the flip does (ADR-010). *"Validated
      within X%"* and *"your serving log in, a recommendation out"* are true separately and false together
- [ ] **Bootstrap CIs resample sessions/segments, not requests** — and every caption says which
- [ ] **A hypothesis that failed is reported as a finding**, not deleted
- [ ] **No trace containing user prompt text is committed**, ever — hashes only
- [ ] **The upstream PR:** every line reviewed; full test suite run
