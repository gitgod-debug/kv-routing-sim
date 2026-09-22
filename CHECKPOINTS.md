# kv-routing-sim — Checkpoints

> **What this file is.** The living to-do list for the whole build. Every task from the approved plan is
> here as a tickable box. **Tick a box only when the thing is done AND verified** — for this project
> "verified" usually means *a number was measured and committed*, not *the code ran*.
>
> **Companion files:** [`docs/00-START-HERE.md`](docs/00-START-HERE.md) explains what everything is and
> why. This file only tracks *where we are*.
>
> Legend: `[ ]` not started · `[~]` in progress · `[x]` done & verified · `[!]` blocked · `[-]` cut

**Last updated:** 2026-09-20 (**the simulator is built**: traces, cache, 7 policies, event loop, eval,
CLI, 73 tests green. Everything except the three engine-dependent numbers, which stay behind the
Phase-0 gate)
**Current phase:** ✅ **Phase 0 COMPLETE** · ✅ **Phase 2b COMPLETE** (2026-09-22) — the cache model is now
vLLM's block pool, reproducing every live-engine number to the token, histograms included. **Next: the
Phase 3 calibration notebook** (Qwen3-4B, Kaggle 2x T4), with its predictions pre-registered.

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
| 3 | **Real-Engine Calibration Report** | `[ ]` | Predicted-vs-measured on **≥2 shapes**, ≥3 live runs each, error bars, **residual explained with the `kv_block_*` histograms** — tests **H3 + H4** |
| 4 | **Seven** policies + Pareto sweep | `[~]` | Built and running; **gated on a calibrated profile** before any curve is published. Tests **H1 + H5 + H6** |
| 5 | **The crossover finding** | `[ ]` | The sharing-ratio × load boundary published as a curve with CIs — tests **H2**. ⛔ never cut |
| 6 | Sim-to-real 2-replica validation | `[ ]` | Predicted vs measured across two MIG instances, or the documented fallback and why |
| 7 | Package · CLI · static demo · hardening | `[~]` | `pip install` + `kvsim replay` + a **static** demo page (ADR-011) + "what breaks at 10×/100×" answered with measured numbers |
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

- [x] `block_trie.py` — trie of resident block hashes on plain dicts (ADR-002)
- [x] `eviction.py` — LRU order + optional TTL + a hard capacity bound derived from a GPU memory config
- [x] `replica_cache.py` — counters **named to mirror vLLM's own**, so predicted and measured are literally
      the same quantity
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
- [ ] ⭐ **Pre-register the Phase-3 predictions** — moved to the Phase-3 notebook build: they depend on
      the Phase-3 workloads, and on capacity, which is read from each run's own log. What is fixed in
      advance is the model, the workload and the procedure; the number follows deterministically

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

- [ ] ⭐ **Model: `Qwen/Qwen3-4B`** on one T4 — not 8B, whose 15.26 GiB of fp16 weights exceed the card.
      4B's KV cache is architecturally identical to 8B's (144 KiB/token), which is what matters here
- [ ] **Read the true KV block count from EACH run's log** ("GPU KV cache size: N tokens") — measured
      2026-09-22: `capacity.py` overestimates by 7–12.5%, and identical launches differ by 4.8%
- [ ] Profile identity recorded: `gpu="Tesla T4"`, `model="Qwen/Qwen3-4B"`, `engine_version="0.29.0"`
- [ ] Calibrate against **`prompt_tokens_by_source{local_compute}`** as well as the hit rate — prefill
      work actually done, and defined even with caching off (ADR-008 amendment)
- [ ] `calib/metrics_scrape.py` — parse the exposition format; **derive** the hit rate from two counters
- [ ] Calibrate on **≥2 workload shapes**, **≥3 live runs each** (a single run has no error bar) — and at
      least one must **overflow the cache**, or eviction (the whole point of H4) is never exercised
- [ ] Traces are generated **locally by our generator** and uploaded with the notebook — never
      re-implemented inside it (two generators would drift; ADR-011's reasoning)
- [ ] Keep the replay **open-loop and deterministic** (`temperature=0`); record that closed-loop
      decode-block reuse is a known, unmeasured limitation (ADR-002 amendment, gap 4)
- [ ] Predicted-vs-measured table with error bars — ⛔ **published whatever it says** (H3)
- [ ] **Explain the residual with a measurement**, not a narrative: compare the engine's `kv_block_*`
      **distributions** against the simulator's own, emitted by vLLM's rules (H4 amendment) — and report
      naive-LRU vs block-pool error side by side
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
- [ ] ⭐ **Test H6 — re-run the sweep under a rescaled cost function** (a "faster GPU", and a different
      prefill/decode ratio) and compute Kendall's τ between the policy orderings. **Pure CPU, no extra GPU
      time, the sweep already exists.** ⛔ Do this *before* the README makes any tier-1 claim: H6 is what
      says whether a user without a GPU can trust the recommendation at all (ADR-010)

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

- [ ] Sweep **prefix-sharing ratio as its own axis**, crossed with offered load
      *(`kvsim crossover` and `eval/sweep.py:crossover_sweep` are built; awaiting a calibrated profile)*
- [ ] ⭐ **Add turns-per-session as a third axis** — the first sweep suggests it decides whether
      stickiness or affinity wins, and it was not previously in the plan
- [ ] Locate the boundary where least-loaded overtakes prefix-affinity on **p95 TTFT**; publish it as a
      **curve with CIs**
- [ ] Check H2's three directional sub-predictions (monotonicity in each axis; power-of-two in between)
- [ ] If the effect sits inside the CIs, report **that** — *"no detectable crossover in this region, here is
      the region and the intervals"* is a real result
- [ ] Make it the README headline, in one sentence a reader can repeat without opening the repo

---

## Phase 6 — Sim-to-real, two replicas

- [ ] **One vLLM server per physical T4** (`CUDA_VISIBLE_DEVICES=0` / `=1`) — no MIG needed; two real
      GPUs are complete isolation. Notebook v2 already detects this and does it
- [ ] Two vLLM servers, one arrival stream, the top recommended policy actually deployed
- [ ] Predicted vs measured hit rate across the pair *(v2's round-robin arm is the first data point —
      predicted 1,088 tokens computed, hit rate 0.966862, committed before it ran)*
- [ ] ⛔ **TTFT percentiles only from separate GPUs** — never from the memory-split fallback (ADR-005).
      And they are **T4** percentiles
- [ ] **Fallback, decided in advance:** if 2 replicas prove infeasible, ship a **single-replica calibration
      plus a documented, reproducible account of why**. That beats no calibration, and the failure write-up
      is itself an OSS-PR premise

---

## Phase 7 — Package, CLI, the static demo, hardening

- [ ] `pip install kv-routing-sim`; `kvsim replay trace.jsonl --policies all --replicas 1-8
      --format json|md|html`; semver + changelog
- [ ] ⭐ **`kvsim calibrate`** — expose the Phase-3 harness so a USER can fit a profile to their own GPU
      and model (~30 min). The ADR-010 mechanism, and the answer to *"why trust your numbers on my
      hardware?"* — *"you shouldn't; run this."* Needs the `[calib]` extra
- [ ] **`configs/profiles/`** — committed calibration profiles keyed by
      `(gpu, model, block_size, max_num_seqs, engine_version)`; the CLI reports **how far** the user is
      from the nearest one. ⛔ Never interpolate between profiles — an interpolated error bar has no
      measured basis
- [ ] ⭐ **`scripts/build_grid.py`** — sweep sharing × load × replicas × policy and emit ONE JSON file.
      This is the demo's whole backend (ADR-011). Parallel via `joblib` (`[sweep]` extra); commit the
      grid plus its wall-clock and byte size, since both are currently estimates
- [ ] **The static demo page** — the one-screen design in
      the deployment notes: finding + crossover curve +
      calibration table above the fold, then three sliders that **look up** the grid (no server, no
      wait) and whose payoff is watching the recommendation **flip**.
      ⛔ **Bundled + generated traces only, no upload box** — a serving log is someone's users' prompts
- [ ] ⚠️ **Rebuild `demo_grid.json` from the CALIBRATED T4 profile.** The committed grid was built on an
      A100-shaped guess (2026-09-21) that no longer describes any hardware in the plan. It is a
      placeholder that proved the pipeline, nothing more
- [ ] Publish that page to **GitHub Pages** AND to a **free Static Space** — same build, two URLs
- [ ] ⭐ **The same curve images and calibration table go IN THE README**, for the reader who never
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
- [ ] Write the README against the **three readers** in
      the deployment notes: the 30-second reader (needs the
      grep targets *visible* — engine, version, metric names, GPU), the 2-minute reader (needs four
      lines), and the 30–60-minute engineer who asks *"isn't this just Vidur?"* (needs the ADRs and the
      residual)
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
- [ ] ⛔ **No sentence blends a calibrated error bar with an uncalibrated recommendation.** Our error bar
      belongs to our GPU and our model; the ranking is what transfers (ADR-010, pending H6). *"Validated
      within X%"* and *"your serving log in, a recommendation out"* are true separately and false together
- [ ] **Bootstrap CIs resample sessions/segments, not requests** — and every caption says which
- [ ] **A hypothesis that failed is reported as a finding**, not deleted
- [ ] **No trace containing user prompt text is committed**, ever — hashes only
- [ ] **The upstream PR:** every line reviewed; full test suite run
