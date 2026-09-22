# 05 — Codebase map

**Status: 2026-09-20 — built, except the three numbers that need an engine.** The contracts below were
written before the code (ADRs 001–004), and the code now implements them. 73 tests, `ruff` clean.

⛔ **What is deliberately NOT built: `calib/`, and the four cost coefficients in `sim/cost_model.py`.**
Those are claims about how a real GPU behaves, and no GPU has been run. They sit behind
`CostProfile.calibrated = False`, and `eval/` refuses in code to present an uncalibrated run as a
prediction.

Each section marks status: `📐 designed` (contract written, no code) · `🔨 building` · `✅ done + tested`.

---

## The pipeline, in one line

```
trace file  →  adapter  →  NormalizedRequest stream  →  ┌─ policy ─→ replica ─→ cache model ─┐
                                                        │                                     │
                                                        └──── sim loop / metrics collector ───┘
                                                                          ↓
                                              eval: bootstrap CIs → cost model → Pareto → recommendation
                                                                          ↓
                                               calib: predicted vs live /metrics → the residual
```

Two extension points, and only two — **trace adapters** (ADR-004) and **policies** (ADR-003). Both are
one file each. Everything else is core and changes rarely.

---

## `src/kv_routing_sim/traces/` — ✅ built + tested

**Contract:** turn any input format into one versioned, streamed record type.

| Piece | Responsibility |
|---|---|
| `schema.py` | `NormalizedRequest` (pydantic) + `SCHEMA_VERSION`. Fields: `request_id`, `arrival_ts`, `session_id`, `block_hashes: list[int]`, `prompt_tokens`, `output_tokens`. |
| `adapters/` | One file per format: `vllm_jsonl.py`, `sglang.py`, `openai_log.py`. Each is a generator — **never returns a list.** |
| `generator.py` | The synthetic **agentic** traffic generator: shared system prompt + tool definitions + N turns, with a controllable **prefix-sharing ratio** (the H2 sweep axis) and arrival process. |
| `hashing.py` | Prompt → block hashes, mirroring the engine's hash-chain rule (ADR-002): a block's hash depends on its contents **and its predecessors**. |

**Why it is shaped this way:** the prompt is hashed **at the adapter boundary** and never enters the
simulator. That is the memory bound *and* the privacy property — a user's trace can be processed without
their prompt text ever being held or dumped. Adapters stream because traces can be large and the ceiling
must be stated, not discovered.

**Invariants:** a malformed line **logs and continues**, and the skipped count lands in the results header.
A run that dropped 12% of its input must say so.

## `src/kv_routing_sim/cache/` — ✅ built + tested

**Contract:** given a sequence of requests routed to this replica, report cached-prefix length per request
and maintain residency.

| Piece | Responsibility |
|---|---|
| `block_trie.py` | The trie of resident blocks (plain dicts, ADR-002). `lookup(block_hashes) -> matched_depth`. |
| `eviction.py` | LRU order + optional TTL, with a hard block-capacity bound derived from a GPU memory config. |
| `replica_cache.py` | Trie + eviction + counters (`queries`, `hits`) named to mirror vLLM's own (ADR-008), so predicted and measured quantities are literally the same quantity. |
| ⭐ **`block_pool.py`** | ✅ **The default cache model (`cache_model="vllm"`), added 2026-09-22.** vLLM's block pool, every behaviour cited to its v0.30.0 source: reference counts, tail-first free, uncached-LIFO / cached-FIFO, partial and decode blocks in capacity, decode blocks hashed, the lookup cap, and eviction events by vLLM's own (quirky) rule. `block_trie.py` stays as the `"lru"` ablation baseline. |

**The tests that matter most live here:** a known prefix sequence must produce a known hit/miss pattern,
and the partial-block rule must hold (sharing 47 of 48 tokens at block size 16 shares **two** blocks, not
three). If these are wrong, every downstream number is wrong and nothing else will reveal it.

## `src/kv_routing_sim/policies/` — ✅ built + tested (**seven**, not six — see below)

**Contract:** `choose(fleet_state, request, rng) -> replica_index`. One file per policy. No global state.
No policy may mutate the fleet.

`round_robin.py` · `least_loaded.py` · `power_of_two.py` · `consistent_hash_prefix.py` ·
`radix_affinity.py` · **`radix_affinity_balanced.py`** · `session_sticky.py` · `base.py` · `registry.py`.

⭐ **A seventh policy exists that the spec did not list, and the reason is a measurement.** Pure
`radix-affinity` degenerates: every request shares the system-prompt first block, so one replica wins
every longest-prefix match forever (measured: `[0, 0, 0, 900]` across 4 replicas, ~450x worse tail
latency). Real routers — SGLang, Dynamo, and `disagg-sim`'s `kv_aware` — all carry a load term.
Publishing pure affinity as "the prefix-aware policy" would be comparing prior art's blend against our
strawman. So pure affinity stays as the honest extreme, and the balanced variant is the realistic one.

**Every tie is broken from the injected `rng`.** A `min()` over a dict silently prefers low replica indices
and fabricates locality the real engine does not have — that is a correctness bug that looks like a
result (ADR-003).

## `src/kv_routing_sim/sim/` — ✅ built + tested (timings uncalibrated)

**Contract:** the fleet, the clock, and the numbers that come out.

| Piece | Responsibility |
|---|---|
| `replica.py` | One replica: continuous-batching admission bounded by `max_num_seqs`, a prefill/decode cost function **fitted** to the live engine (ADR-001), a waiting queue. Under `cache_model="vllm"` it also follows vLLM's scheduler: running requests first, FCFS preemption of the most recent when one cannot grow, memory-based admission, no admission in a step that preempted. |
| `fleet.py` | N replicas + the router + the arrival stream; the event loop. |
| `metrics.py` | The collector: per-request TTFT, per-replica hit/query counters, queue depths. Emits **distributions, not means** (ADR-006 / methodology §1). |
| `engine.py` | `run(trace, policy, n_replicas, seed) -> RunResult`. The one function the sweep calls. |

## `src/kv_routing_sim/eval/` — ✅ built + tested

| Piece | Responsibility |
|---|---|
| `bootstrap.py` | Percentile bootstrap over requests; **paired** bootstrap for policy-vs-policy deltas. |
| `cost.py` | Reads `configs/cost_model.yaml`. **No constant is defined in Python** (ADR-006). |
| `pareto.py` | The non-dominated frontier over (hit rate, p95 TTFT, ₹/1k requests) + the plots. |
| `recommend.py` | Frontier + an SLO → *one* policy, *one* replica count, and the ₹ delta vs round-robin. **This is the output nobody else produces** — see `04-prior-art.md`. |
| `crossover.py` | The H2 finding: sweep the sharing ratio as its own axis, locate the flip boundary, publish it with CIs. |

## `src/kv_routing_sim/calib/` — 📐 designed, NOT built · ⚠️ the only GPU-touching module

| Piece | Responsibility |
|---|---|
| `metrics_scrape.py` | Scrape vLLM `/metrics`, parse the exposition format, **derive** the hit rate from two counters (ADR-008). |
| `compare.py` | Predicted vs measured per workload shape → the residual, with error bars. |
| `report.py` | Renders the Real-Engine Calibration Report section: engine, version, flags, hardware, fields consumed by name. |

⚠️ **Imports `vllm`/`httpx` lazily, inside functions** (ADR-009), and raises a message naming the
`[calib]` extra. Core install must never need a GPU wheel.

## `src/kv_routing_sim/cli.py` — ✅ built + tested

`kvsim replay trace.jsonl --policies all --replicas 1-8 --format json|md|html`. Fails **loudly at startup**
on a bad config; never silently mid-sweep (ADR-004).

---

## `scripts/` — the reproducibility boundary

Two classes, and the distinction is enforced:

- **`spike_*.py`** — throwaway measurements. Still carry the full results header (date, git SHA, dirty
  flag, seed, hardware, package versions), because a number without provenance cannot be traced back.
  `spike_vllm_metrics.py` ✅ written · `spike_two_servers.py` ✅ written.
- **`run_*.py`** — regenerate a README table. Stricter: seeded, config-driven, byte-identical on re-run.
  `run_sweep.py` ✅ · `build_grid.py` ✅ (the static demo's grid, ADR-011).
- **`phase3_*.py`** — the Real-Engine Calibration, as one pipeline with one source of truth
  (`configs/phase3_workloads.yaml`):

  | Script | Does | Runs where |
  |---|---|---|
  | `phase3_prepare.py` | trace + fingerprint + **both models' predictions** → `artifacts/results/phase3_prereg/predictions.json`, and builds the notebook with all of it embedded | here, **before** the GPU run |
  | `_phase3_notebook.py` | the notebook's cells, as source (the `.ipynb` is generated, never hand-edited) | imported by `phase3_prepare.py` |
  | `phase3_dryrun.py` | runs the notebook against a **mock engine** — a real HTTP server backed by our own vLLM model — across four failure scenarios | here, after any notebook change |
  | `phase3_analyze.py` | the pre-registered analysis: exact comparisons, the timing fit, W3 predicted by the fitted profile, H4 histograms → `calibration_report.md` + `fitted_profile.json` | here, **after** the GPU run |

  `tests/test_phase3_prereg.py` fails if the committed predictions no longer reproduce from the
  committed code, or if the notebook carries anything else.

`_spike_common.py` holds the shared header/results helpers, adapted from the sibling project
(`colpali-economics`) so both repos' artifacts are readable the same way.

## `configs/`, `artifacts/`, `tests/`

- **`configs/`** — `cost_model.yaml` (every constant cited + dated), workload grids, policy grids,
  and `phase3_workloads.yaml` (the calibration experiment). A committed config is what makes a sweep
  re-derivable.
- **`artifacts/results/`** — JSON + **raw metric scrapes**, committed (they are the calibration evidence).
  Bulk traces, sweeps and logs are gitignored. **Never hand-edited.**
- **`tests/`** — cache-model hit/miss tests first; a golden-file test on a small sweep; the property test
  that **prefix-affinity never scores below round-robin on perfectly-shared prefixes** (if it does, the
  cache model or the policy is broken, and no amount of calibration will show it).

---

## The invariant

> **Anything in `artifacts/results/` must be reproducible by running a script in `scripts/`.**
> If you can't delete it and get it back, it doesn't ship.
