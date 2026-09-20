# 05 — Codebase map

**Status: 2026-09-20 — this is the Phase-0 scaffold.** Every module below exists as a package with a
docstring stating its contract; **none of them is implemented yet.** That is deliberate: the layout is a
decision (ADRs 001–004), and writing it down before the code exists is what keeps the contract from
drifting into whatever the first implementation happened to do.

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

## `src/kv_routing_sim/traces/` — 📐 designed

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

## `src/kv_routing_sim/cache/` — 📐 designed

**Contract:** given a sequence of requests routed to this replica, report cached-prefix length per request
and maintain residency.

| Piece | Responsibility |
|---|---|
| `block_trie.py` | The trie of resident blocks (plain dicts, ADR-002). `lookup(block_hashes) -> matched_depth`. |
| `eviction.py` | LRU order + optional TTL, with a hard block-capacity bound derived from a GPU memory config. |
| `replica_cache.py` | Trie + eviction + counters (`queries`, `hits`) named to mirror vLLM's own (ADR-008), so predicted and measured quantities are literally the same quantity. |

**The tests that matter most live here:** a known prefix sequence must produce a known hit/miss pattern,
and the partial-block rule must hold (sharing 47 of 48 tokens at block size 16 shares **two** blocks, not
three). If these are wrong, every downstream number is wrong and nothing else will reveal it.

## `src/kv_routing_sim/policies/` — 📐 designed

**Contract:** `choose(fleet_state, request, rng) -> replica_index`. One file per policy. No global state.
No policy may mutate the fleet.

`round_robin.py` · `least_loaded.py` · `power_of_two.py` · `consistent_hash_prefix.py` ·
`radix_affinity.py` · `session_sticky.py` · `base.py` (the interface) · `registry.py` (name → class, so
the CLI's `--policies all` has one source of truth).

**Every tie is broken from the injected `rng`.** A `min()` over a dict silently prefers low replica indices
and fabricates locality the real engine does not have — that is a correctness bug that looks like a
result (ADR-003).

## `src/kv_routing_sim/sim/` — 📐 designed

**Contract:** the fleet, the clock, and the numbers that come out.

| Piece | Responsibility |
|---|---|
| `replica.py` | One replica: continuous-batching admission bounded by `max_num_seqs`, a prefill/decode cost function **fitted** to the live engine (ADR-001), a waiting queue. |
| `fleet.py` | N replicas + the router + the arrival stream; the event loop. |
| `metrics.py` | The collector: per-request TTFT, per-replica hit/query counters, queue depths. Emits **distributions, not means** (ADR-006 / methodology §1). |
| `engine.py` | `run(trace, policy, n_replicas, seed) -> RunResult`. The one function the sweep calls. |

## `src/kv_routing_sim/eval/` — 📐 designed

| Piece | Responsibility |
|---|---|
| `bootstrap.py` | Percentile bootstrap over requests; **paired** bootstrap for policy-vs-policy deltas. |
| `cost.py` | Reads `configs/cost_model.yaml`. **No constant is defined in Python** (ADR-006). |
| `pareto.py` | The non-dominated frontier over (hit rate, p95 TTFT, ₹/1k requests) + the plots. |
| `recommend.py` | Frontier + an SLO → *one* policy, *one* replica count, and the ₹ delta vs round-robin. **This is the output nobody else produces** — see `04-prior-art.md`. |
| `crossover.py` | The H2 finding: sweep the sharing ratio as its own axis, locate the flip boundary, publish it with CIs. |

## `src/kv_routing_sim/calib/` — 📐 designed · ⚠️ the only GPU-touching module

| Piece | Responsibility |
|---|---|
| `metrics_scrape.py` | Scrape vLLM `/metrics`, parse the exposition format, **derive** the hit rate from two counters (ADR-008). |
| `compare.py` | Predicted vs measured per workload shape → the residual, with error bars. |
| `report.py` | Renders the Real-Engine Calibration Report section: engine, version, flags, hardware, fields consumed by name. |

⚠️ **Imports `vllm`/`httpx` lazily, inside functions** (ADR-009), and raises a message naming the
`[calib]` extra. Core install must never need a GPU wheel.

## `src/kv_routing_sim/cli.py` — 📐 designed

`kvsim replay trace.jsonl --policies all --replicas 1-8 --format json|md|html`. Fails **loudly at startup**
on a bad config; never silently mid-sweep (ADR-004).

---

## `scripts/` — the reproducibility boundary

Two classes, and the distinction is enforced:

- **`spike_*.py`** — throwaway measurements. Still carry the full results header (date, git SHA, dirty
  flag, seed, hardware, package versions), because a number without provenance cannot be traced back.
  `spike_vllm_metrics.py` ✅ written · `spike_two_servers.py` ✅ written.
- **`run_*.py`** — regenerate a README table. Stricter: seeded, config-driven, byte-identical on re-run.
  None exist yet; they arrive with Phase 1.

`_spike_common.py` holds the shared header/results helpers, adapted from the sibling project
(`colpali-economics`) so both repos' artifacts are readable the same way.

## `configs/`, `artifacts/`, `tests/`

- **`configs/`** — `cost_model.yaml` (every constant cited + dated), workload grids, policy grids. A
  committed config is what makes a sweep re-derivable.
- **`artifacts/results/`** — JSON + **raw metric scrapes**, committed (they are the calibration evidence).
  Bulk traces, sweeps and logs are gitignored. **Never hand-edited.**
- **`tests/`** — cache-model hit/miss tests first; a golden-file test on a small sweep; the property test
  that **prefix-affinity never scores below round-robin on perfectly-shared prefixes** (if it does, the
  cache model or the policy is broken, and no amount of calibration will show it).

---

## The invariant

> **Anything in `artifacts/results/` must be reproducible by running a script in `scripts/`.**
> If you can't delete it and get it back, it doesn't ship.
