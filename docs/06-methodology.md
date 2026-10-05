# 06 — Methodology: how we measure

> The differentiator of this project is **rigor on axes that are seldom reported** — and, specifically,
> publishing how wrong the simulator is. *(It read "nobody reports" until 2026-10-05. AISimulate
> publishes an accuracy page and LLMServingSim an error against a real vLLM run; neither covers
> the prefix cache or routing — [`04-prior-art.md`](04-prior-art.md) §2a.)* That makes measurement methodology the actual product. This file is its
> specification.
>
> §1 and §2 are shared in substance with the sibling project `colpali-lite` (that is deliberate — the two
> ship as one method instantiated twice). ⚠️ **They are not shared in code yet, and must not be** — see
> ADR-007.

---

## 1. The statistics protocol

### The mistake this section exists to prevent

It is tempting to write *"every curve carries error bars (multi-seed mean±std, bootstrap CIs)"* and treat
those as one thing. **They are two different things and they apply to different quantities.** Getting it
wrong produces error bars that do not mean what the caption says.

### Which uncertainty applies where

| Quantity | Source of randomness | Correct treatment |
|---|---|---|
| **Cache hit rate on a fixed trace** | Which *requests* happened to be in the trace | **Bootstrap over requests** (block-resampled — see below). Resample, recompute, take percentiles. |
| **A policy's result at fixed trace + replicas** | The policy's own **seeded tie-breaking** | **Multiple seeds → mean ± std.** ≥5 seeds; they are cheap here, unlike a fine-tune. |
| **Difference between two policies** | Both are run on the **same** trace — the comparison is paired | **Paired bootstrap.** Resample requests, compute the *difference* per resample. |
| **TTFT** | Per-request queueing and system noise | Report the **distribution** (p50/p95/p99), never a mean ± std. |
| **Predicted vs measured (calibration)** | Both — sim seeds *and* live-engine run-to-run variance | **≥3 live runs**, report the measured spread, and state the sim's seed spread separately. A single live run is an anecdote. |

### ⚠️ The resampling unit is a trace segment, not a request

This is the one place this project's statistics differ from the sibling's, and getting it wrong inflates
confidence badly. **Requests in a trace are not independent** — an agentic session's turns share prefixes
by construction, which is the entire phenomenon under study. Resampling individual requests breaks that
dependence and produces intervals that are far too tight.

> **Resample whole sessions (or contiguous trace blocks), not individual requests.** State the unit in
> every caption. If a result is ever reported with request-level resampling, the caption must say so and
> the number must be treated as a lower bound on the true interval.

### Why the paired test matters

Comparing prefix-affinity at 0.41 hit rate against least-loaded at 0.33, each with its own CI, and noting
the intervals overlap, tells you almost nothing. Both ran on the **same trace**, so trace-difficulty
variance is *shared and cancels*. The right question is *what is the CI on the **difference***, which is
usually far tighter than either individual interval.

**Every policy-vs-baseline row in every table gets a paired bootstrap CI on the delta.**

### Defaults

- Bootstrap resamples: **1,000** (10,000 for anything in the README headline)
- Confidence level: **95%**, percentile method
- Policy seeds: **5** minimum
- Live calibration runs per workload shape: **3** minimum
- **Every seed pinned and committed.** Bootstrap seeds too.

### Determinism contract

A sweep that cannot be re-derived makes its Pareto curves unfalsifiable, which makes them worthless.

- **Seeded tie-breaking in every policy** (ADR-003). No `min()` over a dict.
- **Pinned dependencies**, and the *actually installed* versions recorded in every results header.
- **Committed workload-generator configs** — a generated trace is reproduced from a seed + a config, never
  shipped as data.
- Re-running a `run_*.py` script with the same config must produce a **byte-identical** results file, apart
  from the timestamp. If it doesn't, something is reading unseeded randomness and must be found.

### Latency measurement

- **Warm up first; discard warm-up requests.** Cold-start is a *separate*, separately-reported number.
- **Report the request count behind every percentile.** A p99 from 100 requests is one request.
- Report the load generator, the concurrency level, and the duration.
- ⛔ **Never report TTFT percentiles from the two-servers-on-one-card fallback** — SM contention there is a
  test-rig artifact, not a routing effect (ADR-005). Every results file records which hardware path
  produced it.

---

## 2. The cost model

### Why this needs care

**₹ per 1k requests is the headline number and the easiest thing in the project to get caught on.** The
calibration box is not a metered instance, so any published figure is a **reference-price** calculation.
That is fine and normal — *if the basis is explicit.* If it is hidden or hand-waved, the number is
worthless and a reviewer will say so in public.

### The rule

> Every constant in the cost model lives in a **committed config file**, with a **citation** and a
> **date**. A skeptical reader must be able to disagree with a constant, change it, re-run, and get their
> own number.

No hidden divisors. No `≈₹0.003/request` with no derivation. See `configs/cost_model.yaml` — every value
ships as `<TBD>` with `source:` and `date:` fields present, so an uncited constant is **structurally
obvious** rather than merely discouraged.

### The shape of the calculation

```
₹ per 1k successful requests
  = (replicas × ₹ per GPU-hour) ÷ (successful requests per hour) × 1000
```

Three things must be said out loud about that formula, because each is a place to be wrong:

1. **The denominator is *successful* requests** — requests that met the SLO. A configuration that serves
   more traffic by blowing p95 has not got cheaper; it has changed the product. The SLO is a committed
   config value.
2. **₹ per GPU-hour is a named cloud SKU** at a published price, with a URL and a date — not the cost of
   the box the calibration ran on. State the SKU in the caption.
3. **Replica count is the axis being swept**, so cost is not a constant multiplier: this is the whole point
   of H5 — the recommendation may save money by *removing* a replica.

### What the cost model deliberately does not include

Stated so a reader knows it is a choice, not an oversight: no networking egress, no control-plane
overhead, no engineering time, no storage (the KV cache is ephemeral). **Compute is >90% of the bill for
this workload, and adding poorly-sourced small terms makes the number less trustworthy, not more.**

---

## 3. The reproducibility contract

Every file in `artifacts/results/` opens with a header recording:

`script` · `date_utc` · `git_sha` · **`git_dirty`** · `seed` · `hardware` (platform, CPU count, GPU name,
**compute capability**, driver) · `packages` (actually-installed versions) · `schema_version` ·
**`hardware_path`** (single-replica / MIG / memory-split — ADR-005) · **`skipped_lines`** (malformed trace
records dropped).

`git_dirty` is recorded because **a number produced from an uncommitted tree is not re-derivable**, and the
file should say so rather than imply otherwise.

The header is produced by one function — `scripts/_spike_common.py:results_header()` — so it cannot drift
between scripts.

---

## 4. Calibration: what counts as a pass

The Calibration Report is the credential, so its own standard has to be explicit.

| Requirement | Why |
|---|---|
| **≥2 workload shapes** | One shape can be fitted by luck. Two make it a model. |
| **≥3 live runs per shape** | Otherwise the "measured" side has no error bar and the comparison is meaningless. |
| **The residual explained, with a measurement** | ⭐ *"The remaining N% is eviction-order skew"* is a narrative until it is checked against `vllm:kv_block_idle_before_evict_seconds`. See H4. **A matching number with no account of the gap reads as luck.** |
| **A limits section** | What the simulator gets wrong, owned publicly. One failure mode named. |
| **Raw scrapes committed** | The `/metrics` text, not just the parsed JSON. |
| **Predicted-vs-measured published even if it is bad** | ⛔ **This is the non-negotiable one.** The contribution is *publishing how wrong the model is*. A large, honestly-reported, explained error is a result. A quietly re-tuned model that matches is fraud. |

### On the target number

A published simulation paper reports **~2%** error for this class of model. That is **the bar to aim at,
cited as such — not a number to claim.** Report where it actually lands. The spec's target is
single-digit-%; if it lands at 15% and the reason is understood and stated, that is a publishable result
and a better answer than a suspicious 2%.
