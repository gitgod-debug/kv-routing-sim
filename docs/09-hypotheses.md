# 09 — Pre-registered hypotheses (H1–H5)

> ## ⭐ Why this file exists, and why it is dated
>
> **Committed 2026-09-20 — before any simulator code existed, and therefore before any number existed.**
> That ordering *is* the claim. A sweep run first and interpreted afterwards produces a table; a
> prediction committed first and then tested produces a result. The difference is not rhetorical — it is
> the difference between reporting the cells that looked interesting and reporting what you said you would
> measure.
>
> This is the sibling project's hardest-won lesson, transferred: *"a grid with no hypothesis is a dataset
> dump."*
>
> **Read this before running any phase that produces a number.**

**The rules for this file:**

1. A hypothesis may be **revised only by adding a dated amendment**, never by editing the original text.
2. If a hypothesis is **falsified, that is a result** — it goes in the README. H2 being false would be a
   *more* interesting finding than H2 being true.
3. Every hypothesis names **what would falsify it** and **what decision changes if it holds**. A
   prediction with no decision attached is a horoscope.

---

## H1 — Prefix-affinity dominates on hit rate under high sharing *(the baseline, not the finding)*

**Prediction.** On agentic traffic with a high prefix-sharing ratio (≳70% of prompt tokens shared),
**radix-prefix-affinity achieves a strictly higher cache hit rate than least-loaded at every replica count
from 2 to 8**, and the gap **widens** with replica count (more replicas ⇒ more ways for a cache-blind
router to scatter a shared prefix).

**Falsified if.** Least-loaded matches or beats prefix-affinity on hit rate at any replica count under high
sharing, or the gap fails to widen with replica count.

**Decision it changes.** Almost none — and that is the point of stating it. **This is the expected result**;
`disagg-sim` already published ≈2× hit rate for its `kv_aware` router. H1 is a **sanity gate on the
simulator**, not a contribution. If H1 fails, the cache model or the policy is broken and no downstream
number means anything.

⚠️ **Never present H1 as the finding.** *"Cache-aware routing improves cache hit rate"* is a tautology
dressed as a result, and a reader will say so.

---

## H2 — ⭐ The crossover exists, and it has a boundary *(this is the finding)*

**Prediction.** There exists a region — below some prefix-sharing ratio **S\*** and above some offered load
**λ\*** — where **least-loaded beats prefix-affinity on p95 TTFT despite a lower cache hit rate**, because
locality concentrates queueing on the replicas holding the hot prefixes. The boundary is a **curve** in
(sharing ratio × offered load), not a point, and it **moves with replica count**.

**Directional sub-predictions**, so this is falsifiable rather than merely vague:

- At fixed load, raising the sharing ratio moves the boundary **toward prefix-affinity** (more sharing ⇒
  locality pays more).
- At fixed sharing, raising offered load moves it **toward least-loaded** (queueing dominates the prefill
  saving).
- **Power-of-two-choices sits between them** and its crossover happens at a *higher* load than
  least-loaded's.

**Falsified if.** Prefix-affinity wins on p95 TTFT across the entire swept region (no crossover exists at
realistic loads), **or** the boundary is not monotonic in the two axes, **or** the effect is inside the
bootstrap CIs — in which case the honest report is *"no detectable crossover in this region, here is the
region and here are the intervals."*

**Decision it changes.** Everything about the recommender. If a crossover exists, `recommend.py` **cannot**
return a single policy for a fleet — it must return a policy *as a function of measured sharing ratio and
load*, which is a materially better product than "use prefix-affinity." It also becomes the README
headline and the Show HN title.

**Why it is credibly unclaimed.** `azrabano23/disagg-sim` published *that* the locality-vs-load
counter-result exists. **Nobody published the boundary.** And it needs no new machinery — it is a reporting
axis on the sweep that Phase 4 already builds.

---

## H3 — The simulator's hit rate matches a live engine to single-digit % *(the credential)*

**Prediction.** On ≥2 workload shapes, the simulated prefix-cache hit rate is within **<10% relative
error** of the live engine's `vllm:prefix_cache_hits ÷ vllm:prefix_cache_queries`, with the sign of the
error **consistent across shapes** (a consistent bias is a model defect that can be explained; a sign flip
between shapes means something unmodelled is dominating).

**Falsified if.** Relative error ≥10% on either shape, or the error changes sign between shapes.

**Decision it changes.** Whether the Pareto sweep may be published as *predicted deployment behaviour* at
all. If H3 fails, the sweep is still publishable — as a **comparative** study ("policy A vs B on the same
model") rather than a **predictive** one ("your fleet will see X"). That is a weaker but honest claim, and
choosing it deliberately is far better than publishing predictions the calibration does not support.

**The bar, cited honestly.** A published simulation paper reports **~2%** for this class of model. That is
the bar **to aim at**, not a number to claim. ⛔ **The predicted-vs-measured table is published whatever it
says.** A large, explained error is a result; a quietly re-tuned model that matches is fraud.

---

## H4 — The residual is eviction-order skew, and the `kv_block_*` histograms will show it

**Prediction.** The gap in H3 is dominated by **eviction-order skew**: the simulator's LRU order diverges
from the engine's real eviction sequence (which is perturbed by chunked prefill, block-table churn and
scheduler decisions the simulator does not model). Specifically:

- The simulator **over-predicts** hit rate, because modelled blocks survive longer than real ones.
- `vllm:kv_block_idle_before_evict_seconds` will show a **left-heavier** distribution than the simulator's
  implied idle-before-evict times — i.e. real blocks are evicted sooner than modelled ones.
- `vllm:kv_block_reuse_gap_seconds` will show a **tail beyond the modelled cache lifetime**: reuse attempts
  arriving after the block is gone. Those are exactly the mispredicted misses.
- Forcing the simulator's block capacity **down** to match the observed idle-before-evict distribution
  should **shrink the H3 residual** — which is the strongest available test, because it is a prediction
  about an intervention, not just a correlation.

**Falsified if.** The residual's sign is the other way (simulator *under*-predicts), or the histograms show
no such divergence, or capacity-matching does not reduce the residual — in which case the real cause must
be found and named. Candidates to check next, in order: hash-chain granularity mismatch, requests preempted
and re-prefilled, and cross-request block sharing the simulator credits to the wrong replica.

**Decision it changes.** ⭐ **This is the hypothesis that converts a narrative into a measurement.** The
spec's own words: *"the remaining N% is eviction-order skew"* is a story until it is checked. Whether H4
holds or fails, the Calibration Report gains a measured account of the gap — and that account is the
difference between a careful answer and a plausible one.

⚠️ **H4 is only testable if `--kv-cache-metrics-sample` actually exposes those three histograms on the
installed version.** Confirming that is a **Phase-0 checkbox**. If they are absent, H4 becomes
untestable-as-stated and must be rewritten around what *is* observable — record that as a dated amendment
here rather than silently keeping the narrative.

---

## H5 — Prefix-affinity's cost win comes from *removing* a replica

**Prediction.** At a fixed p95 TTFT SLO, the **cost-optimal replica count is strictly lower** under
prefix-affinity than under round-robin — i.e. the recommendation saves money by needing **fewer GPUs for
the same SLO**, not by serving more traffic on the same fleet. Concretely: there is a load at which
round-robin needs N+1 replicas to hold the SLO and prefix-affinity holds it with N.

**Falsified if.** The cost-optimal replica count is the same under both policies across the swept load
range (the win is purely per-request efficiency, not a sizing change), or prefix-affinity needs *more*
replicas anywhere in the range.

**Decision it changes.** How the headline is phrased, and it is the difference between an interesting
number and a budget line. *"Cuts cost ~X%"* is an efficiency claim. *"Holds the same p95 with one fewer
GPU"* is a procurement decision, and it is the sentence an infra lead repeats to their manager. It also
makes the ₹ cost model load-bearing rather than decorative: the saving is a **discrete** step (one
replica), so the cost model's SKU price is what sets its size.

---

## H6 — ⭐ Policy *ranking* is invariant to hardware, even when absolute latency is not

*(Added 2026-09-20, still before any code. This is the hypothesis ADR-010 rests on, and it is the most
load-bearing one in the file for anyone who is not us: it is what every user who never runs a calibration
is implicitly trusting.)*

**Prediction.** The **ordering** of routing policies — and the **existence and rough position** of the H2
crossover — depend on **workload shape** (prefix-sharing ratio, offered load, session structure) and
**not** on the GPU model, the LLM, or the absolute speed of prefill. Concretely:

- Re-run the Phase-4 sweep with the prefill/decode cost function scaled by a constant factor (a "faster
  GPU") and by a different prefill/decode *ratio* (a different model shape). **The recommended policy at
  each (sharing ratio, load) point should not change**, even though every latency number does.
- The crossover boundary **S\*** may **shift**, but should not **invert** — prefix-affinity should not
  become the wrong answer at high sharing just because the hardware got faster.
- ⭐ The strongest form, and the one to actually test: a **rank correlation** (Kendall's τ) between the
  policy orderings under two different cost-function regimes should be **high (τ > 0.8)** at matched
  workload points.

**Falsified if.** The recommended policy flips at any swept point under a pure cost-function rescale, or
τ drops below ~0.8, or the crossover boundary inverts rather than shifting.

**Why it is plausible but not obvious — state both halves.** The mechanism behind the crossover is a
*competition between two effects*: cache locality reduces work, and concentration increases queueing. Both
scale with the same prefill cost, so their **ratio** — which is what decides the winner — should be
largely invariant. ⚠️ But that argument is not airtight: `max_num_seqs` and the KV-cache block capacity are
**absolute** limits, not scaled ones, so a faster GPU with the same memory reaches saturation at a
different point. **That is precisely why this is a hypothesis and not an assumption.**

**Decision it changes.** ⭐ **Everything about who can use the tool.**

- **If H6 holds:** tiers 1 and 2 of ADR-010 are real. A user gets a trustworthy policy recommendation
  **without owning a GPU or running a calibration**, and `kvsim calibrate` is an optional upgrade for
  people who want absolute numbers. The project is a **decide-now** tool.
- **If H6 fails:** tiers 1 and 2 collapse into tier 3. `kvsim calibrate` becomes **mandatory**, the README
  can promise nothing to a user who has not run it, and the project is a **calibrate-then-decide** tool —
  a materially smaller product with a much narrower audience.

**Cost to test: near zero, and it needs no extra GPU time.** It is a re-run of the Phase-4 sweep under a
rescaled cost function — pure CPU, and the sweep already exists. ⛔ **There is no excuse for shipping the
ADR-010 claims untested**, and testing it early is what stops the README from writing a cheque the
calibration cannot cash.

---

## Status

| | Hypothesis | Tested in | Status |
|---|---|---|---|
| **H1** | Prefix-affinity dominates hit rate under high sharing | Phase 4 | `[ ]` untested |
| **H2** | ⭐ The crossover boundary exists and is monotonic | Phase 5 | `[ ]` untested |
| **H3** | Single-digit-% calibration error on ≥2 shapes | Phase 3 | `[ ]` untested |
| **H4** | The residual is eviction-order skew (histogram-checkable) | Phase 3 | `[ ]` untested · ⚠️ gated on a Phase-0 check |
| **H5** | Cost win = one fewer replica at fixed SLO | Phase 4 | `[ ]` untested |
| **H6** | ⭐ Policy ranking is hardware-invariant (τ > 0.8 under cost rescale) | Phase 4 (CPU-only re-run) | `[ ]` untested · **gates ADR-010's tier-1/2 claims** |

**Amendments:** *(none yet — add dated entries below, never edit above)*
