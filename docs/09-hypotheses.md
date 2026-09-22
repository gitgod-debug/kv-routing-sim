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
| **H1** | Prefix-affinity dominates hit rate under high sharing | Phase 4 | `[x]` ⛔ **FALSIFIED 2026-09-22 — by load.** Holds exactly at 1–2 sessions/s; gone or reversed from 3/s, where concentration forces preemption |
| **H2** | ⭐ The crossover boundary exists and is monotonic | Phase 5 | `[ ]` untested |
| **H3** | Single-digit-% calibration error on ≥2 shapes | Phase 3 | `[x]` ✅ **PASSED 2026-09-22** — **zero** error on 2 sequential shapes under eviction pressure (3 runs each, every| **H4** | The residual is eviction-order skew (histogram-checkable) | Phase 3 | `[x]` ✅ **supported, in its second amendment's form (2026-09-22):** plain LRU's residual on a live engine is **capacity** skew (−11.9 pts when requests hold decode blocks, 0.17 without); histogram counts exact, W2 distributions identical |-predicts — but the cause is **capacity** accounting, not order |
| **H5** | Cost win = one fewer replica at fixed SLO | Phase 4 | `[~]` **holds only for balanced affinity at LOW sharing** (1–3 fewer T4s); falsified elsewhere. Session-sticky (untested) needs the fewest everywhere |
| **H6** | ⭐ Policy ranking is hardware-invariant (τ > 0.8 under cost rescale) | Phase 4 (CPU-only re-run) | `[ ]` untested · **gates ADR-010's tier-1/2 claims** |

**Amendments:** *(add dated entries below, never edit above)*

### 2026-09-22 — H4 is NOT dead. It has not been tested yet, because we used the wrong flag.
v1 of the spike reported the `kv_block_*` histograms absent. The engine's own config dump shows why:
`kv_cache_metrics=False, kv_cache_metrics_sample=0.01`. The **enable** switch is `--kv-cache-metrics`;
`--kv-cache-metrics-sample` (the flag this file and ADR-008 named) only sets the rate. The metric
names ARE in vLLM 0.29.0's source. H4 stands as written; notebook v2 passes the right flag.

### 2026-09-22 — H3 gets its first data point, and it is honest about how small it is.
On the spike workload (one shared 512-token prefix, 64 requests) the corrected cache model reproduces
vLLM's counters **to the token**: 33,344 queried, 32,768 hit, 576 computed. That validates the
*accounting* — unit, denominator, the partial-block rule. It says nothing about eviction, because the
workload never evicts. It is the first calibration point, not the calibration.

### 2026-09-22 — H2 must be tested on the vLLM cache model, not the lru one
H2 is defined on **p95 TTFT**. Building the block-pool model showed that under memory pressure TTFT is
dominated by memory-based admission control and preemption — which only that model has. On the
T4-shaped profile the two models' hit rates differ by ≤ 1.6 points while their p95 TTFT differs by
**1.8–8.5×**. The prediction itself is unchanged; only the instrument that can test it is now fixed.

### 2026-09-22 — ✅ v2's prediction HELD EXACTLY. First multi-replica sim-to-real point.
Round-robin across two physical T4s (vLLM 0.30.0): **1,088 prefill tokens computed, hit rate 0.966862 —
predicted 1,088 and 0.966862, committed before the run. Error: 0 tokens.** Per replica: 32 requests,
16,416 queried, 15,872 hit, 544 computed — each replica paid for exactly one cold copy of the prefix. The
1.89x cost of routing without affinity was measured on real hardware, not simulated.

⚠️ **Scope, stated as plainly as the result:** sequential requests, one shared prefix, no eviction, no
concurrency. It validates the *accounting* across replicas — not eviction, not batching, not timing. It
is the first sim-to-real point, and a strong one, not the calibration.

### 2026-09-22 — ⛔ H4 AMENDED. Its direction is probably wrong, and one sub-prediction cannot be measured.
Reading vLLM 0.30.0's block pool before designing Phase 3 changed H4 in three ways. The original text
above stands as what was predicted; this is what the engine's source says.

**1. The direction probably reverses.** H4 predicted the simulator would **over**-predict hit rate
because modelled blocks outlive real ones. But vLLM frees a finished request's blocks **tail-first**
(`single_type_kv_cache_manager.py:580`) and reuses uncached blocks LIFO (`block_pool.py:789`), so under
pressure it keeps shared prompt **heads** alive. Our plain LRU evicts **heads** first — the worst
possible choice for prefix reuse, since a missing head strands everything after it. So under pressure
the current model should **under**-predict. *New prediction, pre-registered here:* on an eviction-heavy
workload the naive-LRU model under-predicts the engine's hit rate, and the block-pool model (ADR-002
amendment) closes most of that gap.

**2. ⛔ The `reuse_gap_seconds` sub-prediction was mis-specified — withdrawn.** It said the histogram
would show "reuse attempts arriving after the block is gone". The instrument cannot see those: it
records gaps between **successful** accesses within one block's life (`kv_cache_metrics.py`), and an
attempt after eviction is a miss on a block that no longer exists. It also keeps only the **last 4**
accesses (`deque(maxlen=4)`, line 24) — at most three gaps per block, biased toward the end of its life.

**3. ⚠️ The three `*_evict_*` histograms mix two populations.** `on_block_evicted` is called **before**
the check for whether the block held a cached prefix (`block_pool.py:739` vs `741`), so recycling a
block that never held one — e.g. every request's partial tail block, reused LIFO — is recorded as an
"eviction". v2 shows it exactly: **64 samples with the cache never under pressure**, lifetime ≈ idle ≈
81 ms (one request), zero reuse gaps — the 64 tail blocks, not evictions.

**So H4 is tested differently from now on:** the simulator emits the **same** events by the **same**
rules — an event whenever a tracked block is recycled, hashed or not, with the last-4-accesses
truncation — and the **distributions** of lifetime, idle-before-evict and reuse-gap are compared against
the engine's. Comparing an idealised quantity against a differently-defined instrument would make the
"residual" an artefact of the definitions.

### 2026-09-22 (later) — ⛔ the H4 amendment above got the DIRECTION wrong. Measured, not argued.
With the block-pool model built, I compared it against the plain-LRU model on the same agentic traces
(sequential replay, so pinning and preemption are out of the picture) before writing the amendment's
prediction into anything else. **It predicted the pool would score higher. It scores lower.**

Decomposed by varying only the output length, at 60% cross-session sharing across capacities:

| Effect | Isolated by | pool − LRU | Size |
|---|---|---|---|
| **Eviction order** (tail-first, LIFO/FIFO) — what the amendment argued | 1-token outputs (no decode blocks) | **+0.0000 to +0.0020** | real, the predicted sign, and **negligible** |
| **Capacity accounting** — vLLM's cache also holds decode and partial blocks; LRU ignores them | 120-token outputs | **−0.0015 to −0.0192** | **dominates** |

So under pressure plain LRU **over**-predicts the faithful model's hit rate — **H4's original direction**.
The original's *mechanism* ("modelled blocks survive longer than real ones") also turns out closer to
right than the amendment's: LRU's blocks do survive longer, because LRU thinks it has more room. What
the original called *eviction-order skew* is really **capacity skew**. The order argument was correct
and irrelevant.

**Two consequences for Phase 3:**
1. **The quantity to calibrate is capacity accounting, not eviction order.** A workload whose outputs
   are long relative to its prompts should show the largest LRU-vs-engine gap; one with 1-token outputs
   almost none. That contrast is a sharper test than any histogram — and it costs nothing to run both.
2. **Open-loop replay understates vLLM's real agentic hit rate.** Here the cached output blocks are
   dead weight: no replayed prompt contains the engine's real output, so they only crowd out prompt
   blocks. In closed-loop agent traffic the next turn re-sends that output and those same blocks
   *hit*. Our calibration can only measure the first case, which is worth saying in the report, not
   discovering in review.

*The lesson is the one this file exists for: the amendment's reasoning was careful and still wrong, and
only the measurement knew. The prediction stood in the file for a few hours; this entry records it
failing, and neither is edited.*

### 2026-09-22 — a pre-registered sim-to-real prediction for notebook v2
For v2's round-robin arm (the same workload alternated across 2 replicas) the simulator predicts
**1,088 prefill tokens computed, hit rate 0.966862** — versus 576 for affinity routing, i.e. **1.89x
more prefill without affinity**. Committed before v2 runs. The notebook scores itself against it.

### 2026-09-22 — ⭐ Phase 3 PRE-REGISTERED: H3 and H4 get numbers that can fail
Committed before any GPU run, in [`artifacts/results/phase3_prereg/predictions.json`](../artifacts/results/phase3_prereg/predictions.json)
(predictions sha256 `055f57c9…`, trace fingerprint `54472296…`), generated by `scripts/phase3_prepare.py`
from [`configs/phase3_workloads.yaml`](../configs/phase3_workloads.yaml). One vLLM 0.30.0 engine, Qwen3-4B,
**512 GPU blocks by override (511 usable)**, 120 agentic requests (40 sessions × 3 turns, prompts of
383 / 483 / 583 tokens):

| | `vllm` (block-pool) model | `lru` model |
|---|---|---|
| **W2** sequential, 1-token outputs | 35,296 hits / 57,960 queried = **0.6090**; histogram count **963**, reuse gaps **99** | 35,200 / 57,960 = **0.6073** |
| **W1** sequential, 64-token outputs | 28,304 / 57,960 = **0.4883**; histogram count **1,880**, reuse gaps **3** | 35,200 / 57,960 = **0.6073** |
| **W3** concurrent, at arrival times | **preemptions > 0** (holds across a 4× timing bracket) | **0 preemptions**, hit **0.6073** (timing-independent) |

**What this makes of H3 and H4.** On W1 and W2 the predictions are exact integers: for one-at-a-time
requests the engine is deterministic, so H3's "<10% relative error" becomes "**zero** error, or a
specific named mismatch" — a stricter bar, and the one the data can actually support. H4's second
amendment said capacity accounting, not eviction order, is what separates LRU from the engine; **W1
vs W2 tests exactly that**, since they differ only in whether requests hold decode blocks. If the
engine lands on the `vllm` column for W1, the lru model's −11.9-point error *is* the capacity effect,
measured on a real engine. H4's histograms are now tested by their **counts** (exact, above) before
their shapes (a distribution comparison, run afterwards by `scripts/phase3_analyze.py`).

**Pre-registered alternative, so a miss is diagnosed rather than argued:** if the engine commits one
token more per request than its source says, W2 reads **34,928 / 0.6026** and a histogram count of
**986**. W1 cannot distinguish the two (no request ends on a block boundary) — by design, since W1's
job is the decode-block question.

**W3 is pre-registered as a procedure, not a number:** fit the timing model on W1, W2 and a separate
decode-batch sweep (never on W3), simulate W3 at its recorded send times, publish the error. Its
timing-independent claims are the only W3 numbers registered. ⚠️ One named simulator assumption
matters only here: vLLM 0.30.0 schedules **asynchronously** by default, the simulator synchronously.

The decision rule for each outcome is in the prereg file (`decision_rules`), written before the run —
including the one that downgrades concurrent claims from predictive to comparative, with no re-tuning.

### 2026-09-22 (evening) — ✅ Phase 3 RAN. H3 passes with zero error; H4's second amendment holds.
Kaggle, one Tesla T4, vLLM 0.30.0, Qwen3-4B, 511 usable blocks — against the predictions committed
above (`055f57c9…`). Full report: [`calibration_report.md`](../artifacts/results/phase3_kaggle_t4/calibration_report.md).

| | Predicted (vllm model) | Engine, 3 runs each |
|---|---|---|
| **W2** hits · computed · histogram events · reuse gaps | 35,296 · 22,664 · 963 · 99 | **35,296 · 22,664 · 963 · 99** — all three runs |
| **W1** hits · computed · histogram events · reuse gaps | 28,304 · 29,656 · 1,880 · 3 | **28,304 · 29,656 · 1,880 · 3** — all three runs |
| **per-request cached tokens** | 120 values per workload | **120 / 120** match, every run |
| **W3** preemptions | > 0 | **27**, every run |

**H3 — passed, at a stricter bar than it set itself.** "<10% relative error" became *zero*: on both
sequential shapes, under real eviction pressure, every counter and every request matched to the
token, and the three runs were identical. On the concurrent shape — predicted by a timing profile
fitted without it — hit rate **+0.3%**, computed tokens −0.3%, preemptions 24 vs 27, TTFT p50 −3.8%,
p95 and p99 −6.7%. The error's sign is consistent where it is not zero.

**H4 — supported in its second amendment's form.** On the same live runs, plain LRU over-predicts by
**11.9 points** when requests hold decode blocks (W1) and by **0.17** when they do not (W2): the gap is
**capacity** accounting — decode blocks sharing a small cache — not eviction order, exactly as the
second amendment predicted, and now measured on a real engine rather than sim-vs-sim. The histograms:
counts exact on every run; W2's distributions identical bucket for bucket; W1's small distribution
residual (max CDF distance 0.042) traced — **post-hoc, and labelled so** in
[`posthoc.md`](../artifacts/results/phase3_kaggle_t4/posthoc.md) — to the *timing* fit's batch-1
decode step, not the cache model (0.042 → 0.015 with that one number measured; cache counts unchanged).

**The pre-registered alternative was refuted:** the engine commits exactly *prompt + output − 1*
tokens per request, as its source says — W2 would have read 34,928 otherwise.

**What it did NOT show, stated as plainly:** the timing model is the weak part. Its decode fit has R²
0.58 — on this T4 one sequence decodes *slower* (63.6 ms) than two to eight — and its step model
double-charges the fixed per-pass cost when a step prefills (fitted prefill base −21 ms, clamped to 0).
Neither touches a cache count; both are owned in the README's limits rather than tuned away. And
everything here is open-loop: closed-loop decode-block reuse remains unmeasured.


### 2026-09-22 (night) — Phase 4's first calibrated sweep, REGISTERED before it runs: H1 and H5
Design and rules in [`configs/phase4_sweep.yaml`](../configs/phase4_sweep.yaml); procedure in
`scripts/run_phase4_sweep.py`. Both committed before the sweep was run.

**The setup.** The Phase-3 T4 profile at the engine's own natural capacity (**1,537 blocks**, ADR-012);
cost on AWS **g4dn.xlarge** ($0.526/h, one T4) at ₹95.87/$; SLO **p95 TTFT ≤ 2,000 ms** (MLPerf's
Conversational TTFT limit, applied at p95). Two agentic shapes — **low sharing** (achieved 0.300) and
**high sharing** (0.804, above H1's "≳70%") — at six loads (1–8 sessions/s), all seven policies, replicas
1–8, five seeds. 3,360 runs. The load range was chosen from a probe of **round-robin alone**: no
comparison between policies was looked at before this entry.

**How H1 is tested** (high sharing, at every load): radix-affinity — and, reported separately,
radix-affinity-balanced — against least-loaded on hit rate, replicas 2–8. *Strictly higher* = the
paired session-bootstrap 95% CI of the difference has its lower bound above zero at every replica count.
*Widens* = the difference at 8 replicas exceeds the one at 2, with no step down larger than 0.005.
Holds only if both hold at every load.

**How H5 is tested** (per shape): R = the fewest replicas whose cell meets the SLO on every seed.
Holds if at some load round-robin needs more than prefix-affinity, and at no load does
prefix-affinity need more than round-robin.

**What would be uncomfortable, written down now:** the uncalibrated first sweep saw pure
radix-affinity collapse onto one replica at high sharing. If that survives calibration, it will need
*more* replicas, not fewer — and **H5 fails for it** while perhaps holding for the balanced variant.
That outcome gets reported as it lands.

### 2026-09-22 (night) — ⛔ Phase 4 RAN. H1 is falsified by load; H5 holds in one case of four
Report: [`artifacts/results/phase4_sweep_t4/report.md`](../artifacts/results/phase4_sweep_t4/report.md)
(3,360 runs; re-running reproduced every cell exactly).

**H1 — falsified, and the way it fails is the interesting part.** At **1–2 sessions/s** it holds
exactly as written: radix-affinity's hit-rate lead over least-loaded is positive at every replica
count, with every CI's lower bound above zero, and widens from 2 replicas to 8 (**+0.043 → +0.110**
at 1/s). From **3/s** it collapses. Pure radix-affinity sends *every* request to the one replica
holding the shared prefix (load imbalance exactly N), preempts **110–212 times per run**, and
preemption throws away the cached blocks it concentrated traffic to keep — so its lead turns
**negative** (−0.044 at 4/s on 2 replicas). The balanced variant lasts to 3–4/s and fades to zero at
6–8/s.

What that falsification does and does not mean. H1 was written as a sanity gate — *"if H1 fails, the
cache model is broken."* It is not that: wherever there is no memory pressure, H1 holds as predicted,
so the model does credit locality. What H1 got wrong is an assumption — that hit rate is a property
of routing alone. Under load, **concentration causes preemption and preemption destroys reuse**. That
mechanism is precisely what Phase 3's concurrent run validated on a real engine (27 preemptions, hit
rate within 0.3%). It is H2's crossover showing up in the hit rate itself.

**H5 — holds in one case of four.**

| | round-robin needs (2 → 8 sessions/s) | prefix-affinity needs | verdict |
|---|---|---|---|
| low sharing · balanced | 3, 4, 5, 7, 8 | **2, 3, 3, 4, 5** | ✅ holds — 1–3 fewer T4s at every load |
| low sharing · pure | 3, 4, 5, 7, 8 | never meets the SLO | ⛔ falsified |
| high sharing · balanced | 3, 4, 5, 7, — | never meets the SLO | ⛔ falsified |
| high sharing · pure | 3, 4, 5, 7, — | never meets the SLO | ⛔ falsified |

⚠️ **One caveat that must travel with the balanced result.** Its load weight — "one queued request
costs as much as one matched block" (alpha = 1) — was **never swept**; its own docstring says so. At
high sharing the shared prefix is **85 blocks**, so that penalty barely registers and it concentrates
anyway (imbalance 1.6–1.8, up to 121 preemptions). The registered test used alpha = 1 and failed.
Whether a swept alpha rescues it is a **new experiment, registered separately in Phase 5** — not a
re-run of this one until it passes.

**Not tested, reported — and it is the result a reader will care about.** **Session-sticky** holds
the SLO with the fewest T4s at every load on both shapes (tied with balanced affinity at low sharing):
**2, 3, 3, 4, 5** against round-robin's 3, 4, 5, 7, 8. Wherever round-robin meets the SLO at all, the
recommendation is **25–42% cheaper** (₹3.89–4.79 per 1,000 requests against ₹6.39–7.25). At 8/s on
high sharing, round-robin cannot hold the SLO even with 8 T4s; session-sticky holds it with 5. It
never preempts: it spreads sessions evenly (imbalance ≤ 1.09), and each replica caches the shared
system prompt once while every conversation's own history stays with it. **The early, uncalibrated
observation survived calibration.**

**Scope, as plainly:** generated traffic — one system prompt shared by every session, 4-turn
sessions, open-loop replay — Phase 3's T4 timings, one model. Whether the ranking transfers to other
hardware is **H6**, not yet tested; until it is, none of this goes in the README as a claim about
anyone else's fleet.

