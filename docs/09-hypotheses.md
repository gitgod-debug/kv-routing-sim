# 09 — Pre-registered hypotheses (H1–H6)

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
| **H2** | ⭐ The crossover boundary exists and is monotonic | Phase 5 | `[x]` ⛔ **FALSIFIED 2026-09-28** — pure affinity loses at every sharing > 0 (the opposite direction); the balanced policy has no monotone boundary |
| **H2b** | ⭐ Prefix-affinity pays once the tenants' prompts outgrow one replica's cache | Phase 5, then re-tested on C2's timing model | `[x]` ✅ held 2026-09-28 **on the first, context-free timing model** — sticky wins at K = 1; affinity holds the SLO at K = 64 where sticky cannot on 8 · ⛔ **FALSIFIED 2026-09-30 on the validated model**: at K = 64 affinity wins at no load (ties at 0.25–1 session/s, 11 T4s against sticky's 10 at 2) · ⚠️ **2026-10-05: that verdict does not survive a second calibration** — the same rules HOLD on two further calibrations of the same T4 (reported and post hoc, C3): the deciding point moves by 1–2 replicas |
| **H3** | Single-digit-% calibration error on ≥2 shapes | Phase 3 | `[x]` ✅ **PASSED 2026-09-22** — **zero** error on 2 sequential shapes under eviction pressure (3 runs each, every counter and all 120 requests' cached tokens); **+0.3%** on the concurrent shape |
| **H4** | The residual is eviction-order skew (histogram-checkable) | Phase 3 | `[x]` ✅ **supported, in its second amendment's form (2026-09-22):** plain LRU's residual on a live engine is **capacity** skew (−11.9 pts when requests hold decode blocks, 0.17 without); histogram counts exact, W2 distributions identical |
| **H5** | Cost win = one fewer replica at fixed SLO | Phase 4 | `[~]` **holds only for balanced affinity at LOW sharing** (1–3 fewer T4s); falsified elsewhere. Session-sticky (untested) needs the fewest everywhere |
| **R1** | Real traffic (Mooncake): affinity beats sticky on tool-agent, never on conversation | R1 | `[x]` ⛔ **FALSIFIED 2026-09-28** — both directions wrong; sticky and balanced affinity within one replica; pure affinity fails everywhere |
| **H6** | ⭐ Policy ranking is hardware-invariant (τ > 0.8 under cost rescale) | Phase 4 + 5 (CPU-only re-runs) | `[x]` ⛔ ranking half **FALSIFIED** — τ-b down to 0.51 · ✅ crossover half **HOLDS** — H2b's flip shifts but never inverts in all four regimes (both 2026-09-28) |

| **R2** | Real traffic (AgentX): with nothing shared across users, sticky and balanced affinity tie, and pure affinity does not collapse | R2 | `[x]` ✅ **HOLDS 2026-09-28** (on the context-free timing model) |
| **Phase 6** | The simulator's hit rates hold on two live replicas: each within 10%, policies in the predicted order | Phase 6 | `[x]` ✅ **HOLDS 2026-09-28**, run twice — every hit rate within 1.4% · ⚠️ latency came out 5–222× the prediction (not a registered rule; explained by C2) |
| **C1** | `kvsim calibrate` works on a real GPU: its cache check exact, the capacity it reads equal to the engine's | C1 | `[x]` ✅ **HOLDS 2026-09-28** — 48 of 48 requests; its straight-line timing fit is rough |
| **C2** | Timing: decode slows with context (C2a); a busy second GPU slows the first (C2b); a context-aware model predicts held-out traffic within 2× (C2c) | C2 | `[x]` **2026-09-30, all three as predicted** — C2a ✅ 1.92× · C2b ⛔ 1.00× · C2c ✅ within 1.41× |
| **C3** | The context-aware `kvsim calibrate` on a real GPU: it works (C3a); its profile predicts traffic it never saw within 2× (C3b); C2's profile predicts another day within 2×, from numbers registered before (C3c) | C3 | `[x]` ✅ **2026-10-05, all three as predicted** — C3a exact · C3b within 1.41× · C3c within 1.38× · ⚠️ reported: a second run of the command is 4.17× off on one number at the knee, and a second calibration reverses H2b's verdict |

| **T1** | The simulator's per-token latency matches the engine's: mean inter-token latency within 1.5× on the five held-out runs on disk | T1 | `[x]` ⛔ **DOES NOT HOLD 2026-10-05** (predicted: holds) — within 1.07× on the three runs off the knee, **1.54× and 1.71× too slow** on the two at it |
| **T2** | Under a target with both halves (p95 TTFT ≤ 2 s **and** p95 time per output token ≤ 100 ms), on three calibrations: sticky still best or tied with 1 and 4 prompts (T2a), never worse than round-robin or least-loaded (T2b), and the first-token counts were optimistic by ≥ 1.5× (T2c) | T2 | `[x]` ✅ **2026-10-05, all three as predicted, on all three calibrations** — T2a 12 of 12 · T2b no exception · T2c 1.93–2.00× · ⚠️ these are the model's counts, and T1 says its per-token latency is unchecked where they live |

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

### 2026-09-28 — H6's ranking half, REGISTERED before it runs
Design and rules: [`configs/h6_ranking.yaml`](../configs/h6_ranking.yaml). Procedure:
`scripts/run_h6.py`. Both are committed before the run.

**The setup.** Phase 4's design, unchanged — 2 traces × 6 loads × 7 policies × replicas 1–8 × 5
seeds, the vllm cache model, the SLO and the price — re-run under five regimes. One is the measured
T4 profile. The other four multiply its timing coefficients: **2× and 4× faster** (prefill and decode
together), and **prefill alone 2× cheaper or 2× dearer**. KV capacity, `max_num_seqs`, the SLO and
the price do not change. Those fixed limits are exactly why H6's own text says it might fail. The
baseline must reproduce Phase 4's required-replica table exactly, or the script stops.

**How it is decided,** at every (trace, load) point, for each rescaled regime against the baseline.
R(policy) is the fewest replicas that hold the SLO on every seed (9 if none).
- **Flip:** the baseline's recommendation is not among the regime's lowest-R policies — following our
  advice on that hardware would cost at least one more replica. *n/a* where no policy holds the SLO.
- **Kendall's τ-b** between the two R vectors (`eval/rank.py`). *n/a* where either one is all ties.

The ranking half **holds** only if every rescaled regime has **zero flips** and **every defined τ-b ≥
0.8**. A regime with no defined τ-b anywhere is *untestable*, and does not count as holding.

**Why "flip" is defined this way — written before the result.** H6's text says the recommended
policy "should not change". Read literally, that is decided by rupee differences of a few percent
between policies that need the same number of replicas: noise, not ranking. The rule above asks what
a user needs to know instead: does our pick cost them a GPU on their hardware? The literal reading
(*strict*) and a kinder one (the two best sets share no policy, *lenient*) are both reported beside
it, so neither can be hidden.

**What would be uncomfortable, written down now.** At 1–2 sessions/s a faster GPU fits almost every
policy on the same few replicas, so many points will be all ties and the test will be decided at the
loaded points. And this entry does **not** test H6's crossover half: that needs Phase 5's grid, and
gets its own entry.

### 2026-09-28 — ⛔ H6's ranking half RAN: FALSIFIED. The full order moves with the hardware; the winner did not.
Report: [`artifacts/results/h6_ranking/report.md`](../artifacts/results/h6_ranking/report.md) — 16,800
runs in 690 s, from a clean tree at `ff2ba87`. The baseline reproduced Phase 4's required-replica
table exactly, so the rescaled regimes are compared with the numbers Phase 4 published.

**As registered: FALSIFIED in all four rescaled regimes.** Every regime has a point with τ-b below
0.8 — minimums **0.52** (2× faster), **0.65** (4× faster), **0.65** (prefill 2× cheaper) and **0.51**
(prefill 2× dearer). The dearer-prefill regime also has **one flip**: at low sharing and 3 sessions/s,
our T4 pick (radix-affinity-balanced, 3 replicas on the T4) needs 5 replicas there, where
session-sticky needs 4. Strict flips: 4, 7, 2, 2. Lenient flips: 0 in every regime.

**What moved.** Mostly the middle and the bottom of the order. With a faster GPU, whole groups of
policies tie on the same few replicas, and those ties resolve differently from the T4's order. With
prefill 2× dearer, least-loaded falls far behind round-robin: at low sharing it cannot hold the SLO at
2 sessions/s even on 8 replicas, where round-robin needs 4. ⚠️ **Not yet understood** — logged as an
open question rather than explained after the fact. The full order of seven policies is not a property
of the workload alone.

**What did not move — noticed after the run, not registered, labelled so.** Session-sticky was in the
best set (fewest replicas) at **all 48 points** in every regime. The one registered flip sits where the
T4 baseline itself was tied: radix-affinity-balanced and session-sticky both needed 3 replicas there,
and the recommender picked the first on a rupee difference of **0.1%** (₹4.7796 vs ₹4.7842 per 1k).

**Decision, as H6's own text says it must be.** Tiers 1 and 2 of ADR-010 do not survive as stated: the
README no longer says a ranking transfers to other hardware for free (changed the same day). What it
may say is narrower and measured: on this workload, across these five cost regimes, session-sticky
needed the fewest replicas at every point. `kvsim calibrate` moves up, because it is now the only
honest way to give a user a full ranking for their own hardware. And one product change the result
points at, to be made as its own change and not folded into this entry: when policies tie on replicas,
the recommender should report the tie rather than break it on rupee noise.

### 2026-09-28 — Phase 5 REGISTERED before it runs: A1 (the alpha sweep), H2's procedure, and a new H2b
Design and every rule: [`configs/phase5_crossover.yaml`](../configs/phase5_crossover.yaml). Procedure:
`scripts/run_phase5.py`. Both are committed before the run. As for H6, a pipeline check ran first on a
6-session, 0.5 sessions/s config outside the design, where every policy fits on one replica.

**Why a new hypothesis — from the traffic generator, not from any result.** Every trace so far had ONE
system prompt for the whole fleet: 85 blocks at high sharing, against 1,536 usable blocks per T4
replica. A prompt that small fits in every replica's cache many times over, so any policy ends up with a
copy everywhere, and prefix-aware routing has nothing left to win by choosing where it lives. That is
the most likely reason session-sticky beat every prefix-aware policy in Phase 4 — and it means H2, run on
one prompt, may not test prefix-aware routing where it can pay at all. Real fleets serve many tenants
and agents, each with its own long prompt. The generator now takes K distinct prompts with Zipf
popularity (`n_shared_prefixes`, `prefix_zipf_s`); K = 1 reproduces every earlier trace byte for byte,
and the Phase 3 fingerprint test still passes.

**A1 — alpha.** radix-affinity-balanced scores `matched blocks − α × queued requests`; Phase 4 used
α = 1 and never swept it. A1 sweeps α ∈ {1, 2, 4, 8, 16, 32, 64} on Phase 4's two traces, unchanged.
**α\*** = the value with the fewest required replicas summed over both traces and all six loads (9 for
"not met at 8"); ties go to the smaller α. α\* is then **fixed** for H2 and H2b. Guard: α = 1 must
reproduce Phase 4's row exactly. Choosing α on Phase 4's traces and testing it on new ones keeps the
choice out of the test.

**H2 — the procedure for the hypothesis registered on 2026-09-20.** One shared prompt; sharing 0.0–0.8
× 1–8 sessions/s; p95 TTFT at a fixed 4 and 8 replicas; paired session-bootstrap 95% CIs on p95(X) −
p95(P), for X = least-loaded and X = power-of-two. The four checks — a crossover exists, monotone along
sharing, monotone along load, power-of-two in between — are written out in the config. H2 holds for a
policy only if all four hold at both replica counts. It is tested on radix-affinity (H2's
"prefix-affinity", as in H1) and, reported separately under the same rules, on radix-affinity-balanced
at α\*.

**What would be uncomfortable, written down now.** Pure radix-affinity sends every new session to the
replica that already holds block 0 of the shared prompt. With ANY shared prompt (sharing > 0) that is
one replica, so its collapse should not depend on how long the prompt is — and sub-prediction 1 ("more
sharing moves the boundary toward prefix-affinity") may fail for it for exactly that reason. At sharing
0.0 there is no shared block 0, and pure affinity should behave like session-sticky.

**H2b (new) — the shared-prefix working set.**
*Prediction.* With K tenant prompts of 85 blocks each (sharing 0.8, Zipf exponent 1), balanced affinity
at α\* needs **fewer** replicas than session-sticky once the prompts no longer fit in one replica's cache
— and not before. *Concretely:* (a) at K = 1 affinity wins at no load; (b) at K = 64 (a 4,080-block
working set, 2.7× one replica) it wins at one load or more; (c) at every load, no sticky win at a larger
K than an affinity win. *Falsified if* any of the three fails.
*Mechanism.* Session-sticky spreads each tenant's sessions over every replica, so every replica has to
cache every popular prompt. Affinity keeps a tenant's sessions together, so the fleet caches each prompt
about once.
*Decision it changes.* Whether the headline can be "use prefix-aware routing when your tenants' prompts
outgrow one GPU's cache; otherwise keep each user on one GPU" rather than "session-sticky wins". It also
decides which real trace matters most: Mooncake (many prompts, global hashes) is where H2b can show up;
AgentX (hashes scoped to one session) cannot show it.

### 2026-09-28 — Phase 5 RAN. H2 is FALSIFIED for both affinity policies; H2b HOLDS.
Report: [`artifacts/results/phase5_crossover/report.md`](../artifacts/results/phase5_crossover/report.md)
— 14,592 runs in 570 s, from a clean tree at `3967a56`. The α = 1 guard reproduced Phase 4 exactly.

**A1 — α\* = 16.** Total replicas over both Phase-4 traces: 64 at α = 1 (Phase 4's setting), falling to
**45 at α = 16**, then rising again (48 at 32, 49 at 64). Even at α\*, balanced affinity needs more
replicas than session-sticky did in Phase 4 on the same traces (45 in total against 36).

**H2 — FALSIFIED for both policies, as registered.**
- *Pure radix-affinity:* a crossover exists, but it runs the opposite way to sub-prediction 1. At
  sharing 0.0 it beats least-loaded at every load, on 4 and 8 replicas. Above 0.0 it loses at every
  load from 2 sessions/s — by **60–260 s** of p95 TTFT — and at 1 session/s in 7 of 8 cells (one tie),
  by 0.1–0.17 s. How long the shared prompt is does not matter; that there is one does. This is the
  outcome written down as uncomfortable before the run: every new session goes to the one replica
  holding block 0.
- *Balanced affinity at α\* = 16:* no monotone boundary. On 4 replicas it beats least-loaded in 23 of 30
  cells and loses in 1, winning by most under load (+4 to +17 s at 4–8 sessions/s). On 8 replicas
  least-loaded wins 4 cells, three of them at sharing 0.8, and not monotonically. The shape H2 predicted
  — least-loaded taking over above some load — does not appear on 4 replicas at all: there, under load,
  locality helps.

**H2b — HOLDS.** (a) At K = 1 affinity wins nowhere; session-sticky needs fewer replicas at 4 of 6
loads. (b) At K = 64 (48 prompts actually used: a 4,080-block working set, 2.66× one replica's cache)
balanced affinity holds the SLO on 5 replicas at 2 sessions/s and on 8 at 3/s, where session-sticky
cannot hold it on 8. (c) No reversal along K. The mechanism shows in the hit rates (4 replicas,
2 sessions/s): from K = 1 to 64, round-robin falls from 0.84 to 0.53 and session-sticky from 0.92 to
0.82, while pure affinity stays at 0.87–0.92.

**Reported, not tested — selected after the run, labelled so.** The best α falls as K grows: 8–16 at
K = 1 under load, **1** in nearly every cell at K = 16 and K = 64. With that α, affinity already beats
session-sticky at K = 16 (3 vs 5 replicas at 3/s, 4 vs 6 at 4/s), and pure radix-affinity needs 3, 4,
5, 5 replicas at 1–4 sessions/s at K = 64, where session-sticky needs 3 and then cannot hold the SLO at
all. α\* was chosen on one-prompt traffic and is too cautious for many prompts. **The weight on prefix
matching should grow with the number of distinct prompts** — the kind of evidence a router's hand-set
prefix and load weights (llm-d's scorers) need.

**Open question, second sighting.** Least-loaded again does worse than round-robin — at K = 64 it needs
8 replicas at 1 session/s where round-robin needs 5 — as it did with dearer prefill in H6. One candidate
cause, not yet checked: a balancer that counts requests treats a replica stuck in one long prefill as
lightly loaded, and keeps sending it more.

**Decision.** The README gets its first measured finding, with its scope: *keep each user on one GPU
until your tenants' prompts outgrow one GPU's cache; then route by prefix, with a small load weight.*
It is a T4 result (H6) on generated traffic. Whether the flip survives other hardware is H6's crossover
half, registered next; Mooncake is the real trace it points at.

### 2026-09-28 — H6's crossover half REGISTERED before it runs: does H2b's flip survive other hardware?
Design and rules: [`configs/h6_crossover.yaml`](../configs/h6_crossover.yaml). Procedure:
`scripts/run_h6_crossover.py`. Both are committed before the run; a pipeline check ran first on a
6-session, 0.5 sessions/s config outside the design.

**The setup.** Phase 5's H2b grid unchanged — K = 1, 4, 16, 64 tenant prompts × 1–8 sessions/s ×
replicas 1–8 × 3 seeds, radix-affinity-balanced at α\* = 16 against session-sticky, plus round-robin,
least-loaded and pure radix-affinity for the record — under the five cost regimes of H6's ranking
half. Guard: the t4 regime must reproduce Phase 5's H2b table exactly.

**How it is decided.** H6's own text: the boundary "may shift, but should not invert". So a rescaled
regime holds only if (i) at K = 1 affinity beats session-sticky at no load, (ii) at K = 64
session-sticky beats affinity at no load, and (iii) at no load does sticky win at a larger K than
affinity. H6's crossover half holds only if all four rescaled regimes hold. Whether the flip is still
*visible* — an affinity win at K = 64 — is reported, not tested: a regime where everything ties has
shifted, not inverted.

**What would be uncomfortable, written down now.** Nothing in these regimes changes the cache size,
and H2b's mechanism is about the cache, so the flip should survive. But a faster GPU makes ties more
common (H6's ranking half saw whole groups of policies tie), so the flip may stop being visible. And
dearer prefill makes every cache hit worth more, which could let affinity win already at K = 1 — an
inversion under rule (i).

### 2026-09-28 — ✅ H6's crossover half RAN: HOLDS. The flip shifts with the hardware; it never inverts.
Report: [`artifacts/results/h6_crossover/report.md`](../artifacts/results/h6_crossover/report.md) —
14,400 runs in 570 s, from a clean tree at `16fcc53`. The t4 regime reproduced Phase 5's H2b table
exactly.

**As registered: HOLDS in all four rescaled regimes.** In none of them does balanced affinity (α\* =
16) beat session-sticky at K = 1, does session-sticky beat it at K = 64, or does the order reverse
along K.

**How it shifted.** With a GPU 2× or 4× faster, or prefill 2× cheaper, the flip is still visible at
K = 64 and now appears already at K = 16 — for example, 2× faster: 4 vs 5 replicas at 4 sessions/s at
K = 16. With prefill 2× dearer, no policy holds the SLO at K = 64 on 8 replicas (every cell ties at
"not met"), and the flip shows at K = 16 instead: 5 replicas against not held at 1 session/s. At
K = 1, session-sticky still needs fewer replicas than balanced affinity at every load in that regime.

**Decision.** H6's two halves now say different things, and both belong in the README: the full order
of seven policies moves with the hardware (ranking half, falsified), but the one decision the headline
makes — session-sticky with one shared prompt, prefix-affinity once many prompts outgrow the cache —
survived every rescale (crossover half, holds). So the headline may drop "T4 only" for the *direction*
of the flip; the replica counts and rupees stay T4 figures. Real traffic is next (R1, Mooncake).

### 2026-09-28 — R1 REGISTERED before the comparison runs: real traffic, the Mooncake FAST'25 traces
Design and rules: [`configs/realtrace_mooncake.yaml`](../configs/realtrace_mooncake.yaml). Procedure:
`scripts/run_realtrace.py`, reading the traces through the new `mooncake` adapter. Both are committed
before the comparison runs. The traces (Apache-2.0) are not committed; the results record each file's
sha256.

**What was measured before registering — trace structure, and round-robin alone.** The tool-agent
trace (23,608 requests in one hour) and the conversation trace (12,031) carry no session ids; the
adapter infers them from prompts that extend an earlier request's full blocks, never through a prefix
that 100 or more different blocks follow (three shared system prompts have fan-outs of 3,335–8,894;
no other block exceeds 15). Inferred: 26% of tool-agent and 50% of conversation requests belong to
multi-turn sessions; 39% and 10% of block uses are shared across sessions. A round-robin-only probe at
token scale 1/16 could not hold the SLO at any speed — the longest prompts' prefill alone passed 2 s —
so the scale is **1/32**, the coarsest that keeps one of our blocks per trace block. At 1/32 the
blocks shared across sessions come to **2.4×** (tool-agent) and **2.7×** (conversation) one T4's
cache, and round-robin needs 4, 5, 8 (tool-agent) and 5, 5, 8 (conversation) replicas at 1×, 2×, 4×
the recorded arrival rate — the three speeds, fixed now.

**Prediction.** From H2b's mechanism and the structure above:
- *Tool-agent:* balanced affinity at α\* = 16 needs **fewer** replicas than session-sticky at one speed
  or more, and more at none — its shared prompts outgrow one cache, and 39% of its reuse is between
  users, which only a prefix-aware router can place.
- *Conversation:* balanced affinity needs fewer replicas than session-sticky at **no** speed — only
  10% of its reuse is between users; the rest is a user's own history, which session-sticky already
  keeps together.

*Falsified if* either fails. *Reported, not tested:* every policy's replicas and hit rates, α = 1, pure
radix-affinity and consistent-hash (expected to pile the few system-prompt families onto a few
replicas), and the within/cross-session reuse split — the plan's *"how much real agentic reuse is
between users"* question.

**What would be uncomfortable, written down now.** The shared blocks are spread over an hour; if few
are hot at the same time, the working set that matters is far smaller than 2.4×, and sticky may tie or
win on tool-agent too. And every session-sticky number rests on inferred sessions.

### 2026-09-28 — ⛔ R1 RAN: FALSIFIED on both traces. On real traffic the two winners are within one replica.
Report: [`artifacts/results/realtrace_mooncake/report.md`](../artifacts/results/realtrace_mooncake/report.md)
— 1,152 runs in 235 s, from a clean tree at `3f61f9e`; trace sha256s recorded in the results file.

**As registered: FALSIFIED — and in both directions.** *Tool-agent:* balanced affinity at α\* = 16 never
needed fewer replicas than session-sticky (tie at 1× and 4×; one more at 2×: 5 vs 4). *Conversation:*
it did, once — 4 against 5 replicas at 1× — and tied at 2× and 4×. Both predictions came from H2b's
mechanism plus the measured size of the cross-session shared blocks (2.4× and 2.7× one cache), and both
were wrong.

**What the real traces show instead.**
- Session-sticky and balanced affinity (either α) are **within one replica** of each other at every
  speed on both traces; their hit rates at 4 replicas are within about one point (tool-agent 0.42 vs
  0.42; conversation 0.14 vs 0.13–0.15).
- Both beat the load balancers under load: on tool-agent at 4×, session-sticky holds the SLO on 6 T4s
  where round-robin needs 8; the hit rate is 0.42 against 0.37.
- **Pure radix-affinity fails the SLO at every speed on both traces**, as expected: a handful of shared
  system prompts start almost every request, so it piles them onto a few replicas.
- Consistent-hash on the prefix gets the best tool-agent hit rate (0.43) and still needs more replicas
  (5, 6, not met) — locality without load feedback.

**Why H2b's many-prompt advantage did not appear — the "uncomfortable" outcome written down before the
run, not yet tested.** The shared blocks are counted over a whole hour; few of them are in use at the
same time, so the working set a cache must hold at once is probably far smaller than 2.4×. H2b's
generated tenants, by contrast, were all active together. A measure of the *concurrently* shared working
set is the natural next test, not this entry's claim.

**Decision.** The README headline keeps its mechanism (generated traffic, H2b; hardware-robust, H6) but
must now say what real traffic showed: on the Mooncake traces, session-sticky and prefix-affinity with a
load weight tie within a replica, both beat round-robin, and pure prefix-affinity fails. *"Route by
prefix once prompts outgrow the cache"* is not yet shown on real traffic.

### 2026-09-28 — Phase 6 REGISTERED before the GPU run: H2b's mechanism on two live replicas
Design and rules: [`configs/phase6_sim2real.yaml`](../configs/phase6_sim2real.yaml). Predictions:
[`artifacts/results/phase6_prereg/predictions.json`](../artifacts/results/phase6_prereg/predictions.json),
generated from a clean tree at `8951a1e` together with
[`notebooks/phase6_sim2real.ipynb`](../notebooks/phase6_sim2real.ipynb), which embeds their hash and the
traces' fingerprints.

**The setup.** Two vLLM 0.30.0 servers, Qwen3-4B, one per physical T4 (1,537 blocks each, fixed by
override — the calibrated profile's capacity). H2b's traffic at 1 session/s, 120 sessions: one shared
prompt (K1) and 64 tenant prompts (K64). Three routers — round-robin, session-sticky, radix-affinity —
run in the notebook as ports of the simulator's own policies, with the same tie-breaking stream. The
notebook regenerates the traffic with a port of the generator and stops unless the fingerprints match;
the prepare script checked that port against the real generator and hashing before writing it.

**The predictions (simulator, calibrated T4 profile, 2 replicas).**

| | round-robin | session-sticky | radix-affinity |
|---|---:|---:|---:|
| K1 hit rate | 0.8795 | 0.9214 | 0.9230 (all 480 requests on one replica) |
| K64 hit rate | 0.6944 (34 preemptions) | 0.8214 | **0.8436** |

**How it is decided.** S1 — each run's measured hit rate (Σ prefix_cache_hits ÷ Σ prefix_cache_queries
over both servers) within 10% of its prediction. S2 — every pair of policies whose predicted hit rates
differ by at least 0.02 comes out in the same order when measured; that includes the pair the headline
rests on, K64 radix-affinity above session-sticky (0.022 apart). K1's session-sticky vs radix-affinity
(0.002 apart) is inside the noise and not tested. Phase 6 holds only if S1 holds for all six runs and S2
for both workloads.

**What would be uncomfortable, written down now.** The routers decide on in-flight counts, and a real
engine's timing differs from the simulator's, so load-aware routing can place sessions differently than
predicted (the dry-run already shows this in the request splits); S1 and S2 are on hit rates for that
reason. And async scheduling, on by default in vLLM 0.30.0, is not modelled — Phase 3 found it cost
nothing sequentially and 0.3% concurrently.

### 2026-09-28 — Amendment: the headline's load-weight clause, reworded after the demo grid
The demo grid ([`scripts/build_grid.py`](../scripts/build_grid.py)) reproduces Phase 5's H2b table
exactly and puts load weight 1 beside α\* = 16 and pure radix-affinity (no load weight), so the
post-hoc numbers of the Phase 5 entry can be read side by side. All three were already in Phase 5's
committed table (`h2b.required_replicas` in `phase5.json`); only α\* was written up. At K = 16 and
K = 64, weight 1 and no weight need the **same** number of replicas at every load (K = 16: 2, 3, 3, 4 at
1–4 sessions/s; K = 64: 3, 4, 5, 5). At K = 4, 3 sessions/s, no weight needs one fewer (3 against 4).
Nowhere in the grid does weight 1 need fewer replicas than no weight. α\* = 16 needs up to 4 more
(K = 16 at 4/s: 8 against 4), or cannot hold the SLO on 8 at all (K = 64 at 4/s).

So "then route by prefix, **with** a small load weight" credited the weight with a benefit the data does
not show. The headline now reads "then route by prefix, **and keep its load weight small**". That is
what the numbers support where that half of the headline applies — 16 and 64 prompts, working sets of
0.89× and 2.66× one replica's cache: there, small (1 or none) beats large (16). Below that the weight
cuts the other way: at K = 4 and 4 sessions/s only α = 16 holds the SLO among the affinity policies (on
6), and with one prompt, weight 1 and no weight both fail from 2 sessions/s (H2's collapse). But there
the headline's first half applies: session-sticky needs the fewest replicas, or ties for them, at every
load at K = 1 and K = 4. Nothing registered changes: H2b holds as written, at α\* = 16, and everything
in this entry stays *reported, not tested*. The Phase 5 entry above is left as written.

### 2026-09-28 — C1 REGISTERED before it runs: `kvsim calibrate` on a real GPU
Design and rules: [`configs/c1_calibrate.yaml`](../configs/c1_calibrate.yaml). Notebook:
[`notebooks/c1_calibrate_t4.ipynb`](../notebooks/c1_calibrate_t4.ipynb), built by
`scripts/c1_prepare.py` with the package's wheel **embedded** — built from the clean commit
`5a34c6d`, sha256 `ac5209ef2899b896…` — so the GPU box runs exactly the committed code. Prediction:
[`artifacts/results/c1_prereg/prediction.json`](../artifacts/results/c1_prereg/prediction.json).
`scripts/c1_dryrun.py` ran the whole notebook against the mock engine first: C1 held there, as it
must, since the mock is the model.

**Why.** H6's ranking half was falsified, so a replica count from our T4 does not transfer; users
must measure their own GPU, and `kvsim calibrate` is how. It has only met the mock so far.

**The setup.** One vLLM 0.30.0 server, Qwen3-4B in fp16, one T4 — Phase 3's flags without the block
override, so the engine allocates its natural capacity. `kvsim calibrate` with its defaults.

**How it is decided.** C1 holds only if both hold:
- **C1a** — the command exits 0: its check's hit-rate error is at most 5% **and** every one of the
  48 check requests' cached-token counts equals the model's. **Predicted: exact** — hit rate
  **0.714286**, 9,840 of 13,776 tokens, as the block-pool model was exact on sequential workloads in
  Phase 3.
- **C1b** — the KV capacity it reads from the engine equals the engine log's
  `GPU KV cache size` ÷ 16.

*Reported, not tested:* the fitted timings against Phase 3's (0.70 ms per uncached token; 52.9 ms +
1.23 ms × batch), the capacity against the Phase 4 profile's 1,537 blocks, and the wall-clock time.

**What would be uncomfortable, written down now.** The timing fit may not match Phase 3's: the
prompt lengths and batch sizes differ, and one sequence decodes slower than the line predicts.
That is reported, not a failure. A failure of C1a would mean a rule in this engine that the check
exposes and Phase 3's workloads did not — the check's lengths are chosen so that even a wrong block
size shows, which the first version of the check missed.

### 2026-09-28 — R2 REGISTERED before the comparison runs: real traffic, SemiAnalysis's AgentX traces
Design and rules: [`configs/realtrace_agentx.yaml`](../configs/realtrace_agentx.yaml). Procedure: R1's
`scripts/run_realtrace.py` with that config, reading the traces through the new `agentx` adapter.
The traces (Apache-2.0, `huggingface.co/datasets/semianalysisai/cc-traces-weka-042026`, sha256
`d9e30ba2…`) are not committed; the results record the file's hash.

**Why this trace.** Real coding-agent sessions — 739 of them, 59,274 requests — with each request's
prompt as 64-token block hashes that are **local to the session**. So it shows reuse *within* a
session and nothing *across* users: the complement of Mooncake. It tests the other half of H2b's
mechanism — what happens when there is no shared prompt at all.

**What was measured before registering — structure, and round-robin alone.** Sessions are real
here (one per trace; a sub-agent is its own); requests to a model other than the session's main
one are dropped and counted (390 in the window). The sessions were recorded separately, so they are
replayed a fixed number at a time, each at its recorded pace, for three hours with 30 minutes of
warm-up. The scale, the number of sessions and the speeds were then set by **round-robin-only**
probes, by rules written into the config before each probe:
- At R1's scale (1/32) and 32 sessions, round-robin could not hold the SLO on 8 replicas at any
  speed, 0.25× included — so, by the rule, **1/64**.
- At 1/64, round-robin needed 1–3 replicas with 32 sessions, and held the SLO at no speed with 56,
  64 or 128: a cliff, not a slope, because under round-robin every replica ends up caching every
  session. By the next rule — the most sessions at which round-robin still holds at the recorded
  pace — **48 sessions**, and speeds **0.5×, 1×, 2×**, where round-robin needs **2, 2 and 5**.
- The window: 6,063 requests from 136 sessions; median prompt 1,591 tokens (95th percentile 4,824);
  95.9% of block uses are reuse, **0.0% across sessions** (by construction). The live sessions'
  contexts come to roughly five T4 caches.

**Prediction.** With no block shared across sessions, prefix-affinity can only use the reuse that
session-sticky already keeps together. So:
- **tie_within_one** — balanced affinity (α\* = 16) and session-sticky need replicas within one of
  each other at every testable speed;
- **pure_affinity_ties_sticky** — so does **pure** affinity. Its collapse on Mooncake and in H2 came
  from piling every session onto the replica holding a shared prompt; with nothing shared, there is
  nothing to pile onto.

R2 holds only if both hold. A speed where neither policy holds the SLO on 8 is untestable; a rule
with no testable speed does not hold. *Reported, not tested:* every policy's replicas and hit rates,
and α = 1.

**What would be uncomfortable, written down now.** Round-robin's cliff says the working set, not
the request rate, decides who holds the SLO here — and session-sticky places a new session by load
at its first request only, while prefix-affinity re-decides every turn. If sessions' contexts grow
unevenly, sticky may pile large sessions onto one replica, and affinity with a load weight could
then win outright; that would falsify tie_within_one and would be the first real-traffic case where
re-deciding per turn pays. The scale (1/64) and the replay (48 sessions at a time, open-loop) are
choices, not facts about the traffic, and every number carries them.

### 2026-09-28 — R2 RAN: HOLDS. With nothing shared across users, sticky and affinity tie — and pure affinity does not collapse.
Report: [`artifacts/results/realtrace_agentx/report.md`](../artifacts/results/realtrace_agentx/report.md)
— 576 runs in 177 s, from a clean tree at `cf74aec`; the trace file's sha256 is recorded.

**As registered: HOLDS, on both rules.** Session-sticky, balanced affinity (α\* = 16) and pure
affinity each need **2 T4s at 0.5×, 1× and 2×** — equal at every speed, not merely within one; so
does α = 1. Their hit rates at 4 replicas are 0.933–0.937.

**What else it shows (reported).**
- Round-robin needs 2, 2, **5**: at 2× it takes two and a half times the GPUs the locality-keeping
  policies take. Least-loaded and power-of-two need 3 at 2×. All three hit 0.59 at 4 replicas,
  against 0.93: a session's next turn lands on a replica that never saw its context, or holds an
  older, partly evicted copy. The probes before registering had already shown the extreme: above 48
  sessions, round-robin held the SLO on no number of replicas.
- Consistent-hash on the prompt's first blocks does as well (2, 2, 3; hit rate 0.933): with every
  session's prefix private, hashing it *is* session affinity, and there is no hot prompt to pile onto.

**What it means for the finding.** The two real traces now bracket H2b's mechanism. On Mooncake,
where prompts are shared across users, pure affinity failed at every speed; on AgentX, where nothing
is, it tied session-sticky everywhere. So its collapse comes from shared prompts — as H2 found on
generated traffic — not from routing by prefix as such. On neither trace did affinity beat
session-sticky by more than a replica: the many-prompt advantage (H2b) is still shown only on
generated traffic. For agent traffic like this, the practical rule is simple: keep each session on
one replica, by any policy that does, and never scatter it.

### 2026-09-28 — Phase 6 RAN (twice): HOLDS. The cache model predicts two live replicas; the timing model does not.
Report: [`artifacts/results/phase6_kaggle_t4/report.md`](../artifacts/results/phase6_kaggle_t4/report.md),
generated by `scripts/phase6_analyze.py` from the raw files. The registered notebook ran **twice** on
Kaggle's 2× T4, both times with the registered traffic and predictions: run 1 at 09:41 UTC (its
printed results, from the executed notebook) and run 2 at 15:07 UTC (`phase6.json` and both server
logs; S1 and S2 recomputed from its raw per-replica counters). Both are reported; neither was chosen.

**As registered: HOLDS, in both runs.**
- **S1** — every run's hit rate within 10% of its prediction. Largest error: **1.40%** (run 1) and
  **0.22%** (run 2). All three K1 runs match to four decimals in both runs (0.8795, 0.9214, 0.9230).
- **S2** — all five tested pairs in the predicted order, in both runs — including the pair the
  finding rests on: at K64, prefix-affinity above session-sticky (predicted +0.022; measured +0.034
  and +0.023).

**Reported, not tested — and it is the larger news.** The timing model under-predicted p95 TTFT by
**5–222×**. Predicted: under 0.5 s for every K1 policy, 1.3–5.8 s at K64. Measured: session-sticky
1.8–1.9 s at K1 (the only run to hold the 2 s target), 11–15 s round-robin, ~106 s prefix-affinity
(every request on one replica); 32–111 s at K64. It predicted no preemptions in five of six runs;
the engines had 15–83 in four of those five. No request failed and the client sent every request
within 10 ms of its schedule: the engines were overloaded. From vLLM's own stats lines, a decode step
took **2.7–4.0×** what the profile predicts at every batch size, which at two requests per second per
engine is enough to tip an engine into the overload spiral (queue, full cache, preemption,
recomputation).

**Post hoc, and labelled so.** (1) The *order* of the three policies by p95 TTFT matched the
prediction in both workloads and both runs: at K1 session-sticky < round-robin < prefix-affinity, at
K64 prefix-affinity < session-sticky < round-robin. So the finding's flip appeared live, in latency as
well as hit rate — as an observation, not a test. (2) Two candidate causes for the slow steps: the
profile has no context-length term (these prompts are ~1,700 tokens; per-request decode cost is about
0.2 ms at ~100 tokens in C1 and about 2 ms at 448 in Phase 3's sweep), or CPU contention (two engines
and a 200-stream client on 4 vCPUs). Neither is tested.

**What it changes.** The cache model's claim now covers two live replicas under pressure, not only
one. The finding's *direction* has live support. Its *replica counts* do not: they come from a timing
model that, for ~1,700-token agentic prompts on a T4, is optimistic by a factor large enough to turn
"holds on 2" into "overloaded on 2". Every replica count in the README and the preprint is marked
as a simulator figure until a context-aware timing model is fitted and tested on held-out traffic.

### 2026-09-28 — C1 RAN: HOLDS. `kvsim calibrate` works on a real GPU; its straight-line timing fit does not.
Report: [`artifacts/results/c1_kaggle_t4/report.md`](../artifacts/results/c1_kaggle_t4/report.md),
generated by `scripts/c1_analyze.py`. The wheel's hash matched the registration, and the check
workload's prediction matched the registered one exactly.

**As registered: HOLDS.** **C1a:** exit 0; the engine's counters 9,840 of 13,776 tokens, exactly the
registered prediction, and all 48 requests' cached tokens equal. **C1b:** it read 1,537 blocks from
`vllm:cache_config_info`, the same as the engine log. 48 seconds wall clock.

**Reported, not tested.** Prefill up to 1,024 tokens costs **0.71 ms per token** — Phase 3's 0.70 —
and gets steeper beyond, to 1.12 ms per token at 2,040; one line over 64–2,040 tokens fits a negative
intercept (−212 ms), which the command clamps to 0, and then over-predicts a 512-token prompt by 89%.
Decode repeats Phase 3's oddity (one sequence alone is slower than two to sixteen) and, measured on
32-token prompts, describes short contexts only. So the command's *cache* check is proven on a real
engine; its *timing* fit is not yet fit for sizing a fleet — the same limit Phase 6 exposed.

### 2026-09-28 — C2 REGISTERED before it runs: a context-aware timing model, and what would break it
Design and rules: [`configs/c2_timing.yaml`](../configs/c2_timing.yaml). Notebook:
[`notebooks/c2_timing_t4.ipynb`](../notebooks/c2_timing_t4.ipynb). The whole analysis:
`scripts/c2_analyze.py`, committed with the design at `adfada5`, before any GPU run. Part D's traffic
and the context-free profile's predictions for it:
[`artifacts/results/c2_prereg/predictions.json`](../artifacts/results/c2_prereg/predictions.json),
generated from that clean commit. `scripts/c2_dryrun.py` ran the whole notebook against two mock
engines with known context terms, then the analysis: it recovered them (decode 3.98e-6 s per
context token against the mock's 4e-6, R² 1.000) and C2c held within 1.05×.

**Why.** Phase 6 found the timing model optimistic by 5–222× on ~1,700-token prompts. Two suspects:
no context-length term, or CPU contention on a 4-vCPU box running two engines and a client.

**The model** (`sim/cost_model.py`, both new terms default to zero, which reproduces every earlier
result to the bit): prefill(n, c) = p0 + p1·n + p2·n·(c + n/2); decode(b, S) = d0 + d1·b + d2·S,
with S the batch's summed context.

**The setup.** One Kaggle box, two T4s, Phase 6's engine flags. Replica 0 is measured.
- **A** — decode step for 28 cells of context length L (128–1,984) × batch b (1–64), two reps, all
  within 1,300 blocks so nothing is preempted; the step is the mean gap from the 8th token to the
  64th, median over the batch. Replica 1 idle.
- **B** — prefill time for 15 cells of cached prefix c (0–1,280) × new tokens n (64–1,984), three reps.
- **C** — three of A's cells measured alone and with replica 1 kept busy by a separate streaming
  client (48 at a time), alternating, two reps; measured **engine-side** with
  `vllm:inter_token_latency_seconds`.
- **D** — held out: Phase 6's generator on replica 0 alone — K1 at 0.5 sessions/s (Phase 6
  K1-sticky's per-replica load) and K64 at 0.25. The context-free profile predicts p95 TTFT 0.33 s
  and 1.20 s (reported, not tested).

**The fit**, fixed now: least squares with every coefficient ≥ 0, decode on A, then prefill on B with
one fitted decode step taken out of each TTFT (the simulator's own TTFT). The context-free fit on
the same data is reported beside it.

**How it is decided.**
- **C2a** — the context term is real: at batch 8, the step at L = 1,536 is ≥ 1.5× the step at
  L = 128. *Predicted: holds* (~3×, from Phase 3's sweep and C1).
- **C2b** — CPU contention is real: contended ÷ alone engine-side inter-token latency, median over
  C's cells, ≥ 1.2. *Predicted: fails* — context alone roughly accounts for Phase 6's steps.
- **C2c** — the model fitted on A and B, replaying D at the times its requests were really sent,
  gives p50 **and** p95 TTFT within 2× of the measured ones, for **both** workloads. *Predicted:
  holds.* A rule whose data is missing is undecided, and does not hold.

**What would be uncomfortable, written down now.** If C2b holds, Phase 6's latencies were partly an
artefact of the box — and every multi-engine timing on a small CPU carries that caveat. If C2c fails
with C2a holding, the context term is real but not enough: chunked prefill (which the simulator does
not model) or the single-sequence decode oddity may matter under load. And D is one engine: a model
that predicts it still has to be re-checked on two replicas before the README's replica counts
change.

### 2026-09-30 — C2 RAN: all three registered predictions came true. Context, not CPU.
Report: [`artifacts/results/c2_kaggle_t4/report.md`](../artifacts/results/c2_kaggle_t4/report.md),
from the committed `scripts/c2_analyze.py` (unchanged since `adfada5`). The registered notebook ran
once (05:33 UTC); predictions, traffic fingerprints and design match the registration; all 56 decode,
45 prefill, 12 contention and 960 held-out measurements are valid; both engines had 1,537 blocks.

**As registered.**
- **C2a holds** (predicted: holds). At batch 8 the decode step is 87.7 ms at 1,536 tokens of context
  and 45.7 ms at 128 — **1.92×** (needed ≥ 1.5; predicted ~3×).
- **C2b does not hold** (predicted: fails). With replica 1 kept busy by a 48-stream client, replica 0's
  engine-side inter-token latency changed by a median factor of **1.00** (needed ≥ 1.2); the
  client-side steps agree. CPU contention did not slow the engine on this box.
- **C2c holds** (predicted: holds). Fitted on parts A and B only — decode **40.1 ms + 0.34 ms per
  sequence + 3.5 ms per 1,000 context tokens in the batch**; prefill **0.32 ms per new token + 0.76 ms
  per new token per 1,000 tokens attended** — the model predicts the held-out runs' p50 and p95 TTFT
  within **1.10×, 1.41× (K1 at 0.5/s) and 1.31×, 1.11× (K64 at 0.25/s)**. The context-free profile
  was 3.0–7.1× (p50) and 1.7–7.1× (p95) optimistic on the same traffic. Hit rates: exact on one
  replica too (0.9230, 0.8304).

**Reported, not tested.** The decode fit's R² is 0.979 with the context term and 0.315 without it. The
single-sequence oddity is back (batch 1 slower than batch 2 at 128 and 512 tokens). The fitted prefill
reproduces C1's independent 2,040-token measurement (2.28 s predicted against 2.28 s measured).

**Post hoc, and labelled so** ([`posthoc_phase6.md`](../artifacts/results/c2_kaggle_t4/posthoc_phase6.md),
`scripts/c2_posthoc_phase6.py`). The context terms were added *because of* Phase 6, so Phase 6 is not a
clean test of them; but the coefficients above never saw a Phase 6 number. Replaying Phase 6's second
run with each request on the replica that really served it, at the time it was really sent, the model
lands within **1.02–1.79×** of all six runs' p50 and p95 TTFT, and near their preemption counts
(25/18, 0/0, 65/67, 68/62, 88/83, 55/48 model/measured). The context-free profile was 5–222× off. So
the timing model now explains two live replicas too — which is what licenses re-running the finding
on it, next.

### 2026-09-30 — H2b under C2's timing REGISTERED before it runs
Design and rules: [`configs/h2b_c2.yaml`](../configs/h2b_c2.yaml). Procedure: `scripts/run_h2b_c2.py`.
Profile: [`configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json`](../configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json),
C2's fitted profile, unchanged. A pipeline check ran first on a 6-session config outside the design.

**Why.** Every replica count in the finding came from the context-free profile, which C2 and Phase 6
show is optimistic by several times on these prompts. The finding's *direction* survived H6's
rescalings, but those scaled every cost together; a context term changes costs unevenly — a cache
miss now also pays attention over the prompt, and decode pays for context whether it was cached or
not. So the finding is re-tested, not assumed.

**The setup.** Phase 5's H2b traffic unchanged (200 sessions × 4 turns, sharing 0.8, K = 1, 4, 16, 64
tenant prompts, Zipf 1), balanced affinity at α\* = 16 against session-sticky, three seeds, the same
cost model and 2 s p95 SLO. Loads **0.25, 0.5, 1, 2, 3, 4** sessions/s and replicas **1–16**, moved
down and up because each T4 is now slower (6 and 8 sessions/s dropped). Reported beside them: α = 1,
pure affinity, round-robin, least-loaded.

**How it is decided — H2b's rules, unchanged.** (a) at K = 1, balanced affinity needs fewer replicas
than session-sticky at no load; (b) at K = 64 it does at one load or more; (c) no reversal along K.
Holds only if all three hold. **Predicted: holds** — a miss costs more once attention over the prompt
is priced in, so prefix routing should matter more at K = 64, while sticky's balance still wins at
K = 1.

**What would be uncomfortable, written down now.** If decode — which every policy pays in full,
whatever it cached — now dominates the cost, the policies' differences may shrink into ties
everywhere, and (b) can fail for lack of any win. That would say the finding's replica savings were
an artefact of the context-free model.

### 2026-09-30 — ⛔ H2b under C2's timing RAN: FALSIFIED. On the validated timing model, prefix routing no longer saves GPUs at 64 prompts.
Report: [`artifacts/results/h2b_c2/report.md`](../artifacts/results/h2b_c2/report.md) — 6,912 runs in
319 s at `c0c8b18`. The tree was dirty only by the four registered notebooks deleted from disk and
an untracked zip, both under `notebooks/` (listed in the results header); every line of code the run
used is committed.

**As registered: FALSIFIED — predicted to hold.** (a) holds: at K = 1 balanced affinity (α\* = 16)
never needs fewer replicas than session-sticky. (c) holds: no reversal along K. **(b) fails**: at
K = 64 balanced affinity wins at no load — it ties at 0.25, 0.5 and 1 session/s (2, 3, 5 replicas),
loses at 2 (11 against sticky's 10), and at 3 and 4 nothing holds the SLO on 16. So the second half of
the headline — "then route by prefix" — does not survive a timing model that matches the engine.
It rested on the context-free profile, which C2 and Phase 6 showed is optimistic by several times.

**Reported, not tested — found after the run, and labelled so.**
- **Session-sticky needs the fewest replicas, or ties for them, at 22 of the 24 (K, load) points.**
  The two exceptions are prefix routing with a small load weight or none: 3 against 4 replicas at
  K = 16 and 1 session/s (α = 1 and pure affinity), and **8 against 10 at K = 64 and 2 sessions/s**
  (α = 1). Both are post hoc (α = 1 was Phase 5's post-hoc weight), so neither is a finding.
- **The mechanism is still there, and no longer enough.** At K = 64 prefix routing keeps the better
  hit rate (0.87 with α = 1, 0.83 with α = 16, against sticky's 0.80 and round-robin's 0.47 on 8
  replicas at 2 sessions/s). But with context priced in, decode — which every policy pays in full,
  cached or not — is a large share of the cost, and concentrating sessions onto the replicas that
  hold their prompt makes batches uneven. The balance session-sticky keeps is worth more than the
  extra hits.
- **The replica counts roughly double.** At K = 1 and 2 sessions/s: session-sticky 5 T4s (2 on the
  context-free model), round-robin 7 (3), balanced affinity 10 (3). Session-sticky's saving over
  round-robin grows with load: equal at 1 session/s, then 5 against 7, 6 against 9, 7 against 12.

**What it changes.** The README's headline loses its second half. What the evidence now supports:
keep each user on one GPU — on the validated T4 model it is the best or tied-best policy almost
everywhere in this grid, including when tenants' prompts outgrow one cache; routing by prefix raises
the hit rate there but does not, with the registered weight, save a GPU. The earlier finding stays
in this log as written: it was a correct result about the context-free model, and that model was
wrong about the engine. Which of H1–H6, R1 and R2 change on the validated model is not yet tested —
they all ran on the context-free profile.

### 2026-10-01 — POST HOC, no GPU: `kvsim calibrate`'s own cells, checked on C2's data. Cold prompts alone are not enough.
Report: [`artifacts/results/c3_prereg/precheck.md`](../artifacts/results/c3_prereg/precheck.md), made
by `scripts/c3_precheck.py`.

**Why it was run.** Since 2026-09-30 `kvsim calibrate` fits C2's context-aware model — but from its
own, smaller set of cells: 15 decode cells measured once (C2: 28, twice), and prefill on **cold
prompts only** (C2: also after a cached prefix). Before a GPU is spent on that command's first real
run (C3), the question was whether those cells would have been enough on the data C2 already
recorded.

**What it did.** It took from C2's recording the cells the command would have measured, fitted them
with the command's own code, and replayed C2's two held-out workloads with the result — the replay
C2c was decided on.

**Result.** With cold prompts only, K1_half's p95 comes out at 13.2 s and 16.7 s (the two decode
reps) against 2.3 s measured: **5.7× and 7.2×**, far past C2c's 2× limit. With C2's cached-prefix
cells added, every factor is within **1.36×** (K1_half 1.03–1.21×).

**Why.** On a cold prompt the tokens attended to are always half the prompt, so cold cells measure
only n and n². The fit split them as 0.239 ms per token + 0.842 ms per token per 1K attended, with
a 20.8 ms base; C2's full fit has 0.319 + 0.758 and no base. Both describe cold prompts equally
well. Agentic traffic is the other case — about 1,570 cached tokens and 130 new ones per request on
K1_half — and there the cold-only fit charges 14% more prefill time. K1_half runs at the knee of
the load curve, where 14% more service time becomes several times more queueing.

**What changed.** `kvsim calibrate` now also times new tokens after a cached prefix — 64, 256 and
704 new tokens after 256, 768 and 1,280 cached ones, which are C2's part-B cells
(`--prefill-cached`, `--prefill-new`) — and gives the fit the cached-token count the engine reports
for each request, not the one the design intended.

**What this is not.** Not a test. It is post hoc; it uses C2's cells as stand-ins (context 128 for
the command's 32, six cold lengths for its eight, a cache reset before every cell where the command
has none); and the two workloads it replays have now been used to choose the command's design. C3 is
the test, and it adds a workload that no design decision has seen.

### 2026-10-01 — C3 REGISTERED before it runs: the context-aware `kvsim calibrate` on a real GPU
Design and rules: [`configs/c3_calibrate.yaml`](../configs/c3_calibrate.yaml). Notebook:
`notebooks/c3_calibrate_t4.ipynb`, built by `scripts/c3_prepare.py` with the package's wheel inside
it. Analysis: `scripts/c3_analyze.py`. Registered numbers, the wheel's hash and the commit it was
built from: `artifacts/results/c3_prereg/predictions.json`. All committed before the run.

**Why.** `kvsim calibrate` is the project's answer to "why trust your numbers on my hardware?". C1
proved its cache check and its capacity reading on a real T4, with its first, straight-line timing
fit. C2 showed that line up to 7× optimistic and validated a context-aware model — but fitted by an
analysis script, on 101 measured points. The command now fits that model itself, from fewer cells of
its own, and that fit has only ever met a mock engine.

**The setup.** One vLLM 0.30.0 server, Qwen3-4B on one T4, C2's flags (1,537 blocks).
- **Part 1** — the command, every timing cell at its default, seed 0. This is the tested run.
- **Part 2** — three held-out workloads on the same engine, sent and measured by the cell that
  measured C2's part D. Two are C2's own, unchanged (`K1_half`, `K64_quarter`: the same trace
  fingerprints). The third, `T6_K8`, is new: prompts of 630–1,130 tokens instead of 1,550–1,850,
  half the sharing, six turns, eight prompts, another seed, at 0.3 sessions/s. It is there because
  the pre-check above used C2's recording of the first two to change the command's design; for the
  command, only `T6_K8` is still blind.
- **Part 3** — the command again, seed 1, after the traffic. Reported, not tested.

**The rules.**
- **C3a — the command works on a real engine.** Its first run exits 0 and marks the profile
  calibrated; the hits and queries its check measured equal the registered prediction (**9,840 of
  13,776 tokens**); all **48** check requests' cached tokens equal the model's; and the capacity it
  read equals the engine log's `GPU KV cache size` ÷ 16. All five, recomputed from the raw files.
- **C3b — its profile predicts traffic it never saw.** For every workload, p50 and p95 TTFT within
  **2×** of the measured ones, replayed at the times the requests were really sent — C2c's rule and
  C2c's replay code.
- **C3c — C2's profile predicts another day.** For every workload, the numbers below within **2×**
  of the measured ones. They are computed now, at the scheduled times, from the profile committed
  on 2026-09-30:

  | workload | hit rate | p50 TTFT | p95 TTFT | preemptions |
  |---|---:|---:|---:|---:|
  | `K1_half` | 0.9230 | 1,443 ms | 3,277 ms | 6 |
  | `K64_quarter` | 0.8304 | 420 ms | 1,892 ms | 0 |
  | `T6_K8` | 0.8517 | 304 ms | 659 ms | 0 |

A rule whose data is missing is undecided, and an undecided rule does not hold.

**Predicted: all three hold.** C3a: C1's check was exact with this seed, and the timing cells are
C2's, which ran clean. C3b: the pre-check put the command's cells within 1.36× on C2's data. C3c:
C2c held at 1.41× on the two shared workloads.

**What would be uncomfortable, written down now.**
- **`K1_half` sits at the knee, and it can sink C3b and C3c by itself.** C2's profile predicts 6
  preemptions there, where C2's engine had none, and a p95 of 3.3 s against the 2.3 s C2 measured —
  already 1.41×. A session a few percent faster than C2's moves the measured p95 down and that
  factor toward 2; a few percent slower can push the real engine over the edge the model already
  sees. One of a rule's six numbers failing while the other five are close would say "the model
  is fragile at saturation", not "the model is wrong" — and the write-up must say which, with the
  numbers.
- **`T6_K8` is half the prompt length of anything this model was tested on with traffic.** If the
  fixed costs (decode base, per sequence) are off, it shows there and not at 1,700 tokens.
- **The command's cells are not C2's.** 15 decode cells measured once, against 28 measured twice; a
  32-token context where C2 started at 128; no cache reset between cells. The pre-check could not
  test any of that.
- **If C3b fails where C3c holds,** the model is right and the command's measurement is not good
  enough. The README would then have to say that `kvsim calibrate`'s timings are not to be trusted
  yet — in the place where it now says they are untested.
- **If the two seeds disagree** (parts 1 and 3) by more than the factors are away from 2×, a
  single run of the command is luck, and a user who runs it once cannot rely on it.

**Reported, not tested** (listed in the config before the run): the second run of the command; its
coefficients against C2's fit; C2's profile replayed at the real send times; the two shared
workloads against C2's own session; and — `scripts/c3_analyze.py --grid` — the registered H2b-C2
grid re-run on the command's profile against the committed table: does a second calibration of the
same GPU give the same GPU counts?

**Dry-run** (`scripts/c3_dryrun.py`, a mock engine with known context terms, where the model is the
engine): the command recovers them (decode 4.00e-6 against 4e-6 s per context token; prefill
attention 2.2e-8 against 2e-8), its check is exact, and all three rules hold within 1.07×. That
proves the plumbing, and nothing about vLLM.

### 2026-10-05 — C3's first attempt did not start: the GPU box's software changed. The notebook is amended; nothing registered is.
Record: [`artifacts/results/c3_env_failure/`](../artifacts/results/c3_env_failure/) — the stopped
cell's output, and the probe that checked the repair on Kaggle's image.

**What happened.** The notebook stopped in section 5, when it launched the engine: `vllm` exited at
import with *"PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA
version 13.0 whereas TorchAudio has CUDA version 12.8."* No request was sent, `kvsim calibrate`
never ran, and no zip exists — the cell's message said "download the zip" before any cell that
writes one had run. **There is no measurement from this attempt, so nothing was seen.**

**Why.** Kaggle's image changed between C2 (2026-09-30, Python 3.12.13) and now (Python 3.13.15).
It ships `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, `torchvision 0.26.0+cu128`. vLLM 0.30.0
pins `torch==2.13.0`, `torchaudio==2.11.0` and `torchvision==0.28.0`. So `pip install vllm` replaces
torch and torchvision with the builds from PyPI (CUDA 13.0) — and **keeps the image's torchaudio**,
because `2.11.0+cu128` satisfies `==2.11.0`. Everything that imports `transformers` then fails, vLLM
first.

**How the repair was checked, before a GPU hour was spent on it.**
- *On this machine* (Python 3.13.12): a clean `pip install vllm==0.30.0` starts; with
  `torchaudio 2.11.0+cu128` put in, `vllm` cannot start; the pre-flight cell puts it right.
- *On Kaggle's image*, with a probe notebook that starts no engine (`scripts/c3_probe.py`, 400 s):
  the pre-flight found `torchaudio 2.11.0+cu128` failing with the same error, ran
  `pip install --no-deps --force-reinstall --index-url https://pypi.org/simple torchaudio==2.11.0`,
  and after it torchaudio, torchvision and torchcodec import, `vllm serve --help` exits 0, and
  torch sees a Tesla T4.
- A second probe, CPU only: a run that ends in an error still keeps the files it wrote. So "results
  are saved after every part" also holds for a run started through the API.

**What changed — plumbing, and one line of reporting.**
- A **pre-flight cell** right after the install (`scripts/_c3_notebook.py`). In fresh interpreters
  it checks that torch sees a GPU and that the GPU is a **T4** (C3 is registered for one; no other
  hardware is measured); it checks that `torchaudio`, `torchvision` and `torchcodec` import, puts
  one that does not back to the version vLLM itself pins, from PyPI, and removes it if that fails
  too (a text model needs none of them); then `vllm serve --help` must exit 0, or the notebook
  stops there — in the first minute, not after the 8 GB download.
- **Every early stop now writes `c3_results.zip`** (the environment, `pip freeze`, the engine's
  whole log) and says that nothing was measured.
- The pre-flight's record is saved with the results, and the report now states the software of the
  run beside C2's — **one line of reporting** in `scripts/c3_analyze.py`. No rule is touched.
- The run is now started through Kaggle's API (`scripts/kaggle_run.py`: a push is "Save & Run All"),
  as a private notebook on the same T4 ×2 machine with internet on. The notebook, the image and
  the hardware are what a run by hand gets; no browser is attached.

**What did not change:** the config and its rules, the predictions (`predictions_sha256`
`be7f0a76…`), the three traces (the same fingerprints), the command's default cells, and the wheel
— the same bytes (sha256 `4c1c388b…`), re-embedded, not rebuilt. `scripts/c3_prepare.py --amend`
refuses to run unless every one of these comes out as registered, and writes the reason, the commit
and the notebook's hash before and after into `predictions.json`; `tests/test_c3_prereg.py` checks
that chain, and `tests/test_c3_notebook.py` the pre-flight's logic.

**One more thing that would be uncomfortable, written before the run.** C3c asks whether the profile
fitted on 2026-09-30 predicts another day. That other day now has another image as well: Python
3.13.15 instead of 3.12.13, `transformers` 5.16.1 instead of 5.17.0, NVIDIA driver 580.178 instead
of 580.159. torch is the same build (2.13.0, CUDA 13.0), so the GPU's share of each step should not
move; the CPU's share might. **If C3c fails and C3b holds, the image is a suspect alongside the
profile**, and the write-up must say so instead of choosing one.

**"Run it once" still stands.** The attempt produced no data, so the next run is the first.

### 2026-10-05 — C3 RAN: all three hold, as predicted. And a second calibration of the same GPU reverses H2b's verdict.
Report: [`artifacts/results/c3_kaggle_t4/report.md`](../artifacts/results/c3_kaggle_t4/report.md).
One run, started through Kaggle's API from the amended notebook at `1598673`, 2,320 s. The analysis
refuses a run whose predictions, traces or wheel are not the registered ones; these are. The
pre-flight repaired `torchaudio` before anything was measured.

**As registered — predicted holds / holds / holds.**
- **C3a HOLDS.** The command exits 0 in 160 s and marks the profile calibrated; its check measured
  9,840 of 13,776 tokens, the registered prediction; all 48 requests' cached tokens equal the
  model's; it read 1,537 blocks and the engine's log says 24,592 tokens.
- **C3b HOLDS.** The profile it wrote predicts every held-out workload within **1.41×**: `K1_half`
  1.04× · 1.08× (p50 · p95), `K64_quarter` 1.32× · 1.20×, and `T6_K8` — the one no design decision
  had seen, with prompts half as long — 1.41× · 1.21×.
- **C3c HOLDS.** C2's profile, from the numbers registered on 2026-10-01: `K1_half` 1.01× · 1.25×,
  `K64_quarter` 1.35× · 1.21×, `T6_K8` 1.38× · 1.18× — on another day and another software image.
- Every hit rate equals its registered prediction to four decimals (0.9230, 0.8304, 0.8517).

**Reported, not tested — and two things the registration called uncomfortable happened.**

1. **The two runs of the command disagree at the knee.** The second run (part 3, 25 minutes later
   on the same engine) is exact on the check again, and its coefficients are within 5.3% of the
   first run's (decode per context token +3.6%, prefill per token +5.3%, decode base −2.7%). Its
   profile predicts five of the six numbers within 1.35× — and `K1_half`'s p95 at 10.9 s against
   2.6 s measured: **4.17×**. On that 3–5% change the model goes from 4 preemptions to 30; the
   engine had none. C3b holds as registered, on the registered run. On the other run's profile it
   would not have. One run of the command does not promise 2× for a workload at the knee.
2. **A second calibration does not give the same verdict on H2b** (`c3_analyze.py --grid`, the
   registered reported item). H2b-C2's grid on the command's first profile, against the committed
   table: **122 of 144** (K, load, policy) cells need the same number of replicas and 141 are within
   one — but H2b's rules, unchanged, now **hold**: at 64 prompts and 2 sessions/s session-sticky
   needs 12 T4s against balanced affinity's 11, where C2's profile gave 10 against 11.

Also reported: the command's per-sequence decode term came out 0 in both runs (C2: 0.34 ms), and
its decode fit is looser — R² 0.87 and 0.93 from 15 cells measured once, against C2's 0.98 from 56.
C2's profile replayed at the real send times gives the same factors as its registered numbers
(send lag ≤ 5 ms). This session was slower than C2's own on the two shared workloads: `K1_half`
1,431 · 2,615 ms against 1,316 · 2,325 (+9%, +12%), `K64_quarter` 566 · 2,295 against 551 · 2,091
(+3%, +10%) — another image, and the cause is not separated from ordinary session-to-session
difference. All three profiles under-predict the two lighter workloads' p50 by 1.3–1.4×: the
model's known weak spot, a lone sequence decoding slower than the fitted line.

**Post hoc — added after seeing item 2, and labelled so.** The same grid on the command's second
profile too ([`posthoc_three_calibrations.md`](../artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.md),
`scripts/c3_posthoc_grid.py`). Three calibrations of one T4 — **A** C2's fit, **B** and **C** the
command's two runs:
- **H2b: FALSIFIED on A, HOLDS on B and on C.** The deciding comparison, session-sticky against
  balanced affinity at 64 prompts: 5 v 5, 6 v 6, 6 v 5 replicas at 1 session/s; 10 v 11, 12 v 11,
  12 v 10 at 2.
- **102 of 144** cells need the same number of replicas on all three; **138** are within one; at 8
  the calibrations disagree on whether the SLO can be held on 16 replicas at all.
- **What survives all three.** Session-sticky needs the fewest replicas, or ties, at every load
  with 1 prompt and with 4 (12 of 12 points); and it never needs more than round-robin or
  least-loaded, at any of the 24 points. Over all prompts it is best or tied at 22, 20 and 20 of 24
  points — at 19 on all three.
- **What does not:** whether prefix routing saves a GPU at 64 prompts, with the registered weight.
- With a load weight of 1 — Phase 5's post-hoc weight — prefix routing needs **8** T4s at 64
  prompts and 2 sessions/s on all three, where session-sticky needs 10, 12 and 12. Post hoc twice
  over, so not a finding.

**Found while reading the data — post hoc — and a limit on every replica count in this project.**
The SLO here is on the first token only. The engine's own inter-token latency over the three
held-out runs was **521, 174 and 134 ms** per token. The MLPerf rule the 2-second limit was taken
from pairs it with a per-token limit — 2,000 ms / **100 ms** for Llama3.1-8B, "Conversational" —
and under that pair even `T6_K8`, whose p95 TTFT of 774 ms is far inside our SLO, would fail. Not
tested here. It means the replica counts answer "how many GPUs until the first token is on time",
not "until the answer streams at a usable speed".

**What it changes.**
- **`kvsim calibrate`** is proven on a real GPU in its context-aware form. The README says so —
  with the caveat that near saturation one run of it is not enough.
- **The finding.** The verdict of 2026-09-30 stays in this log as what that registered test said.
  But "prefix routing did not save a GPU" was a statement about one calibration of our T4, and two
  more disagree with it. What all three support is narrower and firmer: *keep each user on one GPU*
  — best or tied at every load with one prompt and with four, and never worse than round-robin or
  least-loaded anywhere. With 64 prompts, session-sticky and prefix routing are closer than this
  simulator can tell apart.
- **Every replica count** is to be read as ± one replica: that is how much calibrations of the
  same GPU move it (over the three: 96% of cells within one, 71% identical).
- **What would settle the open half:** a registered test over several fresh calibrations, with the
  load weight fixed beforehand — and an SLO that has its per-token half.

### 2026-10-05 — T1 and T2 REGISTERED before they run: the other half of the SLO, time per output token
Design and rules: [`configs/tpot_slo.yaml`](../configs/tpot_slo.yaml). Procedure:
`scripts/t1_itl_check.py`, `scripts/run_t2_grid.py`, `scripts/t2_analyze.py`. All committed before
either part runs. The measurement itself was added to the simulator first, in its own commit
(`534f5b9`): every run now records each request's time per output token, and no earlier number
moved.

**Why.** Every replica count in this project answers "how many GPUs until the *first* token is on
time" — p95 TTFT ≤ 2,000 ms. The MLPerf row that limit was taken from reads *"TTFT/TPOT: 2000
ms/100 ms"*: it limits the time per output token too, "the average interval between all the tokens
generated", per request. C3's live engine took 134, 174 and 521 ms per token on three workloads
that met, or nearly met, the first-token limit. So the counts are optimistic in a way that had not
been measured, and the fix needs no GPU.

**The target.** First half, as before: p95 TTFT ≤ 2,000 ms. Second half: p95 over requests of
(last token − first token) ÷ (output tokens − 1) ≤ **100 ms** — the same row's limit for
Llama3.1-8B, the model nearest in size to ours. p95, like the first half (MLPerf's Server scenario
uses p99; ours is a little looser on both). A cell meets the target only if every seed meets both
halves.

**T1 — can the simulator's per-token latency be believed?** Five held-out traffic runs are on
disk, each on one live T4: C2's part D (2) and C3's part 2 (3). For each, the notebook recorded the
engine's own mean inter-token latency (Δsum ÷ Δcount of `vllm:inter_token_latency_seconds`). The
simulator replays each run at the times its requests were really sent — C2c's replay — with the
shipped profile (A), and reports the same sum ÷ count.
- **T1 holds only if all five are within 1.5×**, either direction. Tighter than the 2× used for
  first-token latency, on purpose: a mean has no tail to excuse it, and under a per-token limit
  the replicas needed grow roughly in proportion to it.
- ⚠️ **What "registered" means here.** The five measurements were on disk before this was written,
  and I had read three of them (C3's, quoted above). What had never been computed is the
  simulator's side: it could not measure time per token until `534f5b9`. The prediction is blind;
  the data is not fresh.
- **Predicted: holds, narrowly.** First-token latency came within 1.41× on this traffic. The run
  at the knee (`K1_half`) is the one that can break it: there the model already sees preemptions
  the engine did not have.

**T2 — the finding's grid, under the target with both halves.** H2b-C2's grid exactly — the same
traffic, loads, policies, load weights and seeds, read from its config — on the three calibrations
of one T4 that C3 left us with (A: C2's fit; B, C: `kvsim calibrate`'s two runs), with the replica
axis extended from 16 to 32. Every run's p95 TTFT and p95 TPOT is stored, so the count under any
target comes from the same runs. **Guard:** under the first-token target, on replicas 1–16, each
calibration must reproduce its committed table exactly, or the analysis stops.

Each rule must hold on **all three** calibrations — C3's lesson, built in:
- **T2a — "keep each user on one GPU" survives.** At every load with 1 prompt and with 4 (12
  points), no policy in the grid needs fewer replicas than session-sticky.
- **T2b — and it is never worse than plain load balancing.** At all 24 points, session-sticky
  needs no more replicas than round-robin, and no more than least-loaded.
- **T2c — the first-token counts were optimistic by half or more.** Over the points where
  session-sticky's count is defined under both targets, the median of R(both halves) ÷ R(first
  token) is at least 1.5.

T2a and T2b are the two claims that survived three calibrations under the first-token target, with
their definitions unchanged (a point where no policy holds the target is a tie, and the report
counts such points). **Predicted: all three hold.** T2c from a back-of-envelope: at a *mean* of 100
ms per token a T4 carries about 1 request/s of this traffic (a decode step of ~40 ms + ~6 ms per
running sequence, and every arriving request's prefill, ~0.2 s, stalls everyone decoding), against
~1.6 under the first-token target — and a p95 is stricter than a mean. T2a is the one I am least
sure of: ties one replica apart are exactly what a calibration moves.

**Reported, not tested:** every policy's count under both targets on A · B · C; which half binds;
how far the calibrations agree under the two-part target (first-token: 102 of 144 identical, 138
within one); the tables with the per-token limit at 150 and 200 ms (200 is the same document's
limit for Llama2-70b); and H2b's three rules under the two-part target — **exploratory**: the
64-prompt question gets its own registered test, on fresh calibrations, after this.

**What would be uncomfortable, written down now.**
- **T1 cannot reach the regime T2 lives in.** C3's three held-out runs are themselves beyond 100
  ms per token, and C2's two are the same two workloads. T1 checks the model where the engine was
  loaded more heavily than any configuration T2 will call acceptable; whether it is right *below*
  the limit is untested on traffic. That
  needs a GPU run with lighter traffic and per-request token times — the next registration, not
  this one. So even if T1 holds, T2's counts are the model's answer, not yet a checked one.
- **If T1 fails,** T2 still runs and is reported — as what a model that missed its own per-token
  check says. No replica count from it goes into the README as guidance.
- **If T2c comes out near 2×,** every ₹ figure and every "N T4s" in the README is half of what
  someone who needs a streaming answer has to pay. That sentence would have to be at the top, not
  in a scope note.
- **If T2a or T2b fails,** the surviving half of the headline — *keep each user on one GPU* — is a
  first-token result only, and the README must say that instead.
- **100 ms is a datacenter-GPU limit.** One sequence alone on a T4 already takes ~46 ms per token
  at these prompt lengths, so the limit leaves about 2× of headroom. A reader may fairly say the
  limit, not the routing, decides everything; hence the 150 and 200 ms tables.
- **The grid is the same short trace on up to 32 replicas** — 200 sessions, so about six per
  replica at the top. The p95 is over the fleet's 800 requests; no claim is made per replica.

**What this does not change.** The shipped default target stays the first token alone: every
registered result so far is defined on it, and a default should not move ahead of its check. The
simulator now *shows* p95 time per token in every report and says that it is not part of the SLO.

### 2026-10-05 — ⛔ T1 RAN: DOES NOT HOLD. The per-token model is right off the knee and 1.5–1.7× too slow at it.
Report: [`artifacts/results/t1_itl/report.md`](../artifacts/results/t1_itl/report.md), generated at
the registration commit `0be7bec`.

**As registered: does not hold — predicted to hold.** The simulator's mean inter-token latency
against the engine's own counter, profile A, each run replayed at its real send times:

| run | engine | simulator | factor |
|---|---:|---:|---:|
| C2 · `K1_half` | 470 ms | 804 ms | **1.71×** |
| C2 · `K64_quarter` | 161 ms | 173 ms | 1.07× |
| C3 · `K1_half` | 521 ms | 804 ms | **1.54×** |
| C3 · `K64_quarter` | 174 ms | 173 ms | 1.01× |
| C3 · `T6_K8` | 134 ms | 137 ms | 1.02× |

The limit was 1.5×; two of five are past it. Both are the same workload — the one at the knee of
the load curve, which the registration named as the one that could break it. The prediction was
still "holds", and it was wrong.

**Reported, not tested.** The two profiles `kvsim calibrate` wrote in C3 give the same picture on
that session's traffic: 1.51× and 1.71× at `K1_half`, within 1.08× on the other two. In the
simulator `K1_half` has 6 preemptions; the engine had none, in either session.

**Read plainly:** off the knee the model's per-token latency is within 7% of a live engine on three
runs in two sessions (within 8% with the command's two profiles) — closer than its first-token
latency on the same runs (1.11–1.38×). At the knee it is 1.5–1.7× too **slow**: it has the engine
falling off a cliff the real engine stayed on top of.

**Post hoc — written after seeing the result, and labelled so**
([`posthoc_range.md`](../artifacts/results/t1_itl/posthoc_range.md),
`scripts/t1_posthoc_range.py`). The decode model is `step = d0 + d1·b + d2·S`, with S the batch's
summed context. A calibration cell is b sequences with *unique* prompts, so S cannot exceed the KV
cache (24,592 tokens): the largest S any cell ever measured is **20,200** (C2), and 16,960 for
`kvsim calibrate`. Traffic that shares one system prompt stores it once — so far more sequences
fit, and each still attends over its whole context. In the simulator's replay of `K1_half`,
**97% of the tokens** are decoded in steps beyond the fitted range, at a median S of 160,000 —
eight times past it. On the three runs that came out right, the median is inside the range
(13,600–17,400), with a tail to 2–3× beyond.
- So where T1 fails, the model is a straight line drawn eight times past its last data point. That
  is a description of where the failure is, **not** a demonstrated cause: a small error elsewhere
  could equally tip the model over the knee, where its own pile-up then feeds itself. Telling
  those apart needs decode cells with a *shared* prefix, measured on a GPU.
- One consequence that is arithmetic, not evidence: a configuration that meets a 100 ms per-token
  limit has steps of at most ~100 ms, which is S ≲ 17,000 — inside the fitted range. T2's answers
  live where the decode model interpolates. They are still untested on traffic there: all five
  engine measurements above are themselves beyond 100 ms.

**What it changes, as registered.** T2 runs and is reported — as what a model that missed its own
per-token check says. No replica count from it goes into the README as guidance. And the next GPU
run has two more jobs than it had this morning: per-request token times on traffic *below* the
per-token limit, and decode cells with a shared prefix.

### 2026-10-05 — T2 RAN: all three hold, on all three calibrations. Under a target with both halves the counts double, and "keep each user on one GPU" survives.
Report: [`artifacts/results/t2_two_part_slo/report.md`](../artifacts/results/t2_two_part_slo/report.md)
— 41,472 simulator runs, the grids written at the registration commit `0be7bec`. **Guard passed:**
under the first-token target, on replicas 1–16, each calibration reproduces its committed table
exactly.

**As registered — predicted holds / holds / holds.** The target: p95 TTFT ≤ 2,000 ms **and** p95
time per output token ≤ 100 ms. A rule holds only if it holds on calibrations A, B and C.
- **T2a HOLDS.** With 1 prompt and with 4, no policy needs fewer replicas than session-sticky at
  any load: 12 of 12 points on each calibration, and at none of them is it a tie at "nothing
  holds".
- **T2b HOLDS.** At all 24 points, on all three, session-sticky needs no more replicas than
  round-robin and no more than least-loaded.
- **T2c HOLDS.** For session-sticky the two-part count is **1.93×, 2.00× and 2.00×** the
  first-token count (median over 24 points; needed ≥ 1.5). The back-of-envelope in the
  registration said about 2×.

In numbers, one shared prompt at 2 sessions/s, on A · B · C: session-sticky needs **10 · 12 · 12**
T4s where the first-token target said 5 · 5 · 5; round-robin 14 · 15 · 15 (was 7 · 7 · 7);
least-loaded 13 · 14 · 14.

**⚠️ What these counts are — as the registration said they would be if T1 failed.** T1 failed. So
they are what a model that missed its own per-token check says, and no replica count from this
grid goes into the README as sizing guidance. Two things can be said about how far to trust them,
and neither is a test: T1's failure is at the knee, where the model is too *slow*; and a cell that
meets a 100 ms limit decodes in batches inside the range the model was fitted on. The check that
is missing — traffic *below* the limit, timed per request — needs a GPU.

**Reported, not tested.**
- **The per-token half is the one that binds.** Of 432 (calibration, prompts, load, policy)
  cells it raises the replica count at 309; 20 need the same number under both targets; 103 are
  held under neither on 32 replicas.
- **The calibrations agree a little less than under the first-token target:** 81 of 144 cells
  identical on all three and 121 within one (first-token, on the same 1–32 axis: 90 and 126). The
  counts are about twice as large, so ± one is a smaller share of each.
- **A looser limit shrinks the factor, as it should:** 1.50× at 150 ms, 1.33–1.37× at 200 ms (the
  same document's limit for Llama2-70b). Session-sticky is best or tied at 21–22 of 24 points
  under either.
- **Session-sticky best or tied over all 24 points:** 18 · 22 · 22, and at 16 on all three.

**Exploratory — the 64-prompt question under the two-part target.** Not a test; it gets its own, on
fresh calibrations.
- **The load weight of 1 does not survive a per-token limit.** Under the first-token target it
  was the most consistent hint in the grid: 8 T4s at 64 prompts and 2 sessions/s on all three
  calibrations, against session-sticky's 10 · 12 · 12. Under the two-part target the same policy
  holds nothing there on 32 replicas, on any calibration — nor does pure prefix-affinity. They win
  the first token by crowding sessions onto the replicas that hold their prompt, and crowding is
  exactly what makes tokens slow.
- **Session-sticky against balanced affinity (α 16) at 64 prompts: no steady winner.** Loads 0.25
  to 4 sessions/s, sticky / balanced, on A · B · C: 3·3·3 / 3·3·3 — 4·5·5 / 4·4·5 — 7·8·8 / 7·8·7 —
  13·14·14 / 13·14·15 — 19·21·21 / 19·22·22 — 27·27·27 / 24·27·29. H2b's own rules give "holds,
  holds, falsified". One or two replicas apart, in both directions: inside what a calibration
  moves.

**A process fault, and what was done about it.** The three calibrations ran one after another in
one process, started on a clean tree. While B and C were computing I edited other files —
`kvsim calibrate`, the CLI, a new module — so their headers, written as each began, say
`git_dirty: true` and list those paths. The process had loaded the simulator before the edits, and
none of those files is imported by a grid run; but that is an argument, and a header cannot show
it. So 240 random cells of each grid were run again from a clean checkout whose `src/` is
identical to the registration commit's (`scripts/t2_verify_grid.py`, commit `ae4b483`):
**720 of 720 identical**
([`verification.json`](../artifacts/results/t2_two_part_slo/verification.json)). A sample and not
the whole grid because the full re-run is thirty minutes of every core and the machine was on
battery. The rule this breaks is simple, and now written down in `CONTRIBUTING.md`: nothing under
`src/` is edited while a registered run is in flight.

**What it changes.**
- **The README's counts are first-token counts, and the README must say what that leaves out:**
  under the model, a target that also limits token speed needs about twice the GPUs. It says so at
  the top of the scope note, with T1's failure beside it.
- **The headline stands, and a little more firmly:** *keep each user on one GPU* is best or tied
  with one prompt and with four, and never worse than round-robin or least-loaded, under either
  target, on all three calibrations.
- **The open half moves.** "Does prefix routing save GPUs at 64 prompts?" had one hint in its
  favour, the load weight of 1. That hint is a first-token effect. Under a target with both halves
  there is no sign of a saving — exploratory, so the registered test on fresh calibrations asks
  both: does the first-token saving repeat, and is there still none with both halves?
- **Next, on a GPU:** per-request token times on traffic around the per-token limit; decode cells
  that share a prefix; what an arriving request costs the sequences already decoding; and
  `kvsim calibrate` with its repeats.
