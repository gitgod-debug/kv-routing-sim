# 11 — Where we are, in plain words

**Written 2026-09-22.** Everything this project has done so far, what it found, what was decided and
why, what went wrong along the way, and what is left — in plain words, for someone who has forgotten
all of it. The other docs have the detail; this one has the story.

---

## 1. What this project is, in one paragraph

When you run an AI model on several GPUs, every incoming request has to be sent to one of them. The
way you choose — the **routing policy** — quietly decides how often a GPU can **reuse work it has
already done** for an earlier, similar request (its "cache"). More reuse means faster answers and fewer
GPUs, which means a smaller bill. Today people pick a policy by guessing. **kv-routing-sim** is a
simulator that answers the question properly: give it your traffic, and it tells you which policy to
use, how many GPUs you need, and what it costs. What makes it worth trusting is that we **checked it
against a real engine** (vLLM) and published how wrong it is — which, for the cache, turned out to be
**not at all**.

## 2. The words you need

| Word | Plain meaning |
|---|---|
| **KV cache** | The GPU's memory of work already done for a prompt. Reusing it skips recomputing. |
| **Prefix caching** | If two prompts start the same way (same system prompt, same chat history), the second can reuse the first one's cache for that shared start. |
| **Block** | The cache is stored in chunks of 16 tokens. Only full chunks can be reused. |
| **Hit rate** | The share of prompt tokens that came from the cache instead of being recomputed. |
| **Replica** | One copy of the model on one GPU. |
| **Routing policy** | The rule that picks a replica: *round-robin* (take turns), *least-loaded* (the least busy), *session-sticky* (keep a user on one GPU), *prefix-affinity* (send a request to the GPU that already has its prompt cached). |
| **Prefill / decode** | Prefill = reading the prompt (costs time per uncached token). Decode = writing the answer, one token per step. |
| **TTFT, p95** | Time to first token — how long before the answer starts. p95 = the time 95% of requests beat. |
| **Preemption** | When GPU memory runs out, the engine pauses a request, throws away its cache, and redoes it later. |
| **Pre-registration** | Writing a prediction down and committing it **before** running the experiment, so the result cannot be quietly bent to fit. We did this for every result. |

## 3. The story so far

**Day 1 (20 Sep) — plan and build.** We wrote down what we would claim and what would prove us wrong
(hypotheses H1–H6, `docs/09`), then built the whole simulator: traffic generator, cache model, seven
routing policies, the event loop, statistics, cost model, command-line tool. It could not yet be
trusted, because three numbers about real GPUs had never been measured. The software refuses to present
anything as a prediction until they are.

**The hardware turned out to be Kaggle's free 2× Tesla T4**, not the A100/H100 first assumed, and that
was fine: cache behaviour is the same on any GPU; only speed differs, and speed numbers are labelled
"T4". Kaggle runs notebooks, so every GPU experiment became a self-contained notebook.

**Day 2 (21 Sep) — Phase 0, the first real engine.** We started vLLM on the T4 and checked the cache
counters exist, move, and what they count. They count **tokens**, not blocks — so our model was off by
16× in its units, and we fixed it. After that, the simulator matched the engine **to the token**. Three
of that notebook's other verdicts were wrong, and all three were bugs in *our notebook*, not facts about
vLLM.

**Day 3 (22 Sep) — Phase 0 v2, Phase 2b, Phase 3.**
- **v2** fixed those bugs and made the first two-GPU prediction: **1,088 tokens recomputed, predicted
  1,088.** Exact.
- **Phase 2b:** reading vLLM's source showed its cache is **not** a simple "least recently used" list.
  We rebuilt the cache model to follow vLLM's actual rules, line by line, and kept the simple version
  for comparison.
- **Phase 3, the real test:** a deliberately small cache that overflows, 120 realistic chat requests,
  three runs of each workload — against predictions committed before the run. **The new model got every
  number exactly right.** The simple model was off by 11.9 points.

**Tonight — Phase 4 started.** The measured T4 speeds now drive the simulator, costs come from real
cited prices, and the first full calibrated sweep was designed, registered and run. Its headline:
the simple "keep each user on one GPU" policy beat the clever cache-seeking ones (§6).

## 4. What we found

1. **The simulator reproduces a real vLLM's cache exactly — even when the cache is overflowing.**
   Every counter, and the cached-token count of every one of 120 requests, matched on all six runs. On
   top of that, a concurrent run it had never seen was predicted within **0.3%** on hit rate and **~7%**
   on tail latency. *Why it matters:* this is the project's credential — almost nobody publishes how
   wrong their cache model is.
2. **Finished answers crowd the cache.** vLLM keeps the blocks of generated text in its cache too. They
   are almost never reused, but they take space from prompt prefixes that would be. A simple cache model
   ignores them and **overstates the hit rate by 11.9 points** when answers are long, and by only 0.17
   when they are short. *Why it matters:* it is the difference between a model that sounds right and
   one that is.
3. **Under memory pressure, what decides speed is waiting for memory, not the hit rate.** When memory
   runs out, vLLM makes new requests wait and pauses running ones. The simple model does neither, so it
   predicted a **p95 TTFT 73% too low** on the concurrent run. *Why it matters:* the routing question
   is really a latency question under load, so the model has to get this right — and now it does.
4. **What one T4 costs in time (Qwen3-4B):** reading a prompt costs **0.70 ms per uncached token**; one
   decode step costs **43–76 ms**, depending on how many answers are being written at once. Oddly,
   writing **one** answer (64 ms per token) is *slower* than writing two to eight at once (43–58 ms) —
   measured, not yet explained.
5. **Don't estimate cache capacity — read it from the engine.** The arithmetic said a T4 holds ~2,394
   cache blocks for this model; vLLM actually allocated **1,537** — the estimate was **56% too high**,
   because vLLM reserves memory for other things too. For the small model it was only 7–12% off.
6. **vLLM's eviction statistics count more than evictions.** Its code records an "eviction" whenever a
   block is reused, even if the block held nothing reusable. Reproduced exactly by our model, and logged
   as an upstream bug candidate (the upstream log #1).
7. **vLLM 0.30.0 runs its "async" scheduler by default, and nothing in its log says so.** Our
   simulator models the older, synchronous one. We showed from the source that the difference cannot
   change one-at-a-time results, and flagged it for concurrent ones.
8. **Cache-seeking routing backfires under load, and the simple policy wins** — see §6. The early
   hint from the uncalibrated simulator survived calibration.

## 5. What we decided, and why

| Decision | Why |
|---|---|
| **This project first**; the other one (colpali-lite) paused at its Phase-0 gate | This one is the critical path; the other can wait safely at a passed checkpoint |
| **Kaggle 2× T4** is the hardware; A100/H100 is not in the plan | It is what is available, and cache behaviour does not depend on the GPU. Speed numbers are labelled T4 |
| **Qwen3-0.6B** for the first checks, **Qwen3-4B** for calibration; not 8B, not Llama | 8B does not fit one T4; 4B's cache has exactly 8B's shape; Llama needs a licence click and a token |
| **Notebooks** for the GPU box | Kaggle runs notebooks; each one is self-contained (no repo, no data, no token) |
| **The demo is a static web page**, not a hosted app | Free hosted apps now need a paid plan or fall asleep; a static page is free, never sleeps, and every number on it comes from a committed script (ADR-011) |
| **Predict first, then measure** — every result pre-registered | So a result cannot be quietly tuned to match. When a prediction was wrong, we wrote that down too |
| **Keep the simple cache model** alongside the faithful one | It is the comparison that shows what the faithful one adds (the 11.9 points) |
| **A small, forced cache** in Phase 3 (512 blocks) | So the cache overflows, and capacity is an exact input rather than a guess |
| **Phase 4 uses the timing model as fitted** (ADR-012) | It was validated where routing matters; fixing its two known flaws needs a new GPU test, not a re-fit to data we have already seen |
| **Costs from real, cited prices** | AWS g4dn.xlarge (one T4) at $0.526/hour, ₹95.87 per dollar, a 2-second first-token target from MLPerf. Every number has a source and a date |

## 6. Phase 4 — the first calibrated sweep

**What we did.** We simulated a fleet of 1 to 8 T4 GPUs, using the speeds and memory measured on the
real engine, under two kinds of chat traffic — **low sharing** (users share a short system prompt) and
**high sharing** (they share a long one, 80% of every prompt) — at six traffic levels, with all seven
routing policies, five times each: 3,360 simulated runs. The question: which policy holds a
**2-second first-token target** (95% of requests) with the fewest GPUs, and what does that cost?
The tests (H1, H5) were written down and committed before it ran.

**What came out:**

1. **Sending requests to "the GPU that already has the prompt cached" backfires under load.** When
   traffic is light, it does exactly what it promises — more cache reuse than any other policy (H1
   holds at 1–2 sessions per second). But when traffic grows, it sends *everything* to the one GPU
   holding the shared prompt. That GPU runs out of memory, has to pause and redo requests (110–212
   times per run), and in doing so throws away the very cache it was protecting. Its reuse ends up
   **worse** than a policy that ignores the cache entirely. So **H1 is falsified** — by load, not by
   a broken model.
2. **The "balanced" version of that policy works only when the shared prompt is short.** With a short
   shared prompt, it holds the target with 1–3 fewer GPUs than round-robin (**H5 holds**, in that one
   case). With a long one, it still piles traffic up and misses the target. ⚠️ Its balancing weight has
   never been tuned; that is the next experiment, registered separately.
3. ⭐ **The simple policy wins: keep each user on one GPU (session-sticky).** It holds the target with
   the fewest GPUs at every traffic level, on both kinds of traffic — **2, 3, 3, 4, 5 GPUs where
   round-robin needs 3, 4, 5, 7, 8** — which is **25–42% cheaper**. At the heaviest traffic with a
   long shared prompt, round-robin cannot hold the target even with 8 GPUs; session-sticky does it
   with 5. It never runs out of memory, because it spreads users evenly, and it still gets most of
   the reuse: each GPU keeps its own copy of the shared prompt, and each conversation's history stays
   with it.

**How much to trust it.** The cache model behind this is the one that matched the real engine exactly.
But the traffic is generated, not real; the speeds are one T4 and one model; and whether the *ranking*
would hold on other hardware (H6) is not yet tested. So this is a result about this setup, and not yet
a claim about anyone else's fleet.

## 7. Mistakes we made — and caught

Kept here on purpose. Each one was caught by a check we had built for exactly that purpose.

- **The first notebook's three wrong verdicts** — comparing a number that was undefined, using the wrong
  switch for the statistics, and a default setting that ran out of memory. All ours, not vLLM's.
- **The vLLM version drifted between two runs** (0.29 → 0.30) because the install was not pinned. Now
  pinned.
- **We predicted the wrong direction** for how the simple model would err (H4's first amendment). The
  measurement said otherwise; we recorded the wrong prediction and the correction, and edited neither.
- **Our generated prompts used token numbers no real model has** — a live engine would have rejected
  them. Caught while building Phase 3, before any GPU time was spent.
- **A leftover server could have been mistaken for a new one** — found by the dry-run against a fake
  engine, not on the real box.
- **`kvsim replay` silently ignored the calibrated profile** and always reported the uncalibrated one.
  Found and fixed when Phase 4 started.
- **A web search gave an exchange rate from 2024** (₹83); reading the Federal Reserve's own table gave
  ₹95.87. We use the table.
- **Earlier, smaller ones:** a router that fought itself for ownership, a timestamp of zero treated as
  missing, a trace written out of order, and a few over-claims corrected once measured.

## 8. Where we are right now

| Phase | What | Status |
|---|---|---|
| 0 | Prove the engine's counters are real and learn what they count | ✅ done |
| 1–2 | Traffic, cache model, policies, simulator | ✅ done |
| 2b | Cache model rebuilt to follow vLLM's real rules | ✅ done |
| 3 | **The calibration** — model vs real engine, pre-registered | ✅ **done: exact** |
| 4 | Calibrated sweep of policies × GPU counts; the recommendation; H1, H5, H6 | 🔨 **first sweep done**; H6 next |
| 5 | **The finding** — where prefix-aware routing stops paying off | ⬜ next after 4 |
| 6 | Two-GPU real-engine check under pressure | ⬜ |
| 7 | Package, command line, static demo page, "what breaks at 10× and 100×" | ⬜ |
| 8 | Write-up, launch, the upstream pull request | ⬜ |

## 9. What is left

**Phase 4 (in progress):**
- **H6** — re-run the sweep as if on a faster GPU and check the *ranking* of policies does not change
  (if it holds, people can trust the recommendation without calibrating their own GPU);
- tune the balanced policy's weight (its "alpha"), as a new, separately registered experiment;
- a real-world trace, if a public one with prompt-sharing information can be obtained;
- the remaining tests the plan asks for (a small "golden" sweep, and a property test that prefix-aware
  routing never loses on perfectly shared prompts).

**Phase 5 — the headline:** find the boundary — how much prompt sharing, at what load — where sending
requests to the GPU that has them cached stops beating simply sending them to the least busy GPU. It
gets published as a curve with error bars, and it becomes the README's first sentence.

**Phases 6–8:** a two-GPU check on the real engine under pressure; the installable package, the static
demo page (its grid rebuilt from the calibrated profile), and measured answers to "what breaks at 10×
and 100×"; then the write-up, the launch, and an upstream pull request to `vllm-project/guidellm`,
seeded by the friction log (the upstream log, six entries so far).

**Known limits, owned rather than hidden:**
- the timing model is the weak part: it cannot express the slow single-sequence decode, and it charges
  a little too much (~20 ms) for a request arriving at an idle GPU;
- every replay is *open-loop* — real agents wait for an answer before sending the next turn, and would
  then reuse the answer's cache blocks, which we cannot measure this way;
- one GPU type (T4) and one model (Qwen3-4B) so far.

## 10. Where to find things

| You want | Look at |
|---|---|
| The calibration result | [`artifacts/results/phase3_kaggle_t4/calibration_report.md`](../artifacts/results/phase3_kaggle_t4/calibration_report.md), and the README's calibration section |
| What each prediction was, and what happened | [`09-hypotheses.md`](09-hypotheses.md) — its dated entries are the lab notebook |
| The to-do list | [`../CHECKPOINTS.md`](../CHECKPOINTS.md) |
| Why each design choice was made | [`02-architecture-decisions.md`](02-architecture-decisions.md) (ADR-001 to ADR-012) |
| How to run the GPU notebooks | [`../notebooks/README.md`](../notebooks/README.md) |
| The measured T4 profile and the cost numbers | [`../configs/profiles/`](../configs/profiles/), [`../configs/cost_model.yaml`](../configs/cost_model.yaml) |
| Upstream bugs and friction we hit | the upstream log |
