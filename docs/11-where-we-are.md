# 11 — Where we are, in plain words

**Written 2026-09-22; brought up to date 2026-09-28.** Everything this project has done so far, what it found, what was decided and
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

**28 Sep — the finding, and the tests around it** (§6b).
- **H6:** we re-ran the whole Phase 4 sweep as if on four other GPUs (2× and 4× faster; prefill 2×
  cheaper or dearer). The *order* of the policies moved more than we had allowed: **the ranking half
  failed.** A GPU count measured on our T4 is not a GPU count for your GPU.
- **Phase 5, the finding:** we gave the traffic many tenants, each with its own system prompt. With one
  shared prompt, keeping each user on one GPU wins. With 64 prompts — more than one GPU's cache can
  hold — routing by prefix wins, and keep-on-one-GPU cannot hold the target even on 8 GPUs. This flip
  survived all four "other GPUs": **H6's second half held.**
- **A real trace (Mooncake):** on real tool-agent and chat traffic, keep-on-one-GPU and balanced
  prefix routing came out within one GPU of each other. Our prediction was wrong, and is recorded so.
- **guidellm:** while using the tool we plan to contribute to, we found that its trace replay drops
  one token per block with common tokenizers, and wrote the fix with a test.
- **Phase 6 prepared:** the two-GPU notebook is written, tested against fake engines, and its
  predictions are committed. It waits for a Kaggle run.
- **The public side:** the package installs with one dependency; we measured what breaks at 10× and
  100×; the demo page and the README figure are built.

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
9. ⭐ **The finding: keep each user on one GPU — until your tenants' system prompts outgrow one GPU's
   cache; then route by prefix, and keep its load weight small.** With one prompt, session-sticky needs
   2 T4s where round-robin needs 3. With 64 prompts, prefix routing needs 4–5, and session-sticky
   cannot hold the target with 8. *Why it matters:* the choice follows from one number an operator can
   compute — the size of their prompts compared with one GPU's cache.
10. **"Send it to the GPU that has the prompt" fails whenever one prompt is popular:** it piles every
    user onto one GPU (H2, falsified). A load weight fixes that, but the right weight shrinks as prompts
    multiply: 8–16 with one prompt, 1 or none with 16 or more.
11. **The order of the policies depends on the hardware; the direction of the flip does not** (H6). So
    a user must measure their own GPU before trusting a GPU count — which is why `kvsim calibrate` moved
    up the list.
12. **On real traffic the difference is small.** On the Mooncake traces, keep-on-one-GPU and balanced
    prefix routing were within one GPU of each other, and both beat round-robin under load (6 T4s
    against 8 on the tool-agent trace at 4× speed). The many-prompt effect is not yet shown on real
    traffic.
13. **guidellm's trace replay drops one token per block** with tokenizers that add a start token — an
    upstream bug with a ready fix (the upstream log #7).
14. **At 100× the trace size, memory for the block hashes runs out first:** 537 MiB for 80,000 requests
    before the simulation starts. The simulation itself stays linear.
15. **On real coding-agent sessions (AgentX), keep-on-one-GPU and prefix routing tie exactly** — 2 T4s
    at every speed — and even pure prefix routing does not collapse, because nothing is shared across
    users there (R2, predicted and confirmed). Round-robin needs 5 T4s at 2× where they need 2. So
    pure prefix routing fails *because of shared prompts* (Mooncake), not because it routes by prefix.
16. **On two real GPUs, the cache model is right and the timing model is not** (Phase 6, run twice).
    Every hit rate matched the prediction within 1.4% — three of them to four decimals — and the
    policies came out in the predicted order, including prefix routing above keep-on-one-GPU with 64
    prompts. But the answers came 5–222× slower than predicted: each decode step took 3–4× what our
    T4 profile says, and at 2 requests per second per GPU that was enough to overload both engines.
    So every GPU count in this project is a simulator figure, and an optimistic one. *Why it matters:*
    it is the most important known weakness, and now it is measured.
17. **`kvsim calibrate` works on a real GPU** (C1): its cache check was exact on a T4 and it read the
    engine's capacity correctly. Its prefill measurements also show why a straight line is not
    enough: 0.71 ms per token up to 1,024 tokens, 1.12 ms at 2,040.

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

## 6b. Phase 5, H6 and the real trace — the finding

**What we did.** The same simulated T4 fleet, but now the traffic has **tenants**: 1, 4, 16 or 64
companies, each with its own long system prompt. One prompt fits easily in one GPU's cache; 64 prompts
need 2.66 GPUs' worth of cache. Six traffic levels, 1 to 8 GPUs, three repeats each — with the tests
committed before the run.

**What came out** (2 sessions per second, a 2-second first-token target):

| Prompts | Fewest T4s that hold the target |
|---|---|
| 1 | session-sticky **2** · balanced prefix routing 3 · round-robin 3 |
| 64 | balanced prefix routing **5** (4 with a smaller load weight) · session-sticky, round-robin, least-loaded: not with 8 |

**Why.** As prompts multiply, round-robin's hit rate falls from 0.84 to 0.53, because every GPU must
hold every prompt; session-sticky's falls to 0.82. Prefix routing keeps 0.87–0.92, because each prompt
lives on one GPU.

**Then we tried to break it.**
- *Other hardware (H6):* the flip held on all four imaginary GPUs; the GPU counts did not.
- *Real traffic (R1):* on the Mooncake traces the two good policies tied within one GPU. Our best
  guess, written down before the run and not yet tested: the real traces' shared prompts are mostly
  not in use at the same time, so a cache has far less to hold at once than with our 64 tenants.
- *Real traffic (R2):* on AgentX's coding-agent sessions, where nothing is shared across users, the two
  tied exactly, as predicted — and pure prefix routing did not collapse, which shows its collapse on
  Mooncake came from the shared prompts.
- *The demo page* lets anyone click from 1 to 64 prompts and watch the answer change.

**How much to trust it.** A T4 result on generated traffic, from a cache model that matched the real
engine exactly. The direction survives other hardware; the GPU counts do not. Real traffic has not
shown the flip yet. Phase 6, on two real GPUs, is the next check.

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
- **Our headline said "with a small load weight"** — but where the headline applies, a weight of 1 did
  no better than no weight at all. Caught when the demo grid put them side by side; reworded, with a
  dated note in `09-hypotheses.md`.
- **Our first way of finding sessions in the Mooncake trace glued 9,202 requests into one "session"**,
  because they all began with the same system prompt. Fixed: a block that 100 or more requests continue
  from is a shared prompt, not a conversation.
- **Results were marked "dirty" because of an unrelated file** in the folder. The check now looks only
  at code, configs and notebooks.
- **Our first scale for the real trace was too big for a T4** — every policy failed. We changed it
  before the real run and wrote down why.

## 8. Where we are right now

| Phase | What | Status |
|---|---|---|
| 0 | Prove the engine's counters are real and learn what they count | ✅ done |
| 1–2 | Traffic, cache model, policies, simulator | ✅ done |
| 2b | Cache model rebuilt to follow vLLM's real rules | ✅ done |
| 3 | **The calibration** — model vs real engine, pre-registered | ✅ **done: exact** |
| 4 | Calibrated sweep; H1, H5, H6; the load weight; real traces | ✅ mostly — H6's ranking half ⛔, its crossover half ✅; Mooncake (R1) ⛔; AgentX (R2) ✅ |
| 5 | **The finding** — where prefix-aware routing starts paying off | ✅ **done** — H2 ⛔, **H2b ✅** |
| 6 | Two-GPU real-engine check under pressure | ✅ **holds**, run twice — hit rates within 1.4%; ⚠️ latency 5–222× the prediction |
| 7 | Package, command line, static demo page, "what breaks at 10× and 100×" | 🔨 demo, figure, 10×/100× done; `kvsim calibrate` works on a real T4 (C1); PyPI left |
| 8 | Write-up, launch, the upstream pull request | 🔨 the guidellm fix is ready; the preprint is drafted (`paper/`) |

## 9. What is left

**Manual steps (they need an account):**
- File the vLLM issue (the draft is the upstream log #1).
- Open the guidellm issue, then the pull request (the patch is in `docs/upstream/`, the text in
  the upstream log #7–8).
- Ask someone for an arXiv endorsement, by mid-October.
- Create the GitHub repository, push, and turn on Pages (Settings → Pages → Source: *GitHub Actions*).

**Next on the build side:**
- ✅ done 28 Sep: the preprint draft (`paper/`), and `kvsim calibrate` — so a user can measure their
  own GPU, as H6 says they must — which worked on a real T4 (C1);
- ✅ done 28 Sep: Phase 6's write-up — it held on hit rates and exposed the timing model;
- ⭐ **next: a timing model that knows about context length** — a one-GPU Kaggle run that measures
  decode speed at several context lengths and batch sizes, then a model fitted on it and tested on
  data it was not fitted on. Until then, every GPU count stays marked as optimistic;
- ✅ done 28 Sep: the AgentX trace (R2 holds), and a contributor guide with three good first issues;
- later: the PyPI release.

**Known limits, owned rather than hidden:**
- the timing model is the weak part: it cannot express the slow single-sequence decode, it charges
  a little too much (~20 ms) for a request arriving at an idle GPU, it has no term for context
  length — and on two live GPUs with ~1,700-token prompts it was 5–222× optimistic (Phase 6);
- every replay is *open-loop* — real agents wait for an answer before sending the next turn, and would
  then reuse the answer's cache blocks, which we cannot measure this way;
- one GPU type (T4) and one model (Qwen3-4B) so far;
- one real trace family (Mooncake), scaled down 32× to fit a T4, with sessions inferred.

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
| The finding, its figure and the demo | The README's first section; [`../site/index.html`](../site/index.html); [`figures/`](figures/) |
| Every experiment's report | `../artifacts/results/*/report.md` — `phase5_crossover`, `h6_ranking`, `h6_crossover`, `realtrace_mooncake`, `scale` |
| The two-GPU notebook | [`../notebooks/phase6_sim2real.ipynb`](../notebooks/phase6_sim2real.ipynb) |
| The guidellm fix | [`upstream/guidellm-add-special-tokens.patch`](upstream/guidellm-add-special-tokens.patch) |
