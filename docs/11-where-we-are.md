# 11 — Where we are, in plain words

**Written 2026-09-22; brought up to date 2026-10-06.** Everything this project has done so far, what it found, what was decided and
why, what went wrong along the way, and what is left — in plain words, for someone who has forgotten
all of it. The other docs have the detail; this one has the story.

---

## 1. What this project is, in one paragraph

When you run an AI model on several GPUs, every incoming request has to be sent to one of them. The
way you choose — the **routing policy** — quietly decides how often a GPU can **reuse work it has
already done** for an earlier, similar request (its "cache"). More reuse means faster answers and fewer
GPUs, which means a smaller bill. Many people still pick a policy by guessing. *(Until 5 Oct this
said "today people pick by guessing", as if no tool existed. One does — see finding 27.)*
**kv-routing-sim** is a simulator that answers the question: give it your traffic, and it tells you which policy to
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

**5 Oct, later — the second speed.** Until today we only measured how long the *first* word of an
answer takes. We now also measure how fast the *rest* of the words come ("time per token"), and
tested two things, registered first, with no GPU. **T1** asked: is the simulator right about that
second speed? Mostly — within 7% on three real runs — but 1.5–1.7× too slow on the two runs where
the GPU was at its limit. So T1 failed. **T2** asked: if the target covers both speeds, what
changes? The GPU counts about double, and "keep each user on one GPU" still wins — on all three
calibrations (findings 24–26).

**5 Oct — the measuring command proven on a real GPU, and the finding narrowed.** C3 ran on
Kaggle (through its API now; the first attempt did not start because Kaggle had changed its
software). All three predictions came true: `kvsim calibrate` works on a real T4, the profile it
writes predicts traffic it never saw within 1.4×, and so does the older profile from C2. But the
same run measured something we had not measured before: **how much the GPU counts move when the
same GPU is calibrated again.** Mostly by nothing or by one GPU — and the test that had changed the
headline on 30 Sep had been decided by exactly one GPU at one point. On two more calibrations it
gave the opposite answer. So the headline now claims only what all three support (findings 21–23).

**30 Sep — the timing model fixed, and the finding changed.** C2 ran on Kaggle: all three
predictions came true — context length slows decode and prefill, CPU contention does not, and a new
context-aware model predicts traffic it never saw within 1.4×. Re-running the finding's test on that
model (registered first), the "then route by prefix" half failed. The headline is now *keep each user
on one GPU* (findings 18 and 19 below).

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
9. **The first finding (28 Sep) — later overturned, see 18:** *keep each user on one GPU until your
   tenants' system prompts outgrow one GPU's cache; then route by prefix, and keep its load weight
   small.* With one prompt, session-sticky needed 2 T4s where round-robin needed 3; with 64 prompts,
   prefix routing needed 4–5 and session-sticky could not hold the target with 8 — on the first,
   context-free timing model.
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
18. ⭐ **The finding on 30 Sep — narrowed on 5 Oct, see 22: keep each user on one GPU.** C2 (30 Sep, registered) found why the timing
    model was wrong: decode and prefill get slower as the context grows, and a busy second GPU did
    *not* slow the first. A new model with a context term predicted traffic it had never seen within
    1.4×, and — checked afterwards — Phase 6's two live GPUs within 1.8×. We then re-ran the finding's
    test on it, as registered — and **its second half failed**: with 64 prompts, prefix routing no
    longer needed fewer GPUs. Keep-on-one-GPU needs the fewest GPUs, or ties, at 22 of 24 points; at
    2 sessions/s it needs 5 T4s with one prompt (round-robin 7) and 10 with 64 prompts (prefix
    routing 11, round-robin 16). *Why it matters:* the cache model was exact from the start, and it
    was not enough — the routing answer changed with the timing model. We predicted the old finding
    would hold, and wrote down that it did not.
19. *(On calibration A only — see 22.)* **Prefix routing still gets the better hit rate with many prompts** (0.87 against 0.80), but it no
    longer saves GPUs: every policy pays the full decode cost of the context, cached or not, and
    crowding sessions onto the GPUs that hold their prompt makes the work uneven. With a small load
    weight it saved 2 of 10 GPUs at one point — found after the test, so not a finding.
20. **`kvsim calibrate` was missing one measurement, and we found it before spending a GPU.** The
    command timed prefill only on *cold* prompts (nothing cached). We took the data C2 had already
    recorded, kept only what the command would have measured, and asked: would that have been
    enough? No — it put the slow end of a held-out run (p95) off by 5.7–7.2×. Real agent traffic is
    a long cached prefix plus a few new tokens, and cold prompts never measure that case. The
    command now measures it too, and the same check comes out within 1.36×. This was done after
    the fact, on old data, and is labelled so; the real test is C3, on a GPU.
21. **`kvsim calibrate` works on a real GPU in its final form** (C3, 5 Oct, registered; all three
    predictions true). It ran in 160 seconds on a T4, its cache check was exact again, and the
    profile it wrote predicted three kinds of traffic it had never seen within 1.41× — one of them
    with prompts half as long as anything tested before. C2's older profile predicted the same
    traffic within 1.38×, from numbers written down four days earlier.
22. ⭐ **A GPU count from this tool is good to about ± one GPU — and one of our verdicts was inside
    that margin.** We ran the finding's grid on three calibrations of the same T4 (C2's, and the
    command's two runs). Of 144 GPU counts, 102 are the same on all three and 138 are within one.
    That is a good result for the tool. But the test that changed the headline on 30 Sep ("prefix
    routing does not save a GPU at 64 prompts") was decided by 10 against 11 GPUs at one point — and
    on the other two calibrations that point reads 12 against 11 and 12 against 10, so the same test
    says the opposite. *What survives all three:* keep each user on one GPU — it needs the fewest
    GPUs or ties at every load with one prompt and with four, and it is never worse than round-robin
    or least-loaded anywhere. *What does not:* which of sticky and prefix routing wins with 64
    prompts. *Why it matters:* a result is only as sharp as the ruler, and we had not measured the
    ruler. Now it is measured, and the README says "± one".
23. **Two more limits, both found in C3's data.** (a) Near the limit of a GPU, one run of the
    measuring command is not enough: a second run 25 minutes later was 3–5% different, and that
    made one prediction 4.2× wrong (the other five stayed within 1.35×). (b) Our speed target only
    covers the *first* token. The rule we took it from (MLPerf) also limits the time per token — 100
    ms for a model this size — and the live engine took 134–521 ms per token in C3. So the GPU counts
    say when the first token is on time, not when the whole answer streams at a usable speed. Both
    are written down as known limits; neither is fixed yet.
24. ⛔ **The simulator's second speed is right in the middle and wrong at the edge** (T1, 5 Oct,
    registered; predicted to hold, and it did not). "Time per token" is how fast an answer streams
    after its first word. We compared the simulator with the engine's own counter on the five real
    traffic runs we already had. On three it is within 7%. On the two where the GPU was at its
    limit it is 1.5–1.7× too slow: the simulator thinks the engine falls over, and the real engine
    did not. *A likely reason, found afterwards:* our measurements of decode speed could never use
    more text at once than fits in the GPU's cache (about 20,000 tokens). Traffic that shares one
    system prompt can — the shared part is stored once — and at the limit the simulator was working
    8× beyond anything we had measured.
25. ✅ **With a target for both speeds, the GPU counts about double — and the advice holds** (T2,
    5 Oct, registered; all three rules true on all three calibrations). The rule our 2-second
    target comes from also says: no more than 100 ms per token. Under both limits, keeping each
    user on one GPU is still best or tied with one prompt and with four, it is never worse than
    round-robin or least-loaded, and it needs 1.9–2.0× the GPUs we had counted. ⚠️ But because of
    24, this "×2" is what the model says, not a checked number — and no real traffic has been
    measured below 100 ms per token yet. The README says both things.
26. **The one hint in favour of prefix routing was about the first word only.** With 64 prompts, a
    small load weight needed 8 GPUs where keeping users on one GPU needed 10–12. Under a target
    for both speeds that policy holds nothing at all: it gets a fast first word by crowding users
    onto the GPUs that hold their prompt, and crowding is what makes the other words slow.
    (Exploratory, not a test.)
27. **Our main finding was already published, and a bigger tool for the same question exists**
    (found 5 Oct, by reading other people's work again while a GPU run was going).
    - *The finding.* A research group at Shanghai Jiao Tong University and Alibaba published
      "route each user's session to one machine" in July 2026 (the paper is called SMetric), from
      real traffic on 32 real GPUs. The same group showed earlier (LMetric, at the OSDI '26
      conference) why mixing "has the prompt" and "is not busy" with a hand-picked weight never has
      one right weight — which is what we saw with our weights 16 and 1. So our result **agrees**
      with theirs. It is not a discovery.
    - *One thing they saw that we cannot.* Real sessions are very different in size. Then "keep
      each user on one GPU" leaves some GPUs twice as loaded as the average. In our generated
      traffic every session has four turns, so we measured this policy where it looks its best.
    - *The tool.* NVIDIA's Dynamo project has had a simulator since March 2026 (now called
      AISimulate, or DynoSim). It replays a saved trace on pretend GPUs, compares round-robin with
      NVIDIA's own cache-aware router, and recommends a number of GPUs. Its tutorial uses the same
      Mooncake trace we used.
    - *What is still ours.* Three things: our cache model was checked against a real engine and
      matched exactly; every test was written down before it ran, and the failed ones are
      published; and we measured how far a simulator's GPU count can be trusted (findings 22–24).
      The README and the prior-art file now say exactly this, and no more.
28. ✅ **The simulator's second speed is right where it matters** (C4, run on a real T4 on 5 Oct,
    registered first; seven of its nine rules held).
    - We chose three kinds of traffic: one under the 100 ms-per-token limit, one on it, one over
      it. The simulator's number was within 0%, 0% and 7% of the real engine's. And its yes-or-no
      answer ("does this traffic meet the limit?") was the same as the engine's own.
    - Many requests that share one long prompt do not cost extra because they share it.
    - When a new request arrives, the requests already running wait once while its prompt is read.
      The simulator's rule for that wait was within 3–19%.
    - The simulator's decode speed is right (within 3%) up to twice the batch sizes it was
      calibrated on, and **12–15% too slow** at 5–9 times those sizes. That small error is what
      grew into the big one in finding 24: at the edge of a GPU, a slightly slower step means more
      requests pile up, which makes the next step slower again.
    - `kvsim calibrate` with its three repeats works on a real GPU: three runs, six minutes each,
      exact on the cache check every time.
    So the "×2" of finding 25 now stands on a checked model — for one GPU and three kinds of
    traffic. A whole fleet has still not been measured.
29. ⛔ **The "how firm" range the tool prints is too narrow** (C4, rule E; I had predicted it would
    hold). We ran `kvsim calibrate` three times in one hour on the same GPU. Each run gives a
    range of GPU counts from its own three repeats. Another run's answer was inside that range in
    82% of cases; we had asked for 90%. If we make the range one GPU wider on each side, it is 95%.
    The reason: a run's three repeats are taken within six minutes, so they agree with each other.
    The two runs taken one after the other also agreed (97%). The run taken an hour earlier did
    not (75%). So the range is a lower limit, and the honest advice is: take the top of the range
    and add one.
30. **With 64 prompts, the two policies are one GPU apart at most — and my second prediction
    failed** (C4, rules F1 and F2).
    - With a target for the first word only, the small load weight (finding 26) again needed fewer
      GPUs than keeping users on one GPU: 8, 7, 8 against 9, 10, 10 on three fresh calibrations.
      (F1 held.) With a target for both speeds it again holds nothing.
    - With a target for both speeds, I had written that prefix routing (with the registered weight
      16) would need fewer GPUs at no more than 2 of 6 loads. On two of three fresh calibrations it
      needed one GPU fewer at 3 and at 4 loads. (F2 failed.) Every difference is exactly one GPU,
      and one GPU is what two calibrations disagree by (finding 29). So it is not a result for
      prefix routing. But "there is no saving" was wrong as a prediction.
31. **The two real-traffic tests, run again on the checked timing model** (6 Oct, registered
    first; the open box from 30 Sep).
    - *Mooncake (R1):* as predicted, no winner. Keep-on-one-GPU and balanced prefix routing need
      the same number of GPUs on the tool-agent trace (5, 5, 5) and are within one on the
      conversation trace, on two calibrations. So "within one GPU" is now a registered result on
      the checked model. Round-robin is also within one GPU of them at these loads.
    - *The loads are lower than before.* On the first model we tested at 1×, 2× and 4× the
      recorded pace. On the checked model round-robin cannot hold the target above 1× on 8 GPUs,
      so the test ran at 0.25×, 0.5× and 1×. At 1× the checked model needs 1–2 GPUs more than
      the first model said.
    - *AgentX (R2) could not be tested.* The rule for choosing speeds, written down before,
      needs round-robin to hold the target at some speed. It holds at none: at that scale 28% of
      the prompts are longer than the engine allows, and one cache miss on a normal prompt costs
      1.6 seconds. We changed nothing to rescue the test. Its old result ("the two tie exactly")
      is from the first timing model and stays labelled that way.

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
- **I edited code while a registered run was still going** (5 Oct). The run had loaded its code
  before, so nothing changed — but two of its three result files now say "the folder was not
  clean". We did not argue it away: 720 random cells were run again from a clean copy, and all 720
  match. The rule is now in the contributor guide: do not touch the code while a registered run is
  in flight.
- **A glossary line was wrong for two weeks:** it said routing "barely affects" the speed of
  tokens after the first. It does — it is the half of the target that decides the GPU count.
- **Three checks of other people's work missed the closest work** (found 5 Oct). They searched
  GitHub for projects by name, and put NVIDIA's Dynamo on a list of "systems that route, not tools
  that help you choose" without opening its documentation. Nobody searched research papers at all.
  So for two weeks the README claimed a gap that NVIDIA had filled in March, and presented as our
  finding something a paper had published in July. The fourth check found it by following one link
  in a competitor's README. Three rules were added: search papers by the question; read the
  documentation of every "production system" for a simulator; follow the references of every close
  match. The old claims are kept in the docs, struck through, with the date.
- **Two predictions about C4 were wrong** (6 Oct), and both are reported as failed. I expected
  one calibration's own repeats to cover what another calibration gives (82%, not 90%). And I
  wrote a rule about the 64-prompt case from a table I had only looked at, not tested. Also: the
  registered analysis script forgot one item the registration had promised (each fresh
  calibration against the micro-benchmark cells). It was added afterwards, with the registration's
  own arithmetic, in a separate script that is labelled "written after the results were seen".
- **The GPU box changed under us** (5 Oct). Kaggle updated its software, and the engine could no
  longer start: one package it needs was built for a different CUDA version, under the same version
  number, so the installer left it in place. C3's first attempt stopped before measuring anything.
  The notebook now checks the software in its first minute and repairs that package; we tested the
  repair on Kaggle's own image with a small probe notebook before running the real one. Nothing
  that was registered changed.

## 8. Where we are right now

| Phase | What | Status |
|---|---|---|
| 0 | Prove the engine's counters are real and learn what they count | ✅ done |
| 1–2 | Traffic, cache model, policies, simulator | ✅ done |
| 2b | Cache model rebuilt to follow vLLM's real rules | ✅ done |
| 3 | **The calibration** — model vs real engine, pre-registered | ✅ **done: exact** |
| 4 | Calibrated sweep; H1, H5, H6; the load weight; real traces | ✅ mostly — H6's ranking half ⛔, its crossover half ✅ (first timing model only); Mooncake (R1) ⛔ on both timing models, "within one GPU" ✅ on the checked one; AgentX (R2) ✅ on the first model, not testable on the checked one |
| 5 | **The finding** — when prefix-aware routing pays | ✅ **done, changed, then narrowed** — H2 ⛔; H2b ✅ on the first timing model, ⛔ on calibration A of the validated one, ✅ on two more calibrations of the same T4: *keep each user on one GPU* holds on all three; the prefix-routing half is not settled |
| 6 | Two-GPU real-engine check under pressure | ✅ **holds**, run twice — hit rates within 1.4%; ⚠️ latency 5–222× the prediction |
| 7 | Package, command line, static demo page, "what breaks at 10× and 100×" | 🔨 demo, figure, 10×/100× done; `kvsim calibrate` works on a real T4 in all its forms (C1, C3, and with its repeats in C4); the per-token model checked on a real T4 (C4); PyPI left |
| 8 | Write-up, launch, the upstream pull request | 🔨 the guidellm fix is ready; the preprint is drafted (`paper/`) |

## 9. What is left

**Manual steps (they need an account):**
- File the vLLM issue (the draft is the upstream log #1).
- Open the guidellm issue, then the pull request (the patch is in `docs/upstream/`, the text in
  the upstream log #7–8).
- *Optional:* ask someone for an arXiv endorsement, by mid-October — only if the preprint is to be
  published. The project does not need it (see below).
- Create the GitHub repository, push, and turn on Pages (Settings → Pages → Source: *GitHub Actions*).

**Next on the build side:**
- ✅ done 28 Sep: the preprint draft (`paper/`), and `kvsim calibrate` — so a user can measure their
  own GPU, as H6 says they must — which worked on a real T4 (C1);
- ✅ done 28 Sep: Phase 6's write-up — it held on hit rates and exposed the timing model;
- ✅ done 30 Sep: a timing model that knows about context length (C2), and the finding re-tested
  on it — which changed the finding;
- ✅ done 28 Sep: the AgentX trace (R2 holds), and a contributor guide with three good first issues;
- ✅ done 1 Oct: `kvsim calibrate` now also times prefill after a cached prefix; C3 is registered;
- ✅ done 5 Oct: C3 ran (all three rules hold) and is written up, with what it changes;
- ✅ done 5 Oct: a speed target with both halves (T1 failed, T2 held — findings 24–26);
- ✅ done 5 Oct: `kvsim calibrate` repeats every measurement three times, and `kvsim replay` prints
  how much the GPU count moves between those repeats (tested on the mock engine only so far);
- ✅ done 5–6 Oct: **C4**, one GPU run, registered first (findings 28–30): seven of nine rules
  held;
- ✅ done 5 Oct: the check of other people's work (finding 27), and the README corrected;
- ⭐ **next — R3, registered first, no GPU needed.** The two routing policies from the papers
  (LMetric and SMetric's rule) are now built into the simulator. R3 runs them, and our own
  policies, on a public trace from Alibaba's Qwen service: real sessions of very different sizes —
  the case our generated traffic cannot show;
- ✅ done 6 Oct: the two older real traces (R1, R2) re-run on the checked timing model
  (finding 31);
- then: make the target with both speeds the default (its check has passed);
- **your decision:** H6 was tested on the first timing model only. I recommend not re-running it:
  what followed from it ("calibrate your own GPU") is built, and C3 and C4 measured the same thing
  more directly;
- then: a warning in the tool when a workload is at the edge of a GPU (finding 28 says a number
  there cannot be trusted), and a few shared-prompt cells in `kvsim calibrate`;
- later: the PyPI release.

**The preprint (`paper/`) is optional.** It is a 7-page draft that tells the same story as the
README. The project is complete without it: the repo, the README, the demo page and the calibration
reports are the deliverable. Publishing it on arXiv only helps for research-heavy teams, and the
plan itself calls that "optional upside". If it is not published, it can stay as a draft, or its
text can become a blog post.

**Known limits, owned rather than hidden:**
- the timing model is the weak part. The first one had no term for context length, and on two
  live GPUs with ~1,700-token prompts it was 5–222× optimistic (Phase 6). The context-aware one
  (C2) predicts held-out traffic within 1.41× — but it still cannot express the slow
  single-sequence decode, and close to saturation a few percent of error in it moves the slow end
  (p95) by a lot;
- H6 was run on the first timing model only. The AgentX result (R2) is also from the first model:
  on the checked one its test cannot run at the registered scale (finding 31);
- a GPU count is good to about ± one GPU: three calibrations of the same T4 agree exactly on 71%
  of counts and within one on 96%;
- every GPU count in the README is for the first token only; with a per-token limit too, the
  simulator needs about twice as many. Its per-token speed is now checked where that answer lives
  (C4: within 7%, on one GPU) and still wrong at the limit of a GPU (T1); a whole fleet has not
  been measured;
- near the limit of a GPU, one fit is not enough — the tool now shows the spread between its own
  repeats, but that is a floor, and a low one: another run falls outside it in about one case of
  six (finding 29);
- the first-word speed was 23–30% slower on the real GPU than the model said, on light traffic
  (C4) — inside the factor of 2 we had tested for, but always on the same side;
- every replay is *open-loop* — real agents wait for an answer before sending the next turn, and would
  then reuse the answer's cache blocks, which we cannot measure this way;
- one GPU type (T4) and one model (Qwen3-4B) so far;
- every generated session has the same number of turns, so the known weakness of "keep each user
  on one GPU" — a few very long sessions overloading one GPU — cannot appear in our grids; and the
  two routing policies from the papers (LMetric, SMetric) have not been run in our simulator;
- two real trace families, both scaled down to fit a T4: Mooncake (32× smaller, sessions
  inferred) and AgentX (coding-agent sessions, token counts 64× smaller).

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
