# 12 — Story notes

Short pieces written to be **told**: in a talk or a post. Each one is in plain words,
and each number in it has its proof linked at the end. The long history is in
[`11-where-we-are.md`](11-where-we-are.md); the dated record of every test is in
[`09-hypotheses.md`](09-hypotheses.md).

---

## The day the finding changed (2026-09-30)

### C2: why latency was wrong

| Rule | Result |
|---|---|
| **C2a**: decode gets slower as the context grows | **true**: 1.92× slower at 1,536 tokens than at 128 |
| **C2b**: a busy second GPU slows the first | **false** (as predicted): no change at all (1.00×) |
| **C2c**: the new model predicts traffic it never saw, within 2× | **true**: within 1.10–1.41×; the old model was up to 7× optimistic |

The new model also explains Phase 6's two live GPUs, within 1.02–1.79× where the old one was 5–222×
off. That check was done after the fact, and it is labelled so.

### The re-test: the finding changed

I registered a re-test of the finding on the new model, predicting it would hold. **It failed.** With
64 prompts, prefix routing no longer needs fewer GPUs than session-sticky: they tie at low load, and
at 2 sessions/s prefix routing needs 11 T4s against sticky's 10.

- **New headline: "Keep each user on one GPU."** Session-sticky needs the fewest GPUs, or ties, at 22
  of 24 points.
- **Why:** prefix routing still gets the better hit rate (0.87 against 0.80). But now every policy
  pays the full decode cost of the long context, and crowding sessions onto a few GPUs makes the load
  uneven.
- **GPU counts roughly double.** One prompt at 2 sessions/s needs 5 T4s, not 2.

The old finding stays in the log with the date and the reason it changed.

### Where the proof is

| Claim | Proof |
|---|---|
| C2a, C2b, C2c | [`c2_kaggle_t4/report.md`](../artifacts/results/c2_kaggle_t4/report.md) — rules in [`configs/c2_timing.yaml`](../configs/c2_timing.yaml), committed before the run |
| The new model explains Phase 6's two live GPUs (after the fact) | [`c2_kaggle_t4/posthoc_phase6.md`](../artifacts/results/c2_kaggle_t4/posthoc_phase6.md) |
| The old model was 5–222× off on two live GPUs | [`phase6_kaggle_t4/report.md`](../artifacts/results/phase6_kaggle_t4/report.md) |
| The re-test failed; 22 of 24 points; 11 against 10; 5 T4s, not 2 | [`h2b_c2/report.md`](../artifacts/results/h2b_c2/report.md) — rules and the prediction "holds" in [`configs/h2b_c2.yaml`](../configs/h2b_c2.yaml), committed before the run |
| The old finding, kept with its date | [`09-hypotheses.md`](09-hypotheses.md), entries of 2026-09-28 and 2026-09-30 |

⚠️ **Tell this one together with the next.** Five days later the same test was run on two more
calibrations of the same GPU, and it gave the other answer. The story above is true as told — it is
what happened on 30 September — but it is no longer the end of the story.

---

## Five days later: we measured the ruler (2026-10-05)

### C3: the tool that measures a user's own GPU, on a real GPU

| Rule | Result |
|---|---|
| **C3a**: `kvsim calibrate` runs on a real T4 and its cache check is exact | **true**: 48 of 48 requests, in 160 seconds |
| **C3b**: the profile it writes predicts traffic it never saw, within 2× | **true**: within 1.41×, on three kinds of traffic |
| **C3c**: the older profile from C2 predicts a new day, within 2× | **true**: within 1.38×, from numbers written down four days before |

All three were predicted, and all three came true.

### What the same run showed about the finding

I also re-ran the finding's test on the profile the tool wrote — a second calibration of the same
GPU, a few percent different. **The test gave the opposite answer.** With 64 prompts at 2 sessions/s
it read 12 T4s for session-sticky against 11 for prefix routing, where the first calibration had
read 10 against 11.

- So I put three calibrations of the same T4 side by side. Of 144 GPU counts, **102 are the same
  on all three, and 138 are within one.** The tool is good to about ± one GPU.
- The test that changed the headline on 30 September had been decided **by one GPU, at one point**.
  That is inside the tool's own error.
- **What survives all three:** keep each user on one GPU. It needs the fewest GPUs, or ties, at
  every load with one prompt and with four, and it is never worse than round-robin or least-loaded.
- **What does not survive:** which of session-sticky and prefix routing wins with 64 prompts. That
  is now written as "not settled".

The lesson: a result is only as sharp as the ruler. I had tested the model's predictions, but I had
not measured how much the *answer* moves when the same GPU is measured again. Now that number is in
the README.

### Three smaller things from the same day

- **The GPU box changed under us.** Kaggle updated its software and the engine would not start. The
  notebook now checks the software in its first minute and repairs it. The repair was tested on
  Kaggle's own image with a small probe before the real run.
- **A gap found before spending a GPU.** A check on old data showed the tool was missing one
  measurement (prefill after a cached prefix). Without it, one prediction would have been 5.7–7.2×
  wrong. It was added before the run.
- **Two limits written down, not fixed.** Near the limit of a GPU, one run of the tool is not
  enough (a second run was 4.2× off on one number). And the speed target covers only the first
  token: the engine took 134–521 ms per token, and the rule the target came from allows 100.

### The same evening: the second speed

Until that day the project measured one speed: how long until the *first* word of an answer. The
live engine had shown the other one — how fast the rest of the words come — at 134 to 521 ms per
token, where the rule we cite allows 100. So I registered two tests and ran them without a GPU.

- **T1: is the simulator right about the second speed? I predicted yes. It failed.** Within 7% on
  three real runs; 1.5–1.7× too slow on the two where the GPU was at its limit.
- **T2: with a target for both speeds, what happens to the finding? I predicted three things, and
  all three came true, on all three calibrations.** The GPU counts about double (1.9–2.0×). And
  "keep each user on one GPU" still wins.
- **How I report a result that rests on a failed check:** the registration said, before either
  test ran, that if T1 failed, no GPU count from T2 would be offered as sizing advice. So the
  README gives the factor, says it is the model's answer and not a checked one, and prints no
  table of those counts.
- **And my own slip:** I edited code while the long run was going. It could not have changed the
  result, but two result files say "the folder was not clean". I did not argue; I re-ran 720
  random cells from a clean copy. All 720 matched. The rule is now written in the contributor
  guide.

### Where the proof is

| Claim | Proof |
|---|---|
| T1 failed: within 7% on three runs, 1.54× and 1.71× too slow on two | [`t1_itl/report.md`](../artifacts/results/t1_itl/report.md) — rules and the prediction "holds" in [`configs/tpot_slo.yaml`](../configs/tpot_slo.yaml), committed before the run |
| T2: three rules hold on three calibrations; 1.93–2.00× | [`t2_two_part_slo/report.md`](../artifacts/results/t2_two_part_slo/report.md) — same config |
| "If T1 fails, no count from T2 goes into the README as guidance" — written before the run | [`09-hypotheses.md`](09-hypotheses.md), the registration entry of 2026-10-05 |
| 720 of 720 cells match from a clean copy | [`t2_two_part_slo/verification.json`](../artifacts/results/t2_two_part_slo/verification.json) |
| C3a, C3b, C3c; the second run 4.2× off on one number | [`c3_kaggle_t4/report.md`](../artifacts/results/c3_kaggle_t4/report.md) — rules and predictions in [`configs/c3_calibrate.yaml`](../configs/c3_calibrate.yaml), committed before the run |
| The test gives the opposite answer on a second calibration (122 of 144 counts equal, 141 within one) | the last section of the same report — a reported item listed in the config before the run |
| Three calibrations: 102 of 144 equal, 138 within one; what survives all three | [`c3_kaggle_t4/posthoc_three_calibrations.md`](../artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.md) — done after the fact, and labelled so |
| The first attempt did not start; the probe that checked the repair | [`c3_env_failure/`](../artifacts/results/c3_env_failure/) |
| The missing measurement, found on old data | [`c3_prereg/precheck.md`](../artifacts/results/c3_prereg/precheck.md) — after the fact, and labelled so |
| 134–521 ms per token; the rule allows 100 | [`09-hypotheses.md`](09-hypotheses.md), entry of 2026-10-05 |

---

## The same day: someone had been here first (2026-10-05)

### What I found

While the GPU run was going, I did the monthly check of other people's work. It was the fourth
check. The first three had missed the two things that mattered most.

- **A bigger tool already existed.** NVIDIA's Dynamo project has had a simulator for this exact
  question since March 2026 (today it is called AISimulate). It replays a saved trace on pretend
  GPUs, compares round-robin with NVIDIA's cache-aware router, and recommends a number of GPUs. Its
  tutorial uses the same public trace I used.
- **My finding was already in a paper.** "Keep each user's session on one machine" was published in
  July 2026 (SMetric), from real traffic on 32 real GPUs. An earlier paper from the same group
  (LMetric, OSDI '26) explains why a hand-picked weight between "has the prompt" and "is not busy"
  never has one right value. I had seen the same thing with my own weights.

### Why three checks missed it

- I searched GitHub for *projects*, by name. Findings are in *papers*, and I never searched papers.
- I had put Dynamo on a list of "systems that route" and never opened its documentation. Its
  simulator was six months old.
- The fourth check found it by following one link at the bottom of a competitor's README.

### What I did about it

- **Withdrew the claim the same day.** The README said "an intersection nobody occupies". It now
  says the finding is not new, names the papers and the tool, and says what they have that this
  project does not.
- **Kept the old sentences.** They are struck through in the docs, with the date, so anyone can see
  what was believed and when.
- **Wrote down what is still mine.** Three things: a cache model that matched a real engine
  exactly; tests written down before they ran, with the failures published; and a measurement of
  how far a simulator's GPU count can be trusted.
- **Wrote down what the paper saw and I could not.** Real sessions differ a lot in size, and then
  my winning policy leaves the GPUs unevenly loaded. All my generated sessions were the same size.
- **Added three rules to the check,** so the next one starts from the papers.

### Where the proof is

| Claim | Proof |
|---|---|
| What was missed, what was read, and what was not read | [`04-prior-art.md`](04-prior-art.md) §2a, and the log row of 2026-10-05 in §7 |
| The withdrawn claim, struck through | [`04-prior-art.md`](04-prior-art.md) §5 · [`01-why-this-project.md`](01-why-this-project.md) §4 |
| What it changes for each result | [`09-hypotheses.md`](09-hypotheses.md), the entry of 2026-10-05 on the prior-art re-audit |
| SMetric: follow-ups stay on one instance; routing by session ID alone left the load 2.17× out of balance | arXiv 2607.08565 |
| LMetric: no weight to tune; *"a statically tuned weight cannot always achieve competitive performance"* | arXiv 2603.15202 (OSDI '26) |
| Dynamo's offline trace replay since March 2026 | `ai-dynamo/dynamo`, pull requests #7543 and #7553 |
