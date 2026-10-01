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
