# kv-routing-sim

**A calibrated instrument, plus a replication.**

kv-routing-sim is a trace-driven simulator for LLM serving on several replicas. It models the
prefix cache of each replica and the routing of requests between them. The cache model was checked
against the counters of a live vLLM, and the error is published here. The routing result is not
new: others published it first, and this project only repeats it. Every test was written down
before it ran. The tests that failed are in the table below, together with the others.

The simulator has also replayed a public trace of 739 coding-agent sessions end to end, at its
recorded size (53,967 requests, 7.8 billion prompt tokens). That run covers cache accounting and
routing only. It has no timing.

[Run it on your traffic](https://gitgod-debug.github.io/kv-routing-sim/run.html) in the browser.
There is nothing to install, and the log you choose is never uploaded.

## What was checked, and how far off it is

| | |
|---|---|
| **Checked against** | A live vLLM 0.30.0 (`--enable-prefix-caching`) serving `Qwen/Qwen3-4B` on 2× Tesla T4. The check uses the engine's own counters, `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`, and the cached tokens of each request |
| **Cache model** | Exact on one trace of 120 requests sent one at a time, once with 1-token answers and once with 64-token answers, with the cache under eviction pressure. Every counter and every request matched, in three identical runs of each. Exact again on vLLM 0.34.0, with the same test (V1). On two live replicas, hit rates within 1.4% under three routing policies |
| **Timing model** | p95 TTFT within 1.41× on traffic it never saw. p95 time per output token within 7% around a 100 ms limit. ⛔ It fails at the knee of a GPU's load curve: there the mean time per token is 1.5–1.7× off, and a second calibration put one p95 TTFT 4.17× off |
| **Routing result** | On a T4, among the standard policies: keep each session on one GPU. This is not a new result. SMetric (2026) reached it on production traces, and LMetric (OSDI '26) had shown why. This project agrees with them, independently ([the finding](#the-finding)). It also measured how far a GPU count from a simulator can be trusted. Where three calibrations of the same GPU type all give a count (92 of 144 cells), the counts are within one of each other in 89. In 8 more cells, the calibrations disagree on whether the target can be held at all |

## Validity: where these numbers hold

| | What was used | Do not read a number here as |
|---|---|---|
| **Engine and version** | vLLM **0.30.0**. Only the cache model was checked again, on 0.34.0 (the release of 16 November 2026), and it was exact on the same test (V1). Timing, GPU counts and rupees were not checked again | true of "vLLM" in general. For timing and cost, true of any version other than 0.30.0 |
| **GPU** | one Tesla T4 for the exact checks, and two for the two-replica check. Three calibrations of that one GPU type | another GPU's latency, GPU count or rupees |
| **Model** | `Qwen/Qwen3-4B` | another model's numbers. For another model, timing and cache size must be measured again (`kvsim calibrate`) |
| **Replicas** | live: 1 and 2. Simulated: 1 to 16, and to 32 under the two-part target | a measured fleet. Above two replicas, every count comes from the simulator |
| **Block size and cache** | blocks of 16 tokens. 1,537 blocks on a T4, and 512 in the exactness test | valid for another block size or cache size without a new run |
| **Traffic** | for the exactness test, one generated trace of 120 requests (40 sessions of 3 turns, prompts of 383 to 583 tokens), sent one at a time. For the finding, generated agent sessions of 4 turns and about 1,700 prompt tokens. Three public traces, scaled down 16 to 64 times to fit a T4. One of them, AgentX, was also replayed whole at its recorded size, without timing (L1) | production traffic at its real size, sessions of unequal length, or an exact answer for requests that overlap |

## What one agent session costs, by routing policy

Each row uses one shared system prompt and 2 sessions a second, with each policy on the fewest T4s
that meet the target.
A session sends 6,800 prompt tokens in 4 requests.
519 of them are new in this replay. This is text that no earlier request sent, so no routing can
avoid computing it. (180 of the 519 are the model's own earlier replies. The replay is open-loop and
makes them up. A live engine could still have them in its cache from generating them.)
*Re-prefilled* tokens are the tokens the GPUs computed beyond that. A cache shared by all GPUs
would have saved them.

| Target | Policy | T4s | Cache hit rate | Re-prefilled tokens per session (95% interval) | ₹ per session |
|---|---|---:|---:|---:|---:|
| p95 time to first token ≤ 2 s | session-sticky | 5 | 92.0% | 27 (7 to 54) | 0.047 |
| | round-robin | 7 | 82.5% | 669 (627 to 718) | 0.063 |
| | balanced prefix-affinity | 10 | 85.9% | 438 (398 to 483) | 0.091 |
| | least-loaded | 11 | 81.0% | 777 (736 to 824) | 0.098 |
| and p95 time per output token ≤ 100 ms | session-sticky | 10 | 91.5% | 61 (27 to 102) | 0.091 |
| | least-loaded | 13 | 80.5% | 805 (760 to 857) | 0.118 |
| | round-robin | 14 | 80.1% | 834 (785 to 889) | 0.126 |

These numbers are simulated, on calibration A of one T4 (vLLM 0.30.0, Qwen3-4B), with generated
traffic. The intervals are bootstrap intervals over sessions. A policy with no row did not meet the
target at any GPU count of the grid. Session-sticky's 27 tokens are the shared prompt, computed once
on each extra GPU. That is a one-time cost of 4 × 1,360 tokens, not a rate. Per session it would be
half as much in a trace twice as long, and its interval only shows which sessions paid it.
Round-robin's 669 is mostly a rate. Rupees are counted over the whole run of a short trace. A
service that ran all day at this rate would pay about a quarter less. This test was registered
before it ran, as S1: [the report](artifacts/results/session_cost/report.md), made by
`scripts/session_cost.py`.

## Every registered test, and what happened

Each prediction was committed before its test ran. [The log](docs/09-hypotheses.md) has the dated
entries.
Of 21 rows, 11 held (H4 only in an amended form), 4 failed, and 6 held in part or are not settled.

| Test | Predicted | Outcome |
|---|---|---|
| **H1** | prefix-affinity routing has a higher cache hit rate than least-loaded at every replica count | ⛔ failed: true at 1–2 sessions/s, gone or reversed from 3/s |
| **H2** | there is a sharing ratio above which prefix routing wins | ⛔ failed: pure prefix-affinity loses at every sharing ratio above 0 |
| **H2b** | prefix routing pays once tenants' prompts outgrow one GPU's cache | ✅⛔ not settled: held on the first timing model, failed on calibration A of the checked one, held on B and C |
| **H3** | calibration error in single digits (%) on two workload shapes | ✅ held: zero error on two shapes, +0.3% on a third |
| **H4** | the residual of a naive cache model is eviction-order skew | ✅ held, in an amended form: it is capacity skew (−11.9 points) |
| **H5** | the cost win is one fewer replica at a fixed target | ✅⛔ in part: only for balanced prefix-affinity at low sharing |
| **H6** | the ranking of policies does not depend on the hardware | ✅⛔ in part: the ranking moves (τ-b down to 0.51), but the direction of the flip does not. First timing model only |
| **R1** | on the Mooncake trace, prefix-affinity beats session-sticky on tool-agent traffic and never on conversation | ⛔ failed: both directions wrong |
| **R2** | on the AgentX trace, session-sticky and balanced prefix-affinity tie | ✅ held, on the first timing model |
| **R1, R2 again** | the same two on the checked timing model: R1 fails again, R2 holds | ✅⛔ in part: R1 failed again, as predicted. R2 could not be tested as designed |
| **R3** | four rules for the two published policies, ported, on a public production trace | ✅⛔ in part: 2 of 4 held |
| **Phase 6** | hit rates on two live replicas within 10% | ✅ held: within 1.4%, run twice. Latency came out 5–222× off, which led to C2 |
| **C1** | `kvsim calibrate` works on a real GPU, and its cache check is exact | ✅ held: 48 of 48 requests |
| **C2** | three timing predictions. One of them: a busy second GPU does not slow down the first | ✅ held: all three (1.92×, 1.00×, within 1.41×) |
| **C3** | a profile fitted on a real GPU predicts traffic it never saw within 2× | ✅ held: all three (exact, 1.41×, 1.38×) |
| **C4** | nine rules for per-token timing on a real T4 | ✅⛔ in part: 7 of 9 held |
| **T1** | per-token latency within 1.5× on five held-out runs | ⛔ failed: 1.54× and 1.71× at the knee of the load curve. Within 1.07× away from the knee |
| **T2** | with a limit on token speed too, session-sticky is still best or tied | ✅ held, on three calibrations. The GPU counts about double |
| **S1** | three predictions for the session cost table above (the table uses grid cells that had already been read) | ✅ held at the registered cell. One of its rules fails in 4 of 86 cells outside it |
| **V1** | the cache model, unchanged, is exact on vLLM 0.34.0 too, on the same test | ✅ held: every counter and the cached tokens of all 120 requests, three runs each, with 1-token and with 64-token answers |
| **L1** | on the whole AgentX trace at its recorded size: the cache model serves exactly what the raw file allows, round-robin re-prefills at least 4× what session-sticky does, and pure prefix-affinity ties session-sticky | ✅ held: 53,967 of 53,967 requests exact, 35× to 993×, within 0.02 points. No timing |

## If you have five minutes

1. [Every registered test, and what happened](#every-registered-test-and-what-happened): 21 rows. 4 failed, and 6 held only in part.
2. [The calibration report](artifacts/results/phase3_kaggle_t4/calibration_report.md): the prediction against the measurement, request by request, with the residual explained.
3. [The finding, as a replication](#the-finding), next to the work that came first ([`docs/04-prior-art.md`](docs/04-prior-art.md)).
4. [What one agent session costs](artifacts/results/session_cost/report.md) under each routing policy.
5. [What breaks at 10× and 100×](artifacts/results/scale/report.md), as measured.
6. [Ten questions a skeptic should ask](#ten-questions-a-skeptic-should-ask), each with its number and where it does not hold.

<details>
<summary><b>Evidence ledger</b>: for each result above, the file that holds it, the script or notebook that made the file, and the commit</summary>

<!-- ledger:start -->
| Claim | Number | Results file | Made by | Commit |
|---|---|---|---|---|
| Cache model exact with 1-token and with 64-token answers | 3 of 3 runs, on both | [`phase3.json`](artifacts/results/phase3_kaggle_t4/phase3.json) | `notebooks/phase3_calibration.ipynb` | `8efaa7a` |
| The same test on vLLM 0.34.0 (V1) | 3 of 3 runs, on both | [`phase3.json`](artifacts/results/v034_kaggle_t4/phase3.json) | `notebooks/v034_recheck_t4.ipynb` | `9072932` |
| A third, concurrent workload (H3) | +0.3% | [`fitted_profile.json`](artifacts/results/phase3_kaggle_t4/fitted_profile.json) | `scripts/phase3_analyze.py` | `8efaa7a` |
| What a naive cache model misses (H4) | −11.9 points | [`phase3.json`](artifacts/results/phase3_kaggle_t4/phase3.json) | `notebooks/phase3_calibration.ipynb` | `8efaa7a` |
| Hit rates on two live replicas (Phase 6) | within 1.4% | [`analysis.json`](artifacts/results/phase6_kaggle_t4/analysis.json) | `scripts/phase6_analyze.py` | `3a5cd89` |
| Latency on those two replicas, first timing model | 5–222× off | [`analysis.json`](artifacts/results/phase6_kaggle_t4/analysis.json) | `scripts/phase6_analyze.py` | `3a5cd89` |
| The calibration command's cache check (C1) | 48 of 48 requests | [`c1.json`](artifacts/results/c1_kaggle_t4/c1.json) | `notebooks/c1_calibrate_t4.ipynb` | `3a5cd89` |
| Timing: three predictions (C2) | 1.92×, 1.00×, within 1.41× | [`analysis.json`](artifacts/results/c2_kaggle_t4/analysis.json) | `scripts/c2_analyze.py` | `8899fd9` |
| p95 time to first token on traffic the fit never saw | within 1.41× | [`analysis.json`](artifacts/results/c2_kaggle_t4/analysis.json) | `scripts/c2_analyze.py` | `8899fd9` |
| One p95 time to first token at the knee, on a second calibration (C3) | 4.17× off | [`analysis.json`](artifacts/results/c3_kaggle_t4/analysis.json) | `scripts/c3_analyze.py` | `b5f5d6c` |
| A profile fitted on a real GPU, on traffic it never saw; an older profile, another day (C3) | within 1.41× and 1.38× | [`analysis.json`](artifacts/results/c3_kaggle_t4/analysis.json) | `scripts/c3_analyze.py` | `b5f5d6c` |
| p95 time per output token around the limit (C4) | within 7% | [`analysis.json`](artifacts/results/c4_kaggle_t4/analysis.json) | `scripts/c4_analyze.py` | `81012bb` |
| Nine rules for per-token timing (C4) | 7 of 9 held | [`analysis.json`](artifacts/results/c4_kaggle_t4/analysis.json) | `scripts/c4_analyze.py` | `81012bb` |
| One calibration's own spread covers the next one's GPU count (C4, rule E) | 82.4% of cases | [`analysis.json`](artifacts/results/c4_kaggle_t4/analysis.json) | `scripts/c4_analyze.py` | `81012bb` |
| Per-token latency at the knee of the load curve (T1) | 1.54× and 1.71× | [`t1.json`](artifacts/results/t1_itl/t1.json) | `scripts/t1_itl_check.py` | `904765c` |
| Per-token latency off the knee (T1) | within 1.07× | [`t1.json`](artifacts/results/t1_itl/t1.json) | `scripts/t1_itl_check.py` | `904765c` |
| GPU counts under the two-part target (T2) | a median of 1.93–2.00× the first-token counts | [`analysis.json`](artifacts/results/t2_two_part_slo/analysis.json) | `scripts/t2_analyze.py` | `904765c` |
| GPU counts from three calibrations of one GPU type | 92 of 144 cells have a count on all three: 58 equal, 89 within one; 8 more disagree on holding the target | [`posthoc_three_calibrations.json`](artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.json) | `scripts/c3_posthoc_grid.py` | `b5f5d6c` |
| Ranking of policies across hardware (H6) | τ-b down to 0.51 | [`h6.json`](artifacts/results/h6_ranking/h6.json) | `scripts/run_h6.py` | `892b481` |
| Two published policies on a production trace (R3) | 2 of 4 held | [`analysis.json`](artifacts/results/r3_bailian/analysis.json) | `scripts/r3_analyze.py` | `c4109a0` |
| New tokens in a session | 519 of 6,800 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Session-sticky, first-token target | 5 T4s, 27 tokens, ₹0.047 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Round-robin, first-token target | 7 T4s, 669 tokens, ₹0.063 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Balanced prefix-affinity, first-token target | 10 T4s, 438 tokens, ₹0.091 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Least-loaded, first-token target | 11 T4s, 777 tokens, ₹0.098 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Session-sticky, two-part target | 10 T4s, 61 tokens, ₹0.091 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Round-robin, two-part target | 14 T4s, 834 tokens, ₹0.126 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Least-loaded, two-part target | 13 T4s, 805 tokens, ₹0.118 | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| S1's rupee rule outside its registered cell | more than 10% off in 4 of 86 cells | [`session_cost.json`](artifacts/results/session_cost/session_cost.json) | `scripts/session_cost.py` | `a27cbaf` |
| Memory for the block hashes of a 100× trace | 537 MiB for 80,000 requests | [`scale.json`](artifacts/results/scale/scale.json) | `scripts/measure_scale.py` | `f141d73` |
| A public trace replayed whole, at its recorded size (L1) | 739 sessions, 53,967 requests, 7.8 billion prompt tokens | [`l1.json`](artifacts/results/l1_agentx_full/l1.json) | `scripts/run_l1.py` | `73fc093` |
| At that size, one replica that never evicts against the raw file (L1a) | 53,967 of 53,967 requests exact | [`l1.json`](artifacts/results/l1_agentx_full/l1.json) | `scripts/run_l1.py` | `73fc093` |
| Computed again with 1 Mi tokens of cache per replica: round-robin, session-sticky (L1) | 46.64% and 1.32% of prompt tokens | [`l1.json`](artifacts/results/l1_agentx_full/l1.json) | `scripts/run_l1.py` | `73fc093` |
| Engine, block size and cache of the timing profile | vLLM 0.30.0, 16 tokens, 1,537 blocks | [`t4-qwen3-4b-vllm0.30.0-ctx.json`](configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json) | `scripts/c2_analyze.py` | `c0c8b18` |
<!-- ledger:end -->

`python scripts/build_ledger.py --check` reads every number from its file again and compares it
with this table. Settings such as block size, cache size and replica ranges are in the configs that
the validity table uses. Numbers from other people's papers carry their citation.

</details>

[![The in-browser runner: choose traffic, a GPU and a latency target, run it, and read the recommended routing policy and replica count](docs/figures/demo-run.gif)](https://gitgod-debug.github.io/kv-routing-sim/run.html)

## Status and known limits (2026-11-26)

**Known limits.** Replica counts are T4 figures on generated traffic. Another calibration of the
same GPU type usually gives a count within one of them. This is true in 89 of the 92 cells where
all three calibrations give a count. In 8 more cells, they disagree on whether the target can be
held at all. These are also first-token counts. With a limit on token speed too, the simulator
needs about twice as many GPUs. (The command applies that limit by default.) The repeats of one
calibration do not cover the next calibration's count often enough: 82% of cases, where the rule
was 90%. So the tool says "plus one". At the knee of a GPU's load curve no number here is reliable,
and the tool says when a recommendation is there. The grid behind the finding has one generated
trace per point and three seeds of the router.
[`docs/06-methodology.md`](docs/06-methodology.md) asks for five seeds, and a paired interval on
every row that compares a policy with the baseline. The test behind half of the old headline was
decided inside that margin, so that half is "not settled" below.

**Not new.** The routing result was published first (SMetric, LMetric). NVIDIA ships a larger
simulator for the same question. Three more simulators checked against live engines were published
in October and November 2026. See the note under the finding, and the section on related work.

**Upstream.** During calibration, this project found a bug in the trace replay of
`vllm-project/guidellm`: one token is lost per block with tokenizers that add a start token.
Another contributor reported it there, and it is fixed (#1424 and #1425, merged 17 November 2026).
Nothing was filed from this repository. The dated finding is in [`docs/upstream/`](docs/upstream/).

**Demo and docs.** The demo page is live at
[gitgod-debug.github.io/kv-routing-sim](https://gitgod-debug.github.io/kv-routing-sim/), and the
documentation is next to it as [web pages](https://gitgod-debug.github.io/kv-routing-sim/docs/). The
package is on PyPI: `pip install kv-routing-sim` (see Install). Start reading at
[`docs/00-START-HERE.md`](docs/00-START-HERE.md).

## Ten questions a skeptic should ask

| | Question | Answer, with its number | Where it does not hold |
|---|---|---|---|
| 1 | "Exact" on what? | On one trace of 120 requests, sent one at a time to one replica, once with 1-token answers and once with 64-token answers, three identical runs each. Every counter and the cached tokens of every request matched. On a concurrent run: +0.3%. On two live replicas: within 1.4% | Nothing concurrent is exact. One model, one GPU type |
| 2 | vLLM 0.30.0 is four releases old. Is this still true? | For the cache model, yes. The same exactness test ran on vLLM 0.34.0 on 2026-11-25, registered first (V1). It was exact with 1-token and with 64-token answers, three runs each | Timing was not checked again. Latency, GPU counts and rupees are 0.30.0 results, and the two-replica check was not repeated |
| 3 | The timing model was once 5–222× off. Why trust it now? | That model had no term for context length. The model used now predicted unseen traffic within 1.41×, and time per output token within 7% around the limit | At the knee of a GPU's load curve it is 1.54× and 1.71× off (T1 failed). The tool warns when an answer is at the knee |
| 4 | How good is a GPU count from this? | Where three calibrations all give a count (92 of 144 cells), they are equal in 58 and within one in 89 | In 3 cells they are two or three apart. In 8 more they disagree on whether the target can be held at all, once 10 GPUs against "not on 16". The spread of one calibration covers the next one's count in 82.4% of cases, where the rule was 90% (C4). All of this compares calibrations with each other. No count above two replicas has been checked on a live fleet |
| 5 | Is the routing result new? | No. SMetric and LMetric published it first. This project agrees with them, independently, on a small calibrated model | Of four rules for those two policies, ported and run on a production trace, 2 held (R3) |
| 6 | Does session-sticky always win? | With one shared prompt or with four, on all three calibrations, it needs the fewest GPUs or ties | With 64 tenant prompts it is not settled (H2b). Sessions of unequal length are not tested here. SMetric measured a 2.17× imbalance |
| 7 | Rupees per what, and over what time? | Per 1,000 requests that met the target, for a T4 at $0.526 an hour and ₹95.87 a dollar. Both are cited and dated in `configs/cost_model.yaml` | Counted over the whole run of a short trace. A service that ran all day would pay about a quarter less |
| 8 | Real traffic? | Three public traces, each test registered before it ran. One of them (AgentX, 739 coding-agent sessions) was also replayed whole, at its recorded size. The cache model served exactly what the raw file allows. With 1 Mi tokens of cache per replica, round-robin computed 46.64% of the prompt tokens again, and session-sticky 1.32% (L1) | The time-based tests ran on traces scaled down 16 to 64 times. R1 failed twice, R2 could not be tested on the checked timing model, and R3 held in 2 of 4 rules. The full-size replay has no timing, because no GPU here was timed on prompts that long |
| 9 | What breaks first at 100×? | Memory for block hashes: 537 MiB for 80,000 requests, before the simulation starts | Whether many real replicas stay matched cannot be answered on a CPU. Two live replicas are the most that was measured |
| 10 | Can I rebuild your tables myself? | `python scripts/reproduce.py`: thirteen steps, about 20 minutes on a laptop. Every file a step writes is compared with the committed one | It runs the analysis of each GPU run again, on the raw files stored here. It cannot repeat a GPU measurement. Not covered yet: the first two sweeps, H6, the scale test, the real-trace runs, and the stored grids that some analyses start from |

---

## The finding

> **Keep each user on one GPU.** This holds on three calibrations of the same GPU type (a T4).
> Session-sticky routing needs the fewest GPUs of the six policies in the registered grid, or ties
> for the fewest, at every load, with one shared system prompt and with four. It never needs more
> GPUs than round-robin or least-loaded. It is not settled whether routing by prefix saves GPUs
> once the tenants' prompts outgrow the cache. The registered test said no on one calibration and
> yes on the two others. Read every GPU count here as the answer of one calibration. Another
> calibration usually gives a count within one of it.

**This is not a new result.** A fourth check of prior work found this on 2026-10-05, after three
earlier checks had missed it ([`docs/04-prior-art.md`](docs/04-prior-art.md), §2a). SMetric (Wang
et al., arXiv 2607.08565, July 2026) reached the same advice from production traces on 32 real
GPUs: send the follow-up requests of a session to the instance that served it. LMetric (Zhang et
al., OSDI '26) had shown before that a hand-weighted mix of a cache score and a load score has no
right weight. What this repository adds is an independent agreement on a small calibrated model,
from tests registered before they ran.

Both published policies have since been run here, as ports, on a public production trace (R3,
registered first). Two of its four rules held, and two of my predictions failed. Under the
first-token target, neither policy needs more GPUs than session-sticky at any load. LMetric ties it
on three calibrations of four, and needs one fewer on the fourth. SMetric's rule needs one fewer at
the highest load on all four (6 against 7). With a limit on token speed as well, both need two to
three GPUs fewer at the highest load. That is a reported number, not a tested rule. The ports were
written for this repository, so none of this is a result about the two systems themselves.

SMetric also names a weakness of session-sticky routing. Real sessions differ a lot in size, and
then the load goes out of balance: 2.17× on its testbed. This did not show on that trace (1.14× at
most, because its sessions are short), and every generated session here has the same number of
turns. So this weakness is still not tested in this repository.

Since then, others have measured both sides (found by the check of 2026-11-25). On SGLang, session
affinity cut the uncached input by 48.2% and 46.1% compared with round-robin (arXiv 2610.15366).
Vesta (arXiv 2610.14686) measured the cost of the pure form. Sending every turn to the instance
that holds its cache gave a mean time to first token of 2,819 ms, where SMetric gave 2,060 ms. At
higher load it was 5,457 ms against 3,329 ms.

How it was measured: simulated agent traffic, the cache model calibrated against a live vLLM (see
the calibration report below), a context-aware T4 timing model, and a 2-second p95 TTFT target. The
test was registered before it ran, on calibration A, which is C2's fit
([`docs/09-hypotheses.md`](docs/09-hypotheses.md), 2026-09-30). B and C are the two profiles that
`kvsim calibrate` wrote on a real T4 in C3 (2026-10-05). B's table was a registered reported item.
C's table was added after seeing it, so it is post hoc
([all three, side by side](artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.md)).

| Distinct system prompts (85 blocks each) | Working set ÷ one T4's cache | Fewest T4s that hold the target at 2 sessions/s, on A · B · C |
|---|---:|---|
| 1 | 0.06 | session-sticky **5 · 5 · 5** · round-robin 7 · 7 · 7 · balanced prefix-affinity 10 · not held on 16 · not held |
| 64 offered, 48 used | 2.66 | session-sticky 10 · 12 · 12 · balanced prefix-affinity 11 · 11 · 10 · round-robin 16 · 16 · not held · least-loaded: not held *(prefix-affinity with a load weight of 1: **8 · 8 · 8**, a weight found after the tests)* |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/flip-dark.svg">
  <img src="docs/figures/flip-light.svg" width="100%" alt="Left: T4s needed at 2 sessions/s as distinct system prompts grow from 1 to 64, on the context-aware timing model. Session-sticky needs 5 with one prompt and 10 with 64; balanced prefix-affinity needs 10 and 11; round-robin 7 and 16. Right: hit rate on 8 T4s. Round-robin falls from 0.81 to 0.47, session-sticky from 0.92 to 0.80, prefix-affinity with load weight 1 stays near 0.87.">
</picture>

*Drawn by [`scripts/make_figures.py`](scripts/make_figures.py) from
[`demo_grid_ctx.json`](artifacts/results/demo_grid_ctx.json), which reproduces the table of the
registered re-test exactly. There is one generated trace per point and three seeds of the router,
and a point counts only if every seed holds the target. T4 figures, on calibration A only. On B
and C the 64-prompt points move by up to two replicas, as the table above shows.*

**Scope:** generated traffic, one GPU type (T4), one model (Qwen3-4B). The replica counts are T4
figures on the context-aware model, and another calibration usually gives a count within one
replica of them. They are first-token counts. With a limit on the time per output token as well,
the simulator needs about twice as many GPUs (T2).

The details are in [`docs/13-the-finding-in-detail.md`](docs/13-the-finding-in-detail.md): what
holds on all three calibrations and what does not, how this finding changed and why, the live check
on two T4s, the three real traces, and the three tests of the per-token half of the target (T1, T2,
C4).

**Try it.** The [demo page](https://gitgod-debug.github.io/kv-routing-sim/) ([`site/`](site/index.html))
looks up every point of the re-test's grid: prompts × load (0.25–4 sessions/s) × replicas (1–16) ×
six policies, 6,912 simulator runs. [`scripts/build_grid.py`](scripts/build_grid.py) rebuilds the
grid, and accepts it only if it reproduces the registered table exactly. To open the page from a
clone, run `python -m http.server` in the repository root, then open `http://localhost:8000/site/`.

## Calibration report: the simulator against a live vLLM

This is a separate kind of claim, so it has its own section. *"The simulator matches the live
engine within X%"* shows that the method is careful. It says nothing about the world. The finding
above says something about the world. The two are kept apart here.

| | |
|---|---|
| **Engine and version** | vLLM 0.30.0, V1 engine, async scheduling on (its default) |
| **Exact command** | `vllm serve Qwen/Qwen3-4B --dtype half --block-size 16 --num-gpu-blocks-override 512 --max-model-len 2048 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --kv-cache-metrics --kv-cache-metrics-sample 1.0 --enable-prompt-tokens-details --seed 0` |
| **Model and quantization** | `Qwen/Qwen3-4B`, fp16, not quantized. The cache is fixed at 512 blocks (511 usable) by an override, so its capacity is a controlled input and the cache overflows |
| **Hardware** | Kaggle 2x Tesla T4 (compute 7.5, driver 580.159.04, CUDA 12.8). vLLM runs TRITON_ATTN in fp16 here. For requests sent one at a time, the cache counts do not depend on the hardware. Under concurrency they move with timing (the same workload gave 0.810 and 0.822 on two runs of Phase 6). The order of the policies changes with the hardware (H6), and every latency and cost figure is a T4 figure |
| **Hit-rate error against the live engine** | Zero on 2 sequential shapes × 3 runs: every counter, and the cached tokens of each of the 120 requests, to the token. +0.3% (relative) on the concurrent shape, predicted by a timing profile fitted without it. Plain LRU on the same runs over-predicts by 11.9 points (W1) and 19.7% (W3) |
| **The residual, explained** | Sequential: nothing to explain. The engine's `kv_block_*` histograms: the counts are exact (963 and 1,880 events), and the distributions are identical bucket for bucket on W2. W1's small residual (max CDF distance 0.042) was traced to the timing fit, not the cache model. That analysis is post hoc and labelled so |
| **The same test on a later engine** | vLLM 0.34.0, 2026-11-25 (V1, registered first): the same notebook, with only the engine version changed. Zero error again on both sequential shapes × 3 runs, and +0.7% on the concurrent shape ([report](artifacts/results/v034_kaggle_t4/calibration_report.md)). Timing was fitted on it but not checked |
| **Check on 2 replicas** | Round-robin across two physical T4s: 1,088 prefill tokens computed, 1,088 predicted (Phase 0 v2). Under cache pressure (Phase 6, run twice), the hit rates of all 12 runs were within 1.4% (K1 to four decimals), with 1,537 blocks per replica and three routers. The engines preempted requests in 8 of the 10 runs where the model predicted none |
| **Timing under load** | The first profile had no context term and was optimistic. On Phase 6's prompts of ~1,700 tokens, p95 TTFT came out 5–222× its prediction. ✅ Fixed by C2 (2026-09-30, registered): decode cost grows with context (1.92× from 128 to 1,536 tokens), and CPU contention was ruled out (1.00×). The context-aware profile `t4-qwen3-4b-vllm0.30.0-ctx`, fitted on micro-benchmarks, predicted held-out traffic within 1.10–1.41×. Post hoc, it also predicted Phase 6's two live replicas within 1.02–1.79× ([report](artifacts/results/c2_kaggle_t4/report.md)) |
| **Reproduce** | The analysis: `python scripts/reproduce.py --quick` rebuilds this report from the raw files that the GPU run left in this repository, and compares it with the committed one (seconds, no GPU). The measurement: [`notebooks/phase3_calibration.ipynb`](notebooks/phase3_calibration.ipynb) on a Kaggle T4, about 40 minutes ([how](notebooks/README.md)) |

**`/metrics` fields used, by name:** `vllm:prefix_cache_queries` · `vllm:prefix_cache_hits` ·
`vllm:prompt_tokens_by_source{source="local_compute"|"local_cache_hit"}` · `vllm:num_preemptions` ·
`vllm:kv_block_lifetime_seconds` · `vllm:kv_block_idle_before_evict_seconds` ·
`vllm:kv_block_reuse_gap_seconds` (`_count` and `_bucket`) · `vllm:prompt_tokens` · `vllm:generation_tokens` ·
`vllm:num_requests_running` · `vllm:num_requests_waiting`. Also, for each request, the response's
`usage.prompt_tokens_details.cached_tokens`. TTFT is measured at the client, over streamed responses.

### What was predicted, and what the engine did

The predictions were committed before the run
([`predictions.json`](artifacts/results/phase3_prereg/predictions.json)) and measured on the engine
above ([`calibration_report.md`](artifacts/results/phase3_kaggle_t4/calibration_report.md)). The
trace has 120 agent requests (40 sessions × 3 turns, one shared system prefix), and each workload
ran three times.

| Workload | Engine (all 3 runs) | Block-pool model (default) | Plain-LRU model (ablation) |
|---|---|---|---|
| **W2** one at a time, 1-token outputs | hit rate **0.6090** · 22,664 computed | **0.6090** · 22,664, exact | 0.6073 · 22,760 |
| **W1** one at a time, 64-token outputs | hit rate **0.4883** · 29,656 computed | **0.4883** · 29,656, exact | 0.6073 · 22,760 |
| **W3** the same trace, concurrent | hit **0.5076** · **27** preemptions · p95 TTFT **41.0 s** | 0.509 (+0.3%) · 24 · 38.3 s (−6.7%) | 0.6073 (+19.7%) · 0 · 11.2 s (−73%) |

W1 is the important case. The two models agree when requests hold no decode blocks (W2). They
differ by 11.9 points when requests do hold them. vLLM caches full decode blocks, these blocks take
space in the same small cache, and they evict useful prefixes. The block-pool model includes this,
and LRU cannot see it. On a real engine, the block-pool model was right to the token.

**The fitted T4 timing** (all of it a T4 figure): prefill 0.70 ms per uncached token, and decode
52.9 ms + 1.23 ms × batch (R² 0.58). It has two limits. They were measured and recorded, not tuned
away ([`posthoc.md`](artifacts/results/phase3_kaggle_t4/posthoc.md)). First, one sequence decodes
slower than two to eight sequences do (63.6 ms per token), and a straight line cannot express
that. Second, the simulator's step model charges a full decode step on top of every prefill, so it
over-estimates the TTFT of an idle replica by ~20 ms.

> **The hit rate is derived from two counters:** `vllm:prefix_cache_hits` ÷
> `vllm:prefix_cache_queries`. There is no `vllm:prefix_cache_hit_rate` metric (checked against
> `v1/metrics/loggers.py` at v0.29.0). The old V0 gauge was removed together with V0.

---

## What it does

1. **Trace input.** It reads serving logs (vLLM-style JSONL, OpenAI-style request logs) and three
   public trace formats (Mooncake, AgentX, Qwen-Bailian) into one normalized, versioned request
   schema. Prompts are hashed at the adapter boundary and never enter the simulator. So the user
   content of a trace stays out of memory and out of the output files.
2. **A block-level KV cache model** for each replica. The default model follows the block pool of
   vLLM 0.30.0 line by line: reference counts, freeing from the tail, cached decode blocks,
   admission by free memory, and preemption. A plain LRU model is kept next to it for comparison.
   Many simulators have such a model. What this project adds is a measurement of how close it is to
   the real engine.
3. **Nine routing policies.** Seven are standard: round-robin, least-loaded,
   power-of-two-choices, consistent-hashing-on-prefix, radix-prefix-affinity,
   radix-prefix-affinity-balanced (affinity with a load term, as production routers do it) and
   session-sticky. Two are ports of published policies, and you must ask for them by name:
   `lmetric` (LMetric, OSDI '26: new prefill tokens × batch size, with no weight to tune) and
   `smetric-local` (SMetric's session rule, arXiv 2607.08565, without its global cache tier). The
   ports were written for this repository, so a result from them is not a result about the
   authors' systems. Each policy is one file, and a new policy is a new file. `--policies all`
   means the first seven, as in every registered sweep.
4. **Sweep and recommendation.** It runs policy × replica count (1–8) and draws Pareto curves over
   hit rate, p95 TTFT and ₹ per 1k successful requests. Then it recommends one configuration and
   says how much better it is than the naive baseline.
5. **A check against live replicas.** Three fixed routing policies ran on two live vLLM replicas,
   and their hit rates were compared with the predictions (Phase 6). A recommended configuration
   has not been deployed.

### A generator for agent traffic

An agent sends the system prompt, the tool definitions and the whole conversation again in every
turn. So in agent traffic, most of each prompt is shared text. The generator makes traffic of this
shape, and the prefix-sharing ratio is one of its parameters.

---

## Install

**In a browser, with nothing installed:** [run it on your traffic](https://gitgod-debug.github.io/kv-routing-sim/run.html).
The simulator runs inside the page with Pyodide. A log you choose is read on your computer and is
never uploaded. The page shows the `kvsim` commands that give the same answer.

**On your machine:**

```bash
pip install kv-routing-sim   # CPU-only, no GPU dependency
kvsim generate --out trace.jsonl --sessions 100 --turns 4 --rate 0.5   # or use your own log
kvsim replay trace.jsonl --policies all --replicas 1-8 \
  --profile t4-qwen3-4b-vllm0.30.0-ctx --cost-model default
```

That takes about half a minute and ends with a recommendation for a T4.

For the newest code on GitHub instead of the last release, install with
`pip install git+https://github.com/gitgod-debug/kv-routing-sim`.

Set your own target and GPU price with `--ttft-slo-ms`, `--tpot-slo-ms` and `--gpu-price-usd`.

It runs on your laptop, and your trace never leaves your machine. Prompts are hashed at the adapter
boundary, so prompt text never enters the simulator. One caution: a log that has text but no token
ids (`--format-in openai-log`) is hashed with about four characters counted as one token. The block
boundaries then differ from the engine's, and this biases the hit rate. If you can, use a log with
token ids or block hashes.

The profile in that command was measured for this project, on a Tesla T4 with Qwen3-4B and vLLM
0.30.0. It ships with the package, and so does the cited cost model. Without `--profile`, the
timings are a built-in guess. Use the context-aware profile. It is the one checked against live
latency (C2), with prompts up to about 2,000 tokens. The first profile, `t4-qwen3-4b-vllm0.30.0`,
has no context term. It is kept for the record and is optimistic on long prompts.

`kvsim profiles` lists what each profile was measured on and checked against. Give it `--gpu`,
`--model`, `--engine-version`, `--block-size` and `--max-num-seqs` (or `--engine` and a running
vLLM), and it says which of these five differ from your setup and what each difference means.
Nothing is interpolated between profiles. A replay also tells you how many of your prompts are
longer than any prompt its profile was timed on. (`--profile` also takes a path to any profile
file, and `--cost-model` a path to your own YAML. The shipped files are byte-for-byte copies of
`configs/profiles/` and `configs/cost_model.yaml`.)

**The target has two halves by default:** p95 time to first token ≤ 2 s, from the cost model, and
p95 time per output token ≤ 100 ms, from the command (`--tpot-slo-ms`, and `0` turns that half
off). On a T4 the second half is the hard one. Expect about twice the GPUs of the first-token
tables above, and "no configuration met the SLO" on a small fleet.

To get numbers for your own setup, point `kvsim calibrate` at your running vLLM (with prefix
caching on). It fits C2's context-aware timing model to your GPU and model. It times decode steps
at several context lengths and batch sizes, and prefill at several prompt lengths and after a
cached prefix. It reads the KV capacity from the engine, and it checks the cache model against the
engine's own counters. This takes a few minutes and needs nothing beyond the core install:

```bash
kvsim calibrate --engine http://localhost:8000 --out mine.json
kvsim replay trace.jsonl --policies all --replicas 1-8 --profile mine.json --cost-model default
```

If the cache check misses (by more than 5%, or on the cached tokens of any single request), the
profile is written but not marked as calibrated, and the command exits with code 1.

**Tested on a real T4 twice, and both tests were registered before they ran.**
- **C1:** its cache check was exact for the cached tokens of all 48 requests (9,840 of 13,776
  tokens). It read the engine's 1,537 blocks correctly, in 48 seconds
  ([report](artifacts/results/c1_kaggle_t4/report.md)). That version fitted a straight line with
  no context term, and C2 then showed that such a line is up to 7× optimistic.
- **C3 (2026-10-05):** the command as it is now. It fits C2's context-aware model. It also times
  new tokens after a cached prefix, because agent traffic is mostly made of that case. (A check on
  C2's recorded data had shown that timing only cold prompts would have put a held-out p95 off by
  5.7–7.2×. That check is post hoc: [pre-check](artifacts/results/c3_prereg/precheck.md).) On the
  T4 it ran in 160 seconds and was exact on the cache check again. It wrote a profile that
  predicted three workloads it had never seen within 1.41× on p50 and p95 TTFT. One of those
  workloads had prompts half as long as anything the model had been tested on. The profile that
  C2 fitted five days earlier, on another software image, predicted the same traffic within
  1.38×, from numbers written down before the run
  ([report](artifacts/results/c3_kaggle_t4/report.md)).

**Near saturation, one fit is not enough.** A second run on the same engine, 25 minutes later,
gave coefficients 3–5% different. For the one workload at the knee of the load curve, that put a
p95 4.2× off. The other five numbers stayed within 1.36×.

**So the tool now shows how firm its answer is.** Since 2026-10-05, `kvsim calibrate` measures
every timing cell three times. Next to the fit on all the data, it keeps the fit of each repeat
alone. `kvsim replay` sizes the fleet on each of those fits too, and prints the result next to its
recommendation:

```text
How firm: 3 to 4 replicas across the calibration's 3 repeats taken alone; size for the larger, plus one.
```

That range is how much the repeats of one calibration disagree. It is a floor on the uncertainty,
not a confidence interval. It knows nothing about another day or another software image. A profile
without repeats (like the two shipped ones) says that its spread is unknown.

**That floor was measured, and the registered test of it failed (C4, rule E).** The command ran
three times on one live T4 in one session, with three repeats each. Across 18 sizing questions and
both targets, the GPU count of another run was inside a run's own range in 82% of cases. The rule
asked for 90%. With the range widened by one GPU each way, it was 95%. The two runs taken back to
back agree with each other (97%). The run taken an hour earlier does not agree with them (75%), and
the repeats of one run, six minutes apart, cannot see that. So size for the top of the range, plus
one. This is what the tool's own line now says. The command itself works on a real engine with its
repeats: it was exact on its cache check all three times, at six minutes a run
([report](artifacts/results/c4_kaggle_t4/report.md)).

**It also says how close to the edge its recommendation is.** `kvsim replay` runs the recommended
configuration once more with every timing coefficient 5% larger, which is how much a second
calibration of the same GPU can differ. Then it prints how far the p95 latencies moved:

```text
At the edge: with timings 5% slower this configuration no longer meets the SLO (p95 TTFT 619 to 692 ms (×1.12); p95 time per token 93 to 100 ms (×1.07)). Two calibrations of the same GPU differ by that much: allow one replica more.
```

The example above ends with this line. When there is room to spare, it reads "Not at the edge",
and the p95s move by about the same 5%. If a p95 moves by more than 1.5×, it reads "At the knee".
That means the fleet is close to saturation, and there the simulator's own latencies were 1.5–1.7×
off on a live GPU (T1). `--no-edge-check` skips this check.

**The 1.5 was chosen after looking at the data** (post hoc, and labelled so). On T1's five
held-out runs, the 5% slowdown multiplies p95 TTFT by 5.3 and 5.5 on the two runs the simulator got
wrong, and by 1.13–1.14 on the three it got right. Nothing else moves by more than 1.21×
([table](artifacts/results/t1_itl/posthoc_edge.md)). This is five runs of three workloads on one
GPU. It suggests that the check can see the knee, but it is not a validated threshold.

## What carries over to your own fleet

Read this before you trust a number. A calibration is fitted to one GPU, one model and one block
size. So the answer comes in three levels, and they must not be mixed (ADR-010):

| | What you get | What it costs you |
|---|---|---|
| **Policy ranking** | Which policy wins on your workload shape, and roughly by how much | ⛔ Not free. H6 was tested and failed (2026-09-28). With a 2× faster GPU, or with prefill 2× cheaper or more expensive, the full order of the seven policies changed (Kendall's τ-b as low as 0.51). One thing held, noticed after the run and not registered: session-sticky needed the fewest replicas at all 48 points. So read a ranking as a result for the T4 used here, unless you calibrate |
| **Whether prefix routing pays** | Whether prefix-affinity is worth building for your prompts and load | Not settled, even on the T4 used here. With one shared prompt or a few: no. Session-sticky needs the fewest replicas, or ties, on all three calibrations. With 64 prompts, the registered test said no on one calibration (2026-09-30) and yes on two others (2026-10-05). The two policies are one or two replicas apart, and calibrations of the same GPU disagree by that much. On the first timing model (no context term), prefix routing paid once prompts outgrew one GPU's cache, and that flip survived four hardware rescalings (H6's crossover half). But that model was wrong about the engine. So the answer depends on the timing model and on its error bar. Measure yours, more than once |
| **Absolute ms and ₹** | p95 TTFT and ₹/1k requests for your fleet | Run `kvsim calibrate` on your own hardware. It takes a few minutes. The error bar here was measured on one GPU and one model. It does not transfer, and this README will never suggest that it does. On the T4 used here, the first timing model (no context term) was 5–222× optimistic under load. The context-aware model predicts held-out traffic within 1.41× (C2), and so does a profile that `kvsim calibrate` wrote itself (C3). The exception is the knee of the load curve, where a second run of the command was 4.2× off on one p95. All of that is first-token latency. Time per token is now measured too: there the model is within 7% away from the knee, and 1.5–1.7× too slow at it (T1) |

> **These numbers are not for your hardware.** Run `kvsim calibrate` and use your own numbers. The
> cost model works the same way: every constant is in a config file with a source and a date, so
> you can disagree with it and run again.

## Where it runs

A GPU is used only once: for calibration, offline, before anything is published. Everything that
is deployed is plain CPU Python:

- **The package**, on your machine, from PyPI (`pip install kv-routing-sim`). This is the main way
  to use it.
- **A static demo page**, [live on GitHub Pages](https://gitgod-debug.github.io/kv-routing-sim/).
  It is a precomputed grid with selectors. You can see the recommendation change with the prompts
  and the load in ten seconds, with nothing installed. There is no server, and every number on the
  page comes from a committed script with a pinned seed. The page is a demo, not a service. It takes
  no uploads, because your traces contain your users' prompts and should stay on your machine. The
  [in-browser runner](https://gitgod-debug.github.io/kv-routing-sim/run.html) next to it runs the
  same package inside your browser. A log you choose there is read locally and not uploaded.
- **Docs**: static pages.

Replaying 80,000 requests through a dict-based trie took 13.5 to 21 seconds on one CPU core
([measured](artifacts/results/scale/report.md)). There is nothing for a GPU to do here. This is why
a simulator is useful: you pay for the GPU once, for calibration, and then you can ask "what if"
thousands of times on a CPU.

---

## Related work, and what is claimed

Other people work on the same problem, and more of them than this README said before 2026-10-05.
Three checks of prior work missed NVIDIA's simulator and the two papers below. The fourth check
found them. What was missed, and why, is in [`docs/04-prior-art.md`](docs/04-prior-art.md).

| Project | What it is, and how this differs |
|---|---|
| **NVIDIA Dynamo's simulator**: `ai-dynamo/aisim` (called `aisimulate` until November 2026), "DynoSim" (and `aiconfigurator` before it, arXiv 2601.06288) | The closest tool, and a larger one. It replays a saved trace on simulated vLLM or SGLang workers with a block-level prefix cache. (Its tutorial uses the same Mooncake trace as R1 below, and it reads the AgentX traces of R2.) It compares round-robin with Dynamo's KV-aware router, and recommends a minimum GPU count under TTFT and TPOT limits. It has had offline multi-worker replay since March 2026. *What I did not find in it (read, not run, 2026-10-05):* routing policies other than those two; a published check of its cache model against a live engine (its public accuracy page covers TTFT and TPOT on single-node points with prefix caching off); a cost in money. |
| **BLIS** (arXiv 2610.14255) · **a simulator for SGLang clusters** (arXiv 2610.15366) · **ROCm InferaSim** (arXiv 2611.04032), all from October and November 2026 | Three more simulators checked against live engines, found by the check of 2026-11-25. BLIS models routing and KV-cache management, and reports 6.7% median end-to-end error against vLLM with llm-d. The SGLang one reports a token-hit-ratio error within 1.23 points, and it measures session affinity against round-robin with predictions frozen first. InferaSim has a block-level prefix cache and routing across replicas. Each one is larger than this project. A calibrated simulator of this kind is no longer rare. |
| **LMetric** (Zhang et al., OSDI '26, arXiv 2603.15202) · **SMetric** (Wang et al., arXiv 2607.08565) · **Preble** (Srivatsa et al., ICLR '25) | The closest findings, measured on real clusters. LMetric: multiply new prefill tokens by batch size, and then no weight needs tuning. SMetric: route agents by session, and move a session when its instance would miss the SLO. The routing results of this repository agree with them, and add nothing to them. Both are ported here (`lmetric`, and `smetric-local`, which is SMetric's rule without its global cache tier) and were run on a public production trace (R3, above). These are ports, not the authors' systems. |
| **`casys-kaist/LLMServingSim`** (483★, ISPASS '26) | A cycle-level simulator that copies vLLM's scheduler and ports its block pool, for each instance. It reports TTFT and TPOT within 1% of a real vLLM run on one profile. Its router is round-robin, random or least-loaded, with no prefix-aware or sticky policy. |
| **`microsoft/vidur`** (735★, MLSys'24) | Capacity and parallelism planning. Its `canary` branch does have a block-level prefix cache and three cache-aware routers. But that branch has not been merged since June 2025, so `main` (what you install) cannot answer the routing question. |
| **`NetX-lab/Frontier`** (188★, arXiv 2605.21312) | Prefix and hierarchical caching, SLA-aware Pareto search, and session-sticky routing. It has no prefix-affinity router: its sticky routing uses the session identity, not a shared prefix. |
| **`azrabano23/disagg-sim`** | A teaching tool: it reproduces published scheduling arguments on synthetic workloads. Its own README says it does not fit traces, and points to Vidur for that. |
| **`jahnavi-yelamanchi/cachetrace`** | A single-path prompt-construction linter. |
| **CacheWise** (arXiv 2606.16824) · **`gauravapiscean/agentic-kv-cache`** | Prefix-aware scheduling and eviction for coding agents, and single-node eviction policies replayed on the same AgentX and Mooncake traces (that second one finds LRU hard to beat). Both work inside one engine. Neither routes across replicas. |
| **vLLM · SGLang · LMCache · Mooncake · AIBrix · llm-d** | Production engines, caches and gateways. They implement caching and routing, and do not compare policies. Dynamo was in this row until 2026-10-05, and it ships a simulator. The docs of the others were read again on 2026-11-25, and none was found to compare routing policies across replicas against a live engine ([`docs/04-prior-art.md`](docs/04-prior-art.md), §2b). |

**What is claimed, after that check.** Not an empty niche. Only three things that this repository
did and can show:

1. **A measured calibration.** The cache model reproduces the prefix-cache counters of a live vLLM
   exactly, and its hit rates held within 1.4% on two live replicas under three routing policies.
2. **A method.** The design, rules and prediction of every test were committed before it ran. The
   tests that failed are reported as findings.
3. **Numbers on how far sizing from a simulator can be trusted.** A second calibration of the same
   GPU, 5% different, is 4.17× off at the knee of the load curve. Replica counts from three
   calibrations are within one in 89 of 92 cells, and in 8 more cells the calibrations disagree on
   whether the target can be held. The per-token model is right away from the knee and wrong at it.

The routing result itself agrees with published work. It was reached independently, on a free
Kaggle GPU.

So the value of this project is in how it was done, not in the idea: the published calibration
error and the error bars. This section said the same before the check. `Mike-Ew/kv-cache-routing`
specified the same six policies and the same 2-replica validation in April 2026, and never built
it. NVIDIA built a larger one.

---

## What breaks at 10× and 100×

Measured on 2026-09-28 by [`scripts/measure_scale.py`](scripts/measure_scale.py), with one fresh
process per case and one CPU core each. The results are in
[`artifacts/results/scale/`](artifacts/results/scale/report.md). The base is Phase 4's traffic (200
sessions × 4 turns = 800 requests) on 8 replicas, with the calibrated T4 profile. Each case was run
with round-robin and with balanced prefix-affinity.

| Scale | Question | Answer |
|---|---|---|
| **10× trace length** (8,000 requests) | Peak memory (RSS) and sweep time? | 81 MiB peak per process. One run takes 1.3 s (round-robin) to 2.0 s (balanced affinity), after 8 s to generate the trace. Nothing breaks: a default `kvsim replay` sweep (280 runs, one core) takes minutes. |
| **100× trace length** (80,000 requests, 8.5 M block hashes) | What fails first: block-hash memory, the policy inner loop, or the report renderer? | Block-hash memory. The trace alone holds 537 MiB (about 64 bytes per hash, as Python ints) before the simulation starts. The simulation adds 11–35 MiB. The policy loop stays linear: 13.5 s and 21 s, about 10× the 10× case. The renderer does not grow with the trace. Two more limits follow by arithmetic and were not measured. The parallel experiment scripts copy the trace into every worker (11 workers ≈ 6 GiB), and a default one-core sweep is 280 × 13–21 s ≈ 1–1.6 hours. |
| **10× replicas** (80 replicas at 10× load) | Does the cache model stay accurate, or does eviction-order skew add up? | This cannot be answered on a CPU. In the model each replica is its own block pool, so errors cannot add up inside the simulator. Whether real replicas stay matched is what Phase 6 checks, on 2 GPUs. 80 replicas are not tested. What was measured: affinity routing scores every replica on every request, so its run time grew 2.4× (2.0 → 4.8 s for the same 8,000 requests), while round-robin stayed flat (1.3 → 1.4 s). |
| **A real trace, whole** (AgentX: 53,967 requests, 121.6 M block hashes of 64 tokens, prompts up to 894,371 tokens) | Does it run at a recorded size, and does the cache accounting still hold? | Yes, when the block hashes are shared. Reading the trace takes 81 s and 1.3 GiB, because the reader hashes each distinct block of a session only once. Before that change it would have needed about 8 GiB. One run of 8 replicas takes 3.2 to 5.4 minutes on one core. One replica that never evicts served exactly what the raw file allows, for all 53,967 requests (L1, [report](artifacts/results/l1_agentx_full/report.md)). Cache accounting only, no timing |

---

## Reproducibility

The numbers in this README come from files under `artifacts/results/`. Each file was written by a
script in `scripts/` or by a GPU notebook. The evidence ledger near the top names the file behind
each number of the first screen. A file written by a simulation script records its date, the git
SHA (and whether the tree was dirty), the seed, the hardware, the installed package versions and
the hardware path. The raw files of GPU runs and most analysis files carry no commit id of their
own. For those, the last column of the ledger names the commit that last changed the file. Every
cost constant is in `configs/cost_model.yaml`, with a source URL and a date. If you disagree with a
constant, change it and run again to get your own number.

The rule: everything in `artifacts/results/` must be reproducible by running a script in
`scripts/`. A result that cannot be deleted and made again does not go into the repository.

**One command checks it.** `python scripts/reproduce.py` copies the files of the repository to a
temporary folder and runs thirteen steps there. It compares every file a step writes with the
committed one. The steps cover the analysis of each GPU run (from the raw files it left here), the
grid behind the finding, the session cost table and the evidence ledger. It takes about 20 minutes
on a 12-thread laptop. `--quick` leaves out the three slow steps and takes about 20 seconds. Fields
that say when and where a file was made are left out of the comparison. Where the script makes one
more exception, it says so and says why. **Not covered yet:** the first two sweeps (Phase 4 and
Phase 5), H6, the scale test, the real-trace runs (R1 to R3, and the full-size run L1, whose traces
are downloaded and not committed), and the stored grids that some analyses start from (T2's three,
C4's, and those of calibrations B and C). A GPU measurement cannot be made again this way.

---

## Documentation

| | |
|---|---|
| [`docs/00-START-HERE.md`](docs/00-START-HERE.md) | Start here: what this is, and the order to read the docs |
| [`docs/02-architecture-decisions.md`](docs/02-architecture-decisions.md) | ADR-001 to ADR-013: the design decisions and their reasons |
| [`docs/04-prior-art.md`](docs/04-prior-art.md) | Other work on the same question, what to cite, and the sentences this project must never write |
| [`docs/06-methodology.md`](docs/06-methodology.md) | The statistics protocol and the cost model |
| [`docs/09-hypotheses.md`](docs/09-hypotheses.md) | Every test, registered before it ran (H1–H6, H2b, R1–R3, C1–C4, T1, T2, S1, V1, L1 and Phase 6), with dated results, failed ones included |
| [`docs/13-the-finding-in-detail.md`](docs/13-the-finding-in-detail.md) | What stands behind the finding: three calibrations, the live check, the real traces, the per-token half |

**Sibling project:** `colpali-lite`, not public yet. It uses the same method (cost and tail latency
under a configuration choice, bootstrap CIs, an explicit cost model, results in ₹ per unit of
useful work) for late-interaction retrieval instead of LLM serving. Serving cost is usually
measured too little in ML systems: mean latency only, no cost axis, no error bars.

## License

MIT
