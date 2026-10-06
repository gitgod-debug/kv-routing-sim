# kv-routing-sim

**Which KV-cache routing policy should you actually run — and on how many replicas?**

Give it your serving log. It returns a routing policy, a replica count, and a cost number. Its
prefix-cache model is calibrated against a live vLLM's own Prometheus counters, and **the calibration error
is published**.

> ## 🚧 Status (2026-10-06): the cache model, both halves of the timing model and `kvsim calibrate` are checked against a live engine — and the checks measured how far a GPU count can be trusted.
> The Real-Engine Calibration Report below is filled in from a live vLLM, against predictions committed
> before the run. On **two live T4s** (Phase 6, run twice) every hit rate matched within 1.4% — but the
> first timing model was 5–222× optimistic on latency. **C2** (registered) found why — decode and
> prefill cost grow with context length; CPU contention was ruled out — and fitted a context-aware T4
> model that predicts held-out traffic within 1.41×. **C3** (registered, 2026-10-05): `kvsim calibrate`
> ran on a real T4, exact on its cache check, and the profile it wrote predicted three unseen workloads
> within 1.41×. The same run showed that three calibrations of one T4 agree on a GPU count to within
> one replica at 96% of points — **and that the test behind half of the headline was decided inside
> that margin** (below). Replica counts are T4 figures on generated traffic, ± one, **and for the
> first token only: under a target that also limits token speed the simulator needs about twice
> as many GPUs.** **C4** (registered, run 2026-10-05; 7 of its 9 rules hold) checked the model
> behind that factor on a live T4: its p95 time per token is within 7% of the engine's on three
> workloads around the limit. It also found that one calibration's own repeats do **not** cover
> the next calibration's GPU count often enough — 82% of cases, 90% was the rule (the scope note
> and the calibration section, below). ⚠️ **And the routing result is not new:** a prior-art check on 2026-10-05 found it
> published already (SMetric, LMetric), and found that NVIDIA ships a larger simulator for the same
> question — see the note under the finding, and the scope section. Every number
> still written as `TBD — not yet measured` **is exactly that**; see the honesty gates in
> [`CHECKPOINTS.md`](CHECKPOINTS.md). Start at [`docs/00-START-HERE.md`](docs/00-START-HERE.md).

---

## The finding

> **Keep each user on one GPU.** On three calibrations of the same T4, session-sticky routing needs
> the fewest GPUs — or ties for them — at every load with one shared system prompt and with four, and
> it never needs more than round-robin or least-loaded anywhere. **Whether routing by prefix then
> saves GPUs, once tenants' prompts outgrow the cache, is not settled:** the registered test said no
> on one calibration and yes on the two others. Read every GPU count here as ± one.

⚠️ **This is not a new result** — found on 2026-10-05 by a prior-art check that three earlier ones
had failed ([`docs/04-prior-art.md`](docs/04-prior-art.md), §2a). **SMetric** (Wang et al., arXiv
2607.08565, July 2026) reached the same advice from production traces on 32 real GPUs: send a
session's follow-up requests to the instance that served it. **LMetric** (Zhang et al., OSDI '26)
had shown before that why a hand-weighted mix of a cache score and a load score has no right
weight. What this repo adds is an independent agreement on a small calibrated model, from tests
registered before they ran. **And SMetric names a weakness these grids cannot show:** real sessions
differ widely in size, and plain session-sticky routing then leaves the load unbalanced — 2.17× on
its testbed. Every generated session here has the same number of turns. Neither published policy
has been run in this simulator yet.

Measured on simulated agentic traffic, with the cache model calibrated against a live vLLM (the
credential, below), a context-aware T4 timing model, and a 2-second p95 TTFT target. The test was
registered before it ran, on calibration **A** — C2's fit
([`docs/09-hypotheses.md`](docs/09-hypotheses.md), 2026-09-30). **B** and **C** are the two profiles
`kvsim calibrate` wrote on a real T4 in C3 (2026-10-05): B's table was a registered reported item,
C's was added after seeing it and is post hoc
([all three, side by side](artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.md)).

| Distinct system prompts (85 blocks each) | Working set ÷ one T4's cache | Fewest T4s that hold the target at 2 sessions/s — on A · B · C |
|---|---:|---|
| 1 | 0.06 | session-sticky **5 · 5 · 5** · round-robin 7 · 7 · 7 · balanced prefix-affinity 10 · not held on 16 · not held |
| 64 | 2.66 | session-sticky 10 · 12 · 12 · balanced prefix-affinity 11 · 11 · 10 · round-robin 16 · 16 · not held · least-loaded: not held *(prefix-affinity with a load weight of 1: **8 · 8 · 8** — a weight found after the tests)* |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/flip-dark.svg">
  <img src="docs/figures/flip-light.svg" width="100%" alt="Left: T4s needed at 2 sessions/s as distinct system prompts grow from 1 to 64, on the context-aware timing model. Session-sticky needs 5 with one prompt and 10 with 64; balanced prefix-affinity needs 10 and 11; round-robin 7 and 16. Right: hit rate on 8 T4s. Round-robin falls from 0.81 to 0.47, session-sticky from 0.92 to 0.80, prefix-affinity with load weight 1 stays near 0.87.">
</picture>

*Drawn by [`scripts/make_figures.py`](scripts/make_figures.py) from
[`demo_grid_ctx.json`](artifacts/results/demo_grid_ctx.json), which reproduces the registered re-test's
table exactly; 3 seeds per point, a point counts only if every seed holds the target. T4 figures,
**on calibration A only** — on B and C the 64-prompt points move by up to two replicas, as the table
above shows.*

- **One shared prompt, or four — this part holds on all three calibrations:** session-sticky needs
  the fewest GPUs at every load (12 of 12 points), and its saving over round-robin grows with load —
  5 against 7 at 2 sessions/s, 6 against 9 at 3, 7 or 8 against 12 at 4. Across all 24 points it
  never needs more than round-robin or least-loaded. *Pure* prefix-affinity still collapses under
  load: it sends every session to the one replica holding the prompt (H2, falsified).
- **Many prompts (64) — this part does not:** prefix routing gets the better hit rate (0.87 with a
  small load weight, 0.80 for session-sticky, 0.47 for round-robin, on A). Whether that becomes fewer
  GPUs depends on the calibration. With the registered load weight, session-sticky needs 10 against
  11 on A, 12 against 11 on B, 12 against 10 on C. The three calibrations differ by a few percent in
  their coefficients; at this point that is one or two GPUs, and the sign of the answer.
- **Found after the tests, then tested (C4, F1 — holds) — and it is a first-token effect:** with a
  load weight of 1, prefix routing needs 8 GPUs at 64 prompts and 2 sessions/s on calibrations A, B
  and C, where session-sticky needs 10 to 12; on three fresh calibrations, registered beforehand,
  8 · 7 · 8 against 9 · 10 · 10. But under a target that also limits token speed the same policy
  holds nothing there on 32 replicas, on any of the six: it wins the first token by crowding
  sessions onto the replicas that hold their prompt, and crowding is what makes tokens slow.
- **With both halves of the target, the two are one GPU apart at most (C4, F2 — my prediction
  failed):** I had registered that balanced prefix-affinity would need fewer GPUs than
  session-sticky at no more than two of six loads. On two of the three fresh calibrations it
  needed one fewer at three and at four. One GPU of 4 to 27 is inside what two calibrations
  disagree by, so this is not a result for prefix routing — but it is not "no saving" either.
- **How far to trust a GPU count:** of the 144 (prompts, load, policy) cells, 102 need the same number
  of replicas on all three calibrations and 138 are within one.

**How this finding changed.** Until 2026-09-30 it read *"…until your tenants' system prompts outgrow
one GPU's KV cache; then route by prefix"* — at 64 prompts, balanced prefix-affinity held the target
on 5 T4s where session-sticky could not on 8. That was a correct result about the first, context-free
timing model, and that model was wrong about the engine: on two live T4s (Phase 6) latency came out
5–222× its prediction. C2 found the missing term and fitted a model that predicts held-out traffic —
and, post hoc, Phase 6's two replicas within 1.02–1.79×. Re-tested on it, as registered, the "route
by prefix" half was **falsified**, and from 2026-09-30 the headline said that prefix routing "did
not save a GPU".

On 2026-10-05, C3 put `kvsim calibrate` on a real T4. As a registered reported item, the same test
was re-run on the profile the command wrote — a second calibration of the same GPU, a few percent
different in its coefficients. H2b's rules **held** there; and on the profile from the command's
second run (post hoc) they held again. The verdict of 2026-09-30 stands as what that registered
test said on calibration A. But it was decided by one replica at one point, and one replica is
what two calibrations of the same GPU disagree by. So the headline now claims only what all three
support. All versions, and why, are in [`docs/09-hypotheses.md`](docs/09-hypotheses.md).

**The live check (Phase 6, registered; run twice on two T4s).** The traffic at 1 session/s, one prompt
(K1) and 64 (K64), three routers on two live vLLM replicas:

| | K1: sticky · round-robin · affinity | K64: sticky · round-robin · affinity |
|---|---|---|
| **Hit rate**, predicted | 0.921 · 0.880 · 0.923 | 0.821 · 0.694 · **0.844** |
| **Hit rate**, live (two runs) | 0.921 · 0.880 · 0.923 — to four decimals | 0.810–0.822 · 0.695 · **0.844–0.846** |
| **p95 TTFT**, first timing model | 0.35 s · 0.44 s · 0.48 s | 1.55 s · 5.8 s · 1.26 s |
| **p95 TTFT**, context-aware model (post hoc) | 2.0 s · 20.4 s · 128 s | 62 s · 110 s · 39 s |
| **p95 TTFT**, live (two runs) | **1.8–1.9 s** · 11–15 s · ~106 s | 57–72 s · ~110 s · **32–39 s** |

Every hit rate matched within 1.4%, in the predicted order. The first timing model missed latency by
5–222×; the context-aware one, which never saw these numbers, lands within 1.02–1.79×
([Phase 6](artifacts/results/phase6_kaggle_t4/report.md) ·
[post-hoc replay](artifacts/results/c2_kaggle_t4/posthoc_phase6.md)).

**See it yourself:** the demo page ([`site/`](site/index.html)) looks up every point of the re-test's
grid — prompts × load (0.25–4 sessions/s) × replicas (1–16) × six policies, 6,912 simulator runs,
rebuilt by [`scripts/build_grid.py`](scripts/build_grid.py) only if it reproduces the registered table
exactly. To open it from a clone: `python -m http.server` in the repo root, then
`http://localhost:8000/site/`.

**On real traffic** — two trace families, each test registered before it ran, both on the **first**
timing model (not yet re-run on the context-aware one):

| Real trace | What is shared | Session-sticky vs prefix-affinity | Pure prefix-affinity | Round-robin |
|---|---|---|---|---|
| **Mooncake** FAST'25 (R1, scale 1/32, sessions inferred) | prompts across users | within one replica (predicted a win on tool-agent: **falsified**) | fails at every speed | 8 T4s vs 6 at 4× tool-agent |
| **AgentX**, coding-agent sessions (R2, scale 1/64, real sessions) | nothing across users | **identical**: 2 T4s at every speed (**as predicted**) | 2 T4s at every speed — **no collapse** | 5 T4s vs 2 at 2×; hit rate 0.59 vs 0.93 |

Pure prefix-affinity collapses because of prompts shared across users, not because it routes by
prefix; and on agent traffic like AgentX the rule is simply *keep each session on one replica, by any
policy that does* — the same advice as the finding above.

⚠️ **Scope:** generated traffic, one GPU type (T4), one model (Qwen3-4B). The replica counts are T4
figures on the context-aware model, ± one replica.

⚠️ **Every count above is for the first token only — and that is about half the story.** The MLPerf
rule the 2 seconds come from pairs them with a limit on the time per output token (100 ms for a
model of this size). In C3 the live engine took 134–521 ms per token on workloads that met, or
nearly met, the first-token target. Three registered tests followed — two with no GPU, then one
on a live T4:
- **T2 — the same grid under a target with both halves** (p95 TTFT ≤ 2 s *and* p95 time per output
  token ≤ 100 ms), on all three calibrations. All three rules held, as predicted: session-sticky
  is still best or tied at every load with one prompt and with four; it never needs more than
  round-robin or least-loaded; and its replica count is **1.93–2.00× the first-token count**
  (median over 24 points). With the limit at 200 ms the factor is 1.33–1.37×. No table of those
  counts is printed here, on purpose — see T1.
- **T1 — is the simulator's per-token latency right? It failed.** Against the live engine's own
  inter-token counter on five held-out runs, the simulator is within 7% on the three runs off the
  knee of the load curve, and **1.5–1.7× too slow** on the two at it (limit: 1.5×; predicted to
  hold). And all five runs are themselves beyond 100 ms per token, so nothing had been measured
  where T2's answers live.
- **C4 — is it right where the limit is? Yes** (a live T4, 2026-10-05; nine rules registered,
  seven hold). On three workloads chosen to sit under, on and over 100 ms per token, every
  request's token times kept:

  | workload | p95 time per token: live · registered | share of requests within 100 ms: the engine's own histogram · registered |
  |---|---|---|
  | under the limit | 82.6 · 82.9 ms | 100.0% · 100.0% |
  | on it | 104.2 · 104.3 ms | 90.6% · 91.4% |
  | over it | 279.7 · 298.5 ms | 47.7% · 46.1% |

  And it found why T1 failed. Batches that share a prompt follow the fitted line within 3% up to
  twice the range any calibration had measured, and run **12–15% faster** than the line at five to
  nine times that range. At the knee a slower step keeps more sequences in the batch, which slows
  the next step: there, 15% per step became 1.5–1.7×.

So the factor of two now rests on a per-token model that was checked where it lives — **on one
replica and three workloads; a fleet has not been measured.** The counts in this README are still
first-token counts: read them as a *lower bound* for anything that has to stream at a usable
speed. The simulator prints p95 time per token in every report, and a cost model can make it part
of the SLO (`slo.tpot_p95_ms`). It is still off by default: the check it was waiting for has
passed, and switching the default is the next change, not this one
([T1](artifacts/results/t1_itl/report.md) · [T2](artifacts/results/t2_two_part_slo/report.md) ·
[C4](artifacts/results/c4_kaggle_t4/report.md) · [`docs/09-hypotheses.md`](docs/09-hypotheses.md)).

⚠️ **The first token is now the less accurate half.** On the same three workloads the live p95
time to first token was 1.23–1.30× the registered number — inside the 2× that C2 and C3 tested,
and the model too fast every time.

H6 (the policy ranking under faster and slower GPUs) was tested on the first, context-free model
and is not yet re-run. That locality can lose to load balancing is published — LMetric and SMetric
on real clusters, Preble before them, `azrabano23/disagg-sim` in a simulator — and llm-d and Dynamo
weigh prefix and load scores by hand-set weights. On this setup, with timing checked against the
engine, balance won too.

## The credential — the Real-Engine Calibration Report

Separate section, because it is a separate kind of claim. *"The simulator matches the live engine within
X%"* proves **rigor**; it says nothing about the world. The finding above says something about the world.
A README needs both, and needs them apart.

| | |
|---|---|
| **Engine + version** | **vLLM 0.30.0**, V1 engine, async scheduling on (its default) |
| **Exact CLI** | `vllm serve Qwen/Qwen3-4B --dtype half --block-size 16 --num-gpu-blocks-override 512 --max-model-len 2048 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --kv-cache-metrics --kv-cache-metrics-sample 1.0 --enable-prompt-tokens-details --seed 0` |
| **Model + quantization** | `Qwen/Qwen3-4B`, fp16, unquantized. The cache is fixed at **512 blocks (511 usable)** by override, so capacity is a controlled input and the cache overflows |
| **Hardware** | **Kaggle 2x Tesla T4** (compute 7.5, driver 580.159.04, CUDA 12.8). vLLM runs TRITON_ATTN in fp16 here — so hit rates and routing results are hardware-independent, and every latency/cost figure is a **T4 figure** |
| **Hit-rate error vs live engine** | ⭐ **Zero** on 2 sequential shapes × 3 runs — every counter, and every one of 120 requests' cached tokens, to the token. **+0.3%** (relative) on the concurrent shape, predicted by a timing profile fitted without it. Plain LRU on the same runs **over-predicts** by **11.9 points** (W1) and **19.7%** (W3) |
| **The residual, explained** | Sequential: none to explain. The engine's `kv_block_*` histograms: counts exact (963 and 1,880 events); distributions identical bucket for bucket on W2; W1's small residual (max CDF distance 0.042) traced — post-hoc, labelled so — to the **timing** fit, not the cache model |
| **2-replica validation** | Round-robin across **two physical T4s**: 1,088 prefill tokens computed, **1,088 predicted** (Phase 0 v2). **Under pressure (Phase 6, run twice): all 12 runs' hit rates within 1.4%** (K1 to four decimals), 1,537 blocks per replica, three routers, preemptions included |
| **Timing under load** | The first, context-free profile was **optimistic**: on Phase 6's ~1,700-token prompts, p95 TTFT came out 5–222× its prediction. ✅ **Fixed by C2 (2026-09-30, registered):** decode cost grows with context (1.92× from 128 to 1,536 tokens) and CPU contention was ruled out (1.00×); the context-aware profile `t4-qwen3-4b-vllm0.30.0-ctx`, fitted on micro-benchmarks, predicted held-out traffic within **1.10–1.41×** — and, post hoc, Phase 6's two live replicas within 1.02–1.79× ([report](artifacts/results/c2_kaggle_t4/report.md)) |

**`/metrics` fields consumed, by name:** `vllm:prefix_cache_queries` · `vllm:prefix_cache_hits` ·
`vllm:prompt_tokens_by_source{source="local_compute"|"local_cache_hit"}` · `vllm:num_preemptions` ·
`vllm:kv_block_lifetime_seconds` · `vllm:kv_block_idle_before_evict_seconds` ·
`vllm:kv_block_reuse_gap_seconds` (`_count` and `_bucket`) · `vllm:prompt_tokens` · `vllm:generation_tokens` ·
`vllm:num_requests_running` · `vllm:num_requests_waiting` — plus, per request, the response's
`usage.prompt_tokens_details.cached_tokens`. TTFT is measured at the client, over streamed responses.

### What was predicted, and what the engine did

Predictions committed before the run
([`predictions.json`](artifacts/results/phase3_prereg/predictions.json)); measured on the engine above
([`calibration_report.md`](artifacts/results/phase3_kaggle_t4/calibration_report.md)). 120 agentic
requests — 40 sessions × 3 turns, one shared system prefix — three runs of each workload.

| Workload | Engine (all 3 runs) | Block-pool model (default) | Plain-LRU model (ablation) |
|---|---|---|---|
| **W2** one at a time, 1-token outputs | hit rate **0.6090** · 22,664 computed | **0.6090** · 22,664 — exact | 0.6073 · 22,760 |
| **W1** one at a time, 64-token outputs | hit rate **0.4883** · 29,656 computed | **0.4883** · 29,656 — exact | 0.6073 · 22,760 |
| **W3** the same trace, concurrent | hit **0.5076** · **27** preemptions · p95 TTFT **41.0 s** | 0.509 (+0.3%) · 24 · 38.3 s (−6.7%) | 0.6073 (+19.7%) · 0 · 11.2 s (−73%) |

W1 is the experiment's point. The two models agree when requests hold no decode blocks (W2) and part
by **11.9 points** when they do: vLLM caches full decode blocks, they occupy the same small cache, and
they evict useful prefixes. The block-pool model says so; LRU cannot see it. On a real engine, the
block-pool model was right to the token.

**The fitted T4 timing** (all of it a **T4 figure**): prefill **0.70 ms per uncached token**; decode
**52.9 ms + 1.23 ms × batch** (R² 0.58). ⚠️ Two limits, measured and recorded rather than tuned away —
[`posthoc.md`](artifacts/results/phase3_kaggle_t4/posthoc.md): one sequence decodes *slower* than two to
eight do (63.6 ms per token), which a straight line cannot express; and the simulator's step model
charges a full decode step on top of every prefill, over-estimating an idle replica's TTFT by ~20 ms.

> ⛔ **The hit rate is derived — `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`, two counters.**
> `vllm:prefix_cache_hit_rate` **does not exist** (verified against `v1/metrics/loggers.py` at v0.29.0); the
> old V0 gauge went away with V0.

---

## What it does

1. **Trace ingestion** — reads real serving logs (vLLM/SGLang JSONL, OpenAI-style request logs) into one
   normalized, versioned request schema. Prompts are hashed at the adapter boundary and **never enter the
   simulator**, so a trace's user content stays out of memory and out of artifacts.
2. **A block-level KV cache model**, per replica. The default follows vLLM 0.30.0's block pool line by
   line — reference counts, tail-first freeing, cached decode blocks, admission by free memory, and
   preemption. A plain LRU model is kept beside it as the ablation. *This is table stakes, not the
   contribution — the contribution is its measured fidelity.*
3. **Seven routing policies** — round-robin · least-loaded · power-of-two-choices ·
   consistent-hashing-on-prefix · radix-prefix-affinity · radix-prefix-affinity-balanced (affinity with
   a load term, the way production routers do it) · session-sticky. One file each; a new policy is a
   new file.
4. **The sweep + recommendation** — policy × replica count (1–8) → Pareto curves over
   hit-rate / p95 TTFT / **₹ per 1k successful requests**, and one recommended configuration with a
   quantified delta against the naive baseline.
5. **Sim-to-real validation** — deploy the recommendation and compare predicted against measured.

### An agentic-traffic generator, because that is where this matters

An agent re-sends the system prompt, the tool definitions and the whole conversation **every turn**. Prompt
sharing there is not an optimization opportunity — it is most of the workload. The generator produces that
shape with the **prefix-sharing ratio as an explicit parameter**.

---

## Install

```bash
pip install kv-routing-sim          # CPU-only. No GPU dependency, by design.
kvsim replay trace.jsonl --policies all --replicas 1-8 --format md
```

Runs on your laptop. **Your trace never leaves your machine**, and prompts are hashed at the adapter
boundary, so prompt text never enters the simulator at all.

To use measured timings instead of the built-in guess, pass a profile. Ours — Tesla T4, Qwen3-4B,
vLLM 0.30.0 — ships with the package, as does the cited cost model. Use the context-aware one: it is
the one checked against live latency (C2); `t4-qwen3-4b-vllm0.30.0`, the first, context-free profile,
is kept for the record and is optimistic on long prompts.

```bash
kvsim replay trace.jsonl --policies all --replicas 1-16 \
  --profile t4-qwen3-4b-vllm0.30.0-ctx --cost-model default
```

(`--profile` also takes a path to any profile file, and `--cost-model` a path to your own YAML — the
shipped ones are byte-for-byte `configs/profiles/` and `configs/cost_model.yaml`.)

To get **your** numbers, point `kvsim calibrate` at your own running vLLM (prefix caching on). It
fits C2's context-aware timing model to your GPU and model — decode steps timed at several context
lengths and batch sizes, prefill at several prompt lengths and after a cached prefix — reads the KV
capacity from the engine, and checks the cache model against the engine's own counters. That takes a
few minutes and needs nothing beyond the core install:

```bash
kvsim calibrate --engine http://localhost:8000 --out mine.json
kvsim replay trace.jsonl --policies all --replicas 1-8 --profile mine.json --cost-model default
```

If the cache check misses (by more than 5%, or on any single request's cached tokens), the profile is
written but **not** marked calibrated, and the command exits 1.

**On a real T4, twice, both registered before they ran.**
- **C1:** its cache check came out exact — all 48 requests' cached tokens, 9,840 of 13,776 tokens —
  and it read the engine's 1,537 blocks correctly, in 48 seconds
  ([report](artifacts/results/c1_kaggle_t4/report.md)). That version fitted a straight line with no
  context term, which C2 then showed is up to 7× optimistic.
- **C3 (2026-10-05):** the command as it is now. It fits C2's context-aware model, and — since a
  check on C2's recorded data showed that timing cold prompts alone would have put a held-out p95
  off by 5.7–7.2× ([pre-check](artifacts/results/c3_prereg/precheck.md), post hoc) — it also times
  new tokens after a cached prefix, the case agentic traffic is made of. On the T4 it ran in 160
  seconds, was exact on the cache check again, and wrote a profile that predicted three workloads it
  had never seen within **1.41×** on p50 and p95 TTFT — one of them with prompts half as long as
  anything the model had been tested on. The profile C2 fitted five days earlier, on another software
  image, predicted the same traffic within 1.38×, from numbers written down before the run
  ([report](artifacts/results/c3_kaggle_t4/report.md)).

⚠️ **Near saturation, one fit is not enough.** A second run on the same engine, 25 minutes later,
came out 3–5% different in its coefficients. For the one workload at the knee of the load curve,
that put a p95 **4.2×** off; the other five numbers stayed within 1.35×.

**So the tool now shows how firm its own answer is.** Since 2026-10-05 `kvsim calibrate` measures
every timing cell three times and keeps, beside the fit on all of it, the fit of each repeat alone.
`kvsim replay` sizes on each of those too, and prints the result next to its recommendation:

```text
**session-sticky on 3 replica(s)** — cheapest configuration meeting the p95 TTFT SLO
⚠️ how firm: 3 to 4 replicas across the calibration's 3 repeats taken alone — size for the larger
```

That range is what one calibration's own repeats disagree by: a floor on the uncertainty, not a
confidence interval — it knows nothing about another day or another software image. A profile
without repeats (the two shipped ones) says that its spread is unknown instead of staying silent.

⛔ **How low a floor — measured, and the registered test of it failed (C4, rule E).** The command
ran three times on one live T4 in one session, three repeats each. Across 18 sizing questions and
both targets, another run's GPU count was inside a run's own range in **82% of cases**; the rule
asked for 90%. Widened by one GPU each way, 95%. The two runs taken back to back agree with each
other (97%); the one taken an hour earlier does not agree with them (75%) — and no run's repeats,
six minutes apart, can know that. So: **size for the top of the range, plus one.** The command
itself works on a real engine with its repeats — exact on its cache check all three times, six
minutes a run ([report](artifacts/results/c4_kaggle_t4/report.md)).

## ⭐ What transfers to your fleet, and what doesn't

**Read this before trusting a number.** A calibration is fitted to one GPU, one model, one block size —
so the honest answer comes in three tiers, and they must not be blended (ADR-010):

| | What you get | What it costs you |
|---|---|---|
| **Policy ranking** | *Which* policy wins on your workload shape, and roughly by how much | ⛔ **Not free.** **H6 was tested and FALSIFIED (2026-09-28):** under a 2× faster GPU, or prefill 2× cheaper or dearer, the full order of the seven policies changed (Kendall's τ-b as low as 0.51). What held — noticed after the run, not registered: session-sticky needed the fewest replicas at all 48 points. So read a ranking as a result for *our* T4 unless you calibrate |
| **Whether prefix routing pays** | Whether prefix-affinity is worth implementing for *your* prompts and load | ⚠️ **Not settled, even on our own T4.** With one shared prompt or a few: no — session-sticky needs the fewest replicas or ties on all three calibrations. With 64 prompts, the registered test said no on one calibration (2026-09-30) and yes on two others (2026-10-05): the two policies are one or two replicas apart, which is what calibrations of the same GPU disagree by. On the first, context-free timing model it paid once prompts outgrew one GPU's cache, and that flip survived four hardware rescalings (H6's crossover half) — but that model was wrong about the engine. So the answer depends on the timing model, and on its error bar: measure yours, more than once |
| **Absolute ms and ₹** | p95 TTFT and ₹/1k requests for your fleet | **`kvsim calibrate` on your own hardware**, a few minutes. ⛔ Our error bar was measured on our GPU and our model. It does **not** transfer, and this README will never imply that it does. On our own T4 the first, context-free timing model was 5–222× optimistic under load; the context-aware one predicts held-out traffic within 1.41× (C2), and again from a profile `kvsim calibrate` wrote itself (C3) — except at the knee of the load curve, where a second run of the command was 4.2× off on one p95. ⚠️ All of that is first-token latency: time per token is now measured too, and there the model is within 7% off the knee and 1.5–1.7× too slow at it (T1) |

> **So: why trust our numbers on your hardware?** You shouldn't. Run `kvsim calibrate` and use your own.
> That is the same posture as the cost model, which ships every constant in a config with a source and a
> date so you can disagree with it and re-run. **Handing the skeptic the means to disagree, twice.**

## Where it runs — nobody's GPU is involved

The GPU appears **once**, on our calibration box, offline, before anything is published. Everything
deployed is pure CPU Python:

- **PyPI** — on your machine. The real usage path.
- **A static demo page** — a precomputed grid with selectors, so you can see the recommendation change
  with the prompts and the load in ten seconds, with nothing installed. No server, so **nothing to wake up**, and every number on it comes
  from a committed script with a pinned seed. It is a **shop window, not a service**. ⛔ It deliberately
  does not accept uploaded production traces: those are your users' prompts, and they belong on your
  machine.
- **Docs** — static.

Replaying 100k requests through a dict-based trie is milliseconds of CPU work. There is nothing for a GPU
to do. That is the point of a simulator: **pay the GPU cost once, then answer "what if" thousands of times
for free.**

---

## What this is not — scope, stated up front rather than defensively later

**This niche is occupied — by more than this README said before 2026-10-05.** Three prior-art
checks missed NVIDIA's simulator and the two papers below; the fourth found them. What was missed,
and why, is in [`docs/04-prior-art.md`](docs/04-prior-art.md).

| Project | What it is, and how this differs |
|---|---|
| **NVIDIA Dynamo's simulator** — `ai-dynamo/aisimulate`, "DynoSim" (and `aiconfigurator` before it, arXiv 2601.06288) | **The closest tool, and the larger one.** Replays a saved trace — its tutorial uses the same Mooncake trace as R1 below, and it reads the AgentX traces of R2 — on simulated vLLM or SGLang workers with a block-level prefix cache; compares round-robin with Dynamo's KV-aware router; and recommends a minimum GPU count under TTFT and TPOT limits. Offline multi-worker replay since March 2026. *What I did not find in it (read, not run, 2026-10-05):* routing policies other than those two; a published check of its cache model against a live engine — its public accuracy page covers TTFT and TPOT on single-node points with prefix caching off; a cost in money. |
| **LMetric** (Zhang et al., OSDI '26, arXiv 2603.15202) · **SMetric** (Wang et al., arXiv 2607.08565) · **Preble** (Srivatsa et al., ICLR '25) | **The closest findings, measured on real clusters.** LMetric: multiply new prefill tokens by batch size and no weight needs tuning. SMetric: route agents by session, and migrate a session when its instance would miss the SLO. This repo's routing results agree with them; they are not additions to them. Neither policy is implemented here yet. |
| **`casys-kaist/LLMServingSim`** (416★, ISPASS '26) | A cycle-level simulator that mirrors vLLM's scheduler and ports its block pool, per instance; it reports TTFT and TPOT within 1% of a real vLLM run on one profile. Its router is round-robin, random or least-loaded — no prefix-aware or sticky policy. |
| **`microsoft/vidur`** (688★, MLSys'24) | Capacity and parallelism planning. Its `canary` branch **does** have a block-level prefix cache and three cache-aware routers — but it has sat unmerged since June 2025, so `main` (what you install) cannot answer the routing question. |
| **`NetX-lab/Frontier`** (155★, arXiv 2605.21312) | Prefix and hierarchical caching, SLA-aware Pareto search, and **session-sticky** routing. No prefix-affinity router: sticky is *session* identity, not *shared-prefix* identity. |
| **`azrabano23/disagg-sim`** | Pedagogy: reproducing published scheduling arguments on synthetic workloads. Its own README declines trace fitting and defers to Vidur. |
| **`jahnavi-yelamanchi/cachetrace`** | A single-path prompt-construction linter. |
| **CacheWise** (arXiv 2606.16824) · **`gauravapiscean/agentic-kv-cache`** | Prefix-aware scheduling and eviction for coding agents, and single-node eviction policies replayed on the same AgentX and Mooncake traces (which finds LRU hard to beat). Both work **inside one engine**; neither routes across replicas. |
| **vLLM · SGLang · LMCache · Mooncake · AIBrix · llm-d** | Production engines, caches and gateways: they *implement* caching and routing rather than compare policies. ⚠️ Dynamo was in this row until 2026-10-05, and it ships a simulator; the others' docs have not been re-read for one. |

**What is claimed, after that check.** Not a niche — three things this repo did and can show:

1. **A measured credential.** The cache model reproduces a live vLLM's prefix-cache counters
   exactly, and its hit rates held within 1.4% on two live replicas under three routing policies.
2. **A method.** Every test's design, rules and prediction were committed before it ran; the ones
   that failed are reported as findings.
3. **Numbers on how far simulator-based sizing can be trusted** — a second calibration of the same
   GPU, 5% different, that is 4.17× off at the knee of the load curve; replica counts firm to ± one;
   a per-token model that is right off the knee and wrong at it.

The routing result itself is an agreement with published work, reached independently on a free
Kaggle GPU.

⚠️ **The moat is execution quality — the published calibration error and honest error bars — not
the idea.** That was this section's last line before the check as well. `Mike-Ew/kv-cache-routing`
specified the same six policies and the same 2-replica validation in April 2026 and never built
it; NVIDIA built a larger one and did.

---

## What breaks at 10× and 100×

Measured 2026-09-28 by [`scripts/measure_scale.py`](scripts/measure_scale.py) — one fresh process per
case, one CPU core each, results in [`artifacts/results/scale/`](artifacts/results/scale/report.md). The
base is Phase 4's traffic (200 sessions × 4 turns = 800 requests) on 8 replicas with the calibrated T4
profile; each case was run with round-robin and with balanced prefix-affinity.

| Scale | Question | Answer |
|---|---|---|
| **10× trace length** (8,000 requests) | RSS ceiling and sweep wall-clock? | **81 MiB** peak per process. One run: 1.3 s (round-robin) to 2.0 s (balanced affinity), after 8 s to generate the trace. Nothing breaks: a default `kvsim replay` sweep (280 runs, one core) is minutes. |
| **100× trace length** (80,000 requests, 8.5 M block hashes) | What gives out first — block-hash memory, the policy inner loop, or the report renderer? | **Block-hash memory.** The trace alone holds **537 MiB** (about 64 bytes per hash, as Python ints) before the simulation starts; the simulation adds 11–35 MiB. The policy loop stays linear: 13.5 s and 21 s, about 10× the 10× case. The renderer does not grow with the trace. Two walls follow by arithmetic, not measured: the parallel experiment scripts copy the trace into every worker (11 workers ≈ 6 GiB), and a default one-core sweep is 280 × 13–21 s ≈ 1–1.6 hours. |
| **10× replicas** (80 replicas at 10× load) | Does the cache model's fidelity hold, or does eviction-order skew compound? | **Not answerable on a CPU.** In the model each replica is its own block pool, so errors cannot compound inside the simulator; whether real replicas stay matched is what Phase 6 checks, on 2 GPUs. 80 is untested. What *was* measured: affinity routing scores every replica on every request, so its run time grew 2.4× (2.0 → 4.8 s for the same 8,000 requests) while round-robin stayed flat (1.3 → 1.4 s). |

---

## Reproducibility

Every table in this README is regenerated by a script in `scripts/`. Every results file records date, git
SHA (and whether the tree was dirty), seed, hardware, actually-installed package versions, schema version,
and which hardware path produced it. Every cost constant lives in `configs/cost_model.yaml` with a source
URL and a date — **disagree with a constant, change it, re-run, get your own number.**

**Anything in `artifacts/results/` must be reproducible by running a script in `scripts/`. If you can't
delete it and get it back, it doesn't ship.**

---

## Documentation

| | |
|---|---|
| [`docs/00-START-HERE.md`](docs/00-START-HERE.md) | Start here — what this is, and the read order |
| [`docs/02-architecture-decisions.md`](docs/02-architecture-decisions.md) | ADR-001…012: why this shape, this fidelity, this hardware |
| [`docs/04-prior-art.md`](docs/04-prior-art.md) | What exists, what to cite, **and the sentences that must never be written** |
| [`docs/06-methodology.md`](docs/06-methodology.md) | The statistics protocol and the cost model |
| [`docs/09-hypotheses.md`](docs/09-hypotheses.md) | **Every hypothesis, pre-registered** — H1–H6, H2b, R1–R2, C1–C2, Phase 6 and the re-test — with dated results, falsified ones included |
| [`docs/11-where-we-are.md`](docs/11-where-we-are.md) | The whole story so far, in plain words |
| [`docs/12-story-notes.md`](docs/12-story-notes.md) | Short pieces written to be told, each with its proof linked — starting with the day the finding changed |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | Where the build actually is |

**Sibling project:** [`colpali-lite`](../colpali-economics) — the same method (cost and tail latency under a
configuration choice, bootstrap CIs, an explicit cost model, reported as ₹ per unit of useful work) applied
to late-interaction retrieval instead of LLM serving. Serving economics is systematically under-measured in
ML systems: mean latency, no cost axis, no error bars.

## License

MIT
