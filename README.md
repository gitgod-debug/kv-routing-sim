# kv-routing-sim

**Which KV-cache routing policy should you actually run — and on how many replicas?**

Give it your serving log. It returns a routing policy, a replica count, and a cost number. Its
prefix-cache model is calibrated against a live vLLM's own Prometheus counters, and **the calibration error
is published**.

> ## 🚧 Status: the calibration is measured (Phase 3, 2026-09-22). The finding is not yet.
> The Real-Engine Calibration Report below is filled in from a live vLLM, against predictions committed
> before the run. Every number still written as `TBD — not yet measured` **is exactly that**, and stays
> so until it is measured — no placeholder here is ever quietly replaced by a plausible figure; see the
> honesty gates in [`CHECKPOINTS.md`](CHECKPOINTS.md). Start at [`docs/00-START-HERE.md`](docs/00-START-HERE.md).

---

## The finding

> **TBD — not yet measured.** *(One sentence, stating where prefix-affinity routing stops paying: the
> prefix-sharing ratio × offered-load boundary at which least-loaded overtakes it on p95 TTFT. See
> hypothesis H2.)*

`azrabano23/disagg-sim` has published *that* this locality-vs-load counter-result exists. **Nobody has
published the boundary.**

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
| **2-replica validation** | Round-robin across **two physical T4s**: 1,088 prefill tokens computed, **1,088 predicted** (Phase 0 v2). Under pressure: Phase 6 |

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

To use measured timings instead of the built-in guess, pass a profile. The one measured so far is
ours — Tesla T4, Qwen3-4B, vLLM 0.30.0:

```bash
kvsim replay trace.jsonl --policies all --replicas 1-8 \
  --profile configs/profiles/t4-qwen3-4b-vllm0.30.0.json --cost-model configs/cost_model.yaml
```

⚠️ **Not built yet:** `kvsim calibrate`, which will fit a profile to *your* GPU and model (the
`[calib]` extra). Until it exists, every millisecond and rupee figure here is a T4 figure.

## ⭐ What transfers to your fleet, and what doesn't

**Read this before trusting a number.** A calibration is fitted to one GPU, one model, one block size —
so the honest answer comes in three tiers, and they must not be blended (ADR-010):

| | What you get | What it costs you |
|---|---|---|
| **Policy ranking** | *Which* policy wins on your workload shape, and roughly by how much | **Nothing.** No GPU, no calibration. Depends on workload shape, not hardware — *pending* **H6**, which tests exactly that rather than assuming it |
| **The crossover boundary** | Whether prefix-affinity is worth implementing at *your* sharing ratio and utilization | **Nothing** (same caveat) |
| **Absolute ms and ₹** | p95 TTFT and ₹/1k requests for your fleet | **`kvsim calibrate` on your own hardware** *(not built yet)*. ⛔ Our error bar was measured on our GPU and our model. It does **not** transfer, and this README will never imply that it does |

> **So: why trust our numbers on your hardware?** You shouldn't. Once `kvsim calibrate` is built, run it and use your own.
> That is the same posture as the cost model, which ships every constant in a config with a source and a
> date so you can disagree with it and re-run. **Handing the skeptic the means to disagree, twice.**

## Where it runs — nobody's GPU is involved

The GPU appears **once**, on our calibration box, offline, before anything is published. Everything
deployed is pure CPU Python:

- **PyPI** — on your machine. The real usage path.
- **A static demo page** — a precomputed grid with sliders, so you can see the recommendation flip in
  ten seconds with nothing installed. No server, so **nothing to wake up**, and every number on it comes
  from a committed script with a pinned seed. It is a **shop window, not a service**. ⛔ It deliberately
  does not accept uploaded production traces: those are your users' prompts, and they belong on your
  machine.
- **Docs** — static.

Replaying 100k requests through a dict-based trie is milliseconds of CPU work. There is nothing for a GPU
to do. That is the point of a simulator: **pay the GPU cost once, then answer "what if" thousands of times
for free.**

---

## What this is not — scope, stated up front rather than defensively later

**This niche is occupied. The claim here is an intersection, not an empty field.**

| Project | What it is, and how this differs |
|---|---|
| **`microsoft/vidur`** (676★, MLSys'24) | Capacity and parallelism planning. Its `canary` branch **does** have a block-level prefix cache and three cache-aware routers — but it has sat unmerged for 14 months, so `main` (what you install) cannot answer the routing question. It also replays its own trace format, emits no recommendation, and publishes no calibration of that cache model against a live engine. |
| **`NetX-lab/Frontier`** (139★, arXiv 2605.21312) | The closest competitor: prefix + hierarchical caching, SLA-aware Pareto search, and **session-sticky** routing. It has no prefix-affinity router — sticky is *session* identity, not *shared-prefix* identity, so cross-user system-prompt sharing is invisible to it — no adapters for real serving logs, and no recommendation. A research instrument you configure, not a tool that reads your log. |
| **`azrabano23/disagg-sim`** | Pedagogy: reproducing published scheduling arguments on synthetic workloads. Its own README declines trace fitting and defers to Vidur. |
| **`jahnavi-yelamanchi/cachetrace`** | A single-path prompt-construction linter. This is multi-replica policy choice *and* sizing. |
| **vLLM · SGLang · LMCache · Dynamo · Mooncake · AIBrix** | Production engines that *implement* a routing policy. This helps you *choose* one before committing infrastructure. |

**The surviving claim, stated precisely:** trace adapters for real serving logs **+** a calibrated
prefix-cache model **+** policy choice *and* replica sizing **+** ₹/$ per 1k requests **+** sim-to-real
validation. Four repos each own one or two. None owns the set.

⚠️ **That is a narrow gap, not a category** — and `Mike-Ew/kv-cache-routing` specified the same six policies
and the same 2-replica validation in April 2026, then never built it. **The moat is execution quality — the
published calibration error and honest error bars — not the idea.**

---

## What breaks at 10× and 100×

*(Answered with measured numbers in Phase 7 — not with prose. Named here now so it cannot be quietly
skipped.)*

| Scale | Question | Answer |
|---|---|---|
| **10× trace length** | RSS ceiling and sweep wall-clock? | TBD — not yet measured |
| **100× trace length** | What gives out first — block-hash memory, the policy inner loop, or the report renderer? | TBD — not yet measured |
| **10× replicas** | Does the cache model's fidelity hold, or does eviction-order skew compound? | TBD — not yet measured |

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
| [`docs/09-hypotheses.md`](docs/09-hypotheses.md) | **H1–H6, pre-registered before any code existed** — with dated results |
| [`docs/11-where-we-are.md`](docs/11-where-we-are.md) | The whole story so far, in plain words |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | Where the build actually is |

**Sibling project:** [`colpali-lite`](../colpali-economics) — the same method (cost and tail latency under a
configuration choice, bootstrap CIs, an explicit cost model, reported as ₹ per unit of useful work) applied
to late-interaction retrieval instead of LLM serving. Serving economics is systematically under-measured in
ML systems: mean latency, no cost axis, no error bars.

## License

MIT
