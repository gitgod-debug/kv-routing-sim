# kv-routing-sim

**Which KV-cache routing policy should you actually run — and on how many replicas?**

Give it your serving log. It returns a routing policy, a replica count, and a cost number. Its
prefix-cache model is calibrated against a live vLLM's own Prometheus counters, and **the calibration error
is published**.

> ## 🚧 Status: Phase 0 — scaffold. Nothing here is measured yet.
> Every number below is written as `TBD — not yet measured` and **stays that way until it is measured.** No
> placeholder in this file will ever be quietly replaced by a plausible figure; see the honesty gates in
> [`CHECKPOINTS.md`](CHECKPOINTS.md).
> Start at [`docs/00-START-HERE.md`](docs/00-START-HERE.md).

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
| **Engine + version** | TBD — not yet measured |
| **Flags** | `--enable-prefix-caching --kv-cache-metrics-sample <rate> --block-size 16 --max-num-seqs <N>` *(exact CLI: TBD)* |
| **Model + quantization** | TBD |
| **Hardware** | **Kaggle 2x Tesla T4** (compute 7.5, driver 580.159.04, CUDA 12.8). vLLM runs TRITON_ATTN in fp16 here — so hit rates and routing results are hardware-independent, and every latency/cost figure is a **T4 figure** |
| **Hit-rate error vs live engine** | TBD — not yet measured, on ≥2 workload shapes, ≥3 live runs each |
| **The residual, explained** | TBD — measured against the `vllm:kv_block_*` histograms, not narrated |
| **2-replica validation** | TBD — two MIG instances on one card |

**`/metrics` fields consumed, by name:** `vllm:prefix_cache_queries` · `vllm:prefix_cache_hits` ·
`vllm:time_to_first_token_seconds` · `vllm:num_requests_waiting` ·
`vllm:request_prefill_kv_computed_tokens` · `vllm:kv_block_lifetime_seconds` ·
`vllm:kv_block_idle_before_evict_seconds` · `vllm:kv_block_reuse_gap_seconds`.

> ⛔ **The hit rate is derived — `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`, two counters.**
> `vllm:prefix_cache_hit_rate` **does not exist** (verified against `v1/metrics/loggers.py` at v0.29.0); the
> old V0 gauge went away with V0.

---

## What it does

1. **Trace ingestion** — reads real serving logs (vLLM/SGLang JSONL, OpenAI-style request logs) into one
   normalized, versioned request schema. Prompts are hashed at the adapter boundary and **never enter the
   simulator**, so a trace's user content stays out of memory and out of artifacts.
2. **A block-level KV cache model** — hash-block trie with LRU/TTL eviction, per replica, at
   vLLM-APC/RadixAttention granularity. *This is table stakes, not the contribution — the contribution is
   its measured fidelity.*
3. **Six routing policies** — round-robin · least-loaded · power-of-two-choices ·
   consistent-hashing-on-prefix · radix-prefix-affinity · session-sticky. One file each; a new policy is a
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

The inference engine is an optional extra (`pip install "kv-routing-sim[calib]"`), needed only to
**calibrate against your own hardware**:

```bash
kvsim calibrate --engine http://localhost:8000 --out configs/profiles/my-fleet.yaml   # ~30 min, your GPU
```

## ⭐ What transfers to your fleet, and what doesn't

**Read this before trusting a number.** A calibration is fitted to one GPU, one model, one block size —
so the honest answer comes in three tiers, and they must not be blended (ADR-010):

| | What you get | What it costs you |
|---|---|---|
| **Policy ranking** | *Which* policy wins on your workload shape, and roughly by how much | **Nothing.** No GPU, no calibration. Depends on workload shape, not hardware — *pending* **H6**, which tests exactly that rather than assuming it |
| **The crossover boundary** | Whether prefix-affinity is worth implementing at *your* sharing ratio and utilization | **Nothing** (same caveat) |
| **Absolute ms and ₹** | p95 TTFT and ₹/1k requests for your fleet | **`kvsim calibrate` on your own hardware.** ⛔ Our error bar was measured on our GPU and our model. It does **not** transfer, and this README will never imply that it does |

> **So: why trust our numbers on your hardware?** You shouldn't. Run `kvsim calibrate` and use your own.
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
| [`docs/02-architecture-decisions.md`](docs/02-architecture-decisions.md) | ADR-001…009: why this shape, this fidelity, this hardware |
| [`docs/04-prior-art.md`](docs/04-prior-art.md) | What exists, what to cite, **and the sentences that must never be written** |
| [`docs/06-methodology.md`](docs/06-methodology.md) | The statistics protocol and the cost model |
| [`docs/09-hypotheses.md`](docs/09-hypotheses.md) | **H1–H5, pre-registered before any code existed** |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | Where the build actually is |

**Sibling project:** [`colpali-lite`](../colpali-economics) — the same method (cost and tail latency under a
configuration choice, bootstrap CIs, an explicit cost model, reported as ₹ per unit of useful work) applied
to late-interaction retrieval instead of LLM serving. Serving economics is systematically under-measured in
ML systems: mean latency, no cost axis, no error bars.

## License

MIT
