# 02 — Architecture decisions

> Each decision is recorded as: **what we chose · why · what we rejected and why · what would reverse it.**
> Dated, because this landscape goes stale fast — one competitor gained 53★ in 18 days while the spec was
> being written, and the previous OSS-PR target was deprecated the day before it was chosen.
>
> ADR-001 to ADR-009 were written **2026-09-20**, before any simulator code existed.

---

## ADR-001: Simulate at request-and-block granularity, not token-by-token

**Status:** Accepted · 2026-09-20

### The forcing question

A serving simulator can be built at several fidelity levels: per-token (model every forward pass), per-
iteration (model each continuous-batching step), per-request-and-block (model arrivals, admission, and
cache blocks), or per-request-only (queueing theory with a cache hit-rate multiplier).

Higher fidelity is not free. It costs wall-clock per sweep — and the sweep is policy × replicas ×
sharing-ratio × seeds, so the inner loop runs millions of times.

### The decision

**Model the engine at the granularity of requests and cache blocks, with an iteration-level scheduler
loop.** Specifically: arrivals from the trace, per-replica admission with a `max_num_seqs` bound, prefix
lookup at **block** granularity, and a prefill/decode cost function fitted to the live engine — not
derived from FLOP counts.

### Why

**The gradeable quantity decides the fidelity.** What gets checked against reality is the **prefix-cache
hit rate** and the **TTFT distribution**. Hit rate is determined entirely by *which blocks are resident
on which replica when a request arrives* — that is block-granular and routing-dependent, and modelling it
per-token adds no information. TTFT is determined by queueing plus the uncached prefill length, which the
iteration loop captures.

Fitting the cost function to the engine rather than deriving it is the same reasoning: a fitted curve with
a published residual is more honest than an analytical model with hidden assumptions, and it is the thing
the calibration run produces anyway.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **Per-token simulation** | Buys fidelity on an axis nothing checks, and makes the sweep in ADR-003's inner loop intractable. Vidur's own lesson: it predicts *deployment metrics*, not token timings. |
| **Request-level queueing only** (cache as a hit-rate multiplier) | Circular. The hit rate is the thing routing *changes*, so it cannot be an input parameter. This is the shape of the mistake that makes a routing simulator worthless. |
| **Wrapping a real engine** (`llm-d-inference-sim`-style mock) | That is a different product — a mock vLLM for testing routing *code*. We need to sweep thousands of configurations offline; standing up a server per configuration is hours, not seconds. |

### What would reverse this

The calibration residual (H4) turns out to be dominated by something *inside* an iteration that we don't
model — e.g. chunked-prefill interleaving changing effective cache lifetime. Then the iteration loop gets
more detail, still not tokens.

---

## ADR-002: The prefix cache is a hash-block trie on plain dicts, storing hashes only

**Status:** Accepted · 2026-09-20

### The decision

Per replica: a **trie keyed by block hash**, where each node is a resident block, plus an **LRU eviction
order** and optional TTL. A request's cached-prefix length is the depth of the longest matching path.
Implemented on plain `dict`s and a `deque`. **No token payloads are ever stored — only block hashes.**

This mirrors how vLLM's automatic prefix caching and SGLang's RadixAttention actually work, which matters
because the model is going to be compared against one of them by number.

### Why hashes only

It is the difference between a bounded and an unbounded memory profile. A 1M-request sweep must stay in
RAM on a 14 GB laptop; storing prompt tokens makes footprint scale with *trace content*, storing hashes
makes it scale with *distinct blocks*, which is exactly the quantity the cache is bounded by anyway.

It is also a privacy property worth stating in the README: **a trace's prompts never need to enter the
simulator's memory, and never enter its artifacts.** Traces are other people's user content.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **`networkx`** | A graph library dependency to hold a dict-of-dicts. The spec listed it as an option; a trie of ~10⁵ nodes with a hot lookup path does not want a general graph abstraction between it and the inner loop. |
| **Storing token ids per block** | Unbounded in trace content, and it puts user prompts in memory and potentially in a dump. No benefit: a hash collision is not a realistic failure mode at 64-bit width with block-level granularity. |
| **A flat `{prefix_hash: replica}` map** (the "consistent hashing only" shortcut) | Cannot express *partial* prefix reuse, which is the entire phenomenon: agentic turn N shares the first 90% of turn N+1's prompt. A flat map sees a different key and reports a miss. |

### What would reverse this

The engine we calibrate against turns out to use a fundamentally different residency rule (e.g. a global
cache shared across replicas via LMCache-style tiering). Then the *per-replica* assumption breaks, which
changes the whole routing question, not just this data structure.

---

### ⚠️ AMENDMENT 2026-09-22 — vLLM's block pool is NOT an LRU trie. Four structural gaps, from its source.

Phase 0 v2 passed cleanly — but only because its workloads never filled the cache (99,024 tokens of room,
~33,000 used). Every gap below is invisible without capacity pressure, and capacity pressure is exactly
the regime Phase 3 must calibrate and the regime where routing matters. Found by **reading vLLM 0.30.0's
block pool before designing Phase 3**, not by guessing — citations are to the `v0.30.0` tag.

| # | vLLM does | Our `BlockCache` does | Source |
|---|---|---|---|
| 1 | Frees a finished request's blocks **in reverse** — *"so that the tail blocks are freed first"* — so a prompt's **tail** is evicted before its head | Touches blocks head→tail, so under LRU the **head** is the oldest and is evicted first — destroying all downstream reuse and stranding the tail | `single_type_kv_cache_manager.py:580` |
| 2 | Only blocks with `ref_cnt == 0` sit in the free queue; a block held by a running request **cannot** be evicted | No reference counts — under pressure we can evict a block a running request is using | `block_pool.py:770–795` |
| 3 | Splits freed blocks: **no-hash blocks are reused first, LIFO** (*"for better GPU locality"*); **cached blocks FIFO** (*"for LRU eviction behavior"*) | One undifferentiated LRU | `block_pool.py:789, 792` |
| 4 | Capacity includes each request's **partial tail block and its decode blocks** — and **full decode blocks are cached too** (`num_tokens_to_cache` is capped at `request.num_tokens`, which includes output) | Counts full **prompt** blocks only; never inserts output blocks | `kv_cache_manager.py:595–599` |

**What each gap does to accuracy, as pre-registered predictions rather than conclusions** — Phase 3 measures them:

- Gaps 1 and 3 together mean vLLM keeps shared prompt **heads** alive under pressure while we throw them
  away. So under capacity pressure the current model should **UNDER-predict** hit rate — **the opposite
  direction to the one H4 originally predicted.** (See the H4 amendment.)
- Gap 2 matters under concurrency: many in-flight requests pin blocks vLLM cannot evict.
- Gap 4 makes our effective capacity too **large** (we ignore partial and decode blocks), pushing the
  other way. And on real **closed-loop** agentic traffic — where the client re-sends the model's actual
  output — vLLM will *hit* on the previous turn's generated blocks where we miss. On our open-loop
  synthetic replay the re-sent "assistant" tokens never match what the engine generated, so the engine
  misses there too; this gap is real for production traces and invisible in our calibration.

### ✅ Decision: build a faithful block-pool model, and keep the naive LRU as the ablation baseline

Not a silent fix. The Calibration Report is stronger for showing both:

> *"The first model used plain LRU and evicted prompt heads first; vLLM frees tail-first and reuses
> uncached blocks LIFO. Against the live engine, the naive model's error was X%; the block-pool model's
> was Y%."*

That is the **explained residual** the spec asks for — with a measured before/after, not
a narrative. So the block-pool model is an **option** (`eviction="vllm"`), `eviction="lru"` stays, and
Phase 3 replays every workload through both against the same engine run.

⚠️ **This is implementing the engine's documented algorithm from its source, not fitting to data.**
The distinction matters: a model tuned until it matches measurements proves nothing, while a model
that implements the engine's published policy and *then* matches is evidence. Every behaviour the new
model adopts must cite a line of vLLM's source.

---

## ADR-003: Policies are strategy objects behind one interface

**Status:** Accepted · 2026-09-20

### The decision

One interface — given the fleet state and an incoming request, return a replica index. Six
implementations, **one file each**: round-robin, least-loaded, power-of-two-choices,
consistent-hashing-on-prefix, radix-prefix-affinity, session-sticky. Every policy takes a seeded RNG for
tie-breaking; no policy may read global state.

**A new policy is a new file. It is never a core change.**

### Why

Three separate reasons, and the third is the one that is easy to miss:

1. **The sweep needs them uniform.** Policy × replicas × sharing-ratio × seeds only composes if policies
   are interchangeable objects.
2. **Seeded tie-breaking is a correctness requirement, not a nicety.** Least-loaded with ties broken by
   `min()` over a dict silently prefers low replica indices and fabricates locality that the real engine
   does not have. Every tie is broken from an explicit RNG, and the seed is committed.
3. **It makes "add a policy" a genuine good-first-issue.** The project takes contributors (see
   `CHECKPOINTS.md` Phase 7), and the two advertised first issues are *a new trace adapter* and *a new
   policy class*. Those are only honest offers if the extension point really is one file.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **An `if policy == ...` branch in the scheduler** | Makes every new policy a diff against the core loop — the opposite of a contributable surface, and it invites policies that quietly read state they shouldn't. |
| **Policies as plain functions** | They need per-instance state (round-robin's cursor, session-sticky's session→replica map, power-of-two's RNG). A closure is a class with worse discoverability. |

### What would reverse this

Nothing foreseeable. If a policy needs cross-request state the interface doesn't expose, the interface
grows a documented field — it does not grow a back door into the simulator's internals.

---

## ADR-004: Trace adapters are pluggable, and the normalized schema is versioned

**Status:** Accepted · 2026-09-20

### The decision

Every input format — vLLM JSONL, SGLang logs, OpenAI-style request logs, and the synthetic agentic
generator — is an **adapter** that produces the same `NormalizedRequest` record: arrival timestamp,
prompt block hashes (or a prefix identity), prompt/output token counts, session id, and a request id.

The schema carries an explicit **`schema_version`**, and every results file records the version it was
produced under.

### Why versioning is not bureaucracy

The Pareto curves are the product, and they are compared *across weeks*. If the schema gains a field in
Phase 4 and old runs silently lack it, two curves that look comparable aren't, and there is no way to
tell after the fact. Versioning it is what makes "re-derivable" mean something.

**Error semantics, decided here because it is a schema question:** a malformed trace line
**logs-and-continues** (with a count in the results header — a run that skipped 12% of its input must say
so); an unparseable *policy or sweep config* **fails loudly at startup**. Data is dirty and expected to
be; configuration is ours and a typo in it must never produce a plausible wrong number.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **One parser with format flags** | Format quirks leak into the core. Vidur replays its own format and that is precisely the limitation we are differentiating against. |
| **Requiring prompts in the input** | Hashes are enough (ADR-002), and demanding raw prompts means nobody can hand us a trace from a system they actually operate. The adapter hashes at the edge and the prompt never enters the simulator. |

### What would reverse this

Nothing. This is the differentiator's load-bearing component.

---

## ADR-005: ⭐ Calibration hardware — Kaggle 2x T4

**Status:** ⚠️ **Revised 2026-09-22 — Kaggle 2x T4 is the hardware; A100/H100 is not in the plan.**
The 2026-09-20 text below chose A100/H100 + MIG and is kept as the record of that decision. Read the
**final amendment** at the end of this ADR for what holds now.

### What changed

`FLAGSHIP_PROJECTS.md` plans the entire calibration around **Kaggle's dual-T4 session**, and its
2026-09-20 risk block flags — correctly — that *nobody has checked whether that is feasible*, listing
three unvalidated assumptions. An **A100/H100 is available** instead, **one GPU at a time**.

### The decision

- **Calibration (Phase 3, single replica): one A100 or H100.** Named in the README with version, flags,
  driver and compute capability.
- **Sim-to-real validation (Phase 6, two replicas): two MIG instances on the one card.** Two vLLM servers
  bound to two MIG UUIDs via `CUDA_VISIBLE_DEVICES=MIG-<uuid>`.
- **Fallback if MIG is unavailable** (not root, MIG disabled on a rented box, H100 partitioning refused):
  two vLLM servers on the same card with `--gpu-memory-utilization ~0.45` each.
- **Kaggle dual-T4 is demoted to a documented fallback**, not the plan.

### Why this is better, not merely different

| | Kaggle T4 | A100 / H100 |
|---|---|---|
| Compute capability | **7.5** (Turing) | 8.0 / 9.0 |
| bf16 | ✗ | ✓ |
| FP8 | ✗ | H100 ✓ |
| vLLM FlashAttention backend | wants SM80+; **falls back** on T4 | ✓ native |
| Model size that fits | a toy | a real one |
| Session clock | 12 h, weekly quota | none |

The spec's own risk row — *"whether V1 prefix caching behaves as documented on T4 is assumed, not
measured"* — **dissolves**, and calibrating a real model on a datacenter GPU is a materially stronger
README section than calibrating a 0.6B toy on a free T4. Readers who know the field
read the hardware line.

### ⚠️ The honest caveat, which is itself the point

MIG partitions SMs and memory; a memory-split fallback does **not** partition SMs. That matters
differently for the two quantities being measured:

- **Prefix-cache hit rate** — determined by *which blocks are resident where*. Unaffected by SM sharing.
  **Safe to report from either path.**
- **TTFT percentiles** — sensitive to SM contention between the two servers. **Not safe to report from
  the memory-split fallback**, because the contention is an artifact of the test rig, not of the routing
  policy.

So: TTFT percentiles come from the **single-replica run** or from **true MIG isolation**. Every results
file records which path produced it. Publishing a p95 from two servers fighting over one card's SMs, and
attributing the difference to routing, would be the exact kind of error this project exists to be trusted
about.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **Kaggle dual-T4 as planned** | Two genuinely unvalidated assumptions (SM75 prefix-caching behaviour; two servers + generator + scrape inside one notebook process tree under a session clock), for strictly worse hardware. Kept as a documented fallback only. |
| **Two sequential single-GPU runs, stitched** | A 2-replica routing result requires the replicas to be **concurrently** contending for the same arrival stream. Sequential runs cannot produce the queueing interaction, which is half the phenomenon. |
| **Renting a second GPU for one run** | Viable and cheap, and worth doing if MIG proves unavailable. Not the default because MIG is free and already on the card. |

### What would reverse this

MIG unavailable *and* the memory-split fallback shown to distort hit rate (not just TTFT). Then rent a
second card for the single Phase-6 run, or fall back to Kaggle dual-T4 for that run alone — and say so in
the report.

---

### ⚠️ AMENDMENT 2026-09-22 — the spike ran on Kaggle 2x T4, and that sharpened this ADR

**The T4 limitations this ADR predicted are now measured, from the engine's own log:**
*"Your device 'Tesla T4' (with compute capability 7.5) doesn't support torch.bfloat16. Falling back to
torch.float16"* · *"FA2 is only supported on devices with compute capability >= 8"* ·
*"Using TRITON_ATTN attention backend"*.

**So the hardware question splits cleanly by what is being measured:**

| Measurement | Hardware-dependent? | Where it can run |
|---|---|---|
| Cache accounting, hit rates, the partial-block rule | **No** | anywhere — T4 results stand |
| 2-replica **routing** behaviour (hit rate, prefill computed) | **No** | anywhere — and dual-T4 is ideal, see below |
| **TTFT and the prefill/decode cost coefficients** | **Yes** — attention kernel + dtype | A100/H100 only. T4's TRITON_ATTN + fp16 path is not how datacenter GPUs serve |

**⭐ A fourth 2-replica path this ADR missed: separate physical GPUs.** Kaggle's box has two T4s. One
server per GPU (`CUDA_VISIBLE_DEVICES=0` / `=1`) gives **complete** isolation — separate SMs, memory,
and PCIe — which is *stronger* than MIG. v1 of the notebook only knew MIG and memory-split, so it put
both servers on GPU 0 and never used the second card. v2 detects physical GPUs first.

Revised order of preference: **separate GPUs > MIG > memory-split**. The first two permit TTFT
percentiles; memory-split still does not.

---

### ✅ FINAL AMENDMENT 2026-09-22 — Kaggle 2x T4 is THE hardware. This supersedes the text above.

A100/H100 access does not exist and will not be in the plan. That is **sufficient** for this project,
for a reason that is structural rather than a consolation:

**The credential is hardware-independent.** The Calibration Report's headline number is the cache
model's error against the engine's **hit rate** and **prefill tokens computed**. Those are decided by
which blocks are resident on which replica — identical logic on a T4 and an H100. Phase 0 proved it on
T4: the corrected model reproduces vLLM's counters to the token.

| | On Kaggle 2x T4 |
|---|---|
| Cache accounting, hit rate, partial-block rule | ✅ fully valid — **measured 2026-09-21** |
| Eviction behaviour (H4) | ✅ valid — and **easier**: a 15 GiB card evicts under modest traffic, an 80 GiB one would need an enormous workload |
| 2-replica routing (Phase 6) | ✅ **better** — two physical GPUs isolate completely, stronger than MIG |
| TTFT, prefill/decode coefficients, ₹ | ⚠️ **T4 figures.** TRITON_ATTN + fp16 is not how a datacenter GPU serves. Published labelled as T4, priced at a **T4 instance's** rate |
| H6 (does the ranking survive a hardware change?) | tested by rescaling the cost function, as already planned — CPU-only |

**What changes:**

1. **Phase 3 model: `Qwen/Qwen3-4B`, not 8B.** Measured: 8B's fp16 weights are 15.26 GiB against 13.5 GiB
   usable on a T4. 4B is 7.49 GiB, leaving 6 GiB of cache — and its KV cache is **architecturally
   identical** to 8B's (36 layers, 8 KV heads, head_dim 128, 144 KiB/token). A cache-model calibration
   calibrates the cache; the FFN width is irrelevant to it.
   *(Rejected: tensor-parallel 8B across both T4s. It fits, but turns two GPUs into ONE replica, which
   destroys Phase 6.)*
2. **Two replicas = one server per physical GPU.** MIG does not exist on T4, and is not needed.
3. **The claim.** *"Calibrated against live vLLM 0.29.0 on T4"* — the engine and version still go in the
   one-line description. What is given up is the datacenter-hardware line this ADR was originally chasing; what
   is kept is **"ran the real engine and published the error"**, which was always the actual point.
   Nothing ADR-010 permits claiming is lost: it never allowed datacenter latency without datacenter
   calibration.

**What would reverse it:** A100/H100 time appearing. It would then be a **bonus**, not a requirement — a
second real profile, and a T4-vs-A100 comparison would turn H6 from a rescaling exercise into a direct
measurement.

---

## ADR-006: Every cost constant lives in a committed, cited, dated config

**Status:** Accepted · 2026-09-20

### The decision

`configs/cost_model.yaml` holds every constant, each with a `source:` URL and a `date:`. **₹ per 1k
successful requests** is the number this project owns. No hidden divisors, no `≈₹0.003/request` with no
derivation.

A skeptical reader must be able to **disagree with a constant, change it, re-run, and get their own
number.** That property is the point; the specific value is not.

### Why it needs the care

Cost is the easiest thing in the project to get caught on. The calibration box is not a metered cloud
instance, so any published figure is necessarily a **reference-price** calculation. That is fine and
normal — *if* the basis is explicit. If it is hidden, the number is worthless and a reviewer will say so
in public.

Now that the hardware is an A100/H100 rather than a free T4, the reference SKU is a real, nameable cloud
instance at a real published hourly price, which makes the basis stronger than it would have been.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **Constants inline in Python** | Invisible to a reader, and a changed constant leaves no trace in the diff of a results table. |
| **$ only, no ₹** | The ₹ framing is deliberate — see `FLAGSHIP_PROJECTS.md`. Both are reported; the FX rate is a cited, dated constant like any other. |
| **Cost per GPU-hour as the headline** | Not actionable. Cost per unit of *useful work* is; that is why the denominator is successful requests. |

### What would reverse this

Nothing. This is a methodology invariant shared with the sibling project.

---

## ADR-007: Do NOT extract a shared harness with `colpali-lite` yet

**Status:** Accepted · 2026-09-20

### The context

`kv-routing-sim` and `colpali-lite` are deliberately **one method instantiated twice**: cost and tail
latency under a configuration choice, multi-seed with bootstrap CIs, an explicit cost model with no
hidden constants, reported as ₹ per unit of useful work. They should ship as one position, with
cross-linked READMEs and one write-up.

### The decision

**Write this project's own bootstrap/cost/latency code. Extract a shared package only when
`colpali-lite` actually needs it** — which is after this project ships.

### Why

A shared abstraction built up front blocks **two critical paths at once**, and this project ships before
the other resumes. The presentation-level position ("one method, two instantiations") costs nothing and
requires no shared code; the shared *package* is a refactor that can happen at any time, or never.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **Build `serving-econ-core` first** | Blocks both builds on an abstraction designed before either use case is fully known. Classic premature-abstraction failure, and the spec explicitly guards against it. |
| **Copy-paste and never reconcile** | Acceptable actually — two ~200-line implementations that agree is not a crisis. Reconcile if and when a third instantiation appears. |

### What would reverse this

`colpali-lite` resuming and needing the identical bootstrap code. Then extract — with this project's
version as the reference implementation, because it will have been exercised first.

---

## ADR-008: ⛔ Metric names — the hit rate is DERIVED from two counters

**Status:** Accepted · 2026-09-20

### The decision

The prefix-cache hit rate is computed as:

```
vllm:prefix_cache_hits ÷ vllm:prefix_cache_queries
```

**`vllm:prefix_cache_hit_rate` DOES NOT EXIST.** Verified against `vllm/v1/metrics/loggers.py` at
**v0.29.0** (released 2026-09-09): no Prometheus metric has `hit_rate` in its name. `hit_rate` survives
only as an internal Python property feeding a log line. The old V0 gauge
`vllm:gpu_prefix_cache_hit_rate` went away with V0.

Metrics consumed, by exact name — these are **the names a reader will search for**:

| Metric | Used for |
|---|---|
| `vllm:prefix_cache_queries` | denominator of the hit rate |
| `vllm:prefix_cache_hits` | numerator |
| `vllm:time_to_first_token_seconds` | TTFT distribution |
| `vllm:num_requests_waiting` | queue depth, for the queueing model |
| `vllm:request_prefill_kv_computed_tokens` | uncached prefill work actually done |
| `vllm:kv_block_lifetime_seconds` | ⭐ eviction behaviour (H4) |
| `vllm:kv_block_idle_before_evict_seconds` | ⭐ eviction behaviour (H4) |
| `vllm:kv_block_reuse_gap_seconds` | ⭐ eviction behaviour (H4) |

The last three are **gated behind `--kv-cache-metrics-sample`** and must be confirmed present on the
version that actually installs.

### Why this gets its own ADR

Because it is the difference between a report that reads as *"he ran the engine"* and one that reads as
*"he read the docs."*

**The cautionary tale is a competitor.** `Mike-Ew/kv-cache-routing`'s README cites
`vllm:prefix_cache_hit_rate` — the V0-era gauge that no longer exists. That single name is the tell that
they planned against documentation and never started a server. A vLLM-literate reader reads a
nonexistent metric name and concludes exactly that.

Saying *"the hit rate is a ratio of two counters"* out loud in the README is itself evidence that a server was started.

⭐ **And the three `vllm:kv_block_*` histograms are not decoration.** They measure eviction behaviour
**directly** — which is precisely the residual the Calibration Report promises to explain. Without them,
*"the remaining N% is eviction-order skew"* is a narrative. With them it is a measurement. That is the
difference between a careful answer and a plausible one, and it is why H4 is testable at all.

### What would reverse this

The installed vLLM version differs from 0.29.0 and names things differently. **Then re-read ITS
`loggers.py` and update this ADR before trusting any number** — a Phase-0 checkbox, not a formality.

---

### ⚠️ AMENDMENT 2026-09-22 — the live engine corrected this ADR in two places

The Phase-0 spike ran (vLLM **0.29.0** — the pinned version installed exactly — Qwen/Qwen3-0.6B,
Kaggle 2x T4). Evidence: `artifacts/results/phase0_kaggle_t4/`. **The core of this ADR held**: every
metric name we planned against exists, and `vllm:prefix_cache_hit_rate` is correctly absent. Two
details did not:

**1. ⛔ The counters count TOKENS, not blocks — and the denominator includes the partial block.**
Settled two independent ways:

- vLLM's own HELP strings: *"Prefix cache queries, **in terms of number of queried tokens**"* and
  *"Prefix cache hits, **in terms of number of cached tokens**."*
- Empirically: 512 tokens on a cold cache moved `prefix_cache_queries` by **512**, not 32.

And the full scrape reconciles to the token: 1 cold probe of 512 + 64 requests of 513 gives
`queries = 33,344` (= all prompt tokens), `hits = 32,768` (= 64 x 32 full blocks x 16), and
`computed = 576`. So **`queries` counts every prompt token, including the partial trailing block,
while `hits` can only ever be full blocks.** Our cache model counted full blocks on both sides, so it
was biased upward — 0.2% on these 513-token prompts, **20% on a 20-token prompt**. Fixed in
`cache/block_trie.py`; four regression tests now pin the model to the engine's measured counters.

**2. ⛔ `--kv-cache-metrics-sample` does NOT enable the `kv_block_*` histograms.** The engine's own
config dump: `ObservabilityConfig(... kv_cache_metrics=False, kv_cache_metrics_sample=0.01 ...)`.
The **enable** switch is **`--kv-cache-metrics`**; the sample flag only sets the rate. This ADR and
`FLAGSHIP_PROJECTS.md` both said the sample flag enables them — wrong, and it is why v1 of the spike
collected no histograms. H4 is **not dead**; it was never tested (see the H4 amendment).

**3. ⭐ Metrics we did not know about, and they are better for calibration:**

| Metric | Why it is better |
|---|---|
| `vllm:prompt_tokens_by_source{source="local_compute"}` | Prefill tokens **actually computed** — the quantity that drives TTFT and cost. Defined **even with caching off**, where a hit *rate* is undefined |
| `vllm:prompt_tokens_by_source{source="local_cache_hit"}` | Equals `prefix_cache_hits`; a useful cross-check |
| `vllm:prompt_tokens_by_source{source="external_kv_transfer"}` | Cross-instance KV transfer (LMCache-style). Zero here; relevant if ADR-002's per-replica assumption is ever relaxed |
| `vllm:prompt_tokens_cached_total` | Cached prompt tokens, as a plain counter |

`local_compute` is now the primary calibration target alongside the hit rate: it is what the
caching-off comparison needs (v1's notebook compared an undefined hit rate there and wrongly reported
that caching does not help — the real figure is **513x less prefill**).

**3b. ✅ Re-confirmed on vLLM 0.30.0 (v2, 2026-09-22).** Every name present, `prefix_cache_hit_rate`
still absent, and the counters reproduced v1's exactly across the version change. ⚠️ And the three
`kv_block_*` histograms — now enabled with `--kv-cache-metrics` — record every **recycled** tracked block,
not only real evictions: `on_block_evicted` runs before the hash check (`block_pool.py:739` vs `741`).
See the H4 amendment and the upstream log #1.

**4. ✅ Our KV-size arithmetic matches the engine.** vLLM reported *"4.38 GiB KV cache is needed"* for
40,960 tokens = 114,819 B/token; `cache/capacity.py` computes 114,688. The 0.11% gap is vLLM's own
rounding of "4.38".

---

## ADR-009: `vllm` is an optional extra, never a core dependency

**Status:** Accepted · 2026-09-20

### The decision

`pip install kv-routing-sim` installs `numpy`, `pyyaml`, `pydantic`, `matplotlib` — nothing else. The
engine lives in `[calib]`, installed only on the calibration box. The `calib/` module imports `vllm`
and `httpx` lazily, inside functions, and fails with a message naming the extra.

### Why

The user-facing promise is *no GPU dependency for users; GPU only for your own calibration runs.* A
simulator that drags in vLLM — and therefore torch, and therefore multi-gigabyte CUDA wheels — to answer
a CPU question has broken its own pitch before the user reaches the README's second paragraph. It also
makes the static-page deployment possible at all — a precomputed grid is only cheap to build because
the simulator needs no GPU (and see ADR-011: the free *Gradio* Space this originally referred to no
longer exists).

There is a second reason, learned from the sibling project: `colpali-lite`'s `[serving]` extra turned out
**not to install at all** (`ResolutionImpossible` between `optimum` and `transformers`). Keeping the heavy
extra strictly separate means that kind of discovery can never break the core install or the Space.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **vLLM as a core dependency** | Breaks the CPU-only install, the static demo build, and the pitch. |
| **A separate `kv-routing-sim-calib` package** | Two packages, two release processes, for one module. The extra is the right size for this. |
| **Shelling out to a vLLM CLI instead of importing** | Actually fine for *starting servers* (and the spike scripts do exactly that), but `calib/` also parses the exposition format, which wants `prometheus-client`. Both stay in the extra. |

### What would reverse this

Nothing foreseeable.

---

## ADR-010: ⭐ What a calibration transfers — and what it does not

**Status:** Accepted · 2026-09-20 *(added after a "who can actually use this?" review of the scaffold)*

### The problem this exists to fix

A calibration is **fitted to one specific setup**: one GPU, one model, one block size, one set of flags.
The scaffold's README claimed *"your real serving log in → a recommendation out"* and separately claimed
*"matches the live engine within X%"* — and silently invited the reader to combine them.

**Those two claims do not compose.** A user on H100s serving Llama-70B gets nothing from an error bar
measured on an A100 serving a small instruct model. Left unfixed, the first knowledgeable reader to ask
*"transfers to my fleet how?"* collapses the headline, and they will ask.

### The decision — three tiers, stated separately, never blended

| Tier | What is claimed | What it needs |
|---|---|---|
| **1. Policy ranking** | *"Prefix-affinity beats least-loaded on your workload shape"* — the **ordering** of policies, and roughly how big the gap is | **Nothing.** No calibration, no GPU. Depends on workload shape, not hardware. ⚠️ **This is a hypothesis (H6), not an assumption** — it is the load-bearing claim for every user who never calibrates, so it gets tested, not asserted. |
| **2. The crossover boundary** | *"At your sharing ratio and utilization, prefix-affinity stops paying"* — the finding | Nothing, **if H6 holds**. It is a statement about workload geometry. The *position* of the boundary may shift with hardware; H6's test is what says how much. |
| **3. Absolute numbers** | *"Your p95 TTFT will be 84 ms and it will cost ₹Y per 1k requests"* | **A calibration on the user's own hardware and model.** Ours does not transfer, and the README must not imply it does. |

**Tiers 1 and 2 are the product. Tier 3 is a feature you opt into.**

### The three mechanisms that implement this

1. **Say it in the README, in the scope section.** Absolute numbers belong to the calibrated setup;
   what a user gets for free is the ranking and the boundary. ⛔ No sentence may blend a tier-3 error
   bar with a tier-1 claim.
2. **Ship calibration profiles.** A profile is keyed by `(gpu, model, block_size, max_num_seqs,
   engine_version)` and committed under `configs/profiles/`. A user picks the nearest and is told, in
   the output, **how far from it they are** and that error grows with that distance. ⛔ A profile that
   does not record its provenance is worse than no profile.
3. **⭐ Ship `kvsim calibrate`.** The user points it at their own running vLLM for ~30 minutes and gets
   their own profile. This is the important one: it converts the calibration from **a credential we own**
   into **a capability the user has**, and it is what makes *"your log, your fleet"* literally true
   instead of aspirational.

### Why `kvsim calibrate` is worth the build cost

It is the same code path as our own Phase-3 calibration, exposed with a CLI in front. Building it costs
little *because the calibration harness has to exist anyway* — and it changes the project's answer to
the hardest question anyone can ask about a simulator:

> *"Why should I trust your numbers on my hardware?"*
> → **"You shouldn't. Run `kvsim calibrate` and use your own."**

That answer is stronger than any error bar we could publish, and it is the same move ADR-006 makes on
cost constants: **hand the skeptic the means to disagree.** Consistent posture, twice.

⚠️ `vllm` stays in the `[calib]` extra (ADR-009), so `kvsim calibrate` is available only to users who
opt into the heavier install. The CPU-only core is unaffected.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **Claim the calibration transfers** | False, and checkable in one question. It would make the credential the weakest part of the project rather than the strongest. |
| **Publish only tier 3, and tell everyone else to calibrate first** | Kills the tool for everyone who has not yet committed a GPU-hour — i.e. the exact audience deciding *whether* to invest in cache-aware routing. That decision is the main use case. |
| **Drop absolute numbers entirely, ship only rankings** | Throws away the credential, which is the part that proves rigor. And ₹/1k requests is the number the project set out to own. |
| **Auto-detect the user's hardware and interpolate between profiles** | Sounds helpful, is fabrication: an interpolated error bar has no measured basis. A nearest-profile lookup that *states its distance* is honest; interpolation dressed as prediction is not. |

### What would reverse this

**H6 failing.** If policy *ranking* turns out to be hardware-dependent — e.g. the crossover moves so much
between an A100 and an H100 that the recommended policy flips — then tiers 1 and 2 collapse into tier 3,
`kvsim calibrate` stops being optional and becomes **mandatory**, and the project is a
calibrate-then-decide tool rather than a decide-now tool. That is a materially smaller product, and H6 is
how we find out before the README claims otherwise.

---

## ADR-011: ⭐ The demo is a STATIC page with a precomputed grid

**Status:** Accepted · 2026-09-21 · **Supersedes the "free CPU Gradio Space" assumption in ADR-009,
`07-environment.md` and the deployment notes**

### What was wrong

Every deployment note in this repo assumed a **free Gradio Space on CPU Basic (2 vCPU / 16 GB)**. That
is no longer available. Verified against Hugging Face's own docs, 2026-09-21:

> *"Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and **require a paid
> plan to create**: PRO for personal accounts, Team or Enterprise for organizations. Free personal
> accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."*

So Phase 7's demo was **undeliverable as written**. Note the gate precisely, because it is easy to
misread: CPU Basic still costs nothing per hour — what requires PRO ($9/month personal) is *creating a
compute-backed Space at all*.

And the free exception does not rescue it: the ZeroGPU route is **GPU-backed with a small daily
quota**. Pointing a CPU-only simulator at a rationed GPU would exhaust the quota on a handful of
visitors, and a reader who noticed would conclude we do not understand our own workload.

Render was already cut; the reasons are now the measured ones rather than half-remembered. Per Render's
docs: a free web service **spins down after 15 minutes** without inbound traffic, takes **about one
minute** to wake, and the 750 instance-hours are a **monthly cap, not always-on**. Keeping it awake by
pinging is against the spirit of the free tier, so that is not an option either.

### The decision

**Ship a static page driven by a precomputed grid**, published to **GitHub Pages** and (the same build)
to a **free Static Space**. One artifact, two URLs, no compute, no rent, nothing to fall asleep.

`scripts/build_grid.py` sweeps sharing × load × replicas × policy, writes one JSON file, and the page
renders it with sliders. The interaction the demo exists for — *drag prefix-sharing down and watch the
recommendation flip* — works identically against a grid.

### ⭐ Why this is the RIGHT answer, not merely the affordable one

The cost was never the deciding argument. Three things are:

1. **It fixes the defect we had already identified and could not solve.** the deployment notes
   flagged that a sleeping Space makes a cold visitor wait 30–60 s — "3–6× longer than their attention
   lasts" — and the mitigation was a hedge (put the images in the README too). A static page **never
   sleeps**. The problem disappears rather than being worked around. Note that PRO would *not* have
   fixed this: a paid Space still sleeps after 48 h idle.
2. **A precomputed grid is a reproducibility artifact; a live app is not.** Every number on the page
   comes from a committed script, a committed config and a pinned seed — auditable exactly like
   everything in `artifacts/results/`. A Gradio app computing on demand is *less* falsifiable, because
   nobody can check what it computed. The apparent downside ("visitors can only pick grid points")
   inverts into the thing this project claims to care about.
3. **It matches what we already said the demo is for.** This repo's own position is that the Space is
   **distribution, not a product surface**, and that anyone with a real serving log should run the CLI
   locally. Paying monthly rent so strangers can run arbitrary scenarios buys capability for a job
   nobody asked the page to do.

### ✅ The grid, MEASURED — 2026-09-21, not estimated

`scripts/build_grid.py` has run. The estimates in the first draft of this ADR were wrong by an order of
magnitude, in the favourable direction:

| | Estimated | **Measured** |
|---|---|---|
| Grid | 21 × 12 × 8 × 7 | **19 sharing × 8 rates × 8 replicas × 7 policies** |
| Cells | ~14,000 | **8,512** (from 25,536 simulator runs over 152 traces) |
| File, raw | "a few MB" | **327 KiB** |
| File, gzipped | — | ⭐ **79 KiB** |
| Build time | ~8 min on 12 cores | **291 s (4.8 min)**, `joblib` across all cores |

⭐ **79 KiB is the number that settles the argument.** The *entire answer space* — every policy at every
replica count across the full sharing range, measured cross-session sharing spanning 0.000 → 0.900 —
ships in less than the weight of one photograph, on a host that never sleeps, for nothing. There is no
version of a live server that improves on that for this page's job.

The committed grid (`artifacts/results/demo_grid.json`) carries its own provenance header, axes, and
`calibrated: false`, so the page cannot present uncalibrated millisecond figures as predictions.

### Rejected alternatives

| Rejected | Why |
|---|---|
| **PRO Gradio Space, $9/month** | Buys live arbitrary scenarios — a capability for a job the page does not have. Still sleeps (48 h). Recurring rent for a shop window, and the first thing to lapse when attention moves on. Revisit only if a real user asks for scenarios the grid cannot express. |
| **Render free** | 15-minute spin-down, ~60 s wake, 512 MB / 0.1 CPU. Strictly worse than a static host for a static payload, on every axis including honesty. |
| **The free ZeroGPU Gradio exception** | Rationed GPU for a workload with no GPU in it. Quota dies on a few visitors, and it advertises a misunderstanding of our own project. |
| **Pyodide — run the real simulator client-side** | Genuinely attractive: free, static, *and* arbitrary scenarios. But ~10 MB of runtime download, and `pydantic-core` is a compiled Rust extension that is not a given under Pyodide. A real project for a small increment over the grid. ⭐ **Keep as the upgrade path** if someone actually wants arbitrary scenarios. |
| ⛔ **Re-implement the cache model and policies in JavaScript** | **The worst option, and worth stating so a future pass does not revisit it.** It would mean two implementations of the exact thing this project's credibility rests on being *measured for fidelity* — and they would drift, silently, with no test able to see it. Shipping a second unvalidated copy of the component under audit is self-defeating. |

### What would reverse this

A paid plan already being bought for another reason, **plus** a concrete request for a scenario the grid
cannot express. Both, not either.
