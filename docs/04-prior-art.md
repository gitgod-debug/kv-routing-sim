# 04 — Prior art: what exists, what to cite, what never to claim

> ⛔ **Read this before writing any README claim, summary of the project or Show HN comment.**
> There is a list of specific sentences below that are **false**, that were written down in good faith by
> a previous audit, and that a knowledgeable reader will catch instantly.
>
> Evidence re-run **2026-08-25**, re-verified **2026-09-12**, ported into this repo **2026-09-20**.
> ⚠️ **Re-audit monthly, not at kickoff** — one competitor went 86★ → 139★ in 18 days.
>
> ⛔ **Re-audited 2026-10-05, and this pass found what the first three missed** — NVIDIA's own
> simulator, and two papers that published this repo's finding first. The claim in §5 is withdrawn
> and rewritten; §2a says what was missed and why; §4 has six new false sentences.

---

## 1. The method, and why this file exists at all

The first audit of this niche returned **"0 results"** for three searches and concluded the niche was
empty. It was not. The queries were **AND-joined 6-token strings** —
`kv+cache+routing+policy+simulator+replica` — and GitHub repo search requires *every* token to match. The
"0" was a fact about the query, not about the world. Four real competitors were each one **two-word**
search away, and one of them had specified this project's exact design.

### 📏 The audit rules that came out of that

1. **Never AND-join more than 2–3 tokens.** Run several 2-word searches and take the union.
2. **OPEN the top hits — do not count them.** A 666★ repo can be irrelevant; a 2★ repo can be your exact
   spec. Read the tree *and the README prose*, and check whether the code behind the README exists.
3. **List the branches.** ⭐ Two of four findings in the September pass were missed because the previous
   pass read file trees and never read README prose or branch lists. Deprecation notices, successor links
   and *"the feature is on the canary branch"* live in prose.
4. **Query the HF Hub separately** (`/api/models`, `/api/datasets`, `/api/spaces`) — invisible to GitHub
   search. Check **PyPI** for the name.
5. **A "0 results" verdict is never final.** It is a claim about your queries. **Write the queries down**
   so the next pass can falsify them.

*Added 2026-10-05, from the second failure of this audit (§2a):*

6. **Search papers by the question, not by the tool.** arXiv, and the programs of OSDI, NSDI,
   EuroSys, ATC and MLSys: *"LLM request scheduling"*, *"KV cache load balancing"*. Repositories are
   where tools live. Findings live in papers, and three passes never looked there.
7. **"A production engine" is not a reason to skip its docs.** Read each one's docs tree for
   *simulator, mocker, replay, sizing*. Dynamo's offline trace replay was six months old when this
   repo began, and Dynamo was filed here in one line beside vLLM.
8. **Follow the baselines of every close hit.** LMetric compares against six schedulers, SMetric
   against seven. A paper's related-work section is someone else's audit, already done.

**Name status (2026-08-25):** GitHub repo `kv-routing-sim` → 0 · PyPI `kv-routing-sim` → 404. The *name*
is free. The *niche* is not. *(Re-checked 2026-10-05: both still free; PyPI `kvsim` → 404 too.)*

---

## 2. The four competitors — each opened and read, not counted

### `microsoft/vidur` — 676★ · MLSys'24 · the one to be precise about

`vidur/config_optimizer/` holds `capacity_search.py`, `generate_pareto_curves.py`,
`cost_analysis_page.py`, `best_config_page.py`. On **`main`**, `scheduler/global_scheduler/` has exactly
**three** routers — random, round-robin, LOR — and **no prefix cache**.

⛔ **But on the `canary` branch** — advertised in Vidur's own README — it has `vidur/kv_cache/` (block
pool, block queue, replica and disk managers), `get_cached_prefill_length()`, and **eleven** routers,
including three documented *in the code* as **"Prefix cache aware"** (`LOP_UNCACHED`,
`RANKED_STICKY_LOP_UNCACHED`, `TOLERANT_STICKY_LOP_UNCACHED`) plus `STICKY_ROUND_ROBIN` / `STICKY_LOR`.
Merged to main June 2025, reverted July 2025 into `canary`, untouched since **2025-06-25**, carrying
self-declared *"sharp edges."*

**What it structurally cannot do:** read a **production vLLM/SGLang serving log** (it replays its own
trace format), **emit a recommendation**, or **publish a calibration of that cache model against a live
engine's counters**. And because the work has sat unmerged for 14 months, `main` — what anyone actually
installs — cannot answer the routing question at all.

✅ **Re-checked 2026-10-05:** 688★. `main` still has exactly three routers and no `kv_cache/`;
`canary`'s last commit is still 2025-06-25.

### `NetX-lab/Frontier` — 86★ → **139★ in 18 days** · arXiv 2605.21312 · the real competitor

Vidur lineage, pushed 2026-08-24. **Prefix caching + hierarchical caching.** Use case #3 in its README is
literally *"stateful reasoning scheduler validation … with prefix-cache continuity"*, and it ships
`examples/fixtures/prefix_cache_shared_session_trace.csv`. `cluster_scheduler_registry.py` registers
**five** routers: round-robin, random, LOR, **sticky-round-robin, sticky-LOR** (session affinity via
`_session_to_target_map`). Plus SLA-aware Pareto frontier search and a heterogeneous-GPU cost study.

**What it does not have:** a **longest-prefix / radix-affinity or consistent-hash router** — sticky means
*session* identity, not *shared-prefix* identity, so cross-user system-prompt sharing is invisible to it —
**trace adapters for real serving logs**, and a **recommendation** output. It is a research instrument you
configure, not a tool that reads your log and tells you what to run.

✅ **Re-checked 2026-09-12: still exactly five cluster routers, still no prefix-affinity router.** The
single most load-bearing differentiator survives. The star velocity is why the monthly re-audit is real.

✅ **Re-checked 2026-10-05:** 155★. The registry on `main` and on three scheduler branches still
registers the same five; the README's plan (SGLang and TensorRT-LLM, `tair-kvcache`, more models,
faster simulation) names no prefix-affinity router. ⚠️ **But it was never "the real competitor".**
Its acknowledgments name `ai-dynamo/aiconfigurator` — and following that link is what found §2a.

### `azrabano23/disagg-sim` — 0★ · created 2026-08-16

~1,700 lines of pure Python. **Paged KV + radix prefix cache at vLLM-APC/RadixAttention granularity**;
routers `random / round_robin / least_outstanding / kv_aware` (Dynamo-style
`matched_blocks − α·prefill_backlog`). 27 tests, seeded deterministic replay. Published results: kv_aware
≈2× hit rate vs round-robin (37.5% vs 19.1%), TTFT p50 86 ms vs 98 ms, **plus the locality-vs-load
counter-result.**

**What it declines to do, in its own README:** *"not a predictor of absolute numbers for your cluster …
calibrated from public specs, not fitted to traces … for trace-fitted prediction of real deployments, use
Vidur."* Synthetic workloads only, no live-engine calibration, no cost model, no sizing sweep. Its target
is **pedagogy** — reproducing published scheduling arguments — not deployment decisions.

⭐ **It is also the strongest evidence for our finding:** it published *that* the locality-vs-load
counter-result exists. It did not publish **the boundary**. That gap is H2.

### `Mike-Ew/kv-cache-routing` — 2★ · the standing warning

Its README specifies **the identical six policies** (RR, JSQ, power-of-two-choices, session-affinity hash,
prefix-key affinity, load+cache scoring), a `simulator/` for trace-driven scaling, **and validation on 2
real vLLM instances scraped from `/metrics`** — i.e. this project's entire plan, written down in April
2026.

**And nothing was built.** 48 files containing one 126-line file of policy stubs. No simulator, no
analysis, no results. Dead since 2026-04-30.

⭐ **The tell, and it is worth internalizing:** their README cites **`vllm:prefix_cache_hit_rate`** — the
V0-era gauge that **no longer exists**. They planned against the documentation and never started a
server. See ADR-008. *Do not repeat that mistake; it is legible from outside.*

**What this repo proves:** that the design is *reachable by anyone*, not that the work is *done*. **The
plan is not the differentiator. The executed calibration is.**

---

## 2a. ⛔ What three passes missed — found 2026-10-05

The passes of 2026-08-25, 2026-09-12 and 2026-09-23 searched GitHub **repositories** by name and
description, and filed NVIDIA's Dynamo under *"production engines that implement a routing policy"*
without opening its docs. No pass searched **papers** on request scheduling. Both gaps hid work that
is closer to this project than any of the four in §2 — and all of it is older than this repo's first
commit (2026-09-20).

### NVIDIA Dynamo's simulator — `ai-dynamo/aisimulate` and DynoSim · the closest tool

*Read 2026-10-05: both READMEs, the five pages under Dynamo's
`docs/fern/pages/cli/operations/simulation-with-dynosim/`, its replay CLI reference, the replay
module's README, the accuracy page's README, and pull requests #7543, #7553 and #7774.
**Not installed and not run.***

- **Offline trace replay across workers has existed since March 2026.** Dynamo PR #7543 (merged
  2026-03-20) added *"offline mocker replay for Mooncake-style traces"*: *"take a real request trace,
  preserve its prompt/output shapes and cache-sharing structure, and simulate the scheduler directly
  in process"*. #7553 (2026-03-23) made it multi-worker, with the KV router. #7774 (2026-04-02)
  added a replay-based search over configurations. The mock engine underneath it — a block-pool KV
  cache with prefix caching and preemption — dates from 2025.
- **Today it is a product surface.** `aisimulate predict --stack dynamo` replays a synthetic workload
  or a saved trace on N simulated workers. `aisimulate recommend` *"searches simulation trials across
  parallelism, worker split, router knobs, service-level objective (SLO) constraints, and GPU
  budget"* and returns a minimum GPU count with a configuration. `ai-dynamo/aisimulate`: 51★, created
  2026-08-11, on PyPI. It succeeds `ai-dynamo/aiconfigurator` (454★, arXiv 2601.06288), whose
  `recommend` mode was already *"a procurement sizing tool"* under TTFT **and TPOT** limits.
- **Its tutorial is this project's R1.** The DynoSim guide downloads the public FAST'25 Mooncake
  tool-agent trace, replays it at its timestamps, and then has a step titled *"Compare routing
  modes"*: the same trace through four workers under `router.policy=kv_router` against
  `round_robin` — *"review throughput, Time to First Token (TTFT), Inter-Token Latency (ITL), and
  prefix-cache reuse."* It ingests the AgentX (Weka) traces natively too: this project's R2.
- **Its cache model is the same idea as ours.** In vLLM mode: a block pool with block-hash prefix
  matching, least-recently-used eviction and LIFO or FIFO recompute preemption. In SGLang mode: a
  token pool and a radix cache. Timing comes from NVIDIA's measured performance database for the
  model, backend and GPU combinations it supports (others fail rather than fall back), or from a
  fixed or an *"uncalibrated"* polynomial model.

**What I did not find in it.** Each line is a statement about this reading; the next pass should try
to falsify it.

- **Routing policies as peers.** The simulator's `router.policy` is `round_robin` or `kv_router`
  (Dynamo's own, with its overlap and temperature knobs). No session-sticky, consistent-hash or
  least-loaded policy to set beside them.
- **A published check of the cache model against a live engine.** The public accuracy page compares
  predicted with measured mean TTFT and TPOT on *"single-turn, single-node, non-offloaded points"*,
  with the replay pinned to `enable_prefix_caching=False`, and calls its numbers *"advisory"*.
  Prefix-cache hits, and anything with more than one worker, are not on it.
- **A cost figure.** It counts GPUs, not money.
- ⚠️ *Not checked:* whether a user can calibrate it on hardware it has no data for. There is an
  `fpm-self-service` doc this pass did not read. **Do not claim this difference until someone has.**

### LMetric and SMetric — the closest findings

*Read 2026-10-05: both arXiv abstracts in full, and every sentence quoted below found in the
papers' own HTML text. Neither paper was read from start to end: **do that before implementing
either policy.***

- **LMetric** — *Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling*,
  Zhang et al. (Shanghai Jiao Tong University, Alibaba), **OSDI '26**, arXiv 2603.15202 (v1
  2026-03-16). Route each request to the instance with the smallest **P-token × BS**: *"the number
  of new prefill tokens if the request is routed to an instance, considering KV$ hits"*, times
  *"the batch size of the instance"*. SMetric writes it as (q + L − c) × l — prefill work already
  queued there, plus this prompt, minus what is cached, times the load. No weight to tune. Its case
  against the alternatives is this project's own experience, written down first. On a weighted
  sum: *"a statically tuned weight cannot always achieve competitive performance"* — the best
  weight moves with the load. On schedulers that score an instance with a simulator: their
  effectiveness *"relies heavily on the accuracy of the simulator, which requires complex
  per-model, per-hardware, and per-deployment development."* Measured on 16 H20 GPUs under vLLM
  and in production, against vLLM's router, Dynamo, llm-d, Preble, PolyServe and Alibaba's previous
  scheduler. It derives when the product fails — *"extreme KV$ skewness that compromises load
  balance"* — and reports that as rare. Code: `blitz-serving/blitz-router` (15★). ⚠️ The words
  *session* and *sticky* do not occur in its text: it does not compare against keeping a session
  in place.
- **SMetric** — *Rethink LLM Scheduling for Serving Agents with Balanced Session-centric
  Scheduling*, Wang et al., arXiv 2607.08565 (v1 2026-07-09, v2 2026-09-13). Its second insight, in
  its own words: *"given the agent's intra-session locality, routing requests by their sessions can
  balance the load with high KV$ reuse."* First turns go where the load is lowest; follow-ups stick
  to the instance with the highest local cache hit — unless it cannot finish them inside the TTFT
  target, and then the session migrates to the least-loaded instance. On a 32-instance vLLM cluster
  and two production traces, its peak throughput is 9–15% above the best baseline's. It reports
  that the shared system prompt is 18–20% of all reuse on those traces; the rest is sessions
  re-reading their own history.

  ⚠️ **It also names the weakness of plain session-sticky routing,** which is one of its baselines
  (llm-d routing by session ID): *"sticking every follow-up request to its session's instance still
  imbalances the load, because sessions differ widely in size and the size is unknown when the
  session is first routed"* — a mean imbalance of 2.17× on its testbed, against 3.34× for the
  weighted sum. **Every generated session in this repo's grids has the same number of turns and
  the same token ranges** — so those grids cannot show that weakness, and "session-sticky needs the
  fewest GPUs" was measured where sticky is at its best.

**What this does to the finding.** *"Keep each user on one GPU"* was published, from production
traces and real clusters, two months before this repo measured it on a simulated T4. So were its
two companions: pure prefix-affinity crowds sessions onto few instances and loses (both papers'
starting point), and a hand-weighted sum of a cache score and a load score has no weight that is
right everywhere (LMetric's whole case; here: a load weight of 16 against 1, with no steady winner).
And the question itself is older still: **Preble** (Srivatsa et al., UC San Diego, ICLR 2025, arXiv
2407.00023 — abstract read, paper not) called itself *"the first distributed LLM serving platform
that targets and optimizes for prompt sharing"*, with a scheduler that *"co-optimizes KV state reuse
and computation load-balancing"* — in mid-2024.

**The finding stands as an independent agreement on small hardware, not as a discovery.** Neither
published policy has been run in this simulator.

### `casys-kaist/LLMServingSim` — 416★ · IISWC'24, CAL'25, ISPASS'26

*Read 2026-10-05: its README, `serving/core/router.py` and its prefix-caching doc. Not run.*

A cycle-level simulator whose Python frontend mirrors vLLM's continuous-batching scheduler. Its
prefix caching is, in its own words, *"a port of vLLM v0.19.0's block pool"* with chained block
hashes, held per instance (*"a request that lands on instance B can't reuse a prefix cached on
instance A"*), with an optional storage tier shared across instances. Its router has four policies —
round-robin, random, least-loaded and a `CUSTOM` hook that raises `NotImplementedError` — and loads
agentic sessions. It reports a hardware profile *"within 1% of a real vLLM run on TTFT / TPOT /
latency"* (RTX 4090, 2026-08). No prefix-aware or sticky router ships with it.

### Public traces this project did not know about

**`alibaba-edu/qwen-bailian-usagetraces-anon`** (180★, Apache-2.0, since 2025-05). Four two-hour
production traces from Alibaba's Qwen service — chat, API, reasoning, coding — as **16-token** block
hashes, with chat and parent-chat ids, turn numbers and timestamps. Sixteen tokens is vLLM's block
size, so unlike Mooncake (512) and AgentX (64) they need no rescaling of blocks; and the sessions
are real ones, of unequal size. LMetric's evaluation uses them.

---

## 3. The long tail (all small, all narrower)

`200lz/llm-inference-optimization-lab` 31★ (deterministic serving sim with continuous batching + prefix
caching, **single-node**) · `zxuhan/llm-router` (per-worker radix trees, Go) ·
`Joshuaakaspace/kv-prefix-router` · `sundi133/prefix-aware-routing` ·
`AmanPandey28/stateful-cache-router` · `syedazeez337/mesh-lab` · `jahnavi-yelamanchi/cachetrace` (0★, a
**single-path prompt-construction linter**) · `EeroEternal/kvbridge` (0★, a **production routing proxy**).

**Single-engine work on the same traffic** *(found 2026-09-23)*: **CacheWise** (arXiv 2606.16824,
Jun 2026) — prefix-aware scheduling plus reuse-aware eviction for coding-agent workloads, single-node
inside vLLM · **`gauravapiscean/agentic-kv-cache`** (23★) — single-node eviction policies replayed on
the AgentX and Mooncake traces; finds LRU hard to beat. Neither routes across replicas; cite both
wherever those traces are used.

**New on the 2026-10-05 sweep, all small:** **`EleanorLiu12/cache-delay-eval`** (0★, since
2026-09-06) — replays the Qwen-Bailian traces across live vLLM instances behind a client-side
router, with `load`, `lmetric` and `affinity` policies, to measure how often LMetric's failure
condition occurs; the nearest thing found to this repo's Phase 6, and on real traces ·
**`sparkling-snail/kvrouter`** (0★, created 2026-10-05) — a Go router with bounded-load prefix
affinity and a four-instance mock benchmark; its README's result, *"the load bound earns its keep
under skew"*, is H1's again · **`peng1z/prefix-cache-eval`** (0★, 2026-10-01) — compares
per-request hit counts with vendored vLLM cache components under a serial driver, on CPU stubs: the
shape of this repo's Phase 2b check, without the live engine · `Naloam/agent-serving-sim` (1★) and
`asher0913/tiered-kv-cache-lab` (0★) — single-node. *(READMEs read; none run.)* On the HF Hub:
nothing about routing; three prefix-cache datasets and one Space seen by name only, not opened
(`wseaton/prefix-cache-bench`, `jaytonde05/iris-prefix-cache-benchmark`, `aintech/vdf_prefix-cache`,
`atharv6f/prefix-cache-analyzer`).

**Gateways that implement rather than compare:** `smg` 473★ (555★ on 2026-10-05) · `AMD-AGI/Infera` ·
`llm-d/llm-d-inference-sim` 188★ — a mock vLLM built *specifically* so people can test routing and
KV-cache-locality logic. Different product: it helps you test routing **code**; we help you choose a
routing **policy**.

**Every five-figure-star name in the space** — vLLM 89.9K, SGLang 32.4K, LMCache 11.4K, Dynamo 7.8K,
Mooncake 6.4K, AIBrix 5.0K — is a funded-team production engine that *implements* a routing policy rather
than letting you *choose* one before committing infrastructure.

⛔ **That sentence was wrong about Dynamo when it was written** (8.2K★ on 2026-10-05): Dynamo ships
a simulator for exactly that choice. It is kept above as written; §2a is the correction. The other
five were not re-opened for a simulator in this pass — **rule 7 says the next pass must.**

---

## 4. ⛔ Sentences that are FALSE — never write these

| ⛔ Never write | Why it is false | ✅ Write instead |
|---|---|---|
| *"0 results — nobody has built this"* | Artifact of an AND-joined 6-token query. Four competitors exist. | *"An intersection nobody occupies"* — then name the four. |
| *"Vidur has no prefix-cache model"* | Its `canary` branch has a full block-level KV cache **and three cache-aware routers**. One click to check. | *"Vidur's prefix-cache model has sat unmerged on `canary` for 14 months, so `main` — what you install — cannot answer the routing question."* |
| *"Nothing else validates against a live engine"* | Vidur **is** validated against real deployments — that is its paper's headline. | *"the only prefix-cache-aware multi-replica policy comparison whose numbers were checked against a live engine."* |
| *"First trace-driven LLM serving simulator"* | Vidur, 2024, MLSys. | Cite Vidur as the lineage. Claim the intersection. |
| *"Frontier can't do prefix caching"* | It has prefix **and hierarchical** caching. | *"Frontier has prefix caching and session-sticky routing, but no prefix-affinity router — sticky is session identity, not shared-prefix identity."* |
| *"vllm:prefix_cache_hit_rate"* | Does not exist at v0.29.0. A competitor's exact mistake. | *"derived from `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`."* |
| *"The cache model is the core IP"* | Vidur-canary and Frontier both ship one. It is **table stakes.** | *"The contribution is its **fidelity**: single-digit-% error against a live engine, with the calibration curve published."* |
| *"Dynamo implements a routing policy; this helps you choose one"* *(added 2026-10-05, as are the rows below)* | Dynamo ships DynoSim: offline trace replay on simulated workers, round-robin against its KV router, and a recommender that returns a GPU count under TTFT and TPOT limits. Since March 2026. | *"Dynamo's simulator answers this for Dynamo's router. This one sets seven routing policies side by side, and its cache model was checked counter for counter against a live vLLM."* — and only while that stays true. |
| *"The recommendation is the output nobody else produces"* | `aisimulate recommend`, and `aiconfigurator recommend` before it, return a minimum GPU count and a configuration under an SLO. | *"It recommends a routing policy together with the replica count, from your trace."* |
| *"The only policy comparison checked against a live engine"* | LMetric and SMetric compare routing policies on real 16- and 32-GPU clusters; they need no simulator to be checked. `cache-delay-eval` replays traces through live vLLM instances. | *"A simulator whose cache counters were reproduced exactly on a live engine, and whose hit rates held within 1.4% on two live replicas."* A claim about this repo only. |
| *"We found that sessions should stay on one GPU"* | SMetric published session-centric scheduling in July 2026, from production traces. | *"On a calibrated T4 model, this agrees with SMetric."* And cite its caveat: plain session-sticky routing unbalances the load when sessions differ in size, which the generated grids here cannot show. |
| *"Nobody has published when prefix-affinity loses to load balance"* | LMetric derives the failure condition of its own score; SMetric and Preble start from the crowding problem. | *"Whether routing by prefix saves GPUs once the prompts outgrow one cache is not settled here."* |
| *"No one else replays Mooncake or AgentX across replicas"* | DynoSim's tutorial replays the first; AISimulate ingests the second. | Name them. |

---

## 5. ✅ What can still be claimed — rewritten 2026-10-05

**Withdrawn.** Until 2026-10-05 this section claimed an intersection nobody occupied:

> ~~Trace adapters for **real serving logs** **+** a **calibrated** prefix-cache model **+** policy choice
> *and* **replica sizing** **+** **₹/$ per 1k requests** **+** **sim-to-real validation.**
> Four repos each own one or two of those. **None owns the set.**~~

NVIDIA's simulator owns most of that set (§2a), and did before this repo existed. The four repos in
§2 were the wrong four to measure against.

**What is left is not a niche. It is three things this repo did, and can show:**

1. **A measured credential.** The cache model reproduces a live vLLM 0.30.0's prefix-cache counters
   exactly — every counter and every request's cached tokens, on two workloads under eviction
   pressure — and its hit rates held within 1.4% on two live replicas under three routing policies.
   That is a claim about this repo. It is not a claim that nobody else has done the same.
2. **A method.** Every test's design, rules and prediction were committed before it ran, and the
   ones that failed are reported as findings, with their dates
   ([`09-hypotheses.md`](09-hypotheses.md)).
3. **Numbers on how far simulator-based sizing can be trusted.** A second calibration of the same
   GPU, its coefficients within 5.3% of the first's, put p95 latency at the knee of the load curve
   4.17× from the measurement, where the first was within 1.41×; replica counts from three
   calibrations agree to within one at 96% of points; the per-token model is within 7% off the knee
   and 1.5–1.7× wrong at it. LMetric says that a scheduler built on a simulator *"relies heavily on
   the accuracy of the simulator"*, and measures what that costs in scheduling. These numbers are
   the same dependence measured for sizing, on one small GPU.

**And one agreement:** on a calibrated T4 model with generated traffic, the routing results match
the published ones — pure affinity loses under load, sessions belong on one replica, a hand-set
weight has no right value — reached independently, on a free Kaggle GPU.

**Say it as a replication with an open, checked tool. Never as a discovery, and never as an empty
niche.** *The moat is execution quality, not the idea* was this section's last line before; after
this pass it is the only ground left, and it holds only while every sentence above stays true.

---

## 6. The top risk: the gap closes from above

> ⛔ **2026-10-05: it had already closed — from a side this section was not watching.** Not Vidur's
> merge and not Frontier's next router: Dynamo's offline replay, in March 2026. Mitigation 3 below
> reads correctly with the names changed — *pivot to being a layer on top of them.* Three ways to do
> that, **none chosen yet:**
>
> 1. **Replicate the published schedulers here.** Add LMetric and an SMetric-style policy to the
>    seven, and register a test on the public Qwen-Bailian traces — real sessions of unequal size,
>    the case the generated grids cannot show. CPU only.
> 2. **Cross-check against DynoSim.** The same Mooncake trace, the same replica counts, round-robin
>    and a prefix-aware policy in both simulators: do two independent cache models agree on the hit
>    rate, and on the order of the policies? CPU only.
> 3. **Contribute upstream.** AISimulate's public accuracy page has no point with prefix caching on;
>    this repo's counter-exact cache check is one.
>
> *Everything below is the section as written before this pass.*

⛔ **Vidur does NOT "need only a prefix-cache model" — it already has one written.** It needs only a
*merge*, and that merge has been pending 14 months. Treat this as a **live, one-decision-away risk**, not
a build-effort-away one. Frontier needs only a radix-affinity router and a trace adapter, and it is
funded, arXiv-backed, and gaining ~3★/day.

**Mitigations, in order:**

1. **Ship the live-engine calibration early.** It is the one component that costs wall-clock time no
   matter how strong the competing team is.
2. **Re-audit monthly** (§7 below), not once at kickoff.
3. **If either lands a prefix-affinity router mid-build, pivot to being the trace-adapter +
   recommendation layer *on top of* them.** A Vidur/Frontier integration is a better story than a
   duplicate.

---

## 7. Monthly re-audit log

**Write the queries down so the next pass can falsify them.** Add a row per pass; never delete one.

| Date | Queries run (2-word, union) | Vidur ★ / canary merged? | Frontier ★ / prefix-affinity router? | New entrants | Verdict |
|---|---|---|---|---|---|
| 2026-08-25 | `kv routing`, `prefix cache simulator`, `llm router simulator`, `cache aware routing`, + PyPI + HF Hub | 666 / no | 86 / **no** | `disagg-sim` (Aug 16), `Mike-Ew` (Apr) | Open as an intersection |
| 2026-09-12 | re-check of the four + branch lists + README prose | 676 / **no**, untouched since 2025-06-25 | **139** / **no** (registry re-read) | none | Differentiator survives; star velocity noted |
| 2026-09-23 | capability searches (recorded in `FLAGSHIP_PROJECTS.md`; ported here 2026-09-28) | — | — | CacheWise (single-node, inside vLLM); `gauravapiscean/agentic-kv-cache` (single-node eviction on AgentX + Mooncake, no routing); llm-d's scheduler write-ups (hand-set prefix + load weights, no boundary published) | None occupies the intersection; all three cited in the README |
| 2026-10-05 | the four re-read (registries, branch lists, READMEs to the bottom) · PyPI · HF Hub · nine 2-word GitHub sweeps by *recently updated*: `kv routing`, `prefix cache simulator`, `llm router simulator`, `cache aware routing`, `kv cache routing`, `prefix affinity`, `llm serving simulator`, `routing simulator llm`, `inference simulator prefix` · **new:** Frontier's acknowledgments followed (→ AIConfigurator → AISimulate → Dynamo's docs tree) · a web search for the scheduling papers, from a citation in a 0★ repo's README | 688 / **no** — `canary` untouched since 2025-06-25, `main` still three routers | 155 / **no** — five routers on `main` and on three scheduler branches; none in its plan | ⛔ **missed by three passes:** Dynamo's DynoSim and AISimulate (offline multi-worker replay since 2026-03) · **LMetric** (OSDI '26) · **SMetric** (arXiv 2607.08565) · Preble (ICLR '25) · LLMServingSim 2.0 (416★) · the Qwen-Bailian traces. New and small: `cache-delay-eval`, `kvrouter`, `prefix-cache-eval`, `agent-serving-sim`, `tiered-kv-cache-lab` | ⛔ **The intersection claim is withdrawn, and the finding is an agreement with published work, not a discovery.** §2a added; §4, §5, §6 rewritten; rules 6–8 added |
| *(next)* | *due 2026-11-05, before launch — **start from the papers*** | | | | |

**What to check on every pass, minimum:** Vidur's `canary` merge status · Frontier's
`cluster_scheduler_registry.py` router list · both READMEs **to the bottom** · both branch lists · PyPI
name still free · a fresh 2-word GitHub sweep sorted by *recently updated*, not by stars.

*Added 2026-10-05:* AISimulate's `router.policy` choices, and whether its accuracy page has gained a
point with prefix caching on or with more than one worker · its `fpm-self-service` doc (unread) ·
whether SMetric's code and traces were released · arXiv for papers citing LMetric or SMetric · the
docs trees of vLLM's production stack, SGLang, LMCache, Mooncake, AIBrix and llm-d, for a simulator
or a replay mode (rule 7) · LMetric's and SMetric's PDFs, read line by line.
