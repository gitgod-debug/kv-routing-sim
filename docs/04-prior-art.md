# 04 — Prior art: what exists, what to cite, what never to claim

> ⛔ **Read this before writing any README claim, summary of the project or Show HN comment.**
> There is a list of specific sentences below that are **false**, that were written down in good faith by
> a previous audit, and that a knowledgeable reader will catch instantly.
>
> Evidence re-run **2026-08-25**, re-verified **2026-09-12**, ported into this repo **2026-09-20**.
> ⚠️ **Re-audit monthly, not at kickoff** — one competitor went 86★ → 139★ in 18 days.

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

**Name status (2026-08-25):** GitHub repo `kv-routing-sim` → 0 · PyPI `kv-routing-sim` → 404. The *name*
is free. The *niche* is not.

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

**Gateways that implement rather than compare:** `smg` 473★ · `AMD-AGI/Infera` ·
`llm-d/llm-d-inference-sim` 188★ — a mock vLLM built *specifically* so people can test routing and
KV-cache-locality logic. Different product: it helps you test routing **code**; we help you choose a
routing **policy**.

**Every five-figure-star name in the space** — vLLM 89.9K, SGLang 32.4K, LMCache 11.4K, Dynamo 7.8K,
Mooncake 6.4K, AIBrix 5.0K — is a funded-team production engine that *implements* a routing policy rather
than letting you *choose* one before committing infrastructure.

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

---

## 5. ✅ The claim that survives, in full

> Trace adapters for **real serving logs** **+** a **calibrated** prefix-cache model **+** policy choice
> *and* **replica sizing** **+** **₹/$ per 1k requests** **+** **sim-to-real validation.**
>
> Four repos each own one or two of those. **None owns the set.**

**Say it as an intersection. Never as an empty niche.** And say the caveat out loud, because a good
reviewer will find it: this is a **narrow** gap, and one competitor specified the same design in April.
**The moat is execution quality, not the idea.**

---

## 6. The top risk: the gap closes from above

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
| *(next)* | *due 2026-10-20* | | | | |

**What to check on every pass, minimum:** Vidur's `canary` merge status · Frontier's
`cluster_scheduler_registry.py` router list · both READMEs **to the bottom** · both branch lists · PyPI
name still free · a fresh 2-word GitHub sweep sorted by *recently updated*, not by stars.
