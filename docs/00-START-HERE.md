# START HERE

> **You are reading this because you forgot everything, or because you never knew it.**
> That is the assumption this documentation is written under. It explains the project from zero: what
> it is, why every decision was made the way it was, what every piece of code does, and what would
> have to be true for the whole thing to be wrong.
>
> No prior knowledge of KV caches, prefix caching or vLLM is assumed.

---

## The 60-second version

`kv-routing-sim` answers one question: **which KV-cache routing policy should you actually run, and on
how many replicas?** You give it your real serving log. It gives you a policy, a replica count, and a
cost number — and its cache model has been checked against a live inference engine, so the numbers are
not self-referential.

Here is the problem in plain terms. If you self-host an LLM behind more than one replica, something has
to decide which replica each request goes to. The obvious choices are round-robin or least-loaded. But
modern engines keep a **KV cache**: when two requests share a prompt prefix — and under agentic traffic
they nearly always do, because the same system prompt is re-sent on every turn — the second request can
skip recomputing that prefix entirely, *if it lands on the replica that already has it cached.*

So the routing decision silently controls your cache hit rate, which controls your time-to-first-token,
which controls how many GPUs you need, which controls your bill. Today that choice is made by
**guessing, then deploying, then hoping.**

Two deliverables, and the README keeps them in **separate sections** because they are different kinds
of claim:

1. **The simulator + recommender** *(the contribution)* — trace in, policy × replica-count Pareto
   curves out, with a recommended configuration and a ₹/$ delta against the naive baseline.
2. **The Real-Engine Calibration Report** *(the credential)* — the simulator's cache model measured
   against a live vLLM's own Prometheus counters, with the error published and the **residual
   explained**. Anyone can write a cache model. Almost nobody publishes how wrong theirs is.

And separately from both, the thing a reader should be able to repeat without opening the repo:

3. **The finding** *(measured 2026-09-28)* — **keep each user on one GPU until your tenants' system
   prompts outgrow one GPU's KV cache; then route by prefix, and keep its load weight small.** The
   boundary is one ratio: the prompts' working set over one replica's cache. *(It was first framed as
   a sharing-ratio × load crossover against least-loaded — H2 — and that framing was falsified; the
   dated record is in [`09-hypotheses.md`](09-hypotheses.md).)*

---

## Read in this order

| # | File | What it answers |
|---|---|---|
| 00 | **this file** | What is this, where do I start |
| 01 | [`01-why-this-project.md`](01-why-this-project.md) | Why does this project exist? What exactly are we claiming, and which parts are contested? |
| 02 | [`02-architecture-decisions.md`](02-architecture-decisions.md) | Why *this* structure, *this* fidelity level, *this* hardware? What did we reject and why? |
| 03 | [`03-background-concepts.md`](03-background-concepts.md) | What is a KV cache, prefix caching, a radix tree, continuous batching, TTFT? **Start here if the words are unfamiliar.** |
| 04 | [`04-prior-art.md`](04-prior-art.md) | What already exists. What we must cite. **What we must never claim.** |
| 05 | [`05-codebase-map.md`](05-codebase-map.md) | Every module, what it does, why it's shaped that way |
| 06 | [`06-methodology.md`](06-methodology.md) | How we measure. Statistics, the cost model, what makes a number trustworthy |
| 07 | [`07-environment.md`](07-environment.md) | The two machines, what each can and cannot do, and the calibration runbook |
| 08 | [`08-glossary.md`](08-glossary.md) | Every term, acronym and metric name, defined |
| 09 | [`09-hypotheses.md`](09-hypotheses.md) | **The pre-registered predictions** — H1–H6, H2b, the real traces (R1, R2), C1 and Phase 6 — with dated results, falsified ones included. What we expect, what would falsify each, what decision changes if it holds. **Read before running any phase that produces a number** — a sweep with no hypothesis is a dataset dump |
| 11 | [`11-where-we-are.md`](11-where-we-are.md) | ⭐ **The whole story so far, in plain words** — what we found, what we decided and why, the mistakes we caught, where we are, what is left. **Read this first if you have been away.** |

**If you have five minutes:** read this file, then §"The claim, precisely" in
[`01-why-this-project.md`](01-why-this-project.md), then the honesty gates at the bottom of
[`../CHECKPOINTS.md`](../CHECKPOINTS.md).

**If you're about to write code:** read [`05-codebase-map.md`](05-codebase-map.md) and
[`06-methodology.md`](06-methodology.md).

**If you're about to produce a result:** read [`09-hypotheses.md`](09-hypotheses.md) first, and check
that the prediction it tests was committed *before* the number existed. That ordering is the claim.

**If you're about to write a README claim, a summary of the project or a Show HN comment:**
read [`04-prior-art.md`](04-prior-art.md) first. There is a list of specific sentences that are false
and that a knowledgeable reader will catch instantly. *"Nothing else does this"* is on it.

---

## Where everything lives

```
kv-routing-sim/
├── CHECKPOINTS.md          ← the to-do list. Where we are right now.
├── docs/                   ← you are here
├── src/kv_routing_sim/
│   ├── traces/             ← trace adapters + the normalized request schema + the agentic generator
│   ├── cache/              ← the block-level prefix cache model (ours, testable)
│   ├── policies/           ← the seven routing policies, one file each
│   ├── sim/                ← the replica fleet, the queueing model, the metrics collector
│   ├── eval/               ← bootstrap CIs, the cost model, Pareto curves, the recommender
│   ├── calib/              ← `kvsim calibrate`: fit a profile to YOUR engine, check the cache model
│   └── data/               ← the measured T4 profile and cost model, shipped with the package
├── tests/                  ← unit tests; the cache-model hit/miss tests matter most
├── scripts/                ← every results table is regenerated by a script in here
├── configs/                ← cost model, workload grids, policy grids — every constant cited and dated
├── notebooks/              ← the GPU runs, self-contained, for Kaggle (see notebooks/README.md)
├── site/                   ← the static demo page (GitHub Pages)
├── paper/                  ← the preprint draft, and where each of its numbers comes from
└── artifacts/results/      ← results JSON + raw metric scrapes. Regenerable — never hand-edited.
```

**The project plan is not part of this repo.** It is kept privately (working
copies sit beside this repo, gitignored). Its section on this project was
the source for these docs on 2026-09-20, and **where the two ever disagree about `kv-routing-sim`,
these docs win.**

**The rule that keeps this honest:** anything in `artifacts/results/` must be reproducible by running a
script in `scripts/`. If you can't delete it and get it back, it doesn't ship.

---

## The one thing to not get wrong

**This project's defence is an intersection, not an empty niche. Four other repos exist, and one of
them specified this exact design first.**

- `microsoft/vidur` (676★, Microsoft, MLSys'24) **has a block-level prefix cache model and three
  cache-aware routers** — on its `canary` branch. Saying "Vidur has no prefix cache" is false and an
  attentive reader who lists the branches will catch it in one click.
- `NetX-lab/Frontier` (86★ → 139★ in 18 days, arXiv-backed) has prefix caching *and* session-sticky
  routing already.
- `Mike-Ew/kv-cache-routing` wrote down **the identical six policies and the identical 2-replica vLLM
  validation** in April 2026 — and then never built it.

So the honest claim is the **intersection**: real-serving-log trace adapters **+** a calibrated
prefix-cache model **+** policy choice *and* replica sizing **+** ₹/$ per 1k requests **+** sim-to-real
validation. Four repos each own one or two. None owns the set.

**The moat is execution quality — the published calibration error and honest error bars — not the
idea.** `Mike-Ew` proves the idea was reachable by anyone; what nobody has is the executed measurement.
Read [`04-prior-art.md`](04-prior-art.md) for the exact wording, and for the sentences that are
disqualifying.
