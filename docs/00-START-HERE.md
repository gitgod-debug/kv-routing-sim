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
which controls how many GPUs you need, which controls your bill. Often that choice is still made by
**guessing, then deploying, then hoping.** *(This said "today that choice is made by guessing"
until 2026-10-05, as if no tool existed. One does: NVIDIA's Dynamo ships a simulator for it —
[`04-prior-art.md`](04-prior-art.md) §2a.)*

Two deliverables, and the README keeps them in **separate sections** because they are different kinds
of claim:

1. **The simulator + recommender** *(the contribution)* — trace in, policy × replica-count Pareto
   curves out, with a recommended configuration and a ₹/$ delta against the naive baseline.
2. **The Real-Engine Calibration Report** *(the credential)* — the simulator's cache model measured
   against a live vLLM's own Prometheus counters, with the error published and the **residual
   explained**. Anyone can write a cache model. Few publish how wrong theirs is — and, as far as
   one reading of the others found, none for the prefix cache itself.

And separately from both, the thing a reader should be able to repeat without opening the repo:

3. **The finding** *(changed 2026-09-30, narrowed 2026-10-05)* — **keep each user on one GPU.** On
   three calibrations of the same T4, session-sticky routing needs the fewest GPUs, or ties for them,
   at every load with one shared prompt and with four, and never needs more than round-robin or
   least-loaded. **Whether routing by prefix then saves GPUs with 64 tenants' prompts is not
   settled**, and every GPU count is ± one. *(It read "…until your tenants' prompts outgrow one
   GPU's cache; then route by prefix" on the first, context-free timing model. Re-tested on the
   validated one — calibration A — that half was falsified. Re-run on two more calibrations of the
   same GPU in C3, the same test held: the three agree on a GPU count to within one, and that half
   was decided by one. Before all that, a sharing-ratio × load framing — H2 — was falsified too. The
   dated record of all of it is in [`09-hypotheses.md`](09-hypotheses.md).)*

   ⚠️ **It is not a new result.** SMetric (arXiv 2607.08565, July 2026) published "route agents by
   session" from production traces on 32 real GPUs, and names a weakness these grids cannot show —
   real sessions differ widely in size. What this repo has is an independent agreement on a small
   calibrated model, and numbers on how far such a model's GPU count can be trusted: ± one, and
   one calibration's own spread covers the next one's count in 82% of cases, not 90 (C4).

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
| 12 | [`12-story-notes.md`](12-story-notes.md) | Short pieces written to be told — in a talk or a post — each with its proof linked |

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

~~So the honest claim is the **intersection**: real-serving-log trace adapters **+** a calibrated
prefix-cache model **+** policy choice *and* replica sizing **+** ₹/$ per 1k requests **+** sim-to-real
validation. Four repos each own one or two. None owns the set.~~

⛔ **Withdrawn 2026-10-05.** Those four were the wrong four. A fourth prior-art check found that
NVIDIA's Dynamo has shipped a simulator for this question since March 2026 — trace replay across
simulated workers, round-robin against its KV-aware router, a GPU count under an SLO — and that the
routing result this repo reached was already published from real clusters (SMetric, LMetric). What
is left to claim is not a niche but three things this repo did: a cache model reproduced counter for
counter on a live engine, tests registered before they ran with the failures reported, and numbers
on how far a simulator's GPU count can be trusted.

**The moat is execution quality — the published calibration error and honest error bars — not the
idea.** `Mike-Ew` proves the idea was reachable by anyone; NVIDIA proves a funded team got there
first. Read [`04-prior-art.md`](04-prior-art.md) for the exact wording, and for the sentences that
are disqualifying.
