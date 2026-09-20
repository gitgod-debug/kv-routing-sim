# 01 — Why this project exists

**Written 2026-09-20**, from `FLAGSHIP_PROJECTS.md` Project 1. Where this file and that one disagree
about `kv-routing-sim`, **this file wins**.

---

## 1. The problem, and who feels it

Every team self-hosting an LLM behind more than one replica has to pick a routing policy. The candidates
are round-robin, least-loaded, session-sticky, and prefix-aware. The choice swings **prefix-cache hit
rate, TTFT and cost by large factors under agentic traffic**, where a shared system prompt is re-sent on
every turn and tool-call turns re-send the whole conversation so far.

Today that choice is made by **guessing, then deploying, then hoping.** vLLM's router, SGLang's router,
AIBrix, llm-d, Dynamo and Mooncake all *implement* a policy; TrueFoundry sells one. Implementing a
policy is not the same as helping you *choose* one before you commit the infrastructure.

### ⚠️ Say the next part precisely — the loose version is false

**Pre-deployment what-if analysis does exist.** Vidur's `config_optimizer` and Frontier's SLA-aware
Pareto search both answer *"which configuration, how many replicas, at what cost-per-capacity, before I
commit."* Claiming otherwise gets you dismissed by anyone who has read the field.

What none of them does is answer it **as a routing-policy question with a prefix-cache model
underneath, from your own serving log.** Vidur models a prefix cache only on an unmerged branch, and
none of the simulators that *do* model prefix caching will ingest your real serving log or hand you a
sized, costed recommendation.

**On agentic traffic the prefix cache is what decides the answer.** A sizing tool that cannot see the
cache is answering a different question.

---

## 2. The claim, precisely

> **Your real serving log in → a prefix-cache-calibrated policy *and* replica-count recommendation with
> a ₹ number out, validated against a live 2-replica deployment.**

That is the one sentence. Every word in it is load-bearing:

| Clause | Why it is in the sentence |
|---|---|
| *your real serving log* | Vidur and Frontier replay their own trace formats. Trace adapters for real vLLM/SGLang logs are ours. |
| *prefix-cache-calibrated* | The cache model is not trusted because it is plausible. It is trusted because it was measured against a live engine's counters and the error was published. |
| *policy **and** replica-count* | Two axes, not one. Vidur sizes without seeing the cache; Frontier sees the cache but does not recommend. |
| *a ₹ number* | An explicit, configurable cost model with no hidden constants. ₹ per 1k requests is the number we own. |
| *validated against a live 2-replica deployment* | The narrow, strong claim — see §4. |

### ⚠️ And the caveat, said out loud

That is a **narrow gap, not a category**. One of the four competitors specified the same design in April
2026. The defence is execution quality — the calibration error, the error bars, the explained residual —
**not the idea**. Anyone claiming the idea is novel will lose that argument in public.

---

## 3. Credential vs finding — and why the README needs both, separately

This distinction is the single most important thing in the project's presentation.

|  | **The credential** | **The finding** |
|---|---|---|
| The claim | *"My simulator matches the live engine's prefix-cache hit rate within X%, and here is why the residual is what it is."* | *"Prefix-affinity routing stops paying above K replicas at S% prefix sharing."* |
| What it proves | **Rigor.** That the numbers can be trusted. | **Something about the world.** Something an infra team can act on. |
| Who it is for | A reviewer: someone deciding whether to trust the numbers. | Anyone running a fleet — including people who will never open the repo. |
| Where it lives | The **Real-Engine Calibration Report**, its own named README section. | The headline, one sentence, above the fold. |

A credential with no finding is a well-tested tool nobody has a reason to use. A finding with no
credential is a blog post. **Both, in separate sections, or the work reads as half of itself.**

> **The test:** a reader who will never open the repo must be able to state the finding in one sentence.
> If they can only say *"he built a simulator and validated it"*, the README has failed.

---

## 4. The claims we are allowed to make about validation — exact wording

⛔ **Never write: "nothing else in this space validates against a live engine."** It is false in three
different ways, and each one is checkable in a minute:

- `azrabano23/disagg-sim` *deliberately declines* to fit traces and defers to Vidur — that is a stated
  scope choice, not a failure.
- **Vidur *is* validated against real deployments** — that is its paper's headline. What is *not*
  validated is its prefix-cache model, which sits unmerged on `canary`.
- `Mike-Ew/kv-cache-routing` wrote this exact validation design in April and never ran it.

✅ **The accurate claim is narrower and stronger:**

> **the only prefix-cache-aware multi-replica policy comparison whose numbers were checked against a
> live engine.**

Every qualifier in that sentence is doing work. Drop one and it becomes false.

---

## 5. What this project is NOT

- **Not a production router.** `EeroEternal/kvbridge` is that. We help you choose; we don't serve traffic.
- **Not a prompt linter.** `jahnavi-yelamanchi/cachetrace` analyses a single request path. We compare
  policies across a multi-replica fleet.
- **Not a capacity/parallelism planner.** That is Vidur's `config_optimizer`. We answer a
  *cache-aware policy* question; Vidur answers a parallelism-and-capacity question.
- **Not chasing stars.** The honest ceiling for a solo analysis tool in this space is **low hundreds** of
  stars (`vllm-project/guidellm` = 1,598★ *with* the vLLM org's distribution behind it; a genuinely solo
  analysis repo, `bassrehab/triton-kernels`, sits at 41★). What a project of this size can do is send fixes upstream
  for the bugs it finds.

That scope discipline belongs in the README **from day one**, not added defensively after someone asks.

---

## 6. The one thing that costs wall-clock time no matter what

The **live-engine calibration**. Everything else in this repo is Python that a competent team could
write in a week. The calibration takes real hardware, real hours, and produces the one artifact a
competitor cannot get by being smarter: a published predicted-vs-measured curve.

That is why it is **first in the protection order** under schedule compression: three routing policies
calibrated against a real engine beats six policies validated only against themselves. The first is an
infrastructure credential; the second is a spreadsheet.

**The simulator is the contribution. The calibration is the credential. If only one survives a
collapsed week, it is the calibration.**
