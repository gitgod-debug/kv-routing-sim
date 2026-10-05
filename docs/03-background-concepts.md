# 03 — Background concepts

> **Start here if the words are unfamiliar.** Nothing below assumes you have read a serving paper. If you
> can already explain paged attention and RadixAttention, skip to [`04-prior-art.md`](04-prior-art.md).
>
> Every term here is also in [`08-glossary.md`](08-glossary.md) in one-line form. This file explains *why
> each one matters to this project*, which the glossary does not.

---

## 1. The KV cache — why generation is two different problems

A transformer generating text attends, at every new token, over every previous token. Recomputing the keys
and values for the whole prefix at each step would be quadratic and absurd, so they are computed once and
**cached**. That cache is the **KV cache**.

This splits a request into two phases with completely different characters:

| Phase | What happens | Bound by | Cost scales with |
|---|---|---|---|
| **Prefill** | The whole prompt is processed in one parallel pass; its KV entries are written to the cache | **compute** | prompt length |
| **Decode** | One token at a time, each attending over the cache | **memory bandwidth** | output length |

**Why this project cares:** *prefill is the expensive part that can be skipped.* If the prefix is already
in a cache somewhere, that compute simply doesn't happen. Decode cannot be skipped. So every saving this
project chases is a prefill saving — which is why **prompt sharing**, not output length, is the variable
that matters.

### Time-to-first-token vs time-between-tokens

- **TTFT** — arrival to first output token. Queueing delay + prefill. **The metric routing moves
  most visibly** — and, until 2026-10-05, the only one this project measured.
- **TBT / ITL** — the gap between subsequent tokens. Decode-bound — and *not* "barely affected by
  routing", as this line said until 2026-10-05. In a continuous-batching engine every sequence gets
  one token per step, so a step that prefills a newly arrived request makes everyone already
  decoding wait for it; and the more sequences a router sends to one replica, the longer every
  step. On a live T4 the gap was 134–521 ms on traffic whose first token was on time (C3), and
  under a target with both halves it is this half that sets the replica count (T2).
- **TPOT** — time per output token, per request: (last token − first token) ÷ (tokens − 1).

A routing policy that improves mean latency but neither TTFT nor the time per token has probably
improved nothing a user perceives. One that buys a prompt first token by crowding sequences onto a
replica has made every token after it slower — which a first-token target cannot see.

---

## 2. Paged attention — why the cache is made of blocks

Storing each sequence's KV cache in one contiguous slab wastes enormous memory: you must reserve for the
*maximum* possible length. vLLM's **PagedAttention** instead splits the cache into fixed-size **blocks**
(commonly 16 tokens) and keeps a block table per sequence, exactly like virtual memory pages.

**Why this project cares:** it makes the cache **block-granular**, and therefore *shareable*. Two
sequences with the same first 48 tokens can point at the same three blocks. Everything this project models
follows from that — which is why the cache model in ADR-002 is a trie of blocks and not a set of strings.

It also means **partial sharing is the normal case**, and a block is the unit of it. Two prompts sharing
the first 47 tokens share **two** blocks, not 2.9 — the remainder of the third block is not shared. That
quantization is a real effect and the simulator must reproduce it.

---

## 3. Prefix caching — the mechanism this project is about

**Automatic prefix caching (APC)** in vLLM, and **RadixAttention** in SGLang, both do the same thing: hash
each block of a prompt, and if a block with that hash is already resident, **reuse it instead of
recomputing it.**

```
Request A:  [system prompt ......][user question 1]
Request B:  [system prompt ......][user question 2]
                ↑
            identical blocks — B skips prefilling them entirely
```

Two details matter for fidelity:

1. **A hash chain, not a set.** A block's hash includes its *position and its predecessors*. Reuse is
   therefore only possible as a **prefix** — you cannot reuse a middle chunk. This is why the data
   structure is a trie.
2. **Eviction.** The cache is finite. When it fills, blocks are evicted, normally **LRU**. So a hit
   depends not only on whether a prefix was seen before, but on *whether it has survived*. **This is where
   simulator fidelity goes to die** — hit rate is easy to model, eviction order is not, and that is
   exactly what H4 predicts the residual will be.

### Why agentic traffic changes the answer

A single-turn chatbot re-sends little. An **agent** re-sends the system prompt, the tool definitions, and
the entire conversation so far **on every turn**:

```
turn 1:  [system][tools][user]
turn 2:  [system][tools][user][assistant][tool result][user]     ← turn 1's prompt is a prefix of this
turn 3:  [system][tools][user][assistant][tool result][user][assistant][tool result][user]
```

Prompt sharing under agentic traffic is not an optimization opportunity, it is **most of the workload**.
That is why the spec calls prefix-aware routing *"the highest-leverage change for agentic traffic"* — and
why a sizing tool that cannot see the cache is answering a different question.

---

## 4. Continuous batching — why a replica's queue is not a queue

Classic batching waits for a batch, runs it, returns it. Engines instead use **continuous batching**: at
every iteration, finished sequences leave the batch and waiting ones join. A request does not wait for the
batch ahead of it to *finish*; it waits for a *slot*.

Consequences this project models:

- **`max_num_seqs`** bounds concurrent sequences. Beyond it, requests wait — visible as
  `vllm:num_requests_waiting`.
- **Admission is per-iteration**, so queueing delay is a function of slot churn, not of job length.
- **Prefill and decode compete.** A long prefill admitted now delays decode for everyone in the batch
  (which chunked prefill exists to soften). This is why a policy that concentrates long uncached prefills
  on one replica hurts that replica's *whole* batch — and it is the mechanism behind the H2 crossover.

---

## 5. The six routing policies, from zero

A router sees an arriving request and picks a replica. Two things it might optimize, and they conflict:

- **Load balance** — keep every replica equally busy. Classic, cache-blind.
- **Cache locality** — send a request where its prefix already is. Cache-aware, load-blind.

| Policy | Rule | Optimizes | Fails when |
|---|---|---|---|
| **Round-robin** | next replica in order | nothing; it is the baseline | always the cache-blind floor |
| **Least-loaded** (LOR / JSQ) | fewest outstanding requests | load | ignores cache entirely; a warm prefix is wasted |
| **Power-of-two-choices** | sample 2 replicas, take the less loaded | load, cheaply | same cache-blindness, less coordination |
| **Consistent-hashing-on-prefix** | hash the prompt prefix → replica | locality, statelessly | hot prefix ⇒ hot replica; no load feedback at all |
| **Radix-prefix-affinity** | longest actual prefix match against each replica's resident blocks | locality, precisely | needs fleet cache state; can pile load on one replica |
| **Session-sticky** | same session → same replica | locality *by proxy* | **sees session identity, not prefix identity** — cross-user sharing of a common system prompt is invisible to it |

> ⭐ **That last row is the project's central differentiator, and it is worth being able to say cold.**
> Session-stickiness and prefix-affinity look similar and are not. If a thousand different users send the
> *same* system prompt, session-sticky spreads them across the fleet and every replica caches its own
> copy; prefix-affinity concentrates them and one copy serves all. `NetX-lab/Frontier` has sticky routing
> and **no** prefix-affinity router — that is the capability gap, not a marketing one.

### The crossover — why "prefix-affinity wins" is the boring answer

Locality and load pull in opposite directions. Sending every request carrying a hot prefix to one replica
maximizes hit rate and **concentrates queueing on that replica**. At high offered load the queueing delay
can exceed the prefill saving, and **least-loaded wins on p95 TTFT despite a worse hit rate.**

`azrabano23/disagg-sim` has already published *that* this counter-result exists. ~~**Nobody has published
the boundary** — the sharing ratio × load surface where it flips. That boundary is this project's finding
(H2), and it falls straight out of a sweep axis, not out of new machinery.~~

⛔ **Corrected 2026-10-05.** Two things were wrong with that. H2 was tested and **falsified**
(2026-09-28): no such boundary appeared. And the trade-off had been taken much further on real
clusters before this repo began: **LMetric** (OSDI '26) removes the weight between the cache score
and the load score by multiplying them, and derives when that fails; **SMetric** (arXiv 2607.08565)
routes agents by session and migrates a session whose instance would miss its SLO. See
[`04-prior-art.md`](04-prior-art.md) §2a.

---

## 6. Prometheus metrics, and the one that doesn't exist

vLLM exposes counters and histograms at `/metrics` in Prometheus exposition format — a text endpoint you
scrape. Counters only ever increase, so a *rate* is a difference between two scrapes.

⛔ **There is no `vllm:prefix_cache_hit_rate`.** The hit rate is `vllm:prefix_cache_hits` ÷
`vllm:prefix_cache_queries`. See **ADR-008** — this is the one factual detail in the project most likely
to be checked by a reader, and a competitor already got it wrong in public.

---

## 7. MIG — how one GPU becomes two replicas

> ⛔ **Not used in this project (decided 2026-09-22).** The hardware is Kaggle 2x T4, which has no MIG
> — and two physical GPUs isolate replicas completely, which is stronger. Kept because the caveat below
> about what shared hardware does to TTFT still applies to the memory-split fallback.

**Multi-Instance GPU** partitions an A100/H100 into isolated instances, each with its own SMs and its own
slice of HBM, each addressable as a separate device. Two vLLM servers on two MIG instances are, for our
purposes, two replicas.

**What it does and does not isolate** (and this caveat is load-bearing — see ADR-005): MIG partitions
compute and memory, so **hit rate** measurements are safe. The weaker fallback — two servers sharing one
un-partitioned card at ~45% memory each — does **not** isolate SMs, so **TTFT percentiles from it are
contaminated by test-rig contention** and must not be attributed to a routing policy.
