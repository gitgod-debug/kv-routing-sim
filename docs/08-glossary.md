# 08 — Glossary

One line each. [`03-background-concepts.md`](03-background-concepts.md) explains *why* these matter;
this file is for looking one up.

---

## Serving mechanics

| Term | Definition |
|---|---|
| **KV cache** | Cached keys/values for already-processed tokens, so attention need not recompute the prefix at every step. |
| **Prefill** | The parallel pass over the whole prompt. **Compute-bound.** The part prefix caching can skip. |
| **Decode** | Generating one token at a time over the cache. **Memory-bandwidth-bound.** Cannot be skipped. |
| **TTFT** | Time to first token = queueing + prefill. **The metric routing moves.** |
| **TBT / ITL** | Time between tokens / inter-token latency. Decode-bound; barely affected by routing. |
| **PagedAttention** | vLLM's KV cache in fixed-size **blocks** with a per-sequence block table, like virtual memory. Makes blocks shareable. |
| **Block** | The unit of KV cache allocation and sharing, commonly 16 tokens. Partial blocks are **not** shared. |
| **APC** | vLLM's **Automatic Prefix Caching** — hash each block, reuse resident matches. |
| **RadixAttention** | SGLang's equivalent, organised as a radix tree over token sequences. |
| **Hash chain** | A block's hash includes its predecessors, so reuse is only possible as a **prefix**, never as a middle chunk. This is why the cache model is a trie. |
| **Eviction** | Dropping blocks when the cache fills, normally **LRU**. A hit requires the prefix to have *survived*, not just to have been seen — this is where fidelity dies (H4). |
| **Continuous batching** | Per-iteration admission: finished sequences leave the batch and waiting ones join, so a request waits for a **slot**, not for the batch to finish. |
| **Chunked prefill** | Splitting a long prefill across iterations so it doesn't stall everyone else's decode. |
| **`max_num_seqs`** | The cap on concurrent sequences per replica. Past it, requests wait. |
| **Prefix-sharing ratio** | The fraction of a workload's prompt tokens that are shared prefix. **The H2 sweep axis.** |
| **Agentic traffic** | Traffic where the system prompt, tool definitions and whole conversation are re-sent every turn — so sharing is most of the workload, not an edge case. |

## Routing

| Term | Definition |
|---|---|
| **Round-robin** | Next replica in order. The cache-blind baseline. |
| **Least-loaded / LOR / JSQ** | Fewest outstanding requests. Optimizes load, ignores cache. |
| **Power-of-two-choices** | Sample two replicas, take the less loaded. Nearly LOR's quality, far less coordination. |
| **Consistent-hashing-on-prefix** | Hash the prefix → replica. Stateless locality; no load feedback. |
| **Radix-prefix-affinity** | Route by **longest actual prefix match** against each replica's resident blocks. Precise locality. |
| **Session-sticky** | Same session → same replica. Locality *by proxy* — ⚠️ sees **session identity, not prefix identity**, so cross-user sharing of a common system prompt is invisible to it. |
| **The crossover / policy inversion** | The sharing-ratio × load boundary where prefix-affinity stops beating least-loaded, because locality concentrates queueing. **This project's finding (H2).** |

## Measurement

| Term | Definition |
|---|---|
| **Bootstrap CI** | Resample the data, recompute, take percentiles. ⚠️ Here the unit is a **session / trace segment**, not a request — requests are not independent (methodology §1). |
| **Paired bootstrap** | Resample once and compute the *difference* between two policies, since both ran on the same trace. Much tighter and the correct test. |
| **Pareto frontier** | The non-dominated set over (hit rate, p95 TTFT, ₹/1k requests). |
| **₹ per 1k requests** | The owned cost number. Denominator is **successful** (SLO-meeting) requests. |
| **The credential** | *"The simulator matches the live engine within X%, and here is why the residual is what it is."* Proves rigor. |
| **The finding** | *"Prefix-affinity stops paying above K replicas at S% sharing."* Says something about the world. **Different README section.** |
| **Residual** | Predicted minus measured. Publishing it **explained** is the point; publishing a match with no account of the gap reads as luck. |

## vLLM `/metrics` field names — grep targets, exact spelling

| Metric | Use |
|---|---|
| `vllm:prefix_cache_queries` | Hit-rate **denominator** |
| `vllm:prefix_cache_hits` | Hit-rate **numerator** |
| `vllm:time_to_first_token_seconds` | TTFT histogram |
| `vllm:num_requests_waiting` | Queue depth |
| `vllm:request_prefill_kv_computed_tokens` | Uncached prefill work actually performed |
| `vllm:kv_block_lifetime_seconds` | ⭐ Eviction behaviour (H4) — needs `--kv-cache-metrics-sample` |
| `vllm:kv_block_idle_before_evict_seconds` | ⭐ Eviction behaviour (H4) — needs `--kv-cache-metrics-sample` |
| `vllm:kv_block_reuse_gap_seconds` | ⭐ Eviction behaviour (H4) — needs `--kv-cache-metrics-sample` |
| ⛔ `vllm:prefix_cache_hit_rate` | **DOES NOT EXIST** (v0.29.0). The hit rate is `hits ÷ queries`. Writing this name is a competitor's actual mistake — see ADR-008. |

## Hardware

| Term | Definition |
|---|---|
| **Compute capability** | NVIDIA's arch version. T4 and **GTX 1650 are both 7.5** (Turing); A100 = 8.0; H100 = 9.0. Determines bf16/FP8 support and which vLLM attention backend is used. |
| **MIG** | Multi-Instance GPU — partitions an A100/H100 into isolated instances with their own SMs and HBM slice. How one card becomes two replicas (ADR-005). |
| **The memory-split fallback** | Two servers on one un-partitioned card at ~45% memory each. Isolates memory but **not SMs** — so hit rate is reportable, TTFT percentiles are **not**. |

## Project structure

| Term | Definition |
|---|---|
| **Trace adapter** | A pluggable reader turning one input format into `NormalizedRequest`. One of the two extension points. |
| **`NormalizedRequest`** | The versioned record every adapter emits. Carries block **hashes**, never prompt text. |
| **`spike_*.py` vs `run_*.py`** | Spikes are throwaway measurements with full provenance headers. `run_*` regenerates a README table and must be byte-identical on re-run. |
| **The protection order** | Under schedule compression, the **calibration** outranks extra policies. The simulator is the contribution; the calibration is the credential. |
