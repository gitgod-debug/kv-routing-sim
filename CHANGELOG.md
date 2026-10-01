# Changelog

Versions follow [semantic versioning](https://semver.org/). Before 1.0, the command-line interface and
the file formats may still change between minor versions; each change will be listed here.

## 0.1.0 — unreleased

The first public version: planned for PyPI before 2026-11-20.

### Added
- **`kvsim replay`** — a serving trace in; a routing policy, a replica count and ₹ per 1,000
  requests out, with ties on the replica count reported rather than hidden. Seven policies:
  round-robin, least-loaded, power-of-two, consistent-hash-prefix, radix-affinity,
  radix-affinity-balanced (with a load weight) and session-sticky.
- **The cache model** follows vLLM 0.30.0's block pool; plain LRU is kept as an ablation
  (`cache_model="lru"`). Checked against a live vLLM on a Tesla T4: exact on two sequential workloads
  under eviction pressure, +0.3% on a concurrent one.
- **Trace adapters:** `vllm-jsonl`, `openai-log`, and `mooncake` (the FAST'25 traces: 512-token
  block hashes, sessions inferred). Prompts enter as block hashes only; no text is ever read.
- **`kvsim calibrate`** — fit the context-aware timing model to your own running vLLM (decode cells
  over context length × batch, `--decode-contexts`; prefill over prompt length, cold and after a
  cached prefix, `--prefill-cached` and `--prefill-new`), read its KV capacity, and check the cache
  model against its counters, request by request. Standard library only.
- **A context-aware timing model** (C2): prefill and decode cost grow with context length. Both
  terms default to zero, so every earlier profile gives the same results as before.
- **`kvsim generate`** (agentic traffic with shared prompts, including many tenants) and
  **`kvsim crossover`** (sharing × load).
- **Shipped data:** the context-aware T4 profile C2 validated (`--profile
  t4-qwen3-4b-vllm0.30.0-ctx`, the one to use), the first, context-free one kept for the record
  (`t4-qwen3-4b-vllm0.30.0`), and the cited cost model (`--cost-model default`), byte-identical to
  `configs/`.
- The core install needs only PyYAML.

### Known limits
- Every millisecond, rupee and replica count from the shipped profiles is a **T4 figure**. Run
  `kvsim calibrate` for your own numbers.
- `kvsim calibrate`'s cache check and capacity reading were proven on a real T4 (C1); its
  context-aware timing fit so far only against the mock engine, though the same procedure predicted
  held-out traffic on a real T4 (C2).
- `kvsim calibrate` times prompts of up to 2,048 tokens and decode contexts of up to 1,024 by
  default. Longer prompts are an extrapolation unless you measure there (`--prefill-lengths`,
  `--prefill-cached`, `--decode-contexts`).
- `--format html` is not built; `md` and `json` are.
