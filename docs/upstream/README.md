# Upstream findings

Defects found in other projects while building and calibrating this one — and one found in the
tracker of a router whose dispatch-view this project's prefix-affinity policy models — and what was
done about each. Each was checked against that project's `main` on the date given.

## guidellm — trace replay loses one token per hash block with tokenizers that add a start token

**Status (2026-10-06): not yet reported upstream.** The fix is in this folder,
[`guidellm-add-special-tokens.patch`](guidellm-add-special-tokens.patch), made against guidellm
`main` at `246f3b0` (five days after v0.8.0).
*Re-checked 2026-10-18 against `main` at `5164c73`:* the function is unchanged and the patch applies;
the trace tests go from 191 to 192 passing; the numbers below reproduce with transformers 5.20.1 and
come out 2,048 / 1,536 / 256 with the patch; the new test fails without the source change. Draft pull
request #1222 still encodes the same way, now at two sites.

**What happens.** When a trace is replayed (Mooncake, WEKA or the minimal format),
`generate_token_ids()` in `src/guidellm/data/deserializers/trace_common.py` builds every hash
block with `processor.encode(text)`. A tokenizer that adds a start token in `encode()` — Llama,
Mistral, Gemma — puts one at the head of each block. The blocks are then joined and decoded with
`skip_special_tokens=True`, which drops every start token inside the prompt. So each block comes
out one token short: replayed prompts are shorter than the trace's `input_length`, and rows that
share hash IDs share fewer tokens than they should.

**Measured,** with guidellm's own two functions and real tokenizers (transformers 5.19.0):

| Tokenizer | Blocks | Expected tokens | Got |
|---|---|---:|---:|
| TinyLlama (a Llama-2 tokenizer) | 4 × 512 | 2,048 | **2,044** |
| the same, the prefix two rows share | 3 × 512 | 1,536 | **1,533** |
| the same | 4 × 64 (WEKA's default block) | 256 | **252** |
| Qwen3 (adds no start token) | 4 × 512 | 2,048 | 2,048 |

The server adds one start token of its own when it tokenizes the text, so its own count is one
higher in each failing row: 2,045, 1,534 and 253.

**The fix.** `processor.encode(text, add_special_tokens=False)`, a regression test, and the test
doubles updated to accept the argument. With it the same run gives 2,048 of 2,048 and 1,536 of
1,536. On guidellm's `main` with the patch, its 116 trace tests pass; the new test fails without
the source change, which is what it is for.

**Two things to know.** guidellm's unit tests use a whitespace mock that never adds special
tokens, which is why they cannot see this. And a draft pull request there (#1222, opened
2026-10-02) is rewriting the same function; its new code still encodes without
`add_special_tokens=False`.

**A second, smaller effect, not fixed by the patch.** The prompt is sent as text, so the server's
tokenizer can merge tokens differently where two blocks meet: Qwen3 at 64-token blocks gives 255
of 256.

## vLLM — the eviction histograms count recycled blocks that never held a cached prefix

**Status: already reported upstream, with a fix. Nothing was filed from here.** vLLM pull request
#47715, *"[Bugfix] Fix KV cache metrics event handling"*, open since 2026-07-06, names it as its
second defect and makes the same change. Checked against vLLM `main` on 2026-10-06: the code was
still as described below.

**What happens (vLLM 0.30.0).** In `vllm/v1/core/block_pool.py`, `_maybe_evict_cached_block()`
calls `metrics_collector.on_block_evicted(block)` before it checks whether the block holds a
cached hash. So recycling a block that never held a cached prefix — the partial last block of a
finished request — is recorded as an eviction in `vllm:kv_block_lifetime_seconds`,
`vllm:kv_block_idle_before_evict_seconds` and `vllm:kv_block_reuse_gap_seconds`.

**Measured** on a Tesla T4 (`Qwen/Qwen3-0.6B`, `--enable-prefix-caching --kv-cache-metrics
--kv-cache-metrics-sample 1.0`): 65 requests with the same prompt, sent one at a time, on a cache
that was never under pressure, recorded **64** "evictions" — one per request after the first,
each about 81 ms old, with no reuse gap. The raw scrape is
[`metrics_caching_on.txt`](../../artifacts/results/phase0_v2_kaggle_t4/metrics_caching_on.txt).

This repo's cache model counts such events the way the engine does, on purpose, so that its
events can be compared with the engine's histograms
(`src/kv_routing_sim/cache/block_pool.py`, `_maybe_evict`).

## production-stack — the prefix-aware router's trie forgets nothing, so the router grows until it is restarted

**Status (2026-10-18): fix prepared, not yet submitted.** The patch is in this folder,
[`production-stack-prefix-trie-bound.patch`](production-stack-prefix-trie-bound.patch), made against
production-stack `main` at `7b1164d`. Issue #687, open since 2025-09-08 and confirmed by a second
deployment in January 2026; an earlier pull request, #711, stalled in September 2025 on a CI failure
and unanswered review.

**What happens.** `PrefixAwareRouter` (`src/vllm_router/routers/routing_logic.py`) finds the longest
prefix match in a `HashTrie` of 128-character chunk hashes (`src/vllm_router/prefix/hashtrie.py`)
built from the router's own dispatch history. No node is ever removed — the class says "we assume
that there is no eviction of prefix cache". On a workload with ever-new prompts the trie grows for as
long as the router runs, 663 bytes a node and 253 MiB after 100,000 four-chunk prompts (`tracemalloc`,
Python 3.13), and a pod with a memory limit is restarted. It also keeps routing to prefixes the
engines evicted long ago: the drift `policies/radix_affinity.py` here models on purpose, and the
reason its own dispatch map is LRU-bounded.

**The fix.** `--prefix-max-chunks-per-endpoint` (`routerSpec.prefixMaxChunksPerEndpoint` in the Helm
chart). Above 0, each endpoint's least recently used chunks are forgotten first, the tail chunks of a
prompt before its head chunks — the order in which vLLM frees a finished request's KV blocks
(`single_type_kv_cache_manager.py`: "free blocks in reverse order so that the tail blocks are freed
first", the order `cache/block_pool.py` here reproduces) — so the trie's picture of each cache ages
the way the cache does and each endpoint's listings stay prefix-closed. A node is detached once no
endpoint lists it. The default, 0, keeps today's behaviour. Bounded at 10,000 chunks per endpoint
with four endpoints, the trie holds 30 MiB flat through 200,000 prompts. Eight trie tests and one
end-to-end router test; 17 pass with the existing prefix-router tests; black, isort, ruff and
codespell clean on the changed files. **Not measured:** the hit rate the engines see with the bound
against without, which needs live replicas.

**Two things to know.** The bound is sized by hand from `vllm:cache_config_info` (`num_gpu_blocks ×
block_size` tokens, about 32 tokens a chunk); reading it through the router's own stats scraper is a
follow-up. And a bound below the engine's capacity forgets prefixes the engine still holds — the doc
page the patch adds says so.
