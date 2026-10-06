# Upstream findings

Defects found in other projects while building and calibrating this one, and what was done about
each. Each was checked against that project's `main` on the date given.

## guidellm — trace replay loses one token per hash block with tokenizers that add a start token

**Status (2026-10-06): not yet reported upstream.** The fix is in this folder,
[`guidellm-add-special-tokens.patch`](guidellm-add-special-tokens.patch), made against guidellm
`main` at `246f3b0` (five days after v0.8.0).

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
