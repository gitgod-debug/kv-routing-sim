"""The block-level KV cache model, per replica.

Contract (ADR-002):
  * `block_pool.py` — the default (`cache_model="vllm"`): vLLM 0.30.0's block pool, every
                      behaviour cited to its source — reference counts, tail-first freeing, cached
                      decode blocks, admission by free memory, preemption.
  * `block_trie.py` — the plain-LRU model (`cache_model="lru"`), kept as the ablation: a trie of
                      resident block hashes on plain dicts, LRU order, a hard capacity bound, and
                      counters named to MIRROR vLLM's own, so the predicted and measured quantities
                      are literally the same quantity rather than two things that sound alike.
  * `capacity.py`   — how many KV blocks a replica can hold, from a model's shape and a memory
                      budget. An estimate: a measured profile carries the engine's own number.

⚠️ This component is TABLE STAKES, not the contribution. Vidur's `canary` branch and Frontier both
ship a block-level prefix-cache model. **The contribution is its measured fidelity** — see Phase 3.
Do not describe it as the core IP anywhere.

The tests that matter most in the whole project live here: a known prefix sequence must produce a
known hit/miss pattern, and the partial-block rule must hold (sharing 47 of 48 tokens at block size
16 shares TWO blocks, not three). If these are wrong, every downstream number is wrong and nothing
else will reveal it.
"""
