"""The block-level KV cache model, per replica. 📐 designed, not built.

Contract (ADR-002):
  * `block_trie.py`    — trie of resident block hashes on plain dicts. `lookup()` returns the
                         matched depth, i.e. the cached prefix length.
  * `eviction.py`      — LRU order + optional TTL + a hard capacity bound derived from a GPU memory
                         config.
  * `replica_cache.py` — trie + eviction + counters named to MIRROR vLLM's own, so the predicted
                         and measured quantities are literally the same quantity rather than two
                         things that sound alike.

⚠️ This component is TABLE STAKES, not the contribution. Vidur's `canary` branch and Frontier both
ship a block-level prefix-cache model. **The contribution is its measured fidelity** — see Phase 3.
Do not describe it as the core IP anywhere.

The tests that matter most in the whole project live here: a known prefix sequence must produce a
known hit/miss pattern, and the partial-block rule must hold (sharing 47 of 48 tokens at block size
16 shares TWO blocks, not three). If these are wrong, every downstream number is wrong and nothing
else will reveal it.
"""
