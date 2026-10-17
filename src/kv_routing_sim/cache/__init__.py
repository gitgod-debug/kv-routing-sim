"""Block-level KV cache models, one instance per replica (ADR-002).

`block_pool` models vLLM's block pool and is the default (`cache_model="vllm"`); `block_trie` is a
plain LRU kept as an ablation baseline (`cache_model="lru"`); `capacity` estimates how many KV
blocks a replica can hold from a model's shape and a memory budget.
"""
