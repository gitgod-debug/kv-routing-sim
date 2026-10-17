"""Turn prompts into chained block hashes, following the engine's hash-chain rule.

A block's hash depends on its contents and on every block before it, so blocks can be reused only as
a prefix, and only full blocks are hashed: two prompts sharing their first 47 tokens at block size
16 share two blocks, not three. Hashes use blake2b rather than the built-in `hash()`, which is
randomised per process, so a sweep gives the same hashes on every run and machine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

DEFAULT_BLOCK_SIZE = 16  # vLLM's default


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def chain_hashes(
    token_ids: Sequence[int], block_size: int = DEFAULT_BLOCK_SIZE, salt: int = 0
) -> tuple[int, ...]:
    """Hash-chain a token sequence into one hash per full block.

    `salt` separates caches that share token ids but not KV, such as two models or two LoRA
    adapters; vLLM folds comparable things into its block keys.

    >>> a = chain_hashes(list(range(48)), block_size=16)
    >>> b = chain_hashes(list(range(48)) + [999], block_size=16)
    >>> a == b[:len(a)]      # a shared prefix produces identical leading hashes
    True
    >>> len(chain_hashes(list(range(47)), block_size=16))   # 47 tokens -> 2 full blocks, not 3
    2
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    n_full = len(token_ids) // block_size
    out: list[int] = []
    parent = _h64(salt.to_bytes(8, "big"))
    for i in range(n_full):
        block = token_ids[i * block_size : (i + 1) * block_size]
        body = b"".join(int(t).to_bytes(8, "big", signed=True) for t in block)
        payload = parent.to_bytes(8, "big") + body
        parent = _h64(payload)
        out.append(parent)
    return tuple(out)


def approx_token_ids(text: str, chars_per_token: float = 4.0) -> list[int]:
    """Return deterministic stand-in token ids for a trace that has text but no token ids.

    Real tokenizers do not split on whitespace, so block boundaries will not line up with the
    engine's: shared prefixes are still detected, but the number of shared blocks can be off by one
    at each boundary, which biases the hit rate. Use it only when a trace has no token ids. A real
    tokenizer would pull `transformers` into the core install (ADR-009).
    """
    words = text.split()
    ids: list[int] = []
    for w in words:
        n = max(1, round(len(w) / chars_per_token))
        base = _h64(w.encode("utf-8"))
        ids.extend((base + i) % (2**31) for i in range(n))
    return ids


def sharing_ratio(requests: Iterable[tuple[int, ...]]) -> float:
    """Return the fraction of all blocks that repeat a block seen earlier in the trace.

    This includes a session re-sending its own history; `cross_session_sharing_ratio` counts only
    blocks shared between sessions.
    """
    seen: set[int] = set()
    total = 0
    repeats = 0
    for blocks in requests:
        for b in blocks:
            total += 1
            if b in seen:
                repeats += 1
            else:
                seen.add(b)
    return repeats / total if total else 0.0


def cross_session_sharing_ratio(
    requests: Iterable[tuple[tuple[int, ...], str | None]],
) -> float:
    """Return the fraction of block occurrences whose block is used by more than one session.

    Unlike `sharing_ratio`, this ignores a session re-sending its own history, which session-sticky
    routing already captures. Blocks shared between sessions, such as a common system prompt, are
    what separate prefix affinity from session stickiness, so this is the sharing measure that
    matters for routing.
    """
    owners: dict[int, str | None] = {}
    multi: set[int] = set()
    occurrences: list[tuple[int, ...]] = []
    for blocks, session in requests:
        occurrences.append(blocks)
        for b in blocks:
            if b not in owners:
                owners[b] = session
            elif owners[b] != session:
                multi.add(b)

    total = sum(len(b) for b in occurrences)
    if not total:
        return 0.0
    shared = sum(1 for blocks in occurrences for b in blocks if b in multi)
    return shared / total
