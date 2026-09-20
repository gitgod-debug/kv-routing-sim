"""Prompt -> block hashes, mirroring the engine's hash-chain rule.

THE RULE (docs/03-background-concepts.md §3): a block's hash depends on its contents AND on every
block before it. That is what makes reuse possible only as a PREFIX — you cannot reuse a middle
chunk — and it is why the cache is a trie rather than a set.

THE PARTIAL-BLOCK RULE: only FULL blocks are cacheable. Two prompts sharing the first 47 tokens at
block size 16 share TWO blocks (32 tokens), not 2.9. The leftover is recomputed every time. Getting
this wrong inflates every hit rate in the project, and no downstream check would catch it.

⚠️ Determinism: Python's built-in `hash()` is randomized per process for str/bytes, so it cannot be
used here — a sweep must be re-derivable across runs and machines (docs/06-methodology.md §1).
blake2b is used instead: stable across processes, platforms and versions.
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
    """Hash-chain a token sequence into one hash per FULL block.

    `salt` distinguishes logically different caches that happen to share token ids — e.g. two
    different models, or a different LoRA. vLLM folds comparable things into its own block keys.

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
    """Deterministic STAND-IN token ids for a trace that only has text, not token ids.

    ⚠️ THIS IS AN APPROXIMATION AND IT IS NOT FREE. Real tokenizers do not split on whitespace, so
    the block BOUNDARIES this produces will not line up with the engine's. Shared prefixes are still
    detected (identical text produces identical ids), but the number of shared BLOCKS can be off by
    one at each boundary, which biases the hit rate.

    Use it only when a trace has no token ids. Prefer `prompt_token_ids` in the trace. The bias this
    introduces is a named limitation in the Calibration Report, not something to discover later.

    No tokenizer dependency is taken on purpose: it would pull `transformers` into the CPU-only core
    install and break ADR-009.
    """
    words = text.split()
    ids: list[int] = []
    for w in words:
        n = max(1, round(len(w) / chars_per_token))
        base = _h64(w.encode("utf-8"))
        ids.extend((base + i) % (2**31) for i in range(n))
    return ids


def sharing_ratio(requests: Iterable[tuple[int, ...]]) -> float:
    """Fraction of all blocks that are repeats of a block seen earlier in the trace.

    The measured counterpart of the generator's `prefix_sharing_ratio` knob, and the x-axis of the
    H2 crossover. Reported so a user can locate their own workload on our published curve without
    running anything.
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
    """Fraction of block occurrences whose block is used by MORE THAN ONE session.

    ⚠️ This is the number that matters for routing, and it is NOT `sharing_ratio`.

    `sharing_ratio` counts every repeated block, which in agentic traffic is dominated by a session
    re-sending its own history. A session-sticky router already captures all of that for free. What
    separates prefix-affinity from session-stickiness is the blocks shared BETWEEN sessions — the
    system prompt and tool definitions that every user on the fleet sends.

    So this is the x-axis of the H2 crossover. Reporting total reuse there instead would make every
    workload look highly shared and would hide the effect entirely.
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
