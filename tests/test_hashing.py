"""The hash-chain rules. If these are wrong, every hit rate in the project is wrong."""

from __future__ import annotations

from kv_routing_sim.traces.hashing import (
    chain_hashes,
    cross_session_sharing_ratio,
    sharing_ratio,
)


def test_partial_block_is_never_shared():
    """47 tokens at block size 16 is TWO full blocks, not 2.9. The leftover is recomputed."""
    assert len(chain_hashes(list(range(47)), block_size=16)) == 2
    assert len(chain_hashes(list(range(48)), block_size=16)) == 3
    assert len(chain_hashes(list(range(15)), block_size=16)) == 0


def test_shared_prefix_produces_identical_leading_hashes():
    a = chain_hashes(list(range(48)), 16)
    b = chain_hashes(list(range(48)) + list(range(100, 132)), 16)
    assert b[: len(a)] == a
    assert len(b) > len(a)


def test_hash_is_a_chain_not_a_set():
    """The same block in a different POSITION must hash differently — reuse is prefix-only."""
    forward = chain_hashes(list(range(16)) + list(range(100, 116)), 16)
    reversed_ = chain_hashes(list(range(100, 116)) + list(range(16)), 16)
    assert forward[0] != reversed_[0]
    assert set(forward).isdisjoint(reversed_)


def test_hashes_are_deterministic_across_processes():
    """Built-in hash() is per-process randomized; blake2b is not. The sweep must be re-derivable."""
    assert chain_hashes([1, 2, 3] * 16, 16) == chain_hashes([1, 2, 3] * 16, 16)


def test_salt_separates_logically_different_caches():
    assert chain_hashes(list(range(32)), 16, salt=0) != chain_hashes(list(range(32)), 16, salt=1)


def test_cross_session_ratio_ignores_within_session_reuse():
    """THE distinction the H2 x-axis depends on (docs/03 §5).

    One session re-sending its own blocks is high TOTAL reuse and ZERO cross-session sharing —
    a session-sticky router already captures all of it, so it must not count toward the axis that
    separates prefix-affinity from stickiness.
    """
    blocks = (1, 2, 3)
    one_session = [(blocks, "s1"), (blocks, "s1"), (blocks, "s1")]
    assert cross_session_sharing_ratio(one_session) == 0.0
    assert sharing_ratio([b for b, _ in one_session]) > 0.5

    two_sessions = [(blocks, "s1"), (blocks, "s2")]
    assert cross_session_sharing_ratio(two_sessions) == 1.0
