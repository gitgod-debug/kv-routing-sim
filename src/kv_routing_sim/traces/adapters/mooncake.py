"""Adapter for Mooncake FAST'25 traces: real serving traffic as block hashes, with no text.

The format (kvcache-ai/Mooncake, `FAST25-release/README.md`) is one JSON object per line:

    timestamp      ms from the trace start
    input_length   prompt tokens
    output_length  generated tokens
    hash_ids       remapped prefix-block hashes; block size 512 tokens; identical ids = reusable KV

The trace hashes 512-token blocks while the simulator caches 16-token blocks, so each trace block is
expanded into sub-block hashes derived from its id. Sharing is exact at the trace's granularity and
invisible below it: two prompts sharing 700 tokens share one trace block (512 tokens) here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

from ..hashing import DEFAULT_BLOCK_SIZE
from ..schema import NormalizedRequest, TraceStats
from .base import open_maybe_gzip

TRACE_BLOCK_TOKENS = 512  # FAST25-release/README.md: "The block size is 512 tokens."


def _sub_hash(trace_id: int, j: int) -> int:
    """Hash sub-block j of trace block `trace_id`; the trace id already identifies the prefix."""
    data = trace_id.to_bytes(8, "big") + j.to_bytes(4, "big")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"mooncake").digest(), "big")


def _rows(path: str | Path, stats: TraceStats | None) -> Iterator[tuple[int, dict]]:
    """Yield valid rows; malformed ones are counted by reason (ADR-004) and skipped."""
    with open_maybe_gzip(path) as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if stats is not None:
                    stats.skip("json_decode_error")
                continue
            try:
                ids = [int(h) for h in obj["hash_ids"]]
                n_in, n_out = int(obj["input_length"]), int(obj["output_length"])
                ts = float(obj["timestamp"])
            except (KeyError, TypeError, ValueError):
                if stats is not None:
                    stats.skip("missing_or_bad_fields")
                continue
            if n_in < 1 or n_out < 0 or any(h < 0 for h in ids):
                if stats is not None:
                    stats.skip("invalid_record")
                continue
            if len(ids) != math.ceil(n_in / TRACE_BLOCK_TOKENS):
                if stats is not None:
                    stats.skip("block_count_mismatch")
                continue
            yield index, {"ids": ids, "n_in": n_in, "n_out": n_out, "ts": ts}


def _hot_prefixes(path: str | Path, hot_fanout: int) -> set[int]:
    """Blocks followed by `hot_fanout` or more different next blocks: shared prompts, not histories.

    A first pass over the file. Memory is one small set per distinct block id, capped at the
    threshold, so it is bounded by the trace's number of distinct blocks, never by its number of
    requests.
    """
    children: dict[int, set[int]] = {}
    hot: set[int] = set()
    for _, row in _rows(path, None):
        for a, b in pairwise(row["ids"]):
            if a in hot:
                continue
            nxt = children.setdefault(a, set())
            nxt.add(b)
            if len(nxt) >= hot_fanout:
                hot.add(a)
                del children[a]
    return hot


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    token_scale: float = 1.0,
    infer_sessions: bool = True,
    hot_fanout: int = 100,
    stats: TraceStats | None = None,
    **_: object,
) -> Iterator[NormalizedRequest]:
    """Stream a Mooncake trace as `NormalizedRequest`s.

    Args:
        token_scale: Factor applied to every token count; at 1/16 a 512-token block becomes 32
            tokens, two simulator blocks. Which requests share which blocks is kept exactly while
            sizes shrink, and on a fixed cache a scaled replay also stands for a proportionally
            larger cache.
        infer_sessions: The trace has no sessions. When set, a request whose prompt extends an
            earlier request's full blocks joins that request's session, since an agent re-sends its
            history every turn. This is an inference, and results about session-sticky routing
            depend on it.
        hot_fanout: A block followed by this many different next blocks or more is a shared prompt,
            never a conversation history. On the tool-agent trace three shared prompts are followed
            by 3,335-8,894 different blocks, and no other block by more than 15.
    """
    scaled_block = TRACE_BLOCK_TOKENS * token_scale
    per_block = scaled_block / block_size
    if token_scale <= 0 or per_block < 1 or per_block != int(per_block):
        raise ValueError(
            f"token_scale {token_scale} makes a {TRACE_BLOCK_TOKENS}-token trace block "
            f"{scaled_block:g} tokens, which is not a whole number of {block_size}-token blocks"
        )
    per_block = int(per_block)
    stats = stats if stats is not None else TraceStats()
    hot = _hot_prefixes(path, hot_fanout) if infer_sessions else set()

    first_ts: float | None = None
    registered: dict[int, str] = {}  # the id ending a request's full blocks -> its session
    for index, row in _rows(path, stats):
        ids, n_in = row["ids"], row["n_in"]
        full = n_in // TRACE_BLOCK_TOKENS
        hashes = [_sub_hash(t, j) for t in ids[:full] for j in range(per_block)]
        tail = int((n_in % TRACE_BLOCK_TOKENS) * token_scale) // block_size
        if tail and full < len(ids):
            hashes.extend(_sub_hash(ids[full], j) for j in range(tail))

        session = None
        if infer_sessions:
            # The longest earlier request whose full blocks this prompt starts with, never matched
            # through a shared prompt, which thousands of unrelated requests also start with.
            for pos in range(len(ids) - 2, -1, -1):
                s = registered.get(ids[pos])
                if s is not None and ids[pos] not in hot:
                    session = s
                    break
            if session is None:
                session = f"m{index}"
            if len(ids) >= 2:
                registered.setdefault(ids[-2], session)

        if first_ts is None:
            first_ts = row["ts"]
        stats.requests += 1
        yield NormalizedRequest(
            request_id=f"m{index}",
            arrival_ts=max(0.0, (row["ts"] - first_ts) / 1000.0),
            block_hashes=tuple(hashes),
            prompt_tokens=max(1, round(n_in * token_scale)),
            output_tokens=max(1, round(row["n_out"] * token_scale)),
            session_id=session,
        )
