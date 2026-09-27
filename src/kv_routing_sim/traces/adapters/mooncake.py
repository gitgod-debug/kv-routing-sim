"""Adapter for Mooncake FAST'25 traces — real serving traffic as block hashes, with no text at all.

The format (kvcache-ai/Mooncake, `FAST25-release/README.md`) is one JSON object per line:

    timestamp      ms from the trace start
    input_length   prompt tokens
    output_length  generated tokens
    hash_ids       remapped prefix-block hashes; block size 512 tokens; identical ids = reusable KV

⚠️ THREE THINGS THIS ADAPTER DOES THAT CHANGE WHAT A RESULT MEANS — each is a parameter, and
each must be stated next to any number produced from it:

1. BLOCK SIZE. The trace hashes 512-token blocks; vLLM, and this simulator, cache 16-token
   blocks. Each trace block is expanded into sub-block hashes derived from its id, so sharing is
   exact at the trace's granularity — and at nothing finer. Two prompts that share 700 tokens
   share one trace block here (512 tokens), not 43 vLLM blocks: coarse blocks under-count
   partial-prefix hits.
2. TOKEN SCALE. `token_scale` shrinks every token count by one factor (1/16: a 512-token block
   becomes 32 tokens, two of our blocks). The sharing STRUCTURE — which requests share which
   blocks — is kept exactly; the sizes shrink, so prompts of up to 126K tokens fit a profile
   calibrated on ≤ 2K. A scaled replay is a statement about structure. On a fixed cache it also
   stands for a proportionally BIGGER cache: 1/16 of the trace on one T4's cache is the whole
   trace on a 16× larger one.
3. SESSIONS. The trace has none. With `infer_sessions`, a request whose prompt extends an earlier
   request's full blocks joins that request's session — an agent re-sends its history every
   turn. A shared system prompt looks like a history too, so a prefix that `hot_fanout` or more
   different next blocks follow is treated as a shared prompt, never as a conversation. (On the
   tool-agent trace three such prompts are followed by 3,335-8,894 different blocks; no other
   block by more than 15.) It is an inference, and every result that depends on it — anything
   about session-sticky — says so.
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
    """The j-th 16-token block inside trace block `trace_id`. The trace's ids already identify the
    whole prefix up to them, so (id, j) is enough to make equal prefixes hash equal."""
    data = trace_id.to_bytes(8, "big") + j.to_bytes(4, "big")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"mooncake").digest(), "big")


def _rows(path: str | Path, stats: TraceStats | None) -> Iterator[tuple[int, dict]]:
    """Valid rows, streamed; malformed ones counted by reason (ADR-004), never fatal."""
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
    threshold — bounded by the trace's number of distinct blocks, never by its number of requests.
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
    """Stream a Mooncake trace as `NormalizedRequest`s. See the module docstring for the three
    parameters that change what a replay means."""
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
            # The longest earlier request whose full blocks this prompt starts with — never through
            # a shared prompt, which thousands of unrelated requests also start with.
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
