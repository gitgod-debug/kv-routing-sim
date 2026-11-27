"""Adapter for SemiAnalysis's AgentX traces: real coding-agent sessions as block hashes, no text.

The format (huggingface.co/datasets/semianalysisai/cc-traces-weka-042026, README.md) has one JSON
object per line, one agent session each:

    id             the session, e.g. "trace_0001"
    models         the session's models; the first is its main one
    block_size     64: each hash id covers one full 64-token block of input
    hash_id_scope  "local": ids are comparable only within one session
    requests       in order; each has t (seconds since the session began), type ("n", "s", or
                   "subagent", which carries its own nested requests, timed from its own start),
                   model, in, out (tokens), hash_ids, api_time, think_time

Because hash ids are local, each session's blocks are kept distinct from every other session's: the
replay shows reuse within a session and none across users, unlike Mooncake's global ids. Sessions
are the trace's own, not inferred. A file whose name ends in .gz is read compressed.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..hashing import DEFAULT_BLOCK_SIZE
from ..schema import NormalizedRequest, TraceStats

TRACE_BLOCK_TOKENS = 64  # README.md: "KV block size: 64 tokens"
ID_LIMIT = 2**64  # a hash id is hashed as 8 bytes


class _Malformed(ValueError):
    """A session the replay cannot place in time. It is skipped and counted as invalid_record."""


def _hash(scope: str, prev: int, ids: list[int]) -> int:
    """One simulator block: the session, the chain so far, and the trace blocks it covers."""
    data = scope.encode() + prev.to_bytes(8, "big") + b"".join(i.to_bytes(8, "big") for i in ids)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"agentx").digest(), "big")


def _open(path: str | Path):
    p = Path(path)
    return gzip.open(p, "rb") if p.suffix == ".gz" else p.open("rb")


def _seconds(value: Any, *, signed: bool = False) -> float:
    """A finite number of seconds, not negative unless `signed`."""
    try:
        s = float(value)
    except OverflowError:  # an integer too large for a float
        raise _Malformed(f"{value!r} seconds") from None
    if not math.isfinite(s) or (s < 0 and not signed):
        raise _Malformed(f"{value!r} seconds")
    return s


def _flat(d: Any) -> Iterator[tuple[float, str, dict[str, Any]]]:
    """Yield (time from session start, session id, request); a sub-agent is its own session.

    Raises _Malformed for a line that is not an object, a `models` that is not a list, a request
    that is not an object, and a `t` that is negative or not finite.
    """
    if not isinstance(d, dict) or not isinstance(d.get("models") or [], list):
        raise _Malformed("not a session")
    for r in d.get("requests") or []:
        if not isinstance(r, dict):
            raise _Malformed("a request is not an object")
        if r.get("type") == "subagent":
            sid = f"{d['id']}/{r.get('agent_id', 'subagent')}"
            for q in r.get("requests") or []:
                if not isinstance(q, dict):
                    raise _Malformed("a request is not an object")
                yield _seconds(r["t"]) + _seconds(q["t"]), sid, q
        else:
            yield _seconds(r["t"]), str(d["id"]), r


def _index(path: str | Path, stats: TraceStats) -> list[tuple[int, float]]:
    """First pass: (line number, span in seconds) per session. Memory: two numbers per session."""
    out = []
    with _open(path) as fh:
        for line, raw in enumerate(fh):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
                span = max((t + _seconds(q.get("api_time") or 0.0, signed=True)
                            for t, _, q in _flat(d)), default=0.0)
            except _Malformed:
                stats.skip("invalid_record")
                continue
            except (KeyError, TypeError, ValueError):  # including a line that is not JSON
                stats.skip("bad_session")
                continue
            out.append((line, span))
    return out


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    token_scale: float = 1 / 32,
    concurrency: int = 32,
    window_s: float | None = None,
    stagger_s: float = 600.0,
    seed: int = 0,
    models: str = "main",
    stats: TraceStats | None = None,
    **_: object,
) -> Iterator[NormalizedRequest]:
    """Replay AgentX sessions `concurrency` at a time, as `NormalizedRequest`s in arrival order.

    The sessions were recorded separately, each from its own zero. Each of `concurrency` slots plays
    sessions back to back in a seeded shuffle, a session starting when the one before it would have
    finished, and a request arrives at its session's start plus its recorded `t`, which includes the
    recorded think and tool times (open loop). Results depend on these choices, so state them with
    any number produced.

    Args:
        token_scale: Factor applied to every token count. At the default 1/32, eight 64-token trace
            blocks (512 tokens) become one 16-token block: reuse finer than 512 tokens is not seen,
            and a partial group at the end is never shared.
        concurrency: Number of sessions replayed at once.
        window_s: Keep only requests that arrive before this time, in seconds; None keeps all.
        stagger_s: The slots start at random times spread over this many seconds.
        seed: Seed for the session order and the slots' start times.
        models: "main" drops requests to any model other than the session's main one, counted as
            `other_model`; "all" keeps them. A sub-agent is its own session either way.
    """
    if not math.isfinite(token_scale) or token_scale <= 0:
        raise ValueError(f"token_scale must be a number above 0, got {token_scale}")
    group = block_size / (TRACE_BLOCK_TOKENS * token_scale)
    if group < 1 or group != int(group):
        raise ValueError(
            f"token_scale {token_scale} makes a {TRACE_BLOCK_TOKENS}-token trace block "
            f"{TRACE_BLOCK_TOKENS * token_scale:g} tokens, which does not divide a "
            f"{block_size}-token block")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if models not in ("main", "all"):
        raise ValueError(f"models must be 'main' or 'all', got {models!r}")
    group = int(group)
    stats = stats if stats is not None else TraceStats()
    # Every number produced depends on these choices, so the trace's statistics carry them.
    stats.scaling = {"token_scale": token_scale, "concurrency": concurrency,
                     "stagger_s": float(stagger_s)}

    # The schedule: which session starts when. Seeded, and decided before any session is parsed.
    index = _index(path, stats)
    order = list(range(len(index)))
    random.Random(f"agentx-{seed}").shuffle(order)
    rng = random.Random(f"agentx-slots-{seed}")
    slots = [(rng.uniform(0.0, stagger_s), j) for j in range(concurrency)]
    heapq.heapify(slots)
    starts: list[tuple[int, float]] = []  # (session index, start time)
    for k in order:
        free, j = heapq.heappop(slots)
        if window_s is not None and free >= window_s:
            break
        starts.append((k, free))
        heapq.heappush(slots, (free + index[k][1], j))

    # Second pass, through the file in order, since a compressed file is slow to read out of
    # order: parse only the scheduled sessions, and keep what arrives inside the window.
    scheduled = {index[k][0]: (rank, start) for rank, (k, start) in enumerate(starts)}
    rows = []
    with _open(path) as fh:
        for line, raw in enumerate(fh):
            if line not in scheduled:
                continue
            rank, start = scheduled[line]
            d = json.loads(raw)
            main = (d.get("models") or [None])[0]
            # A session sends the same leading blocks on nearly every turn. Each distinct block is
            # hashed once, and its value is one shared object, so a full-size trace fits in memory.
            known: dict[tuple, int] = {}
            for place, (t, sid, q) in enumerate(_flat(d)):
                arrival = start + t
                if window_s is not None and arrival >= window_s:
                    continue
                if models == "main" and q.get("model") != main:
                    stats.skip("other_model")
                    continue
                try:
                    ids = [int(h) for h in q["hash_ids"]]
                    n_in, n_out = int(q["in"]), int(q["out"])
                except (KeyError, TypeError, ValueError):
                    stats.skip("missing_or_bad_fields")
                    continue
                except OverflowError:  # 1e999 is read as an infinity
                    stats.skip("invalid_record")
                    continue
                if n_in < 1 or n_out < 0 or len(ids) != n_in // TRACE_BLOCK_TOKENS:
                    stats.skip("block_count_mismatch")
                    continue
                if any(not 0 <= h < ID_LIMIT for h in ids):
                    stats.skip("invalid_record")
                    continue
                try:
                    sizes = (max(1, round(n_in * token_scale)), max(1, round(n_out * token_scale)))
                except OverflowError:  # an output length too large to scale
                    stats.skip("invalid_record")
                    continue
                hashes, prev = [], 0
                for g in range(len(ids) // group):
                    key = (sid, prev, *ids[g * group:(g + 1) * group])
                    block = known.get(key)
                    if block is None:
                        block = known[key] = _hash(sid, prev, ids[g * group:(g + 1) * group])
                    hashes.append(block)
                    prev = block
                rows.append((arrival, rank, place, sid, tuple(hashes), sizes))

    # By arrival, then by the schedule, then by each session's own order of requests.
    rows.sort(key=lambda r: r[:3])
    for i, (arrival, _, _, sid, hashes, (prompt_tokens, output_tokens)) in enumerate(rows):
        stats.requests += 1
        yield NormalizedRequest(
            request_id=f"a{i}",
            arrival_ts=arrival,
            block_hashes=hashes,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            session_id=sid,
        )
