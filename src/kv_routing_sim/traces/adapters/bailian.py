"""Adapter for the Qwen-Bailian traces: Alibaba Qwen production traffic as block hashes, no text.

The format (alibaba-edu/qwen-bailian-usagetraces-anon, `README.md`; Apache-2.0) is one JSON object
per line:

    chat_id          this request
    parent_chat_id   the request it follows; -1 for the first request of a session
    timestamp        seconds from the trace start
    input_length     prompt tokens
    output_length    generated tokens
    turn             the turn number within the session
    hash_ids         salted hashes of the prompt's 16-token blocks, remapped to small integers

A request belongs to the session at the root of its `parent_chat_id` chain; sessions are the trace's
own and vary widely in size. The trace hashes each block on its own, so the adapter chains them: a
replay block's hash covers the previous block's hash and the trace ids it spans, so equal prefixes,
and only those, hash equal. A follow-up's prompt contains its parent's answer, whose block hashes
are unknown here, so it can hit only the parent's prompt and reuse within a session is
under-counted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from ..hashing import DEFAULT_BLOCK_SIZE
from ..schema import NormalizedRequest, TraceStats
from .base import open_maybe_gzip

TRACE_BLOCK_TOKENS = 16  # README.md: "Salted SipHash blocks (16 tokens per block)"
_CAPPED = "output_capped"  # kept beside the skip reasons, but it is not a skipped line


def _hash(prev: int, ids: list[int]) -> int:
    """One block of the replay: the chain so far, and the trace blocks it covers."""
    data = prev.to_bytes(8, "big") + b"".join(i.to_bytes(8, "big") for i in ids)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"bailian").digest(), "big")


def outputs_capped(stats: TraceStats) -> int:
    """How many requests had their output cut to fit `max_total_tokens`."""
    return stats.skip_reasons.get(_CAPPED, 0)


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    token_scale: float = 1.0,
    max_total_tokens: int | None = None,
    stats: TraceStats | None = None,
    **_: object,
) -> Iterator[NormalizedRequest]:
    """Stream a Qwen-Bailian trace as `NormalizedRequest`s.

    Args:
        block_size: Must equal the trace's 16-token blocks.
        token_scale: 1/k for a whole number k. Every token count shrinks k-fold, and one replay
            block stands for k trace blocks in a row, shared only if all k are; a system prompt
            shorter than k x 16 tokens is not seen. On a fixed cache a scaled replay also stands for
            a k-times larger cache.
        max_total_tokens: The engine's `max_model_len`. A request whose scaled prompt alone reaches
            it is dropped and counted as `prompt_too_long`; one whose prompt and output together
            exceed it has its output cut to fit, as the engine would stop it (see `outputs_capped`).
    """
    if block_size != TRACE_BLOCK_TOKENS:
        raise ValueError(
            f"the trace hashes {TRACE_BLOCK_TOKENS}-token blocks; a replay with {block_size}-token "
            "blocks would need hashes the trace does not have"
        )
    k = round(1.0 / token_scale) if token_scale > 0 else 0
    if k < 1 or abs(k * token_scale - 1.0) > 1e-9:
        raise ValueError(
            f"token_scale must be 1/k for a whole number k (1, 1/2, 1/16, ...), got {token_scale}"
        )
    stats = stats if stats is not None else TraceStats()

    session_of: dict[int, str] = {}  # chat id -> its session (the root of its parent chain)
    first_ts: float | None = None
    with open_maybe_gzip(path) as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.skip("json_decode_error")
                continue
            try:
                ids = [int(h) for h in obj["hash_ids"]]
                n_in, n_out = int(obj["input_length"]), int(obj["output_length"])
                ts = float(obj["timestamp"])
                chat, parent = int(obj["chat_id"]), int(obj["parent_chat_id"])
            except (KeyError, TypeError, ValueError):
                stats.skip("missing_or_bad_fields")
                continue
            if n_in < 1 or n_out < 0 or ts < 0 or any(h < 0 for h in ids):
                stats.skip("invalid_record")
                continue
            if len(ids) != -(-n_in // TRACE_BLOCK_TOKENS):
                stats.skip("block_count_mismatch")
                continue

            # A child whose parent is not in the file starts a session of its own.
            session = session_of.get(parent) or f"b{chat}"
            session_of[chat] = session

            prompt_tokens = max(1, round(n_in * token_scale))
            output_tokens = max(1, round(n_out * token_scale))
            if max_total_tokens is not None:
                if prompt_tokens >= max_total_tokens:
                    stats.skip("prompt_too_long")
                    continue
                if prompt_tokens + output_tokens > max_total_tokens:
                    output_tokens = max_total_tokens - prompt_tokens
                    stats.skip_reasons[_CAPPED] = stats.skip_reasons.get(_CAPPED, 0) + 1

            # Only full trace blocks can be shared, and only whole groups of k of them make a
            # block of the replay; never more blocks than the scaled prompt has.
            full = n_in // TRACE_BLOCK_TOKENS
            n_blocks = min(full // k, prompt_tokens // block_size)
            hashes, prev = [], 0
            for j in range(n_blocks):
                prev = _hash(prev, ids[j * k:(j + 1) * k])
                hashes.append(prev)

            if first_ts is None:
                first_ts = ts
            stats.requests += 1
            yield NormalizedRequest(
                request_id=f"b{chat}",
                arrival_ts=max(0.0, ts - first_ts),
                block_hashes=tuple(hashes),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                session_id=session,
            )
