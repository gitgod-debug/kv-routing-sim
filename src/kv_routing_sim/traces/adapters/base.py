"""Shared machinery for JSONL trace adapters.

A malformed line is skipped and counted by reason in `TraceStats`, since real serving logs are
dirty; a malformed configuration fails at startup instead (ADR-004).
"""

from __future__ import annotations

import datetime
import gzip
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ..hashing import DEFAULT_BLOCK_SIZE, approx_token_ids, chain_hashes
from ..schema import NormalizedRequest, TraceStats

# What converting one record's fields can raise. Each skips that record, never the whole file.
RECORD_ERRORS = (ValueError, TypeError, OverflowError, AttributeError, KeyError, IndexError)
# What decoding one line can raise: text that is not JSON, an integer of thousands of digits, or
# nesting so deep that the decoder gives up. Each skips that line.
LINE_ERRORS = (ValueError, RecursionError)
# Seconds since the epoch stay below this until the year 5138. A larger time is in milliseconds
# or finer, or is not a time.
LARGEST_SECONDS = 1e11


class RecordMapper(Protocol):
    """Turn one decoded JSON object into the fields an adapter needs, or raise to skip it."""

    def __call__(self, obj: dict[str, Any], index: int) -> dict[str, Any]: ...


def open_maybe_gzip(path: str | Path):
    # utf-8-sig drops a byte-order mark at the start of the file, which would spoil line one.
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8-sig", errors="replace")
    return p.open("r", encoding="utf-8-sig", errors="replace")


def parse_timestamp(value: Any, *, per_second: float = 1.0) -> float | None:
    """Read a timestamp in the trace's own unit, or return None if the value is not one.

    A number or a numeric string is taken as it is. An ISO 8601 string such as
    2026-11-27T10:00:00Z is converted from seconds since the epoch, at `per_second` units per
    second, and one without an offset is read as UTC. NaN, an infinity, a bool and anything else
    give None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if "_" in value:  # Python reads "1_0" as 10, and no log means that
            return None
        try:
            ts = float(value)
        except ValueError:
            try:
                when = datetime.datetime.fromisoformat(value.strip())
            except ValueError:
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.UTC)
            ts = when.timestamp() * per_second
    elif isinstance(value, (int, float)):
        try:
            ts = float(value)
        except OverflowError:  # an integer too large for a float
            return None
    else:
        return None
    return ts if math.isfinite(ts) else None


def whole_number(value: Any) -> int:
    """Read a count: an integer, a float without a fraction, or a string of digits.

    Raises ValueError or TypeError for anything else. A bool, 20.7 and "1_0" are refused, where
    `int()` would give 1, 20 and 10.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a count")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():  # also NaN and the infinities
            raise ValueError(f"{value!r} is not a whole number")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not (text.isascii() and text.lstrip("+-").isdigit()):
            raise ValueError(f"{value!r} is not a whole number")
        return int(text)
    raise TypeError(f"a {type(value).__name__} is not a count")


def unique_id(request_id: str, taken: dict[str, int], stats: TraceStats) -> str:
    """Return the id, or the id with #2, #3, ... appended if an earlier request already has it.

    Two requests with one id would share one entry in the simulator's books. `taken` holds every
    id given out so far, each with the last suffix tried for it.
    """
    if request_id not in taken:
        taken[request_id] = 1
        return request_id
    n = taken[request_id]
    while True:
        n += 1
        renamed = f"{request_id}#{n}"
        if renamed not in taken:
            break
    taken[request_id] = n
    taken[renamed] = 1
    stats.renamed_duplicate_ids += 1
    return renamed


def stream_jsonl(
    path: str | Path,
    mapper: RecordMapper,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    salt: int = 0,
    stats: TraceStats | None = None,
) -> Iterator[NormalizedRequest]:
    """Read a JSONL trace as `NormalizedRequest`s, in order of arrival.

    The prompt is hashed here and discarded (ADR-002), so nothing downstream sees prompt text or
    token ids. Time starts at the earliest valid timestamp, so the file is read to the end before
    the first request is returned. Requests that arrive together keep the file's order.

    Raises:
        ValueError: a timestamp is too large to be seconds. Read as seconds, a log in milliseconds
            would be replayed a thousand times slower, so it is refused.
    """
    stats = stats if stats is not None else TraceStats()
    rows = []  # (timestamp, request id, block hashes, prompt tokens, output tokens, session id)

    with open_maybe_gzip(path) as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except LINE_ERRORS:
                stats.skip("json_decode_error")
                continue
            if not isinstance(obj, dict):
                stats.skip("not_an_object")
                continue
            try:
                fields = mapper(obj, index)
            except RECORD_ERRORS:
                stats.skip("missing_or_bad_fields")
                continue
            ts = parse_timestamp(fields["arrival_ts"])
            if ts is None:
                stats.skip("bad_timestamp")
                continue

            token_ids = fields.get("token_ids")
            if token_ids is None:
                text = fields.get("prompt_text")
                if text is None:
                    stats.skip("no_prompt_tokens_or_text")
                    continue
                try:
                    token_ids = approx_token_ids(str(text))
                except ValueError:  # text that cannot be encoded, such as half a surrogate pair
                    stats.skip("invalid_record")
                    continue
            # A missing output length is unknown, not 0 tokens. An explicit 0 stays 0. Where the
            # log has only the request's limit, the limit stands in, and that is counted.
            ot, from_limit = fields.get("output_tokens"), False
            if ot is None:
                ot = fields.get("output_limit")
                from_limit = ot is not None
            if ot is None:
                stats.skip("no_output_length")
                continue
            try:
                # Every id, not only those in full blocks: a None or "n/a" anywhere is a bad record.
                # A string is not a list of ids, though Python would walk its characters.
                if not isinstance(token_ids, (list, tuple)):
                    raise TypeError("token ids must be a list")
                if not all(type(t) is int for t in token_ids):  # plain integers need no reading
                    token_ids = [whole_number(t) for t in token_ids]
                # `or` would treat a legitimate 0 as missing; be explicit.
                pt = fields.get("prompt_tokens")
                prompt_tokens = whole_number(pt) if pt is not None else len(token_ids)
                output_tokens = whole_number(ot)
                hashes = chain_hashes(token_ids, block_size, salt)
                request_id = str(fields.get("request_id") or f"r{index}")
                sid = fields.get("session_id")
                session_id = str(sid) if sid is not None else None
            except RECORD_ERRORS:
                stats.skip("invalid_record")
                continue
            if prompt_tokens < 0 or output_tokens < 0:
                stats.skip("invalid_record")
                continue
            if prompt_tokens == 0 or not token_ids:
                stats.skip("empty_prompt")
                continue
            if from_limit:
                stats.assume("max_tokens_as_output_length")
            rows.append((ts, request_id, hashes, prompt_tokens, output_tokens, session_id))

    if rows:
        largest = max((row[0] for row in rows), key=abs)
        if abs(largest) >= LARGEST_SECONDS:
            raise ValueError(
                f"a timestamp of {largest:,.0f} is too large to be seconds. This format takes "
                "seconds, since the epoch or from any start. If the log has milliseconds, divide "
                "them by 1,000 first")

    # Elapsed time from the earliest request: absolute epochs differ between logs and are not
    # comparable. The sort is stable, so equal times keep the file's order.
    rows.sort(key=lambda row: row[0])
    taken: dict[str, int] = {}
    for ts, request_id, hashes, prompt_tokens, output_tokens, session_id in rows:
        stats.requests += 1
        yield NormalizedRequest(
            request_id=unique_id(request_id, taken, stats),
            arrival_ts=ts - rows[0][0],
            block_hashes=hashes,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            session_id=session_id,
        )
