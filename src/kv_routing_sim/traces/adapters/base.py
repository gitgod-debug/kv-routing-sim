"""Shared machinery for trace adapters. A new adapter stays one small file (ADR-004).

Error semantics, decided in ADR-004 and implemented here so every adapter inherits them:

  * A malformed LINE logs and continues, and is counted by reason in `TraceStats`. Real serving logs
    are dirty; refusing to read one because of a truncated final line would make the tool useless on
    exactly the input it was built for.
  * A malformed CONFIG fails loudly at startup. Data is dirty and expected to be; configuration is
    ours, and a typo there must never produce a plausible wrong number.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ..hashing import DEFAULT_BLOCK_SIZE, approx_token_ids, chain_hashes
from ..schema import NormalizedRequest, TraceStats


class RecordMapper(Protocol):
    """Turn one decoded JSON object into the fields we need, or raise to skip it."""

    def __call__(self, obj: dict[str, Any], index: int) -> dict[str, Any]: ...


def open_maybe_gzip(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return p.open("r", encoding="utf-8", errors="replace")


def stream_jsonl(
    path: str | Path,
    mapper: RecordMapper,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    salt: int = 0,
    stats: TraceStats | None = None,
) -> Iterator[NormalizedRequest]:
    """Stream a JSONL trace into `NormalizedRequest`s. NEVER loads the file into memory.

    The prompt is hashed HERE, at the adapter boundary, and discarded immediately (ADR-002). Nothing
    downstream ever sees prompt text or token ids. That is the memory bound and the privacy
    property, and it is the reason this function returns a generator rather than a list.
    """
    stats = stats if stats is not None else TraceStats()
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
            if not isinstance(obj, dict):
                stats.skip("not_an_object")
                continue
            try:
                fields = mapper(obj, index)
            except (KeyError, TypeError, ValueError):
                stats.skip("missing_or_bad_fields")
                continue

            token_ids = fields.get("token_ids")
            if token_ids is None:
                text = fields.get("prompt_text")
                if text is None:
                    stats.skip("no_prompt_tokens_or_text")
                    continue
                token_ids = approx_token_ids(str(text))

            ts = float(fields["arrival_ts"])
            if first_ts is None:
                first_ts = ts
            # Normalize to seconds-from-trace-start. Absolute epochs differ between logs and are
            # not comparable; elapsed time is.
            rel = max(0.0, ts - first_ts)

            # `or` would treat a legitimate 0 as missing; be explicit.
            pt = fields.get("prompt_tokens")
            prompt_tokens = int(pt) if pt is not None else len(token_ids)
            ot = fields.get("output_tokens")
            try:
                req = NormalizedRequest(
                    request_id=str(fields.get("request_id") or f"r{index}"),
                    arrival_ts=rel,
                    block_hashes=chain_hashes(token_ids, block_size, salt),
                    prompt_tokens=prompt_tokens,
                    output_tokens=int(ot) if ot is not None else 0,
                    session_id=(
                        str(fields["session_id"]) if fields.get("session_id") is not None else None
                    ),
                )
            except ValueError:
                stats.skip("invalid_record")
                continue

            stats.requests += 1
            yield req
