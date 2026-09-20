"""Adapter for vLLM-style JSONL request logs.

vLLM does not emit one canonical log format, so this accepts the field names seen in practice and
falls back sensibly. Extending it is a one-line change to the alias tuples below — which is the
point of ADR-004, and why "add a trace adapter" is an honest good-first-issue.

⚠️ Prefer a log carrying `prompt_token_ids`. If only text is present the adapter falls back to
`approx_token_ids`, whose block boundaries will not line up with the real tokenizer's — a named
limitation, not a silent one. See `hashing.approx_token_ids`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..hashing import DEFAULT_BLOCK_SIZE
from ..schema import NormalizedRequest, TraceStats
from .base import stream_jsonl

_TS = ("arrival_time", "timestamp", "arrival_ts", "created", "time", "start_time")
_IDS = ("prompt_token_ids", "token_ids", "input_ids")
_TEXT = ("prompt", "text", "input")
_RID = ("request_id", "id", "req_id")
_SID = ("session_id", "conversation_id", "user_id", "session")
_PTOK = ("num_prompt_tokens", "prompt_tokens", "n_prompt_tokens")
_OTOK = ("num_generated_tokens", "output_tokens", "completion_tokens", "max_tokens")


def _first(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _map(obj: dict[str, Any], index: int) -> dict[str, Any]:
    ts = _first(obj, _TS)
    if ts is None:
        raise KeyError("no timestamp field")
    return {
        "arrival_ts": ts,
        "token_ids": _first(obj, _IDS),
        "prompt_text": _first(obj, _TEXT),
        "request_id": _first(obj, _RID) or f"r{index}",
        "session_id": _first(obj, _SID),
        "prompt_tokens": _first(obj, _PTOK),
        "output_tokens": _first(obj, _OTOK),
    }


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    salt: int = 0,
    stats: TraceStats | None = None,
) -> Iterator[NormalizedRequest]:
    return stream_jsonl(path, _map, block_size=block_size, salt=salt, stats=stats)
