"""Adapter for vLLM-style JSONL request logs.

vLLM has no single log format, so each field is read from the first of several names seen in
practice (the alias tuples below). A log with `prompt_token_ids` is preferred, and a `prompt` that
is a list of integers is read as token ids too. With text alone, `hashing.approx_token_ids` stands
in, and its block boundaries will not match the engine's.
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
_OTOK = ("num_generated_tokens", "output_tokens", "completion_tokens")
_LIMIT = ("max_tokens",)  # the request's limit: it stands in for a missing count, and is counted


def _first(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _is_token_ids(value: Any) -> bool:
    """True for a list of integers, which vLLM accepts as a prompt of token ids."""
    return isinstance(value, list) and all(
        isinstance(t, int) and not isinstance(t, bool) for t in value
    )


def _map(obj: dict[str, Any], index: int) -> dict[str, Any]:
    ts = _first(obj, _TS)
    if ts is None:
        raise KeyError("no timestamp field")
    ids = _first(obj, _IDS)
    if ids is None and _is_token_ids(obj.get("prompt")):
        ids = obj["prompt"]
    return {
        "arrival_ts": ts,
        "token_ids": ids,
        "prompt_text": _first(obj, _TEXT),
        "request_id": _first(obj, _RID) or f"r{index}",
        "session_id": _first(obj, _SID),
        "prompt_tokens": _first(obj, _PTOK),
        "output_tokens": _first(obj, _OTOK),
        "output_limit": _first(obj, _LIMIT),
    }


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    salt: int = 0,
    stats: TraceStats | None = None,
) -> Iterator[NormalizedRequest]:
    return stream_jsonl(path, _map, block_size=block_size, salt=salt, stats=stats)
