"""Adapter for OpenAI-style request logs (chat completions, and the `usage` block).

The distinguishing feature versus the vLLM adapter: the prompt is a `messages` LIST, and it must be
flattened IN ORDER. Order is not cosmetic here — a chat prompt's shared prefix is the leading system
and tool messages, so flattening them in any other order destroys the prefix structure the entire
simulation is about.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..hashing import DEFAULT_BLOCK_SIZE
from ..schema import NormalizedRequest, TraceStats
from .base import stream_jsonl


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = m.get("content", "")
        if isinstance(content, list):  # multimodal / content-parts form
            content = " ".join(
                str(c.get("text", "")) for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _first_present(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """First key that is present and non-None.

    ⛔ NOT `a or b or c`. A timestamp of 0 is falsy, so `or` would treat the first request of a
    trace normalized to start at t=0 as having no timestamp and skip it — silently, as a
    "malformed line". Our own writer emits exactly that trace, which is how this was caught.
    """
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _map(obj: dict[str, Any], index: int) -> dict[str, Any]:
    ts = _first_present(obj, ("created", "timestamp", "arrival_time"))
    if ts is None:
        raise KeyError("no timestamp field")

    text: str | None = None
    if isinstance(obj.get("messages"), list):
        text = _flatten_messages(obj["messages"])
    elif isinstance(obj.get("body"), dict) and isinstance(obj["body"].get("messages"), list):
        text = _flatten_messages(obj["body"]["messages"])
    elif obj.get("prompt") is not None:
        text = str(obj["prompt"])

    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    return {
        "arrival_ts": ts,
        "token_ids": obj.get("prompt_token_ids"),
        "prompt_text": text,
        "request_id": _first_present(obj, ("id", "request_id")) or f"r{index}",
        "session_id": _first_present(obj, ("session_id", "conversation_id", "user")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": _first_present(
            {**usage, **{k: v for k, v in obj.items() if k == "max_tokens"}},
            ("completion_tokens", "max_tokens"),
        ),
    }


def read(
    path: str | Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    salt: int = 0,
    stats: TraceStats | None = None,
) -> Iterator[NormalizedRequest]:
    return stream_jsonl(path, _map, block_size=block_size, salt=salt, stats=stats)
