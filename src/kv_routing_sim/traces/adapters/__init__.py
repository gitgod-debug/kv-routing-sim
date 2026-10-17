"""Trace adapters, one module per input format, each a generator of `NormalizedRequest`s.

A new format is a new module registered in `REGISTRY`.
"""

from . import agentx, bailian, mooncake, openai_log, vllm_jsonl

# name -> module: the single list of formats the CLI reads.
REGISTRY = {"vllm-jsonl": vllm_jsonl, "openai-log": openai_log, "mooncake": mooncake,
            "agentx": agentx, "bailian": bailian}


def get(name: str):
    """Look up an adapter by name; an unknown name raises ValueError (ADR-004)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown trace format {name!r}; available: {sorted(REGISTRY)}"
        ) from None
