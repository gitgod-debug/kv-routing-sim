"""One adapter per input format. Each is a generator yielding `NormalizedRequest`.

This is one of the project's two extension points (the other is `policies/`), and it is the one
that separates this simulator from Vidur and Frontier: both replay their own trace formats. An
adapter is a single file, which is what makes "add a trace adapter" an honest good-first-issue.
"""

from . import mooncake, openai_log, vllm_jsonl

# name -> module, so the CLI has ONE source of truth for "which formats do we read?"
REGISTRY = {"vllm-jsonl": vllm_jsonl, "openai-log": openai_log, "mooncake": mooncake}


def get(name: str):
    """Look up an adapter, failing LOUDLY with the valid names (ADR-004 config semantics)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown trace format {name!r}; available: {sorted(REGISTRY)}"
        ) from None
