"""One adapter per input format. Each is a generator yielding `NormalizedRequest`.

This is one of the project's two extension points (the other is `policies/`). Vidur and Frontier
replay their own trace formats; NVIDIA's AISimulate reads Mooncake and AgentX traces too
(docs/04-prior-art.md §2a). An adapter is a single file, which is what makes "add a trace adapter"
an honest good-first-issue.
"""

from . import agentx, mooncake, openai_log, vllm_jsonl

# name -> module, so the CLI has ONE source of truth for "which formats do we read?"
REGISTRY = {"vllm-jsonl": vllm_jsonl, "openai-log": openai_log, "mooncake": mooncake,
            "agentx": agentx}


def get(name: str):
    """Look up an adapter, failing LOUDLY with the valid names (ADR-004 config semantics)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown trace format {name!r}; available: {sorted(REGISTRY)}"
        ) from None
