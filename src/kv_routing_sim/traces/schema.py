"""The normalized request record that every adapter emits, and the stats an adapter reports.

`NormalizedRequest` is a slotted frozen dataclass rather than a pydantic model because a sweep may
construct millions of them: about 160 bytes per instance against about 1.2 KB, which at a million
requests is the difference between roughly 150 MiB and 1.1 GiB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bump when a field is added, removed or reinterpreted. Every results file records the version it
# was produced under, so two curves are only ever compared when they are actually comparable.
SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class NormalizedRequest:
    """One request, as every adapter must produce it.

    The prompt is carried only as a chain of block hashes, with no token ids or text (ADR-002), so a
    trace can be replayed without its prompt text entering the simulator or any output.
    """

    request_id: str
    arrival_ts: float  # seconds from the start of the trace, monotonically non-decreasing
    block_hashes: tuple[int, ...]  # full blocks only; a trailing partial block is never shareable
    prompt_tokens: int
    output_tokens: int
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.arrival_ts < 0:
            raise ValueError(f"arrival_ts must be >= 0, got {self.arrival_ts}")
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be >= 0")


@dataclass(slots=True)
class TraceStats:
    """What an adapter saw while reading, recorded in the results header.

    Skipped lines are counted by reason, so a results file says how much of its input was dropped
    (ADR-004).
    """

    schema_version: int = SCHEMA_VERSION
    requests: int = 0
    skipped_lines: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_lines += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requests": self.requests,
            "skipped_lines": self.skipped_lines,
            "skip_reasons": dict(self.skip_reasons),
        }
