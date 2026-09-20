"""The normalized request record — the one type every adapter emits.

⚠️ AMENDMENT TO ADR-004, 2026-09-20: this is a `@dataclass(slots=True, frozen=True)`, NOT a pydantic
model as that ADR originally said. Recorded here rather than silently contradicted.

Why the change: `NormalizedRequest` is the hot path. A 1M-request sweep constructs a million of them
and the sim holds the in-flight ones, so per-instance cost is a memory-ceiling question — and the
ceiling is something this project promises to state ("what breaks at 10x and 100x").

MEASURED on this box, 2026-09-20, 100k instances under tracemalloc — not estimated:

    dataclass(slots=True, frozen=True)    159 bytes    0.81 us to construct
    pydantic BaseModel                  1,215 bytes    1.15 us
                                         7.65x          1.43x

⚠️ The construction-speed difference is small and is NOT the reason. **Memory is.** At 1M requests
that is 152 MiB versus 1,159 MiB for the records alone, on a box with ~7 GB available — the
difference between a sweep that fits and one that does not.

**Pydantic is still used where ADR-004 wanted it — for CONFIGS** (`configs/*.yaml`): parsed once,
must fail loudly on a typo, and not on any hot path. Validation belongs where the cost is
paid once, not a million times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bump when a field is added, removed or reinterpreted. Every results file records the version it
# was produced under, so two curves are only ever compared when they are actually comparable.
SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class NormalizedRequest:
    """One request, as every adapter must produce it.

    `block_hashes` carries the request's prompt as a chain of BLOCK HASHES and nothing else — no
    token ids, no text (ADR-002). This is the memory bound and also the privacy property: a user's
    trace can be replayed without their prompt text ever entering the simulator or an artifact.
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
    """What an adapter saw while reading. Travels into the results header.

    `skipped_lines` exists because a run that silently dropped 12% of its input is a different run
    from one that read all of it, and the results file must say which it was (ADR-004).
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
