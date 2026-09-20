"""Synthetic agentic traffic — the workload this project exists to reason about.

WHY THIS SHAPE (docs/03-background-concepts.md §3): an agent re-sends the system prompt, the tool
definitions and the whole conversation on every turn. So turn N's prompt is literally a PREFIX of
turn N+1's, and a fixed system prompt is shared across every session on the fleet. Prefix sharing is
most of the workload rather than an edge case, which is why routing decides the answer here and not
on single-turn chat traffic.

Two kinds of sharing exist and they are NOT the same thing — the distinction is the whole reason
session-sticky and prefix-affinity routing diverge:

  * CROSS-SESSION sharing — the system prompt + tool definitions, identical for every user on the
    fleet. Only a prefix-aware router can exploit this. Session-stickiness is blind to it.
  * WITHIN-SESSION sharing — a session's own conversation history. Either router exploits this.

`prefix_sharing_ratio` controls the first one, and it is the x-axis of the H2 crossover.

⚠️ THE KNOB IS A TARGET, NOT A RESULT. The achieved ratio depends on how long sessions actually run,
so `generate()` MEASURES what it produced and returns it alongside. Reporting the requested value as
though it were the observed one would be exactly the kind of quiet substitution this project claims
not to make.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from .hashing import (
    DEFAULT_BLOCK_SIZE,
    chain_hashes,
    cross_session_sharing_ratio,
    sharing_ratio,
)
from .schema import NormalizedRequest

# Token-id namespaces, kept disjoint so a "shared" token can never collide with a private one and
# manufacture sharing that the workload does not actually have.
_SHARED_BASE = 0
_SESSION_BASE = 10_000_000
_TURN_BASE = 500_000_000


class AgenticWorkload:
    """Generates multi-turn agentic traffic with a controllable cross-session sharing ratio.

    Every field is explicit; nothing is hidden in a default that a reader cannot see. The config
    that produced a trace is committed, and a trace is reproduced from (config + seed) rather than
    shipped as data (docs/06-methodology.md §1).
    """

    def __init__(
        self,
        *,
        n_sessions: int = 200,
        turns_per_session: int = 6,
        prefix_sharing_ratio: float = 0.6,
        session_context_tokens: int = 300,
        user_turn_tokens: int = 60,
        assistant_turn_tokens: int = 120,
        output_tokens: int = 120,
        arrival_rate: float = 10.0,  # sessions per second (Poisson)
        think_time_s: float = 1.5,  # gap between a session's turns
        block_size: int = DEFAULT_BLOCK_SIZE,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= prefix_sharing_ratio < 1.0:
            raise ValueError(f"prefix_sharing_ratio must be in [0, 1), got {prefix_sharing_ratio}")
        if turns_per_session < 1:
            raise ValueError("turns_per_session must be >= 1")
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")

        self.n_sessions = n_sessions
        self.turns_per_session = turns_per_session
        self.prefix_sharing_ratio = prefix_sharing_ratio
        self.session_context_tokens = session_context_tokens
        self.user_turn_tokens = user_turn_tokens
        self.assistant_turn_tokens = assistant_turn_tokens
        self.output_tokens = output_tokens
        self.arrival_rate = arrival_rate
        self.think_time_s = think_time_s
        self.block_size = block_size
        self.seed = seed
        self.shared_prefix_tokens = self._solve_shared_tokens()

    def _solve_shared_tokens(self) -> int:
        """Pick the shared-span length that hits the target ratio at the MEAN prompt length.

        A session's prompt grows every turn, so the instantaneous ratio falls as the conversation
        runs. We solve at the mean so the knob means something stable, and then measure what
        actually came out.

            mean_private = session_context + user_turn + mean_history
            shared       = r/(1-r) * mean_private
        """
        turn_growth = self.user_turn_tokens + self.assistant_turn_tokens
        mean_history = turn_growth * (self.turns_per_session - 1) / 2.0
        mean_private = self.session_context_tokens + self.user_turn_tokens + mean_history
        r = self.prefix_sharing_ratio
        if r <= 0.0:
            return 0
        return max(0, round(r / (1.0 - r) * mean_private))

    def _shared_tokens(self) -> list[int]:
        # Identical for every session on the fleet: the system prompt + tool definitions.
        return [_SHARED_BASE + i for i in range(self.shared_prefix_tokens)]

    def generate(self) -> tuple[list[NormalizedRequest], dict[str, object]]:
        """Produce the trace and a report of what was ACTUALLY produced."""
        requests = list(self._iter_requests())
        requests.sort(key=lambda r: r.arrival_ts)
        # TWO different numbers, reported separately because they answer different questions.
        # Cross-session is the one the knob controls and the one routing turns on.
        achieved_cross = cross_session_sharing_ratio(
            (r.block_hashes, r.session_id) for r in requests
        )
        achieved_total = sharing_ratio(r.block_hashes for r in requests)
        duration = requests[-1].arrival_ts if requests else 0.0
        report: dict[str, object] = {
            "requested_prefix_sharing_ratio": self.prefix_sharing_ratio,
            "achieved_cross_session_sharing": round(achieved_cross, 6),
            "achieved_total_block_reuse": round(achieved_total, 6),
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "requests": len(requests),
            "sessions": self.n_sessions,
            "trace_duration_s": round(duration, 3),
            "mean_prompt_tokens": (
                round(sum(r.prompt_tokens for r in requests) / len(requests), 1)
                if requests
                else 0.0
            ),
            "block_size": self.block_size,
            "seed": self.seed,
        }
        return requests, report

    def write_jsonl(self, path: str) -> dict[str, object]:
        """Write the trace as vLLM-style JSONL carrying REAL token ids.

        The in-memory `NormalizedRequest` deliberately drops token ids (ADR-002), so writing a
        trace has to regenerate them. Emitting placeholders instead would make every prompt in the
        file start 0,1,2,... — i.e. share a prefix with everything — and the round-trip would report
        a ~99% hit rate that is an artifact of the writer, not a property of the workload.
        """
        import json

        _, report = self.generate()
        # MUST be sorted the same way generate() sorts, or the written trace is a different trace:
        # the adapter normalizes timestamps against the FIRST line, so an unsorted file silently
        # clamps early arrivals to 0.
        ordered = sorted(self._iter_with_tokens(), key=lambda pair: pair[0].arrival_ts)
        lines: list[str] = []
        for req, ids in ordered:
            lines.append(
                json.dumps(
                    {
                        "request_id": req.request_id,
                        "timestamp": req.arrival_ts,
                        "session_id": req.session_id,
                        "num_prompt_tokens": req.prompt_tokens,
                        "num_generated_tokens": req.output_tokens,
                        "prompt_token_ids": ids,
                    }
                )
                + "\n"
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        return report

    def _iter_with_tokens(self) -> Iterator[tuple[NormalizedRequest, list[int]]]:
        """Same generation, but keeping the token ids. Used only by `write_jsonl`."""
        rng = random.Random(self.seed)
        shared = self._shared_tokens()
        clock = 0.0
        for s in range(self.n_sessions):
            clock += rng.expovariate(self.arrival_rate)
            session_id = f"s{s}"
            ctx_base = _SESSION_BASE + s * 100_000
            private = [ctx_base + i for i in range(self.session_context_tokens)]
            history: list[int] = []
            for turn in range(self.turns_per_session):
                turn_base = _TURN_BASE + s * 100_000 + turn * 1_000
                user = [turn_base + i for i in range(self.user_turn_tokens)]
                prompt_ids = shared + private + history + user
                yield (
                    NormalizedRequest(
                        request_id=f"{session_id}-t{turn}",
                        arrival_ts=round(clock + turn * self.think_time_s, 6),
                        block_hashes=chain_hashes(prompt_ids, self.block_size),
                        prompt_tokens=len(prompt_ids),
                        output_tokens=self.output_tokens,
                        session_id=session_id,
                    ),
                    prompt_ids,
                )
                history = history + user + [
                    turn_base + 500 + i for i in range(self.assistant_turn_tokens)
                ]

    def _iter_requests(self) -> Iterator[NormalizedRequest]:
        rng = random.Random(self.seed)
        shared = self._shared_tokens()

        clock = 0.0
        for s in range(self.n_sessions):
            # Poisson arrivals => exponential gaps between session starts.
            clock += rng.expovariate(self.arrival_rate)
            session_id = f"s{s}"
            ctx_base = _SESSION_BASE + s * 100_000
            private = [ctx_base + i for i in range(self.session_context_tokens)]

            history: list[int] = []
            for turn in range(self.turns_per_session):
                turn_base = _TURN_BASE + s * 100_000 + turn * 1_000
                user = [turn_base + i for i in range(self.user_turn_tokens)]
                # The real agentic structure: turn N's full prompt is a prefix of turn N+1's.
                prompt_ids = shared + private + history + user
                yield NormalizedRequest(
                    request_id=f"{session_id}-t{turn}",
                    arrival_ts=round(clock + turn * self.think_time_s, 6),
                    block_hashes=chain_hashes(prompt_ids, self.block_size),
                    prompt_tokens=len(prompt_ids),
                    output_tokens=self.output_tokens,
                    session_id=session_id,
                )
                # What the assistant said becomes part of the next turn's prompt.
                history = history + user + [
                    turn_base + 500 + i for i in range(self.assistant_turn_tokens)
                ]
