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

⭐ HOW MANY DIFFERENT SHARED PREFIXES (added 2026-09-28, for H2b). Until this date every generated
trace had ONE system prompt for the whole fleet. One prompt of 85 blocks fits in every replica's
cache many times over, so every policy ends up caching it everywhere, and routing cannot win by
deciding where it lives — which is why session-sticky beat every prefix-aware policy in Phase 4.
Real fleets serve many tenants and agents, each with its own long prompt. `n_shared_prefixes` = K
distinct system prompts, and each session draws one with Zipf popularity (exponent
`prefix_zipf_s`) from its OWN random stream. With K = 1 (the default) no draw is made at all, so
every earlier trace is reproduced byte for byte — tests/test_phase3_prereg.py pins one by
fingerprint.
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
_PREFIX_STRIDE = 100_000  # shared prefix k lives at [k * stride, k * stride + length)
_SESSION_BASE = 10_000_000
_TURN_BASE = 500_000_000
MAX_SHARED_PREFIXES = _SESSION_BASE // _PREFIX_STRIDE  # 100: the namespaces must stay disjoint

# ⚠️ Those raw ids are far outside any real vocabulary, and a live engine REJECTS them. Found while
# building the Phase-3 notebook, 2026-09-22: the generator had only ever fed the simulator, which
# just hashes. So by default every raw id is mapped into ENGINE_SAFE_IDS. Qwen3's vocabulary is
# 151,936 with special tokens at 151,643-151,668; [1000, 100000) is ordinary text tokens there and
# in every mainstream tokenizer. The map is applied HERE, in the one generator, so the simulator and
# the engine see byte-identical ids. Accidental sharing needs all 16 ids of a block to collide at
# once — ~(1/99,000)^16 per pair of blocks, i.e. never.
ENGINE_SAFE_IDS = (1000, 100_000)
_MIX_MUL, _MIX_ADD, _MASK64 = 0x9E3779B97F4A7C15, 0x632BE59BD9B4E019, 0xFFFFFFFFFFFFFFFF


def _to_engine_id(raw: int, low: int, high: int) -> int:
    # A fixed 64-bit multiplicative mix: deterministic across processes, platforms and versions.
    return low + ((raw * _MIX_MUL + _MIX_ADD) & _MASK64) % (high - low)


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
        token_id_range: tuple[int, int] | None = ENGINE_SAFE_IDS,
        n_shared_prefixes: int = 1,
        prefix_zipf_s: float = 1.0,
    ) -> None:
        if not 0.0 <= prefix_sharing_ratio < 1.0:
            raise ValueError(f"prefix_sharing_ratio must be in [0, 1), got {prefix_sharing_ratio}")
        if turns_per_session < 1:
            raise ValueError("turns_per_session must be >= 1")
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")
        if not 1 <= n_shared_prefixes <= MAX_SHARED_PREFIXES:
            raise ValueError(
                f"n_shared_prefixes must be in [1, {MAX_SHARED_PREFIXES}], got {n_shared_prefixes}"
            )
        if prefix_zipf_s < 0:
            raise ValueError(f"prefix_zipf_s must be >= 0, got {prefix_zipf_s}")

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
        self.token_id_range = token_id_range
        self.n_shared_prefixes = n_shared_prefixes
        self.prefix_zipf_s = prefix_zipf_s
        self.shared_prefix_tokens = self._solve_shared_tokens()
        if self.shared_prefix_tokens >= _PREFIX_STRIDE:
            raise ValueError(
                f"a shared prefix of {self.shared_prefix_tokens} tokens would overlap the next "
                f"prefix's namespace ({_PREFIX_STRIDE} ids each)"
            )

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

    def _shared_tokens(self, k: int = 0) -> list[int]:
        # Identical for every session that uses prefix k: its system prompt + tool definitions.
        base = _SHARED_BASE + k * _PREFIX_STRIDE
        return [base + i for i in range(self.shared_prefix_tokens)]

    def prefix_assignment(self) -> list[int]:
        """Which shared prefix each session sends, by session index.

        Drawn from its OWN random stream, and only when K > 1: with one prefix nothing is drawn,
        so the arrival times — and every earlier trace — are unchanged.
        """
        if self.n_shared_prefixes == 1:
            return [0] * self.n_sessions
        rng = random.Random(f"shared-prefix-{self.seed}")
        ks = range(self.n_shared_prefixes)
        weights = [1.0 / (k + 1) ** self.prefix_zipf_s for k in ks]
        return rng.choices(ks, weights=weights, k=self.n_sessions)

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
        if self.n_shared_prefixes > 1:
            # Only for K > 1, so that every earlier trace's report is unchanged too. The working
            # set is what a replica must hold to serve every prefix the trace actually used.
            used = len(set(self.prefix_assignment()))
            prefix_blocks = self.shared_prefix_tokens // self.block_size
            report.update({
                "n_shared_prefixes": self.n_shared_prefixes,
                "prefix_zipf_s": self.prefix_zipf_s,
                "distinct_prefixes_used": used,
                "shared_prefix_blocks": prefix_blocks,
                "shared_working_set_blocks": used * prefix_blocks,
            })
        return requests, report

    def write_jsonl(self, path: str) -> dict[str, object]:
        """Write the trace as vLLM-style JSONL carrying the REAL token ids.

        The in-memory `NormalizedRequest` deliberately drops token ids (ADR-002), so writing a
        trace regenerates them from the same loop. The ids are what a live engine will be sent, so
        they are engine-safe by default (`token_id_range`).
        """
        import json

        _, report = self.generate()
        ordered = sorted(self._iter_with_tokens(), key=lambda pair: pair[0].arrival_ts)
        lines = [
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
            for req, ids in ordered
        ]
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        return report

    def iter_with_tokens(self) -> Iterator[tuple[NormalizedRequest, list[int]]]:
        """Public: every request with the exact token ids an engine must be sent, arrival-sorted."""
        return iter(sorted(self._iter_with_tokens(), key=lambda pair: pair[0].arrival_ts))

    def _iter_with_tokens(self) -> Iterator[tuple[NormalizedRequest, list[int]]]:
        """THE generation loop. Every other view of the trace is derived from this one."""
        rng = random.Random(self.seed)
        prefixes = [self._shared_tokens(k) for k in range(self.n_shared_prefixes)]
        assignment = self.prefix_assignment()
        remap = self.token_id_range
        clock = 0.0
        for s in range(self.n_sessions):
            # Poisson arrivals => exponential gaps between session starts.
            clock += rng.expovariate(self.arrival_rate)
            shared = prefixes[assignment[s]]
            session_id = f"s{s}"
            ctx_base = _SESSION_BASE + s * 100_000
            private = [ctx_base + i for i in range(self.session_context_tokens)]
            history: list[int] = []
            for turn in range(self.turns_per_session):
                turn_base = _TURN_BASE + s * 100_000 + turn * 1_000
                user = [turn_base + i for i in range(self.user_turn_tokens)]
                # The real agentic structure: turn N's full prompt is a prefix of turn N+1's.
                raw = shared + private + history + user
                ids = [_to_engine_id(x, *remap) for x in raw] if remap else raw
                yield (
                    NormalizedRequest(
                        request_id=f"{session_id}-t{turn}",
                        arrival_ts=round(clock + turn * self.think_time_s, 6),
                        block_hashes=chain_hashes(ids, self.block_size),
                        prompt_tokens=len(ids),
                        output_tokens=self.output_tokens,
                        session_id=session_id,
                    ),
                    ids,
                )
                # What the assistant said becomes part of the next turn's prompt.
                history = history + user + [
                    turn_base + 500 + i for i in range(self.assistant_turn_tokens)
                ]

    def _iter_requests(self) -> Iterator[NormalizedRequest]:
        for req, _ in self._iter_with_tokens():
            yield req
