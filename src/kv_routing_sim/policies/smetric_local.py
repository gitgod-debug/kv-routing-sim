"""SMetric's routing rule, without its global cache tier: stick a session's follow-ups to the
replica that holds its history — unless that replica would miss the first-token deadline; then
move the session.

Wang et al., "SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric
Scheduling", arXiv 2607.08565 (v2, 2026-09-13). ⭐ NOT THIS PROJECT'S IDEA (docs/04-prior-art.md
§2a). Written from the paper's Figure 27 — its pseudocode for engines that prefill and decode on
the same instance, which is what this simulator models:

    c    = hit length of the request on each instance
    q    = prefill cost already queued at each instance
    prev = argmax c                               # "likely served last turn"
    if turn != 0 and session_cached(prev) and meets_ttft(prev):   stick to prev
    else:                             balance: argmin  estimate_load(i) · num_requests(i)

    session_cached(i):  c[i] > HIT_RATIO · est_hit        # est_hit: the hit the history implies
    meets_ttft(i):      estimate_load(i) / PREFILL_RATE <= SLACK · ttft_slo
    estimate_load(i):   q[i] + prefill_cost(request, c[i])
    prefill_cost:       C_LIN · n + C_ATT · n · (L − n/2)       # n new tokens of a prompt of L

That cost is the form of this simulator's own prefill model (sim/cost_model.py; the paper takes
it from KunServe, EuroSys '26), so the run's timing profile supplies C_LIN and C_ATT, costs are in
seconds, and PREFILL_RATE is 1.

⚠️ FOUR DEPARTURES FROM THE PAPER — each could move a result, so a result from this policy is a
result about THIS rule, not about SMetric:

1. NO GLOBAL TIER. SMetric assumes a shared KV store (Mooncake, LMCache) from which a migrated
   session's cache is fetched. This simulator has none: a session that moves recomputes its
   history. The paper itself reports that its lead does not hold "for setups without any global
   KV$ store". That is the setting here.
2. NO TPOT FILTER. Figure 27 first drops instances whose running decodes the new prefill would
   push past their per-token deadline. That needs each decode's remaining slack, which this
   router does not track. Left out, and said.
3. THE TURN COMES FROM THE SESSION ID. The paper counts the messages in the request, because a
   provider's requests carry no session id. A trace's do; the answer is the same.
4. THE BALANCE SCORE has LMetric's `+ 1` — estimate_load · (requests + 1), as its authors' router
   computes it (policies/lmetric.py) — where the paper's simplified figure has none.

`SLACK` and `HIT_RATIO` are the paper's two settings. It reports that results move by about 1%
over SLACK 0.5–2 and 2% over the whole range of HIT_RATIO, and its text names no default: 1.0 and
0.5 are this file's choices.
"""

from __future__ import annotations

import random

from ..cache.block_pool import max_hit_blocks
from ..traces.schema import NormalizedRequest
from .base import FleetState, argmin_random
from .lmetric import LMetric


class SMetricLocal(LMetric):
    name = "smetric-local"

    def __init__(
        self,
        n_replicas: int,
        *,
        ttft_slo_s: float = 2.0,
        slack: float = 1.0,
        hit_ratio: float = 0.5,
        max_sessions: int = 1_000_000,
        **kwargs: object,
    ) -> None:
        profile = kwargs.get("profile")
        if profile is None or not hasattr(profile, "prefill_s"):
            raise ValueError(
                "smetric-local estimates prefill time with the run's timing profile; build it "
                "through run(), or pass profile=<CostProfile>"
            )
        if ttft_slo_s <= 0 or slack <= 0 or not 0 < hit_ratio < 1:
            raise ValueError("need ttft_slo_s > 0, slack > 0 and 0 < hit_ratio < 1")
        super().__init__(n_replicas, **kwargs)
        self._profile = profile
        self.ttft_slo_s = ttft_slo_s
        self.slack = slack
        self.hit_ratio = hit_ratio
        self.max_sessions = max_sessions
        self._queued_cost = [0.0] * n_replicas  # seconds of prefill committed and not yet done
        self._pending_cost: dict[str, tuple[int, float]] = {}
        self._history: dict[str, int] = {}  # session -> the prompt tokens of its last request
        self._last_cost: tuple[str, list[float]] | None = None
        self.stuck = self.migrated = self.balanced = 0  # what the rule did, for the report

    def _prefill_cost(self, request: NormalizedRequest, new_tokens: int) -> float:
        return self._profile.prefill_s(new_tokens, request.prompt_tokens - new_tokens)

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        new = self.new_tokens(state, request)
        cost = [self._prefill_cost(request, n) for n in new]
        load = [self._queued_cost[i] + cost[i] for i in range(self.n_replicas)]
        self._last = (request.request_id, new)
        self._last_cost = (request.request_id, cost)

        sid = request.session_id
        before = self._history.get(sid) if sid is not None else None
        if before is not None:
            # The hit this request's history implies: its last prompt's full blocks, and never
            # more than a lookup may return for this one (cache/block_pool.py).
            bs = self.block_size
            est = min(before // bs, max_hit_blocks(request.prompt_tokens, bs)) * bs
            hit = [request.prompt_tokens - n for n in new]
            prev = argmin_random([-h for h in hit], rng)  # the longest hit; ties to the rng
            if est > 0 and hit[prev] > self.hit_ratio * est:
                if load[prev] <= self.slack * self.ttft_slo_s:
                    self.stuck += 1
                    return prev
                self.migrated += 1
            else:
                self.balanced += 1  # its history is cached nowhere: a new session, in effect
        else:
            self.balanced += 1
        return argmin_random(
            [load[i] * (state.outstanding(i) + 1) for i in range(self.n_replicas)], rng
        )

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        super().observe(replica, request)  # LMetric's token account, and the dispatch index
        assert self._last_cost is not None and self._last_cost[0] == request.request_id
        cost = self._last_cost[1][replica]
        self._queued_cost[replica] += cost
        self._pending_cost[request.request_id] = (replica, cost)
        sid = request.session_id
        if sid is not None and (sid in self._history or len(self._history) < self.max_sessions):
            self._history[sid] = request.prompt_tokens

    def on_first_token(self, replica: int, request: NormalizedRequest) -> None:
        super().on_first_token(replica, request)
        entry = self._pending_cost.pop(request.request_id, None)
        if entry is not None:
            self._queued_cost[entry[0]] -= entry[1]

    def stats(self) -> dict[str, object]:
        """Follow-ups that stayed, follow-ups moved because their replica would have missed the
        deadline, and requests placed by the balance score (first turns, and follow-ups whose
        history was cached nowhere)."""
        return {"stuck": self.stuck, "migrated": self.migrated, "balanced": self.balanced}
