"""Tests for the session cost table's own arithmetic (scripts/session_cost.py).

They use hand-made traces and made-up rows only, so nothing here says anything about the table's
result. The script itself runs the registered cells and checks each against the stored grids.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.traces.schema import NormalizedRequest

REPO = pathlib.Path(__file__).resolve().parents[1]

TOY = CostProfile(
    name="toy", prefill_base_s=0.0, prefill_per_token_s=0.0005, decode_base_s=0.02,
    decode_per_seq_s=0.0005, kv_cache_blocks=2_000, block_size=16, max_num_seqs=32,
    calibrated=True, prefill_attn_s=1e-7, decode_per_ctx_token_s=1e-6,
)
B = TOY.block_size


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load("session_cost")


def _request(name: str, at: float, blocks: tuple[int, ...], tail: int, session: str):
    return NormalizedRequest(request_id=name, arrival_ts=at, block_hashes=blocks,
                             prompt_tokens=len(blocks) * B + tail, output_tokens=4,
                             session_id=session)


# Two sessions of two turns, far apart in time so that no two requests overlap. Session "a" grows
# from three blocks to five. Session "b" shares only the first block with it.
TRACE = [
    _request("a1", 0.0, (11, 12, 13), 5, "a"),
    _request("a2", 100.0, (11, 12, 13, 14, 15), 2, "a"),
    _request("b1", 200.0, (11, 21), 1, "b"),
    _request("b2", 300.0, (11, 21, 22), 0, "b"),
]
NEW = [3 * B + 5, 2 * B + 2, B + 1, B]


def test_new_tokens_are_the_blocks_nobody_sent_before_plus_every_unfinished_block():
    assert sc.new_tokens(TRACE, B) == NEW
    # The same total, counted the other way: each distinct full block once, plus the tails.
    distinct = {block for q in TRACE for block in q.block_hashes}
    assert sum(NEW) == len(distinct) * B + (5 + 2 + 1 + 0)


def test_a_block_seen_before_counts_only_at_the_head_of_a_prompt():
    """A cache lookup stops at the first block it does not hold, and so does this count."""
    trace = [_request("x", 0.0, (1, 2, 3), 0, "s"), _request("y", 1.0, (9, 2, 3), 0, "t")]
    assert sc.new_tokens(trace, B) == [3 * B, 3 * B]


def _measure(n_replicas: int):
    return sc.measure(TRACE, policy="round-robin", alpha=None, n=n_replicas, seed=0, profile=TOY,
                      cache_model="vllm", cost=CostModel.from_yaml("default"))


def test_one_replica_with_room_re_prefills_nothing():
    """Every block sent before is still in the one cache, so only new tokens are computed."""
    m = _measure(1)
    assert m["preemptions"] == 0
    assert sc.check_run(TRACE, m, sum(NEW)) == 0
    assert m["cached"] == [0, 3 * B, B, 2 * B]


def test_round_robin_on_two_replicas_re_prefills_what_the_other_replica_holds():
    """Turns alternate between the replicas, so each session's second turn finds only block 11."""
    m = _measure(2)
    assert m["cached"] == [0, 0, B, B]
    # a2 computes blocks 11 to 13 again, and b2 computes block 21 again.
    assert sc.check_run(TRACE, m, sum(NEW)) == 4 * B


def test_a_run_that_disagrees_with_the_trace_is_refused():
    m = _measure(1)
    with pytest.raises(sc.CheckFailed, match="negative"):
        sc.check_run(TRACE, m, sum(NEW) + 1)
    with pytest.raises(sc.CheckFailed, match="cache_queries"):
        sc.check_run(TRACE, {**m, "queries": m["queries"] + 1}, sum(NEW))


def test_replica_counts_are_read_from_the_stored_tables():
    """The two cells named in the registration, and a policy that met the target nowhere."""
    assert sc.required("first_token", 1, 2, "session-sticky") == 5
    assert sc.required("first_token", 1, 2, "round-robin") == 7
    assert sc.required("first_token", 1, 0.5, "radix-affinity") is None


def _row(target: str, policy: str, replicas: int, again: float, inr: float) -> dict:
    return {"target": target, "prompts": 1, "load": 2, "policy": policy, "replicas": replicas,
            "new_tokens_per_session": 519.0, "re_prefilled_tokens_per_session": again,
            "inr_per_session": inr}


def test_the_three_predictions_are_scored_by_their_registered_rules():
    rows = [
        _row("first_token", "session-sticky", 5, 27.0, 0.05),
        _row("first_token", "round-robin", 7, 669.0, 0.07),
        _row("two_part", "session-sticky", 10, 20.0, 0.104),  # rupees x2.08 for replicas x2
        _row("two_part", "round-robin", 14, 600.0, 0.14),
    ]
    scored = sc.predictions(rows)
    assert [scored[name]["holds"] for name in ("S1a", "S1b", "S1c")] == [True, True, True]
    assert scored["S1c"]["rupee_ratio_over_replica_ratio"] == {
        "session-sticky": 1.04, "round-robin": 1.0}

    rows[0]["re_prefilled_tokens_per_session"] = 100.0  # not below 100
    rows[2]["inr_per_session"] = 0.112  # rupees x2.24 for replicas x2: 12% off
    for r in rows:
        r["new_tokens_per_session"] = 541.0
    scored = sc.predictions(rows)
    assert [scored[name]["holds"] for name in ("S1a", "S1b", "S1c")] == [False, False, False]


def test_the_stored_table_agrees_with_the_grid_it_was_derived_from():
    """Read-only: the committed result against the committed grid, row by row."""
    results = REPO / "artifacts" / "results"
    table = json.loads((results / "session_cost" / "session_cost.json").read_text())
    grid = json.loads((results / "demo_grid_ctx.json").read_text())
    ax = grid["axes"]
    first = [r for r in table["rows"] if r["target"] == "first_token"]
    assert (len(first), len(table["rows"])) == (99, 195)
    for r in first:
        cell = grid["cells"][ax["prompts"].index(r["prompts"])][ax["loads"].index(r["load"])][
            ax["policies"].index(r["policy"])][ax["replicas"].index(r["replicas"])]
        assert (r["hit_rate"], r["inr_per_1k_requests"], 1) == (cell[0], cell[3], cell[4])
    per_session = table["requests_per_session"]
    for r in table["rows"]:
        assert r["inr_per_session"] == pytest.approx(
            r["inr_per_1k_requests"] * per_session / 1000, abs=1e-5)
        assert r["prefilled_tokens_per_session"] == pytest.approx(
            r["new_tokens_per_session"] + r["re_prefilled_tokens_per_session"], abs=0.16)
        assert r["re_prefilled_tokens_per_session"] >= 0
        assert r["recomputed_after_preemption_per_session"] == 0
    assert table["git_dirty"] is False
    assert [table["predictions"][name]["holds"] for name in ("S1a", "S1b", "S1c")] == [True] * 3
