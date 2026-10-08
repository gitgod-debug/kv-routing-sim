"""The edge check: is a configuration at the knee of the fleet's load curve?

`kvsim replay` runs its recommended configuration once more on timings 5% slower and says how far
the p95 latencies moved (src/kv_routing_sim/eval/edge.py). With room to spare 5% in is 5% out; at
the knee it is 50% or more.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from kv_routing_sim.cli import main
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.edge import KNEE_FACTOR, SLOWDOWN, edge_check, slowed
from kv_routing_sim.eval.sweep import sweep
from kv_routing_sim.sim.cost_model import TIMING_FIELDS, load_profile
from kv_routing_sim.traces.generator import AgenticWorkload

CTX = "configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json"
CITED = "configs/cost_model.yaml"


def test_a_slowed_profile_differs_in_its_six_timing_coefficients_and_nothing_else():
    base = load_profile(CTX)
    slow = slowed(base, 1.05)
    for f in dataclasses.fields(base):
        if f.name in TIMING_FIELDS:
            assert getattr(slow, f.name) == pytest.approx(getattr(base, f.name) * 1.05)
        elif f.name != "name":
            assert getattr(slow, f.name) == getattr(base, f.name), f.name
    assert len(TIMING_FIELDS) == 6 and slow.calibrated is base.calibrated
    assert SLOWDOWN == 1.05 and KNEE_FACTOR == 1.5


@pytest.fixture(scope="module")
def three_counts():
    """One workload on 2, 3 and 4 replicas: under, at and over what it needs."""
    trace, _ = AgenticWorkload(n_sessions=200, turns_per_session=6, arrival_rate=0.25,
                               seed=0).generate()
    profile, cost, seeds = load_profile(CTX), CostModel.from_yaml(CITED), (0, 1, 2)
    points, _ = sweep(trace, policies=["session-sticky"], replica_counts=[2, 3, 4], seeds=seeds,
                      profile=profile, cost=cost)
    return {p.n_replicas: (p, edge_check(trace, chosen=p, seeds=seeds, profile=profile,
                                         cost=cost)) for p in points}


def test_at_the_knee_five_percent_in_is_more_than_fifty_percent_out(three_counts):
    point, e = three_counts[2]
    assert not point.meets_slo and e["verdict"] == "knee"
    assert e["ttft_factor"] > KNEE_FACTOR and e["p95_ttft_ms"][1] > e["p95_ttft_ms"][0]


def test_at_the_edge_the_slo_holds_and_stops_holding_on_slower_timings(three_counts):
    point, e = three_counts[3]
    assert point.meets_slo and e["verdict"] == "edge" and e["still_meets_slo"] is False
    assert 1.05 < e["ttft_factor"] <= KNEE_FACTOR


def test_with_room_to_spare_five_percent_in_is_about_five_percent_out(three_counts):
    point, e = three_counts[4]
    assert point.meets_slo and e["verdict"] == "clear" and e["still_meets_slo"] is True
    assert 1.0 < e["ttft_factor"] < 1.1 and 1.0 < e["tpot_factor"] < 1.1
    assert (e["policy"], e["n_replicas"], e["slowdown"]) == ("session-sticky", 4, SLOWDOWN)


def test_replay_says_how_close_to_the_edge_its_recommendation_is(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    capsys.readouterr()
    args = ["replay", str(trace), "--policies", "session-sticky", "--replicas", "3-5",
            "--seeds", "2", "--profile", CTX, "--cost-model", CITED, "--tpot-slo-ms", "0"]

    assert main(args) == 0
    out = capsys.readouterr().out
    assert "with timings 5% slower" in out
    assert sum(s in out for s in ("⛔ at the knee", "⚠️ at the edge", "not at the edge")) == 1

    report = tmp_path / "r.json"
    assert main([*args, "--format", "json", "--out", str(report)]) == 0
    edge = json.loads(report.read_text())["edge"]
    assert edge["verdict"] in ("knee", "edge", "clear") and edge["slowdown"] == 1.05
    assert edge["policy"] == "session-sticky" and len(edge["p95_ttft_ms"]) == 2

    assert main([*args, "--no-edge-check"]) == 0
    assert "with timings 5% slower" not in capsys.readouterr().out


def test_no_edge_check_where_nothing_met_the_slo(tmp_path):
    """With the per-token half on, this small fleet meets nothing: there is no recommended
    configuration to be at the edge of, and the report does not pretend there is."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    report = tmp_path / "r.json"
    assert main(["replay", str(trace), "--policies", "session-sticky", "--replicas", "3-4",
                 "--seeds", "1", "--profile", CTX, "--cost-model", CITED, "--format", "json",
                 "--out", str(report)]) == 0
    assert json.loads(report.read_text())["edge"] is None
