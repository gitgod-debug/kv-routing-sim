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


# ── post hoc: the check against T1's five held-out runs ─────────────────────────────────────────

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
POSTHOC = REPO / "artifacts" / "results" / "t1_itl"


def _script():
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "edge_posthoc_t1", REPO / "scripts" / "edge_posthoc_t1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["edge_posthoc_t1"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_post_hoc_table_reproduces_and_separates_the_runs_the_model_got_wrong():
    """POST HOC, and labelled so: on T1's five held-out runs the 5% slowdown multiplies p95 TTFT
    by more than five on the two where the simulator's per-token latency was 1.54x and 1.71x the
    engine's, and by 1.13-1.14 on the three where it was within 1.07x. The module's docstring and
    the README quote these numbers."""
    stored = json.loads((POSTHOC / "posthoc_edge.json").read_text())
    script = _script()
    fresh = json.loads(json.dumps(script.analyze()))
    assert {k: stored[k] for k in fresh} == fresh
    assert stored["post_hoc"] is True and stored["git_dirty"] is False
    assert (POSTHOC / "posthoc_edge.md").read_text() == script.markdown(stored, stored)

    runs = {(r["session"], r["workload"]): r for r in stored["runs"]}
    assert len(runs) == 5
    wrong = [runs[("c2", "K1_half")], runs[("c3", "K1_half")]]
    right = [runs[("c2", "K64_quarter")], runs[("c3", "K64_quarter")], runs[("c3", "T6_K8")]]
    assert [r["t1_factor"] for r in wrong] == [1.712, 1.544]
    assert round(max(r["t1_factor"] for r in right), 2) == 1.07  # "within 1.07x", as T1 reports it
    assert [r["ttft_factor"] for r in wrong] == [5.469, 5.332]
    assert [r["ttft_factor"] for r in right] == [1.133, 1.133, 1.14]
    assert sorted({r["tpot_factor"] for r in stored["runs"]}) == [1.129, 1.134, 1.161, 1.206]
    assert all(r["flagged_as_knee"] for r in wrong) and not any(r["flagged_as_knee"] for r in right)
    assert [r["preemptions"] for r in wrong] == [[6, 36], [6, 33]]
    assert stored["separating_range"] == [1.206, 5.332]
    assert stored["separating_range"][0] < KNEE_FACTOR < stored["separating_range"][1]
