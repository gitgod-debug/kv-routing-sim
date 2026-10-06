"""How firm is a replica count? The sizing re-run on each repeat of the calibration.

C3 measured why this exists: two runs of `kvsim calibrate` on one engine differed by 3–5% in their
coefficients, and three calibrations of one T4 agreed on a replica count exactly at 102 of 144
points and within one at 138. A count printed alone claims more than that.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from kv_routing_sim.cli import main
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point
from kv_routing_sim.eval.spread import replica_spread, required_replicas
from kv_routing_sim.eval.sweep import sweep
from kv_routing_sim.sim.cost_model import load_profile, replicate_profiles
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
CTX = REPO / "configs" / "profiles" / "t4-qwen3-4b-vllm0.30.0-ctx.json"
TIMING = ("prefill_base_s", "prefill_per_token_s", "prefill_attn_s", "decode_base_s",
          "decode_per_seq_s", "decode_per_ctx_token_s")


def _p(policy, n, hit, p95, inr, meets=True):
    return Point(policy, n, hit, p95, p95 / 2, inr, 10.0, 1.0, meets, True)


def _with_repeats(tmp_path: pathlib.Path, scales=(1.0, 1.3, 0.9)) -> pathlib.Path:
    """The shipped T4 profile with made-up repeats: each one the pooled fit, scaled."""
    base = json.loads(CTX.read_text())
    reps = [{"rep": i, **{k: base[k] * x for k in TIMING}} for i, x in enumerate(scales)]
    out = tmp_path / "with_repeats.json"
    out.write_text(json.dumps({**base, "name": "toy-with-repeats",
                               "_source": {"replicates": reps}}))
    return out


def test_required_replicas_is_the_fewest_that_meet_the_slo_or_none():
    pts = [_p("a", 1, 0.5, 9.0, 1.0, meets=False), _p("a", 3, 0.5, 9.0, 1.0, meets=True),
           _p("a", 2, 0.5, 9.0, 1.0, meets=True), _p("b", 1, 0.5, 9.0, 1.0, meets=False),
           _p("c", 4, 0.5, 9.0, 1.0, meets=None)]
    assert required_replicas(pts) == {"a": 2, "b": None, "c": None}


def test_a_profile_without_repeats_has_no_spread_to_show(tmp_path):
    """Empty means "unknown", never "zero": the two shipped profiles, and anything written before
    the calibration kept its repeats."""
    assert replicate_profiles(CTX) == []
    assert replicate_profiles("t4-qwen3-4b-vllm0.30.0-ctx") == []  # by its shipped name
    assert replicate_profiles("no-such-profile") == []


def test_each_repeat_is_the_same_engine_with_its_own_timing(tmp_path):
    path = _with_repeats(tmp_path)
    pooled, reps = load_profile(path), replicate_profiles(path)
    assert [r.name for r in reps] == [f"toy-with-repeats#repeat{i}" for i in range(3)]
    for r, x in zip(reps, (1.0, 1.3, 0.9), strict=True):
        assert r.decode_per_ctx_token_s == pytest.approx(pooled.decode_per_ctx_token_s * x)
        assert r.prefill_attn_s == pytest.approx(pooled.prefill_attn_s * x)
        # Capacity, block size and the calibrated flag are the engine's, not a repeat's.
        assert (r.kv_cache_blocks, r.block_size, r.max_num_seqs, r.calibrated) == (
            pooled.kv_cache_blocks, pooled.block_size, pooled.max_num_seqs, True)


def test_the_spread_is_the_count_on_every_repeat_beside_the_pooled_one(tmp_path):
    path = _with_repeats(tmp_path)
    trace, _ = AgenticWorkload(n_sessions=30, turns_per_session=3, arrival_rate=3.0,
                               seed=0).generate()
    cost = CostModel.from_yaml(REPO / "configs" / "cost_model.yaml")
    kw = {"policies": ["session-sticky", "round-robin"], "replica_counts": list(range(1, 7)),
          "seeds": (0,)}
    points, _ = sweep(trace, profile=load_profile(path), cost=cost, **kw)
    s = replica_spread(trace, cost=cost, pooled_points=points,
                       replicates=replicate_profiles(path), **kw)
    assert (s["repeats"], s["swept_up_to"]) == (3, 6)
    sticky, rr = s["replicas"]["session-sticky"], s["replicas"]["round-robin"]
    # Repeat 0 IS the pooled fit; repeat 1 is 30% slower and needs one replica more.
    assert sticky == {"pooled": 3, "repeats": [3, 4, 3], "min": 3, "max": 4, "firm": False}
    # 30% slower, round-robin no longer fits on the six replicas swept: "not met" is the maximum.
    assert rr["pooled"] == 5 and rr["repeats"] == [5, None, 5]
    assert (rr["min"], rr["max"], rr["firm"]) == (5, None, False)


def test_replay_shows_how_firm_its_recommendation_is(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    capsys.readouterr()
    args = ["replay", str(trace), "--policies", "session-sticky,round-robin", "--replicas",
            "1-8", "--seeds", "1", "--cost-model", "configs/cost_model.yaml"]

    assert main([*args, "--profile", str(_with_repeats(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "**session-sticky on 3 replica(s)**" in out
    assert "⚠️ how firm: 3 to 4 replicas across the calibration's 3 repeats" in out
    assert "| session-sticky | 3 | 3 · 4 · 3 |" in out and "| round-robin | 5 | 5 · 7 · 5 |" in out
    assert "a floor on the uncertainty, not a confidence interval" in out
    # C4's rule E failed: a run's own range covers another run's count in 82% of cases, not 90.
    # The tool therefore says to add one, and says where that comes from.
    assert "size for the larger, plus one" in out
    assert "outside a run's own range in 18% of cases" in out and "plus one." in out

    # Repeats that agree say so, in one line, without a warning sign — and still allow one more:
    # in C4, 24 of the 38 misses were of a range that was a single count.
    agree = _with_repeats(tmp_path, scales=(1.0, 1.0))
    assert main([*args, "--profile", str(agree)]) == 0
    out = capsys.readouterr().out
    assert "how firm: the same count on each of the calibration's 2 repeats" in out
    assert "still allow one more" in out
    assert "⚠️ how firm" not in out

    # A profile with no repeats: the report says the spread is unknown, it does not stay silent.
    assert main([*args, "--profile", str(CTX)]) == 0
    out = capsys.readouterr().out
    assert "carries no per-repeat fits" in out and "How firm are the replica counts?" not in out

    # --no-spread: one sweep, no spread, no note.
    assert main([*args, "--profile", str(_with_repeats(tmp_path)), "--no-spread"]) == 0
    out = capsys.readouterr().out
    assert "how firm" not in out.lower() and "per-repeat" not in out


def test_the_spread_reaches_the_json_output(tmp_path):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    out = tmp_path / "r.json"
    assert main(["replay", str(trace), "--policies", "session-sticky", "--replicas", "2-5",
                 "--seeds", "1", "--cost-model", "configs/cost_model.yaml", "--profile",
                 str(_with_repeats(tmp_path)), "--format", "json", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["replica_spread"]["replicas"]["session-sticky"]["repeats"] == [3, 4, 3]
    assert payload["replica_spread_note"] is None
    # No cost model: no SLO, so no replica count to be firm or not about.
    assert main(["replay", str(trace), "--policies", "session-sticky", "--replicas", "2-3",
                 "--seeds", "1", "--profile", str(_with_repeats(tmp_path)), "--format", "json",
                 "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["replica_spread"] is None and payload["replica_spread_note"] is None
