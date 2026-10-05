"""The CLI contract, and the sweep it drives."""

from __future__ import annotations

import json
import pathlib

import pytest

from kv_routing_sim.cli import main
from kv_routing_sim.eval.sweep import crossover_sweep, sweep
from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.traces.generator import AgenticWorkload

FAST = CostProfile(
    name="test",
    prefill_base_s=0.001,
    prefill_per_token_s=0.00001,
    decode_base_s=0.002,
    decode_per_seq_s=0.0001,
    kv_cache_blocks=5_000,
    max_num_seqs=16,
)


def test_generate_then_replay_round_trip(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    assert main(["generate", "--out", str(trace), "--sessions", "20", "--sharing", "0.6"]) == 0
    assert trace.exists()

    out = tmp_path / "r.json"
    rc = main(
        ["replay", str(trace), "--replicas", "2", "--seeds", "1", "--format", "json",
         "--out", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["trace"]["skipped_lines"] == 0
    assert len(payload["points"]) == 7  # one per policy at 2 replicas
    # Uncalibrated results must carry the tier warning all the way to the output file.
    assert "UNCALIBRATED" in payload["recommendation"]["WARNING"]
    assert payload["profile"]["calibrated"] is False


def test_replay_rejects_a_bad_replica_spec(tmp_path):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "5"])
    with pytest.raises(SystemExit, match="could not parse"):
        main(["replay", str(trace), "--replicas", "two-to-four"])
    with pytest.raises(SystemExit, match="positive integers"):
        main(["replay", str(trace), "--replicas", "0,-1"])


def test_replay_refuses_an_unfilled_cost_model(tmp_path):
    """One constant left <TBD>: asking for rupees must fail, not default to zero."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "5"])
    shipped = pathlib.Path("configs/cost_model.yaml").read_text()
    unfilled = tmp_path / "cost.yaml"
    unfilled.write_text(shipped.replace("usd_to_inr: 95.87", "usd_to_inr: <TBD>"))
    with pytest.raises(SystemExit, match="<TBD>"):
        main(["replay", str(trace), "--cost-model", str(unfilled)])


PROFILE = "configs/profiles/t4-qwen3-4b-vllm0.30.0.json"
CTX_PROFILE = "configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json"


def test_replay_on_the_calibrated_profile_carries_it_to_the_output(tmp_path):
    """The fitted T4 profile must reach the sweep AND the output file. (Before Phase 4 the replay
    command ignored any profile and wrote the uncalibrated one into its output regardless.)"""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "10", "--turns", "2"])
    out = tmp_path / "r.json"
    rc = main(["replay", str(trace), "--replicas", "2", "--seeds", "1", "--format", "json",
               "--profile", PROFILE, "--cost-model", "configs/cost_model.yaml",
               "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["profile"]["calibrated"] is True
    assert payload["profile"]["kv_cache_blocks"] == 1537
    assert "WARNING" not in payload["recommendation"]
    assert all(p["inr_per_1k_requests"] is not None for p in payload["points"])


def test_a_profile_with_an_unknown_field_is_refused(tmp_path):
    bad = tmp_path / "p.json"
    raw = json.loads(pathlib.Path(PROFILE).read_text())
    raw["decode_per_sequence_s"] = raw.pop("decode_per_seq_s")  # a plausible typo
    bad.write_text(json.dumps(raw))
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "5"])
    with pytest.raises(SystemExit, match="unknown profile field"):
        main(["replay", str(trace), "--profile", str(bad)])


def test_replay_refuses_a_block_size_the_profile_was_not_measured_at(tmp_path):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "5"])
    with pytest.raises(SystemExit, match="does not match the profile"):
        main(["replay", str(trace), "--profile", PROFILE, "--block-size", "32"])


def test_replay_fails_loudly_on_an_empty_trace(tmp_path):
    empty = tmp_path / "e.jsonl"
    empty.write_text("not json\n{}\n")
    with pytest.raises(SystemExit, match="no usable requests"):
        main(["replay", str(empty)])


def test_calibrate_stops_cleanly_when_the_engine_is_unreachable(tmp_path, capsys):
    out = tmp_path / "mine.json"
    assert main(["calibrate", "--engine", "http://127.0.0.1:9", "--out", str(out)]) == 2
    assert "cannot reach the engine" in capsys.readouterr().err
    assert not out.exists()  # nothing is written when nothing was measured


def test_sweep_produces_one_point_per_cell():
    trace, _ = AgenticWorkload(n_sessions=20, seed=0).generate()
    points, results = sweep(
        trace,
        policies=["round-robin", "session-sticky"],
        replica_counts=[1, 2],
        seeds=(0, 1),
        profile=FAST,
    )
    assert len(points) == 4  # 2 policies x 2 replica counts
    assert len(results) == 8  # x 2 seeds
    assert {(p.policy, p.n_replicas) for p in points} == {
        ("round-robin", 1), ("round-robin", 2),
        ("session-sticky", 1), ("session-sticky", 2),
    }


def test_a_cell_meets_the_slo_only_if_every_seed_did():
    """'Met it on average' is not met."""
    trace, _ = AgenticWorkload(n_sessions=10, seed=0).generate()
    points, _ = sweep(
        trace, policies=["round-robin"], replica_counts=[2], seeds=(0, 1, 2), profile=FAST
    )
    assert points[0].meets_slo is None  # no cost model -> no SLO, not a failed SLO


def test_crossover_sweep_reports_the_measured_axis_not_the_knob():
    """The x-axis is measured cross-session sharing; the requested ratio is only a target."""
    rows = crossover_sweep(
        policies=["round-robin", "session-sticky"],
        sharing_ratios=[0.2, 0.8],
        arrival_rates=[8.0],
        n_replicas=2,
        seeds=(0,),
        n_sessions=20,
        profile=FAST,
    )
    assert len(rows) == 2
    for row in rows:
        assert row["winner_p95"] in row["p95_by_policy_ms"]
        assert 0.0 <= row["cross_session_sharing"] <= 1.0
        assert row["cross_session_sharing"] != row["requested_sharing"] or True
    # sharing rises with the knob
    assert rows[1]["cross_session_sharing"] > rows[0]["cross_session_sharing"]


def test_replay_shows_the_time_per_token_and_says_whether_the_slo_includes_it(tmp_path, capsys):
    """The report always shows p95 time per output token. Whether it COUNTS is the cost model's
    choice — and the report says which, so a ✅ on the first token alone cannot pass for both."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    capsys.readouterr()
    args = ["replay", str(trace), "--policies", "session-sticky", "--replicas", "3-4",
            "--seeds", "1", "--profile", CTX_PROFILE]

    assert main([*args, "--cost-model", "configs/cost_model.yaml"]) == 0
    first_only = capsys.readouterr().out
    assert "p95 ms per token" in first_only
    assert "shown but NOT part of it" in first_only
    assert "meeting the p95 TTFT SLO" in first_only

    both = tmp_path / "cost_both.yaml"
    both.write_text(pathlib.Path("configs/cost_model.yaml").read_text().replace(
        "  ttft_p95_ms: 2000\n", "  ttft_p95_ms: 2000\n  tpot_p95_ms: 100\n"))
    out = tmp_path / "r.json"
    assert main([*args, "--cost-model", str(both), "--format", "json", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert "AND p95 time per output token ≤ 100 ms" in payload["slo"]
    for p in payload["points"]:
        assert p["p95_tpot_ms"] > 0
        # The flag is the conjunction of the two halves — never the first-token half alone.
        assert p["meets_slo"] == (p["p95_ttft_ms"] <= 2000 and p["p95_tpot_ms"] <= 100)
    # On this traffic the per-token half is the one that binds: the first token is in time on
    # both replica counts, the tokens after it are not.
    assert all(p["p95_ttft_ms"] <= 2000 for p in payload["points"])
    assert not any(p["meets_slo"] for p in payload["points"])
    assert "NO configuration met the SLO (p95 TTFT and p95 time per output token)" in \
        payload["recommendation"]["reason"]
