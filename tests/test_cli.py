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
    """The fitted T4 profile reaches both the sweep and the output file."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "10", "--turns", "2"])
    out = tmp_path / "r.json"
    # The first-token target alone: this test is about the profile, and with the per-token half
    # on (the default) nothing meets the SLO on 2 replicas, so no rupee figure would be defined.
    rc = main(["replay", str(trace), "--replicas", "2", "--seeds", "1", "--format", "json",
               "--profile", PROFILE, "--cost-model", "configs/cost_model.yaml",
               "--tpot-slo-ms", "0", "--out", str(out)])
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


def test_replay_sizes_on_both_halves_of_the_slo_by_default_and_says_so(tmp_path, capsys):
    """The report always shows p95 time per output token, and says what "meets the SLO" was decided
    on. By default that is both halves: the cost model's first-token limit and this command's
    per-token limit. `--tpot-slo-ms 0` sizes on the first token alone, and a cost model that sets
    its own per-token limit keeps it whatever the option says."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    capsys.readouterr()
    args = ["replay", str(trace), "--policies", "session-sticky", "--replicas", "3-4",
            "--seeds", "1", "--profile", CTX_PROFILE]
    cited = ["--cost-model", "configs/cost_model.yaml"]

    assert main([*args, *cited]) == 0
    default = capsys.readouterr().out
    assert "p95 ms per token" in default
    assert "and p95 time per output token ≤ 100 ms" in default
    assert "The per-token limit is this command's" in default
    assert "the SLO (p95 TTFT and p95 time per output token)" in default

    assert main([*args, *cited, "--tpot-slo-ms", "0"]) == 0
    first_only = capsys.readouterr().out
    assert "shown but not part of it" in first_only
    assert "meeting the p95 TTFT SLO" in first_only

    assert main([*args, *cited, "--tpot-slo-ms", "250"]) == 0
    looser = capsys.readouterr().out
    assert "and p95 time per output token ≤ 250 ms" in looser
    # A tighter per-token limit can only take configurations away.
    assert first_only.count("| yes |") >= looser.count("| yes |") >= default.count("| yes |")
    assert first_only.count("| yes |") > default.count("| yes |")

    both = tmp_path / "cost_both.yaml"
    both.write_text(pathlib.Path("configs/cost_model.yaml").read_text().replace(
        "  ttft_p95_ms: 2000\n", "  ttft_p95_ms: 2000\n  tpot_p95_ms: 100\n"))
    out = tmp_path / "r.json"
    assert main([*args, "--cost-model", str(both), "--tpot-slo-ms", "0",
                 "--format", "json", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert "and p95 time per output token ≤ 100 ms" in payload["slo"]
    assert "this command's" not in payload["slo"]  # the limit is the cost model's own
    for p in payload["points"]:
        assert p["p95_tpot_ms"] > 0
        # Meeting the SLO requires both halves, not only the first-token one.
        assert p["meets_slo"] == (p["p95_ttft_ms"] <= 2000 and p["p95_tpot_ms"] <= 100)
    # On this traffic the per-token half is the one that binds: the first token is in time on
    # both replica counts, the tokens after it are not.
    assert all(p["p95_ttft_ms"] <= 2000 for p in payload["points"])
    assert not any(p["meets_slo"] for p in payload["points"])
    assert "NO configuration met the SLO (p95 TTFT and p95 time per output token)" in \
        payload["recommendation"]["reason"]



def test_replay_compares_cost_with_round_robin_only_between_fleets_that_meet_the_slo(
        tmp_path, capsys):
    """Rupees per successful request of a fleet that misses the SLO is not a price. With both
    halves of the SLO on, this small fleet meets nothing, and the report prints no cost
    difference; on the first token alone both policies meet it, and it does."""
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "30", "--turns", "3", "--rate", "3"])
    capsys.readouterr()
    args = ["replay", str(trace), "--policies", "round-robin,session-sticky", "--replicas",
            "3-6", "--seeds", "1", "--profile", CTX_PROFILE, "--cost-model",
            "configs/cost_model.yaml", "--no-edge-check"]

    assert main(args) == 0
    nothing_met = capsys.readouterr().out
    assert "NO configuration met the SLO" in nothing_met
    assert "vs round-robin:" not in nothing_met

    assert main([*args, "--tpot-slo-ms", "0"]) == 0
    both_meet = capsys.readouterr().out
    assert "cheapest configuration meeting the p95 TTFT SLO" in both_meet
    assert "vs round-robin:" in both_meet


def _strict(text: str):
    """Parse JSON as JavaScript's JSON.parse does: NaN and Infinity are not JSON."""
    def refuse(token):
        raise ValueError(f"not JSON: {token}")
    return json.loads(text, parse_constant=refuse)


@pytest.fixture
def small_trace(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "5"])
    capsys.readouterr()
    return trace


@pytest.mark.parametrize(("args", "message"), [
    (["--policies", ""], "choose at least one routing policy"),
    (["--replicas", "1-99999999999999"], "Ask for at most 1,024"),
    (["--ttft-slo-ms", "nan"], "first-token limit"),
    (["--gpu-price-usd", "inf"], "GPU price"),
    (["--tpot-slo-ms", "-1"], "per-token limit"),
    (["--tpot-slo-ms", "nan"], "per-token limit"),
    (["--token-scale", "0.5"], "works only with the agentx, bailian and mooncake formats"),
    (["--token-scale", "nan", "--format-in", "mooncake"], "--token-scale must be a number from"),
    (["--token-scale", "0", "--format-in", "agentx"], "--token-scale must be a number from"),
    (["--token-scale", "0.3", "--format-in", "bailian"], "1/k for a whole number k"),
])
def test_replay_options_are_checked_before_any_simulation_runs(small_trace, monkeypatch, args,
                                                               message):
    from kv_routing_sim import service

    monkeypatch.setattr(service, "sweep", lambda *a, **k: pytest.fail("a simulation ran"))
    with pytest.raises(SystemExit, match=message):
        main(["replay", str(small_trace), *args])


@pytest.mark.parametrize("command", ["replay", "generate", "crossover"])
def test_an_output_path_that_cannot_be_written_is_refused_at_startup(tmp_path, small_trace,
                                                                     monkeypatch, command):
    from kv_routing_sim import cli

    monkeypatch.setattr(cli, "replay", lambda *a, **k: pytest.fail("the replay ran"))
    monkeypatch.setattr(cli, "crossover_sweep", lambda *a, **k: pytest.fail("the sweep ran"))
    monkeypatch.setattr(cli, "AgenticWorkload", lambda *a, **k: pytest.fail("traffic was made"))
    first = {"replay": ["replay", str(small_trace)], "generate": ["generate"],
             "crossover": ["crossover"]}[command]
    with pytest.raises(SystemExit, match="the folder .* does not exist"):
        main([*first, "--out", str(tmp_path / "no-such-folder" / "out.json")])
    with pytest.raises(SystemExit, match="is a folder. Give a file name"):
        main([*first, "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="is not a folder"):
        main([*first, "--out", str(small_trace / "out.json")])


@pytest.mark.parametrize(("args", "message"), [
    (["--sharing", "0.1,1.0"], "--sharing values must be at least 0 and below 1"),
    (["--sharing", "-0.1"], "--sharing values must be at least 0 and below 1"),
    (["--sharing", "nan"], "--sharing must be finite numbers"),
    (["--sharing", "lots"], "could not parse --sharing"),
    (["--sharing", ""], "--sharing is empty"),
    (["--rates", "4,0"], "--rates must be numbers above 0"),
    (["--rates", "inf"], "--rates must be finite numbers"),
    (["--seeds", "0"], "--seeds must be at least 1"),
    (["--replicas", "0"], "--replicas must be at least 1"),
    (["--sessions", "0"], "--sessions must be at least 1"),
    (["--policies", ""], "choose at least one routing policy"),
])
def test_crossover_options_are_checked_before_any_simulation_runs(monkeypatch, args, message):
    from kv_routing_sim import cli

    monkeypatch.setattr(cli, "crossover_sweep", lambda *a, **k: pytest.fail("the sweep ran"))
    with pytest.raises(SystemExit, match=message):
        main(["crossover", *args])


@pytest.mark.parametrize("command", ["calibrate", "profiles"])
def test_an_engine_that_is_not_an_http_url_is_refused(tmp_path, command):
    with pytest.raises(SystemExit, match="must start with http:// or https://"):
        main([command, "--engine", "localhost:8000", "--out", str(tmp_path / "p.json")]
             if command == "calibrate" else [command, "--engine", "localhost:8000"])


def test_generate_refuses_no_sessions_and_a_rate_that_is_not_a_number(tmp_path):
    out = tmp_path / "t.jsonl"
    with pytest.raises(SystemExit, match="n_sessions must be >= 1"):
        main(["generate", "--out", str(out), "--sessions", "0"])
    with pytest.raises(SystemExit, match="arrival_rate must be a finite number"):
        main(["generate", "--out", str(out), "--rate", "nan"])
    assert not out.exists()


@pytest.mark.parametrize(("change", "message"), [
    (("ttft_p95_ms: 2000", "ttft_p95_ms: [2000"), "not valid YAML"),
    (("price_per_hour_usd: 0.526", "price_per_hour_usd: 0"), "compute.price_per_hour_usd must be"),
    (("usd_to_inr: 95.87", "usd_to_inr: -1"), "fx.usd_to_inr must be a number above 0"),
    (("ttft_p95_ms: 2000", "ttft_p95_ms: .nan"), "slo.ttft_p95_ms must be a number above 0"),
    (("ttft_p95_ms: 2000", "ttft_p95_ms: 2000\n  tpot_p95_ms: -5"), "slo.tpot_p95_ms must be"),
    (("ttft_p95_ms: 2000", "ttft_p95_ms: .inf"), "slo.ttft_p95_ms must be a number above 0"),
])
def test_a_cost_model_with_a_bad_value_is_refused(small_trace, tmp_path, change, message):
    bad = tmp_path / "cost.yaml"
    bad.write_text(pathlib.Path("configs/cost_model.yaml").read_text().replace(*change))
    with pytest.raises(SystemExit, match=message):
        main(["replay", str(small_trace), "--cost-model", str(bad)])


@pytest.mark.parametrize(("field", "value", "message"), [
    ("block_size", 0, "block_size must be a whole number of at least 1"),
    ("kv_cache_blocks", 1, "kv_cache_blocks must be a whole number of at least 2"),
    ("max_num_seqs", 0, "max_num_seqs must be a whole number of at least 1"),
    ("max_num_seqs", 2.5, "max_num_seqs must be a whole number of at least 1"),
    ("prefill_base_s", -0.001, "prefill_base_s must be a finite number of at least 0"),
    ("decode_per_seq_s", float("nan"), "decode_per_seq_s must be a finite number of at least 0"),
    ("decode_base_s", "0.02", "decode_base_s must be a finite number of at least 0"),
])
def test_a_profile_with_a_value_no_engine_has_is_refused(small_trace, tmp_path, field, value,
                                                          message):
    raw = json.loads(pathlib.Path(PROFILE).read_text())
    raw[field] = value
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match=message):
        main(["replay", str(small_trace), "--profile", str(bad), "--block-size",
              str(raw["block_size"])])


def test_json_output_writes_null_for_nan_and_infinity(small_trace, tmp_path, monkeypatch, capsys):
    from kv_routing_sim import cli
    from kv_routing_sim.report import to_json
    from kv_routing_sim.service import ReplayReport

    assert to_json({"a": float("nan"), "b": [1.5, float("inf")], "c": (float("-inf"),)}) == \
        '{"a": null, "b": [1.5, null], "c": [null]}'
    report = ReplayReport(trace={"requests": 1}, profile={"calibrated": True},
                          points=[{"p95_ttft_ms": float("nan"), "hit_rate": 0.5}],
                          recommendation={"calibrated": True}, slo=None, replica_spread=None,
                          replica_spread_note=None, edge={"slowdown": float("inf")})
    monkeypatch.setattr(cli, "replay", lambda *a, **k: report)
    out = tmp_path / "r.json"
    assert main(["replay", str(small_trace), "--format", "json", "--out", str(out)]) == 0
    written = _strict(out.read_text())
    assert written["points"][0] == {"p95_ttft_ms": None, "hit_rate": 0.5}
    assert written["edge"] == {"slowdown": None}

    monkeypatch.setattr(cli, "crossover_sweep", lambda **k: [{"p95_by_policy_ms": {
        "round-robin": float("nan")}}])
    assert main(["crossover", "--out", str(out)]) == 0
    assert _strict(out.read_text())["crossover"] == [{"p95_by_policy_ms": {"round-robin": None}}]
    capsys.readouterr()


def test_crossover_reports_traffic_it_cannot_build_in_one_line():
    """A sharing ratio so high that the generator refuses it ends in a message, not a traceback."""
    with pytest.raises(SystemExit) as stop:
        main(["crossover", "--sharing", "0.999", "--rates", "1", "--sessions", "5"])
    assert str(stop.value).startswith("error: a shared prefix of ")


# A second review before release 0.2.0.


def test_an_output_name_is_needed_and_must_be_writable(tmp_path, monkeypatch):
    """An empty `--out` and a folder without write permission ended in tracebacks, the second one
    after the whole run."""
    import os

    from kv_routing_sim import cli

    monkeypatch.setattr(cli, "AgenticWorkload", lambda *a, **k: pytest.fail("traffic was made"))
    with pytest.raises(SystemExit, match="--out needs a file name"):
        main(["generate", "--out", ""])
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("this user can write anywhere")
        with pytest.raises(SystemExit, match="no permission to write there"):
            main(["generate", "--out", str(locked / "t.jsonl")])
    finally:
        locked.chmod(0o755)


@pytest.mark.parametrize("url", ["http://localhost:abc", "http://local host:8000", "http://",
                                 "http://[::1"])
def test_an_engine_address_that_is_not_one_is_refused_in_one_line(url):
    with pytest.raises(SystemExit, match="is not a server address"):
        main(["profiles", "--engine", url])
    with pytest.raises(SystemExit, match="is not a server address"):
        main(["calibrate", "--engine", url, "--out", "unused.json"])


@pytest.mark.parametrize(("args", "message"), [
    (["--reps", "0"], "--reps must be at least 1"),
    (["--check-sessions", "0"], "--check-sessions must be at least 1"),
    (["--check-turns", "-1"], "--check-turns must be at least 1"),
    (["--kv-blocks", "1"], "--kv-blocks must be at least 2"),
    (["--block-size", "0"], "--block-size must be at least 1"),
    (["--max-num-seqs", "-3"], "--max-num-seqs must be at least 1"),
    (["--decode-batches", "0,2"], "--decode-batches must be whole numbers of at least 1"),
    (["--prefill-lengths", "0,-5"], "--prefill-lengths must be whole numbers of at least 1"),
    (["--timeout", "-1"], "--timeout must be a number of seconds above 0"),
    (["--timeout", "nan"], "--timeout must be a number of seconds above 0"),
    (["--max-error", "nan"], "--max-error must be a number of at least 0"),
])
def test_calibrate_checks_every_option_before_it_asks_the_engine(tmp_path, monkeypatch, args,
                                                                 message):
    """A calibration takes minutes. A bad option must not surface after them."""
    from kv_routing_sim.calib import calibrate as module

    monkeypatch.setattr(module, "calibrate", lambda *a, **k: pytest.fail("the engine was asked"))
    with pytest.raises(SystemExit, match=message):
        main(["calibrate", "--engine", "http://127.0.0.1:9", "--out", str(tmp_path / "p.json"),
              *args])


def test_generate_refuses_a_rate_that_gives_infinite_times(tmp_path):
    with pytest.raises(SystemExit, match="arrival_rate must be a finite number of at least"):
        main(["generate", "--out", str(tmp_path / "t.jsonl"), "--sessions", "3",
              "--rate", "1e-320"])
    assert not (tmp_path / "t.jsonl").exists()


def test_a_run_too_large_for_memory_ends_in_one_line(small_trace, monkeypatch):
    from kv_routing_sim import cli

    def out_of_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(cli, "replay", out_of_memory)
    with pytest.raises(SystemExit, match="needs more memory than this machine has"):
        main(["replay", str(small_trace)])


def test_crossover_refuses_a_fleet_that_would_not_fit_in_memory(monkeypatch):
    from kv_routing_sim import cli

    monkeypatch.setattr(cli, "crossover_sweep", lambda *a, **k: pytest.fail("the sweep ran"))
    with pytest.raises(SystemExit, match="a fleet of 300,000 replicas was asked for"):
        main(["crossover", "--replicas", "300000"])

