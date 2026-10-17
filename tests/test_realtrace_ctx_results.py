"""Pins R1 and R2's registered result on the validated timing model (configs/realtrace_ctx.yaml).

The traces are not in the repository, so these tests tie the summary and quoted numbers to the
stored per-calibration results, and the stored runs to the registered design.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "realtrace_ctx.yaml").read_text())
RESULTS = REPO / CFG["results_dir"]
STICKY, STAR, RR = "session-sticky", "radix-affinity-balanced@16", "round-robin"
PURE = "radix-affinity"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load("run_realtrace_ctx")
SUMMARY = json.loads((RESULTS / "summary.json").read_text())


def _run(cal: str) -> dict:
    return json.loads((RESULTS / "R1" / cal / "realtrace.json").read_text())


def test_the_summary_is_what_the_stored_runs_say():
    assert rc.summarize(CFG, RESULTS) == SUMMARY
    assert (RESULTS / "report.md").read_text() == rc.markdown(SUMMARY, CFG)


def test_the_speeds_are_the_registered_rules_answer_to_the_round_robin_probe():
    probe = json.loads((RESULTS / "probe.json").read_text())
    assert probe["calibration"] == "A" and probe["policy"] == "round-robin"
    r1, r2 = probe["tests"]["R1"], probe["tests"]["R2"]
    assert r1["needs"] == {
        "toolagent": {"0.25": 5, "0.5": 5, "1": 6, "2": None, "4": None},
        "conversation": {"0.25": 6, "0.5": 6, "1": 7, "2": None, "4": None}}
    assert set(r2["needs"]["agentx"].values()) == {None}  # held at no probe speed
    for test, entry in probe["tests"].items():
        chosen = rc.choose_speeds(entry["probe_speeds"], entry["needs"])
        assert chosen == entry["speeds_by_the_rule"] == CFG["speeds"][test]
    assert CFG["speeds"] == {"R1": [0.25, 0.5, 1], "R2": []}


def test_each_stored_run_is_the_registered_design_with_only_the_profile_and_speeds_replaced():
    base = yaml.safe_load((REPO / CFG["tests"]["R1"]).read_text())
    for cal, profile in CFG["calibrations"].items():
        run = _run(cal)
        want = rc.derive(base, "R1", cal, profile, CFG["speeds"]["R1"],
                         f"{CFG['results_dir']}/R1/{cal}")
        assert run["config"] == want
        assert yaml.safe_load((RESULTS / "R1" / cal / "config.yaml").read_text()) == want
        assert run["git_dirty"] is False and run["n_runs"] == 2 * 3 * 8 * 8 * 3
        assert run["profile"]["calibrated"] is True
    # Both from the commit that fixed the speeds, and the same trace bytes as the first model's run.
    first = json.loads((REPO / "artifacts/results/realtrace_mooncake/realtrace.json").read_text())
    assert {_run(c)["git_sha"] for c in CFG["calibrations"]} == {"3fd95df"}
    assert all(_run(c)["trace_sha256"] == first["trace_sha256"] for c in CFG["calibrations"])


def test_r2_is_untestable_as_designed_and_was_not_run():
    r2 = SUMMARY["tests"]["R2"]
    assert r2["verdict"] == "UNTESTABLE" and r2["calibrations"] == {}
    assert not (RESULTS / "R2" / "A").exists() and not (RESULTS / "R2" / "s0").exists()
    assert r2["first_model"]["holds"] is True  # the first model's verdict is kept


def test_the_verdicts_the_write_up_quotes():
    """R1 is falsified and R1w holds, as predicted; R2 could not be tested."""
    assert SUMMARY["predicted_outcomes"] == {"R1": "falsified", "R1w": "holds", "R2": "holds"}
    r1 = SUMMARY["tests"]["R1"]
    assert r1["verdict"] == "FALSIFIED"
    assert {c: v["holds"] for c, v in r1["calibrations"].items()} == {"A": False, "s0": False}
    assert SUMMARY["R1w"] == {"by_calibration": {"A": True, "s0": True}, "verdict": "HOLDS"}
    # Tool-agent: no winner at any speed on either calibration, where R1's rule required one.
    for cal in ("A", "s0"):
        assert set(r1["calibrations"][cal]["winner_by_speed"]["toolagent"].values()) == {"tie"}
        assert r1["calibrations"][cal]["rules"]["toolagent"] == {"affinity_wins": False}
    # Conversation: affinity wins once on A (6 against 7 at 1x), never on s0.
    assert r1["calibrations"]["A"]["winner_by_speed"]["conversation"] == {
        "0.25": "tie", "0.5": "tie", "1": "affinity"}
    assert set(r1["calibrations"]["s0"]["winner_by_speed"]["conversation"].values()) == {"tie"}
    assert r1["calibrations"]["s0"]["rules"]["conversation"] == {"affinity_never_wins": True}


def test_the_replica_counts_the_write_up_quotes():
    def row(cal, trace, policy):
        req = SUMMARY["tests"]["R1"]["calibrations"][cal]["required_replicas"][trace]
        return [req[sp][policy] for sp in ("0.25", "0.5", "1")]

    for cal in ("A", "s0"):
        assert row(cal, "toolagent", STICKY) == row(cal, "toolagent", STAR) == [5, 5, 5]
        assert row(cal, "toolagent", RR) == [5, 5, 6]
        assert row(cal, "conversation", RR) == [6, 6, 7]
        assert row(cal, "conversation", STAR) == [6, 6, 6]
        # Pure prefix-affinity holds the target on no replica count, on either trace.
        assert row(cal, "toolagent", PURE) == row(cal, "conversation", PURE) == [9, 9, 9]
    assert row("A", "conversation", STICKY) == [6, 6, 7]
    assert row("s0", "conversation", STICKY) == [6, 6, 6]
    # No sensible policy is more than one replica from round-robin at these speeds.
    for cal in ("A", "s0"):
        for trace in ("toolagent", "conversation"):
            for policy in (STICKY, STAR, "radix-affinity-balanced@1"):
                assert all(abs(a - b) <= 1 for a, b in zip(row(cal, trace, policy),
                                                           row(cal, trace, RR), strict=True))
    # The two calibrations differ at 9 of the 48 cells, never by more than one replica.
    a = SUMMARY["tests"]["R1"]["calibrations"]["A"]["required_replicas"]
    b = SUMMARY["tests"]["R1"]["calibrations"]["s0"]["required_replicas"]
    gaps = [abs(a[tr][sp][p] - b[tr][sp][p]) for tr in a for sp in a[tr] for p in a[tr][sp]]
    assert (len(gaps), sum(1 for g in gaps if g), max(gaps)) == (48, 9, 1)
    # At the one speed both models share (1x), the validated model needs one or two replicas more.
    first = SUMMARY["tests"]["R1"]["first_model"]["required_replicas"]
    assert [first["toolagent"]["1"][p] for p in (RR, STICKY, STAR)] == [4, 4, 4]
    assert [first["conversation"]["1"][p] for p in (RR, STICKY, STAR)] == [5, 5, 4]
