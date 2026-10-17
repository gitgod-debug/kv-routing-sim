"""Tests for the wrapper that reruns R1 and R2 on the validated timing model
(configs/realtrace_ctx.yaml).

It derives each config, applies the registered speed rule to a round-robin probe, and sets the
verdicts side by side; the policies themselves are run by scripts/run_realtrace.py unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load("run_realtrace_ctx")
CFG = yaml.safe_load((REPO / "configs" / "realtrace_ctx.yaml").read_text())
STICKY, STAR = "session-sticky", "radix-affinity-balanced@16"


def test_the_design_changes_the_profile_and_the_speeds_and_nothing_else():
    for test, base_path in CFG["tests"].items():
        base = yaml.safe_load((REPO / base_path).read_text())
        for cal, profile in CFG["calibrations"].items():
            d = rc.derive(base, test, cal, profile, [0.5, 1], f"artifacts/results/x/{test}/{cal}")
            same = [k for k in base if k not in ("profile", "speeds", "results_dir", "name",
                                                 "title")]
            assert all(d[k] == base[k] for k in same)
            assert set(d) - set(base) <= {"results_dir", "name", "title"}
            assert d["profile"] == profile and d["speeds"] == [0.5, 1]
            # The rules, the replica axis, the seeds and the adapter are the registered ones.
            assert d["rules"] == base["rules"] and d["replicas"] == list(range(1, 9))
            assert d["seeds"] == [0, 1, 2] and d["adapter"] == base["adapter"]
            assert f"calibration {cal}" in d["title"]
    # Both calibrations are context-aware profiles a live engine was measured for.
    from kv_routing_sim.sim.cost_model import load_profile

    for path in CFG["calibrations"].values():
        p = load_profile(REPO / path)
        assert p.calibrated and p.context_aware


def test_the_probes_lines_are_read_as_written():
    out = ("probe toolagent speed 0.25: round-robin needs 3\n"
           "probe toolagent speed 4: round-robin needs more than 8\n"
           "something else\n"
           "probe conversation speed 0.25: round-robin needs 8\n")
    assert rc.parse_probe(out) == {"toolagent": {"0.25": 3, "4": None},
                                   "conversation": {"0.25": 8}}


def _needs(**by_trace):
    return {t: {f"{sp:g}": n for sp, n in zip([0.25, 0.5, 1, 2, 4], row, strict=True)}
            for t, row in by_trace.items()}


def test_the_speed_rule_is_the_highest_held_speed_and_the_two_below_it():
    probe = [0.25, 0.5, 1, 2, 4]
    # R1 on the first model: round-robin needed 4, 5, 8 and 5, 5, 8 at 1x, 2x, 4x.
    first = _needs(toolagent=[2, 3, 4, 5, 8], conversation=[3, 4, 5, 5, 8])
    assert rc.choose_speeds(probe, first) == [1, 2, 4]
    # A speed counts only if every trace of the test holds there.
    assert rc.choose_speeds(probe, _needs(a=[2, 3, 4, 5, 8], b=[3, 4, 5, None, None])) == [
        0.25, 0.5, 1]
    assert rc.choose_speeds(probe, _needs(a=[2, 3, None, None, None])) == [0.25, 0.5]
    assert rc.choose_speeds(probe, _needs(a=[7, None, None, None, None])) == [0.25]
    # Held nowhere: untestable as designed, so no speeds are chosen.
    assert rc.choose_speeds(probe, _needs(a=[None] * 5)) == []
    assert rc.choose_speeds(probe, {}) == []


def _req(*rows):
    """{trace: {speed: {policy: R}}} from (trace, speed, balanced, sticky) rows."""
    out: dict = {}
    for trace, speed, bal, sticky in rows:
        out.setdefault(trace, {})[speed] = {STAR: bal, STICKY: sticky, "round-robin": 8}
    return out


def test_within_one_is_decided_only_where_one_of_the_two_holds():
    ok = rc.within(_req(("t", "1", 4, 4), ("t", "2", 5, 4), ("c", "1", 9, 8)), 1)
    assert ok == {"testable": 3, "apart": [], "decided": True, "holds": True}
    far = rc.within(_req(("t", "1", 4, 6), ("t", "2", 9, 9)), 1)
    assert far["testable"] == 1 and not far["holds"]
    assert far["apart"] == [{"trace": "t", "speed": "1", "balanced": 4, "sticky": 6}]
    none = rc.within(_req(("t", "1", 9, 9)), 1)  # neither holds on 8: not testable
    assert not none["decided"] and not none["holds"]


def test_a_test_holds_only_on_every_calibration():
    assert rc.verdict({"A": True, "s0": True}) == "HOLDS"
    assert rc.verdict({"A": False, "s0": False}) == "FALSIFIED"
    assert rc.verdict({"A": True, "s0": False}) == "CALIBRATION-DEPENDENT"
    assert rc.verdict({"A": True, "s0": None}) == "UNTESTABLE"
    assert rc.verdict({}) == "UNTESTABLE"


def test_the_registration_names_its_rule_and_its_predictions():
    assert CFG["rules"] == {"r1w_within": 1}
    assert CFG["predictions"] == {"R1": "falsified", "R1w": "holds", "R2": "holds"}
    assert set(CFG["speeds"]) == set(CFG["tests"]) == {"R1", "R2"}
    for base_path in CFG["tests"].values():
        base = yaml.safe_load((REPO / base_path).read_text())
        assert base["probe_speeds"] == [0.25, 0.5, 1, 2, 4]
        # The first model's result this re-test is set beside is on disk.
        first = REPO / base.get("results_dir", "artifacts/results/realtrace_mooncake")
        assert json.loads((first / "realtrace.json").read_text())["n_runs"] > 0


def test_the_summary_says_untestable_when_a_test_has_no_speeds(tmp_path):
    cfg = {**CFG, "speeds": {"R1": [], "R2": []}, "results_dir": str(tmp_path)}
    s = rc.summarize(cfg, tmp_path)
    assert s["tests"]["R1"]["verdict"] == s["tests"]["R2"]["verdict"] == "UNTESTABLE"
    assert s["R1w"]["verdict"] == "UNTESTABLE"
    assert s["tests"]["R1"]["first_model"]["holds"] is False  # R1 was falsified on the first model
    assert s["tests"]["R2"]["first_model"]["holds"] is True
    assert "untestable as designed" in rc.markdown(s, cfg)


@pytest.mark.parametrize("field", ["profile", "speeds"])
def test_derive_refuses_to_touch_anything_else(field):
    base = yaml.safe_load((REPO / CFG["tests"]["R2"]).read_text())
    d = rc.derive(base, "R2", "A", "p.json", [1], "out")
    assert d[field] != base[field]
    assert d["policies"] == base["policies"] and d["alphas"] == base["alphas"]
