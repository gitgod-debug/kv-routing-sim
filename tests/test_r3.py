"""Tests for R3's grid runner and analysis on made-up data, and checks on its registered design.

The made-up trace is in the Qwen-Bailian format; nothing here says anything about R3's result.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import random
import subprocess
import sys

import pytest
import yaml

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run

REPO = pathlib.Path(__file__).resolve().parents[1]

TOY = CostProfile(
    name="toy", prefill_base_s=0.0, prefill_per_token_s=0.0005, decode_base_s=0.02,
    decode_per_seq_s=0.0005, kv_cache_blocks=2_000, block_size=16, max_num_seqs=32,
    calibrated=True, prefill_attn_s=1e-7, decode_per_ctx_token_s=1e-6,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r3 = _load("run_r3")


def _made_up_trace(path: pathlib.Path, n_sessions: int = 30, seed: int = 0) -> int:
    """Sessions of 1 to 6 turns behind two shared prompts, in the Bailian format."""
    rng = random.Random(seed)
    rows, chat, next_id = [], 0, 1000
    for s in range(n_sessions):
        t = s * 1.5 + rng.random()
        ids = [(s % 2) * 100 + j for j in range(8)]  # one of two 8-block "system prompts"
        parent = -1
        for turn in range(1, rng.randint(1, 6) + 1):
            ids = [*ids, *range(next_id, next_id + 4)]
            next_id += 4
            rows.append({"chat_id": chat, "parent_chat_id": parent, "timestamp": round(t, 3),
                         "input_length": len(ids) * 16 - 3, "output_length": 12, "type": "coder",
                         "turn": turn, "hash_ids": list(ids)})
            parent, chat = chat, chat + 1
            t += 2.0 + rng.random()
    rows.sort(key=lambda r: r["timestamp"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


@pytest.fixture
def design(tmp_path):
    n = _made_up_trace(tmp_path / "trace.jsonl")
    (tmp_path / "toy.json").write_text(json.dumps(dataclasses.asdict(TOY)))
    cfg = {
        "results_dir": str(tmp_path / "out"), "trace": str(tmp_path / "trace.jsonl"),
        "adapter_name": "bailian", "adapter": {"token_scale": 1.0, "max_total_tokens": 2048},
        "window_s": 10_000, "warmup_s": 5, "cache_model": "vllm", "replicas_max": 3,
        "seeds": [0, 1], "probe_speeds": [1, 4], "speeds": [1, 2],
        "slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 100},
        "calibrations": {"X": str(tmp_path / "toy.json")},
        "policies": [
            {"label": "round-robin", "policy": "round-robin"},
            {"label": "session-sticky", "policy": "session-sticky"},
            {"label": "radix-affinity-balanced@16", "policy": "radix-affinity-balanced",
             "kwargs": {"load_penalty": 16}},
            {"label": "lmetric", "policy": "lmetric"},
            {"label": "smetric-local", "policy": "smetric-local", "kwargs": {"ttft_slo_s": 2.0}},
        ],
    }
    (tmp_path / "r3.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path, cfg, n


def _run(tmp_path, *extra):
    # Run in its own process, as in real use, so its worker pool is not forked from pytest's.
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_r3.py"), "--config",
                           str(tmp_path / "r3.yaml"), "--workers", "2", *extra],
                          check=True, capture_output=True, text=True, cwd=REPO)


def test_the_probe_runs_round_robin_alone(design):
    tmp_path, _, _ = design
    out = _run(tmp_path, "--probe").stdout
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["probe"] is True
    assert probe["axes"]["policies"] == ["round-robin"] and probe["axes"]["seeds"] == [0]
    assert probe["axes"]["speeds"] == ["1", "4"] and probe["n_runs"] == 2 * 3
    assert set(probe["cells"]["X"]["4"]) == {"round-robin"}
    assert "round-robin at 1×" in out and "round-robin at 4×" in out
    assert not (tmp_path / "out" / "grid.json").exists()


def test_the_grid_runner_writes_every_run_and_decides_nothing(design):
    tmp_path, cfg, n_requests = design
    _run(tmp_path)
    grid = json.loads((tmp_path / "out" / "grid.json").read_text())
    labels = [p["label"] for p in cfg["policies"]]
    assert grid["axes"] == {"calibrations": ["X"], "speeds": ["1", "2"], "policies": labels,
                            "replicas": [1, 2, 3], "seeds": [0, 1]}
    assert grid["n_runs"] == 2 * 5 * 3 * 2 and grid["probe"] is False
    assert grid["adapter_stats"]["requests"] == n_requests
    assert len(grid["trace_sha256"]) == 64
    assert "verdict" not in json.dumps(grid) and "holds" not in json.dumps(grid)

    s = grid["structure"]
    assert s["requests"] == n_requests and s["sessions"] == 30
    assert s["requests_per_session"]["max"] <= 6
    # Two shared prompts and sessions that re-send their history: reuse of both kinds.
    assert 0 < s["reusable_from_the_same_session"] < s["reusable_from_any_earlier_request"] < 1

    for label in labels:
        cell = grid["cells"]["X"]["2"][label]
        assert [len(cell[m]) for m in ("ttft_p95_ms", "tpot_p95_ms", "hit_rate",
                                       "busiest_over_mean", "preemptions")] == [3] * 5
        assert all(len(row) == 2 for row in cell["ttft_p95_ms"]) and cell["all_completed"]
    # Only a policy that reports on itself has a `router` entry.
    assert set(grid["cells"]["X"]["1"]["smetric-local"]["router"]) == {"stuck", "migrated",
                                                                      "balanced"}
    assert "router" not in grid["cells"]["X"]["1"]["session-sticky"]

    # The stored numbers are the run's own: one cell, recomputed here.
    trace, _ = r3.read_trace(cfg)
    warm = sum(1 for q in trace if q.arrival_ts < cfg["warmup_s"])
    at2 = tuple(dataclasses.replace(q, arrival_ts=q.arrival_ts / 2.0) for q in trace)
    r = run(at2, policy="lmetric", n_replicas=2, seed=1, profile=TOY, cache_model="vllm",
            warmup_requests=warm)
    cell = grid["cells"]["X"]["2"]["lmetric"]
    assert cell["ttft_p95_ms"][1][1] == r.ttft_percentiles()["p95_ms"]
    assert cell["tpot_p95_ms"][1][1] == r.tpot_percentiles()["p95_ms"]


def test_the_runner_refuses_a_trace_that_is_not_the_registered_file(design):
    tmp_path, cfg, _ = design
    cfg["trace_sha256"] = "0" * 64
    (tmp_path / "r3.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(subprocess.CalledProcessError) as err:
        _run(tmp_path, "--probe")
    assert "not the registered file" in err.value.stderr


def test_structure_separates_a_sessions_own_reuse_from_reuse_across_sessions():
    from kv_routing_sim.traces.schema import NormalizedRequest as Req

    shared = (1, 2, 3, 4)
    trace = (
        Req("a1", 0.0, (*shared, 10, 11), 100, 4, "A"),
        Req("b1", 1.0, (*shared, 20, 21), 100, 4, "B"),          # 4 blocks, from A's session
        Req("a2", 2.0, (*shared, 10, 11, 12, 13), 132, 4, "A"),  # 6 blocks, all its own
    )
    s = r3.structure(trace, 16)
    assert (s["blocks"], s["distinct_blocks"], s["sessions"]) == (20, 10, 2)
    assert s["reusable_from_any_earlier_request"] == round(10 / 20, 4)
    assert s["reusable_from_the_same_session"] == round(6 / 20, 4)


# The analysis, on grids written by hand

an = _load("r3_analyze")

ROLES = {"sticky": "session-sticky", "lmetric": "lmetric", "smetric": "smetric-local",
         "balanced": "radix-affinity-balanced@16", "pure": "radix-affinity",
         "baseline": "round-robin"}
SLO = {"ttft_p95_ms": 2000, "tpot_p95_ms": 100}
REPS = [1, 2, 3, 4]


def _cell(first: int | None, both: int | None = None) -> dict:
    """A cell that meets the first-token target from `first` replicas on, and both halves from
    `both` on (None: never), on both of its seeds."""
    def row(n, limit, ok, bad):
        return [ok, ok] if limit is not None and n >= limit else [ok, bad]
    return {"ttft_p95_ms": [row(n, first, 900.0, 5000.0) for n in REPS],
            "tpot_p95_ms": [row(n, both, 60.0, 400.0) for n in REPS],
            "hit_rate": [0.5] * 4, "busiest_over_mean": [1.2] * 4, "preemptions": [0.0] * 4,
            "all_completed": True}


def _grid(by_cal: dict) -> dict:
    """{calibration: {speed: {role: first-token count or (first, both)}}} -> a grid file."""
    cells = {}
    for cal, by_speed in by_cal.items():
        for speed, by_role in by_speed.items():
            for role, v in by_role.items():
                first, both = v if isinstance(v, tuple) else (v, v)
                cells.setdefault(cal, {}).setdefault(speed, {})[ROLES[role]] = _cell(first, both)
    return {"axes": {"calibrations": list(by_cal), "speeds": list(next(iter(by_cal.values()))),
                     "policies": list(ROLES.values()), "replicas": REPS, "seeds": [0, 1]},
            "cells": cells}


def _decide(by_cal, target="first_token"):
    tabs = an.tables(_grid(by_cal), SLO)
    return an.decide(tabs[target], ROLES, 1, 5)


ALL_HOLD = {"sticky": 3, "lmetric": 2, "smetric": 3, "balanced": 4, "pure": None, "baseline": 3}


def test_a_count_is_the_fewest_replicas_on_which_every_seed_meets_the_target():
    tabs = an.tables(_grid({"A": {"1": {**ALL_HOLD, "sticky": (2, 4)}}}), SLO)
    assert tabs["first_token"]["A"]["1"]["session-sticky"] == 2
    assert tabs["two_part"]["A"]["1"]["session-sticky"] == 4
    assert tabs["first_token"]["A"]["1"]["radix-affinity"] == 5  # held on none: one past the axis


def test_the_four_rules_where_all_hold():
    d = _decide({"A": {"1": ALL_HOLD, "2": {**ALL_HOLD, "lmetric": 3}}, "B": {"1": ALL_HOLD}})
    assert all(d[k]["decided"] and d[k]["holds"] for k in ("r3a", "r3b", "r3c", "r3d"))
    assert d["r3a"]["by_calibration"]["A"]["fewer_at"] == ["1"]  # tied at 2x, fewer at 1x


def test_r3a_needs_lmetric_never_above_sticky_and_below_it_somewhere():
    tie = _decide({"A": {"1": {**ALL_HOLD, "lmetric": 3}}})
    assert tie["r3a"]["decided"] and not tie["r3a"]["holds"]  # no more, but nowhere fewer
    worse = _decide({"A": {"1": ALL_HOLD, "2": {**ALL_HOLD, "lmetric": 4}}})
    assert not worse["r3a"]["holds"] and worse["r3a"]["by_calibration"]["A"]["more_at"] == ["2"]


def test_one_calibration_is_enough_to_fail_a_rule():
    d = _decide({"A": {"1": ALL_HOLD}, "B": {"1": {**ALL_HOLD, "smetric": 4}}})
    assert d["r3b"]["by_calibration"]["A"]["holds"]
    assert not d["r3b"]["by_calibration"]["B"]["holds"] and not d["r3b"]["holds"]  # 4 against 2
    assert d["r3a"]["holds"]


def test_within_one_counts_a_policy_that_never_holds_as_one_past_the_axis():
    near = _decide({"A": {"1": {**ALL_HOLD, "balanced": None, "sticky": 4}}})
    assert near["r3c"]["holds"]  # 5 against 4
    far = _decide({"A": {"1": {**ALL_HOLD, "balanced": None, "sticky": 3}}})
    assert not far["r3c"]["holds"] and far["r3c"]["by_calibration"]["A"]["apart_at"] == ["1"]


def test_pure_affinity_fails_if_it_needs_more_than_round_robin_or_never_holds():
    assert _decide({"A": {"1": {**ALL_HOLD, "pure": 4}}})["r3d"]["holds"]      # 4 against 3
    assert _decide({"A": {"1": ALL_HOLD}})["r3d"]["holds"]                      # held on none
    tied = _decide({"A": {"1": {**ALL_HOLD, "pure": 3}}})
    assert not tied["r3d"]["holds"]
    assert tied["r3d"]["by_calibration"]["A"]["not_worse_at"] == ["1"]


def test_a_speed_nobody_holds_is_not_testable_and_no_testable_speed_is_undecided():
    nobody = dict.fromkeys(ROLES)
    d = _decide({"A": {"1": ALL_HOLD, "4": nobody}})
    assert d["r3a"]["by_calibration"]["A"]["testable"] == ["1"] and d["r3a"]["holds"]
    none = _decide({"A": {"1": ALL_HOLD}, "B": {"1": nobody}})
    assert not none["r3a"]["decided"] and not none["r3a"]["holds"]


def test_the_other_target_is_decided_from_the_same_runs():
    by_cal = {"A": {"1": {**ALL_HOLD, "lmetric": (2, 4), "sticky": (3, 3)}}}
    assert _decide(by_cal)["r3a"]["holds"]
    assert not _decide(by_cal, "two_part")["r3a"]["holds"]  # 4 against 3 with both halves


def test_the_analysis_refuses_a_grid_from_another_design_and_writes_a_report(design):
    tmp_path, cfg, _ = design
    cfg.update({"trace_sha256": None, "predictions": dict.fromkeys(("r3a", "r3b", "r3c", "r3d"),
                                                                 "holds"),
                "rules": {"target": "first_token", "within": 1, "roles": ROLES}})
    cfg["policies"].append({"label": "radix-affinity", "policy": "radix-affinity"})
    (tmp_path / "r3.yaml").write_text(yaml.safe_dump(cfg))
    _run(tmp_path)
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "r3_analyze.py"), "--config",
                          str(tmp_path / "r3.yaml")], check=True, capture_output=True,
                         text=True, cwd=REPO).stdout
    a = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert a["all_completed"] and set(a["tables"]) == {"first_token", "two_part"}
    assert all(isinstance(a[k]["holds"], bool) for k in ("r3a", "r3b", "r3c", "r3d"))
    assert "| rule | registered prediction | result |" in out
    assert "## Replicas needed" in out and "own account" in out
    # A grid run with other rules is not this design's grid.
    cfg["speeds"] = [1]
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(cfg))
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "r3_analyze.py"), "--config",
                          str(tmp_path / "other.yaml")], capture_output=True, text=True,
                         cwd=REPO, check=False)
    assert bad.returncode != 0 and "different `speeds`" in bad.stderr


# The registered design (configs/r3_bailian.yaml)

R3_CFG = yaml.safe_load((REPO / "configs" / "r3_bailian.yaml").read_text())


def test_the_registered_design_names_real_policies_calibrations_and_rules():
    from kv_routing_sim import policies
    from kv_routing_sim.sim.cost_model import load_profile
    from kv_routing_sim.traces import adapters

    labels = [p["label"] for p in R3_CFG["policies"]]
    assert len(labels) == len(set(labels)) == 9
    for spec in R3_CFG["policies"]:
        assert spec["policy"] in policies.REGISTRY
    # Every role a rule names is a policy of the grid.
    roles = R3_CFG["rules"]["roles"]
    assert set(roles) == {"sticky", "lmetric", "smetric", "balanced", "pure", "baseline"}
    assert set(roles.values()) <= set(labels)
    assert R3_CFG["rules"]["target"] == "first_token" and R3_CFG["rules"]["within"] == 1
    assert R3_CFG["predictions"] == dict.fromkeys(("r3a", "r3b", "r3c", "r3d"), "holds")
    # Four calibrations, each a measured, context-aware profile that is in the repository.
    assert list(R3_CFG["calibrations"]) == ["A", "s0", "s1", "s2"]
    for path in R3_CFG["calibrations"].values():
        profile = load_profile(REPO / path)
        assert profile.calibrated and profile.context_aware and profile.block_size == 16
    assert R3_CFG["adapter_name"] in adapters.REGISTRY
    assert R3_CFG["adapter"] == {"token_scale": 0.0625, "max_total_tokens": 2048}
    assert (R3_CFG["window_s"], R3_CFG["warmup_s"], R3_CFG["replicas_max"]) == (3600, 300, 16)
    assert R3_CFG["seeds"] == [0, 1, 2] and R3_CFG["probe_speeds"] == [0.25, 0.5, 1, 2, 4]
    assert len(R3_CFG["trace_sha256"]) == 64
    # The trace is other people's data: its path is under the ignored folder.
    assert R3_CFG["trace"].startswith("artifacts/traces/")


def test_the_new_policies_are_built_as_registered():
    """The two published policies, built the way the grid builds them, are the documented ports."""
    from kv_routing_sim import policies
    from kv_routing_sim.sim.cost_model import load_profile

    profile = load_profile(REPO / R3_CFG["calibrations"]["A"])
    by_label = {p["label"]: p for p in R3_CFG["policies"]}
    lm = policies.build("lmetric", 4, profile=profile, **(by_label["lmetric"].get("kwargs") or {}))
    assert lm.index == "events" and lm.wants_cache_view and lm.wants_events
    assert lm.block_size == 16
    sm = policies.build("smetric-local", 4, profile=profile,
                        **by_label["smetric-local"]["kwargs"])
    assert (sm.ttft_slo_s, sm.slack, sm.hit_ratio, sm.index) == (2.0, 1.0, 0.5, "events")
    assert sm.ttft_slo_s * 1000 == R3_CFG["slo"]["ttft_p95_ms"]  # the rule's deadline is the SLO
    disp = policies.build("lmetric", 4, profile=profile, **by_label["lmetric@dispatch"]["kwargs"])
    assert disp.index == "dispatch" and not disp.wants_cache_view
