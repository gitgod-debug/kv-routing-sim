"""T1 and T2 (configs/tpot_slo.yaml) — their analysis, tested before either part runs.

Nothing here touches the held-out runs or the three calibrations the registered parts use: the
"engine" is a made-up profile, and the grids are written by hand. So these tests can show that the
procedure is right, and cannot show any part of its answer.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the grid runner hands its job function to worker processes by name
    spec.loader.exec_module(mod)
    return mod


t1 = _load("t1_itl_check")
t2 = _load("t2_analyze")

# ── T1: the engine is a known profile, so a correct check must say "factor 1" ────────────────

SPEC = {"n_sessions": 12, "turns_per_session": 3, "session_context_tokens": 150,
        "user_turn_tokens": 40, "assistant_turn_tokens": 60, "output_tokens": 16,
        "think_time_s": 1.0, "seed": 0, "prefix_sharing_ratio": 0.8, "arrival_rate": 2.0}
TRUTH = CostProfile(name="truth", prefill_base_s=0.0, prefill_per_token_s=3e-4, decode_base_s=0.04,
                    decode_per_seq_s=3e-4, kv_cache_blocks=1537, block_size=16, max_num_seqs=256,
                    prefill_attn_s=7.5e-7, decode_per_ctx_token_s=3.5e-6, calibrated=True)
SLOW = dataclasses.replace(TRUTH, name="slow", decode_base_s=0.12, decode_per_ctx_token_s=1.05e-5)


def _measured(itl: bool = True) -> dict:
    """What a notebook's held-out run records, with TRUTH as the engine."""
    trace, _ = AgenticWorkload(**SPEC).generate()
    r = run(trace, policy="round-robin", n_replicas=1, seed=0, profile=TRUTH, cache_model="vllm",
            record_per_request=True)
    return {"workload": "W", "preemptions": r.preemptions,
            "server_itl_s": r.engine_itl_mean_ms() / 1e3 if itl else None,
            "requests": [{"request_id": q.request_id, "valid": True, "send_offset_s": q.arrival_ts,
                          "ttft_ms": r.per_request[q.request_id]["ttft_ms"]} for q in trace]}


def test_t1_gives_factor_one_when_the_profile_is_the_engine():
    row = t1.one("W", SPEC, _measured(), TRUTH)
    assert row["factor"] == pytest.approx(1.0)
    assert row["predicted_itl_ms"] == pytest.approx(row["measured_itl_ms"])
    assert row["gaps"] == 36 * 15  # every request's tokens after its first
    assert row["ttft"]["factor"] == pytest.approx({"p50": 1.0, "p95": 1.0})
    assert row["predicted_tpot_p95_ms"] >= row["predicted_tpot_p50_ms"] > 0


def test_t1_sees_a_profile_whose_decode_is_three_times_too_slow():
    row = t1.one("W", SPEC, _measured(), SLOW)
    assert row["factor"] > 1.5 and row["predicted_itl_ms"] > row["measured_itl_ms"]


def _t1_config(tmp_path: pathlib.Path, profile: CostProfile, *, itl: bool = True) -> dict:
    (tmp_path / "p.json").write_text(json.dumps(dataclasses.asdict(profile)))
    (tmp_path / "truth.json").write_text(json.dumps(dataclasses.asdict(TRUTH)))
    (tmp_path / "prereg.json").write_text(json.dumps({"workloads": {"W": SPEC}}))
    (tmp_path / "c.json").write_text(json.dumps({"holdout": [_measured(itl)]}))
    src = {"results": str(tmp_path / "c.json"), "prereg": str(tmp_path / "prereg.json"),
           "workloads": ["W"]}
    return {"slo": {"tpot_p95_ms": 100},
            "t1": {"profile": str(tmp_path / "p.json"), "max_factor": 1.5,
                   "runs": [{"session": "c2", **src}, {"session": "c3", **src}],
                   "reported_profiles": {"B": str(tmp_path / "truth.json"),
                                         "C": str(tmp_path / "p.json")}},
            "predictions": {"t1": "holds"}}


def test_t1_holds_only_if_every_run_is_within_the_factor(tmp_path):
    good = t1.analyze(_t1_config(tmp_path, TRUTH))
    assert good["decided"] and good["holds"] and good["worst_factor"] == pytest.approx(1.0)
    # The command's profiles are reported on C3's traffic only.
    assert "B" not in good["runs"][0] and {"B", "C"} <= set(good["runs"][1])
    bad = t1.analyze(_t1_config(tmp_path, SLOW))
    assert bad["decided"] and not bad["holds"] and bad["worst_factor"] > 1.5
    assert bad["runs"][1]["B"]["factor"] == pytest.approx(1.0)  # reported beside it, not tested
    report = t1.markdown(bad, {"date_utc": "x", "git_sha": "y", "git_dirty": False})
    assert "**T1: DOES NOT HOLD**" in report and "What this cannot show" in report


def test_t1_without_a_measurement_is_undecided_never_passed(tmp_path):
    a = t1.analyze(_t1_config(tmp_path, TRUTH, itl=False))
    assert not a["decided"] and not a["holds"] and a["worst_factor"] is None
    assert "UNDECIDED" in t1.markdown(a, {"date_utc": "x", "git_sha": "y", "git_dirty": False})


# ── T2: replica counts and rules, on grids written by hand ───────────────────────────────────

POLS = ["session-sticky", "radix-affinity-balanced@16", "radix-affinity-balanced@1",
        "round-robin", "least-loaded", "radix-affinity"]


def _cell(first_ok: int, both_ok: int, n: int = 6) -> dict:
    """A cell that holds the first-token limit from `first_ok` replicas on and the per-token limit
    from `both_ok` on (0 = never), on both of two seeds."""
    reps = range(1, n + 1)
    return {"ttft_p95_ms": [[1000.0, 1500.0] if first_ok and r >= first_ok else [2500.0, 1500.0]
                            for r in reps],
            "tpot_p95_ms": [[80.0, 90.0] if both_ok and r >= both_ok else [80.0, 130.0]
                            for r in reps],
            "hit_rate": [0.9] * n}


def _grid(counts: dict[str, tuple[int, int]], n: int = 6) -> dict:
    """One calibration: prompts 1 and 64 × loads 1 and 2, every point with the same counts."""
    cells = {k: {ld: {p: _cell(*counts[p], n=n) for p in POLS} for ld in ("1", "2")}
             for k in ("1", "64")}
    return {"axes": {"replicas": list(range(1, n + 1)), "policies": POLS, "prompts": [1, 64],
                     "loads": [1, 2], "seeds": [0, 1]},
            "cells": cells, "n_runs": 0, "git_sha": "test", "git_dirty": False}


BASE = {"session-sticky": (2, 4), "radix-affinity-balanced@16": (3, 5),
        "radix-affinity-balanced@1": (3, 0), "round-robin": (3, 5), "least-loaded": (4, 6),
        "radix-affinity": (0, 0)}
CFG = {"slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 100},
       "t2": {"replicas_max": 6, "calibrations": {"A": {}, "B": {}},
              "rules": {"t2a_prompts": [1], "t2b_versus": ["round-robin", "least-loaded"],
                        "t2c_min_median_ratio": 1.5, "t2c_min_points": 4},
              "reported_tpot_limits_ms": [150]},
       "predictions": {"t1": "holds", "t2a": "holds", "t2b": "holds", "t2c": "holds"}}


def _committed(grid: dict, top: int = 4) -> dict:
    """The first-token table a calibration is checked against, as run_h2b_c2.py writes it."""
    return {"not_met": top + 1,
            "required_replicas": t2.table(grid, 2000, None, max_replicas=top)}


def test_a_cell_counts_only_if_every_seed_meets_every_half():
    cell = {"ttft_p95_ms": [[2500.0, 900.0], [1900.0, 900.0], [1900.0, 900.0], [800.0, 900.0]],
            "tpot_p95_ms": [[50.0, 50.0], [50.0, 101.0], [100.0, 99.0], [50.0, 50.0]]}
    reps = [1, 2, 3, 4]
    assert t2.required(cell, reps, 2000, None) == 2  # the first token alone
    assert t2.required(cell, reps, 2000, 100) == 3  # at the limit is within it; 101 is not
    assert t2.required(cell, reps, 2000, 40) == 5  # never: one past the axis
    assert t2.required(cell, reps, 500, None) == 5
    # The committed tables stop at fewer replicas: the comparison must stop there too.
    assert t2.required(cell, reps, 2000, 100, max_replicas=2) == 3
    assert t2.required(cell, reps, 2000, None, max_replicas=1) == 2


def test_the_guard_finds_a_first_token_table_that_no_longer_reproduces():
    grid = _grid(BASE)
    assert t2.guard(grid, _committed(grid), 2000) == []
    moved = _committed(grid)
    moved["required_replicas"]["64"]["2"]["round-robin"] = 2
    assert t2.guard(grid, moved, 2000) == ["K=64 load=2 round-robin: 3 here, 2 committed"]
    with pytest.raises(SystemExit, match="does not reproduce"):
        t2.analyze(CFG, {"A": grid, "B": grid}, {"A": moved, "B": _committed(grid)})
    with pytest.raises(SystemExit, match="no grid for calibration"):
        t2.analyze(CFG, {"A": grid}, {"A": _committed(grid), "B": _committed(grid)})


def test_the_three_rules_on_a_grid_where_all_hold():
    grid = _grid(BASE)
    a = t2.analyze(CFG, {"A": grid, "B": grid}, {"A": _committed(grid), "B": _committed(grid)})
    both = a["required"]["two_part"]["A"]["1"]["1"]
    assert both == {"session-sticky": 4, "radix-affinity-balanced@16": 5,
                    "radix-affinity-balanced@1": 7, "round-robin": 5, "least-loaded": 6,
                    "radix-affinity": 7}
    assert a["required"]["first_token"]["A"]["1"]["1"]["session-sticky"] == 2
    assert all(a[r]["decided"] and a[r]["holds"] for r in ("t2a", "t2b", "t2c"))
    assert a["t2a"]["by_calibration"]["A"]["points"] == 2  # K = 1 only: two loads
    assert a["t2b"]["by_calibration"]["A"]["points"] == 4
    c = a["t2c"]["by_calibration"]["A"]
    assert (c["points"], c["median_ratio"]) == (4, 2.0)  # 4 replicas where 2 held the first token
    # Reported: 5 of 6 policies hold the first token somewhere, and each needs more for both.
    assert a["binding"] == {"cells": 48, "per_token_half_adds_replicas": 40,
                            "same_count_under_both": 0, "held_under_neither": 8}
    assert a["agreement"]["two_part"]["identical"] == a["agreement"]["two_part"]["cells"] == 24
    # At 150 ms every hand-made cell is fast enough: the counts fall back to the first-token ones.
    assert a["other_limits"]["150"]["required"]["A"]["1"]["1"]["session-sticky"] == 2
    report = t2.markdown(a, {"A": grid, "B": grid})
    assert report.count("**HOLDS**") == 3 and "EXPLORATORY" in report
    assert "| 1 | 4 / 4 | 5 / 5 | — / — |" in report


def test_one_calibration_is_enough_to_fail_a_rule():
    """C3's lesson, built into the rules: each must hold on EVERY calibration."""
    grid = _grid(BASE)
    other = _grid({**BASE, "round-robin": (3, 3)})  # here round-robin needs 3, sticky 4
    a = t2.analyze(CFG, {"A": grid, "B": other},
                   {"A": _committed(grid), "B": _committed(other)})
    assert a["t2a"]["by_calibration"]["A"]["holds"] and not a["t2a"]["by_calibration"]["B"]["holds"]
    assert not a["t2a"]["holds"] and not a["t2b"]["holds"] and a["t2c"]["holds"]
    beaten = a["t2a"]["by_calibration"]["B"]["beaten"][0]
    assert beaten["fewer"] == {"round-robin": 3} and beaten["sticky"] == 4
    assert len(a["t2b"]["by_calibration"]["B"]["worse"]) == 4  # at all four points
    report = t2.markdown(a, {"A": grid, "B": other})
    assert report.count("**DOES NOT HOLD**") == 2
    assert "T2a, calibration B: at K 1, 1 sessions/s session-sticky needs 4; fewer: round-robin 3"\
        in report
    assert "**5 / 3**" in report  # cells where the calibrations differ are bold


def test_a_point_where_nothing_holds_is_a_tie_and_is_counted():
    never = dict.fromkeys(POLS, (2, 0))  # the per-token limit is never met, by anyone
    grid = _grid(never)
    a = t2.analyze(CFG, {"A": grid, "B": grid}, {"A": _committed(grid), "B": _committed(grid)})
    assert a["t2a"]["holds"] and a["t2a"]["by_calibration"]["A"]["none_holds"] == 2
    # …and with no point where sticky's count is defined under both targets, T2c is UNDECIDED.
    assert not a["t2c"]["decided"] and not a["t2c"]["holds"]
    assert "UNDECIDED" in t2.markdown(a, {"A": grid, "B": grid})


def test_h2bs_rules_are_applied_unchanged_for_the_record():
    tab = {"1": {"1": {t2.STICKY: 2, t2.BAL: 3}, "2": {t2.STICKY: 4, t2.BAL: 4}},
           "64": {"1": {t2.STICKY: 6, t2.BAL: 5}, "2": {t2.STICKY: 9, t2.BAL: 9}}}
    assert t2.h2b_rules(tab) == {"affinity_wins_at_k_min": [], "affinity_wins_at_k_max": ["1"],
                                 "reversals": [], "holds": True}
    tab["1"]["1"][t2.BAL] = 1  # affinity wins with ONE prompt: rule (a) fails
    assert not t2.h2b_rules(tab)["holds"]


# ── the grid runner, end to end on a design that is not the registered one ───────────────────


def test_the_grid_runner_writes_what_the_analysis_reads(tmp_path):
    """A made-up profile, one policy pair, three replicas: the plumbing, and nothing of T2."""
    toy = dataclasses.replace(TRUTH, name="toy")
    (tmp_path / "toy.json").write_text(json.dumps(dataclasses.asdict(toy)))
    p5 = {"session_shape": {**{k: SPEC[k] for k in SPEC if k not in (
              "prefix_sharing_ratio", "arrival_rate")}},
          "h2b": {"sharing": 0.8, "n_shared_prefixes": [1, 2], "prefix_zipf_s": 1.0}}
    (tmp_path / "p5.yaml").write_text(yaml.safe_dump(p5))
    h2b = {"h2b_config": str(tmp_path / "p5.yaml"), "cache_model": "vllm", "alpha": 16,
           "alphas_reported": [], "policies_reported": ["round-robin"], "seeds": [0, 1],
           "loads": [0.5, 2]}
    (tmp_path / "h2b.yaml").write_text(yaml.safe_dump(h2b))
    cfg = {"t2": {"h2b_config": str(tmp_path / "h2b.yaml"), "replicas_max": 3,
                  "calibrations": {"X": {"profile": str(tmp_path / "toy.json")}}}}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(cfg))
    # As it is really run — its own process — so its worker pool is not forked from pytest's.
    subprocess.run([sys.executable, str(REPO / "scripts" / "run_t2_grid.py"), "--config",
                    str(tmp_path / "t.yaml"), "--out", str(tmp_path / "out"), "--workers", "2"],
                   check=True, capture_output=True, cwd=REPO)
    grid = json.loads((tmp_path / "out" / "grid_X.json").read_text())
    assert grid["axes"] == {"prompts": [1, 2], "loads": [0.5, 2], "replicas": [1, 2, 3],
                            "seeds": [0, 1], "policies": ["session-sticky",
                                                          "radix-affinity-balanced@16",
                                                          "round-robin"]}
    assert grid["n_runs"] == 2 * 2 * 3 * 3 * 2 and grid["calibration"] == "X"
    cell = grid["cells"]["2"]["0.5"]["round-robin"]
    assert [len(cell[m]) for m in ("ttft_p95_ms", "tpot_p95_ms", "hit_rate")] == [3, 3, 3]
    assert all(len(row) == 2 and all(x > 0 for x in row) for row in cell["tpot_p95_ms"])
    # The stored numbers are the run's own: one cell, recomputed here.
    trace, _ = AgenticWorkload(**p5["session_shape"], prefix_sharing_ratio=0.8,
                               n_shared_prefixes=2, prefix_zipf_s=1.0,
                               arrival_rate=0.5).generate()
    r = run(trace, policy="round-robin", n_replicas=2, seed=1, profile=toy, cache_model="vllm")
    assert cell["ttft_p95_ms"][1][1] == r.ttft_percentiles()["p95_ms"]
    assert cell["tpot_p95_ms"][1][1] == r.tpot_percentiles()["p95_ms"]
    # And the analysis can count replicas from it.
    tab = t2.table(grid, 2000, 100)
    assert set(tab) == {"1", "2"} and set(tab["1"]) == {"0.5", "2"}
    assert all(1 <= v <= 4 for by_load in tab.values() for row in by_load.values()
               for v in row.values())
