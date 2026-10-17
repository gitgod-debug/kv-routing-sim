"""Tests for C4's analysis (scripts/c4_analyze.py) on hand-built data.

Each case fixes what a rule must report when the model is right, when it is wrong by a known amount,
and when the data is missing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c4 = _load("c4_analyze")
STICKY, BAL16, BAL1, RR = c4.STICKY, c4.BAL16, c4.BAL1, "round-robin"

RULES = {"c4a_repeats": 3, "d1a_max_factor": 1.10, "d1b_max_factor": 1.25, "d2_max_factor": 1.5,
         "t3a_max_factor": 1.25, "t3b_min_margin": 0.15, "e_min_coverage": 0.90, "f1_load": 2,
         "f2_max_loads_won": 2}
STEP = {8: 0.080, 12: 0.100, 16: 0.120, 64: 0.380}  # the "registered" steps, by batch size


def _prereg() -> dict:
    return {
        "config": {"rules": RULES, "slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 100},
                   "decode_shared": {"batches": [8, 12, 16, 64], "controls": [[1408, 8]]},
                   "stall": {"cells": [[8, 128, 2.0], [64, 128, 9.5]]}},
        "fitted_max_summed_context": 20_200,
        "predictions": {
            "decode_shared": {str(b): {"summed_context": b * 1444.0, "step_s": s}
                              for b, s in STEP.items()},
            "stall": {"8+128": {"usual_gap_s": 0.080, "extra_gap_s": 0.180},
                      "64+128": {"usual_gap_s": 0.380, "extra_gap_s": 0.180}}},
    }


def _shared(scale: dict[int, float] | None = None, reps: int = 2) -> list[dict]:
    scale = scale or {}
    return [{"rep": r, "b": b, "valid": True, "step_s": s * scale.get(b, 1.0)}
            for r in range(reps) for b, s in STEP.items()]


# D1: decode with a shared prefix


def test_d1_holds_when_the_line_continues_and_sharing_is_free():
    data = {"decode_shared": _shared(),
            "decode_control": [{"rep": r, "L": 1408, "b": 8, "valid": True, "step_s": 0.082}
                               for r in range(2)]}
    d = c4.rule_d1(data, _prereg())
    assert d["d1a"] == {"decided": True, "holds": True, "max_factor": 1.10}
    assert d["d1b"]["holds"] and d["d1b"]["cells_tested"] == [16, 64]  # past 20,200 only
    assert [c["beyond_fitted"] for c in d["cells"]] == [False, False, True, True]
    assert d["controls"][0]["shared_over_unique"] == pytest.approx(0.080 / 0.082)


def test_d1b_fails_when_the_engine_is_much_faster_than_the_line_at_large_batches():
    """The case T1 suggests: at b = 64 the real step is 0.27 s where the line says 0.38 s."""
    data = {"decode_shared": _shared({64: 0.27 / 0.38}), "decode_control": []}
    d = c4.rule_d1(data, _prereg())
    big = next(c for c in d["cells"] if c["b"] == 64)
    assert big["factor"] == pytest.approx(0.38 / 0.27) and big["factor"] > 1.25
    assert big["predicted_over_measured"] > 1  # signed: the model is too slow
    assert d["d1b"]["decided"] and not d["d1b"]["holds"]
    # A miss inside the fitted range (b = 8) is reported, but D1b is not decided on it.
    inside = c4.rule_d1({"decode_shared": _shared({8: 2.0}), "decode_control": []}, _prereg())
    assert inside["d1b"]["holds"]


def test_d1_is_undecided_without_valid_cells_and_never_passed():
    cells = _shared()
    for c in cells:
        if c["b"] == 64:
            c["valid"] = False  # both repeats preempted: nothing to compare
    d = c4.rule_d1({"decode_shared": cells, "decode_control": []}, _prereg())
    assert not d["d1b"]["decided"] and not d["d1b"]["holds"]
    assert not d["d1a"]["decided"] and not d["d1a"]["holds"]  # no control was measured
    one = _shared()
    one[3]["valid"] = False  # one repeat of b = 64 lost: the other still decides
    assert c4.rule_d1({"decode_shared": one, "decode_control": []}, _prereg())["d1b"]["holds"]


def test_d1a_fails_when_sharing_a_prefix_changes_the_step():
    data = {"decode_shared": _shared(),
            "decode_control": [{"rep": 0, "L": 1408, "b": 8, "valid": True, "step_s": 0.095}]}
    d = c4.rule_d1(data, _prereg())
    assert d["d1a"]["decided"] and not d["d1a"]["holds"]  # 0.095 against 0.080: 1.19x


# D2: the decode stall caused by an arriving prefill


def _stall(extra: dict[int, float | None]) -> list[dict]:
    rows = []
    for b, usual in ((8, 0.082), (64, 0.37)):
        if extra.get(b) is None:
            continue
        rows += [{"rep": r, "b": b, "new_tokens": 128, "valid": True, "usual_gap_s": usual,
                  "gap_at_arrival_s": usual + extra[b], "late_ttft_s": 0.3} for r in range(2)]
    return rows


def test_d2_compares_the_extra_gap_not_the_gap():
    d = c4.rule_d2({"stall": _stall({8: 0.170, 64: 0.150})}, _prereg())
    assert d["decided"] and d["holds"]
    assert [round(c["factor"], 3) for c in d["cells"]] == [round(0.18 / 0.17, 3), 1.2]
    # Half the predicted stall at b = 64: prefill inside a busy batch is cheaper than alone.
    d = c4.rule_d2({"stall": _stall({8: 0.170, 64: 0.085})}, _prereg())
    assert d["decided"] and not d["holds"] and d["cells"][1]["factor"] > 2


def test_d2_no_stall_at_all_is_a_miss_and_a_missing_cell_is_undecided():
    none = c4.rule_d2({"stall": _stall({8: 0.170, 64: -0.01})}, _prereg())
    assert none["decided"] and not none["holds"] and none["cells"][1]["factor"] == float("inf")
    missing = c4.rule_d2({"stall": _stall({8: 0.170, 64: None})}, _prereg())
    assert not missing["decided"] and not missing["holds"]


# T3: held-out traffic


def _holdout(name: str, tpots: list[float], share_engine: float | None = 0.0) -> dict:
    reqs = [{"request_id": f"r{i}", "valid": True, "ttft_ms": 300.0, "tpot_ms": t,
             "send_offset_s": float(i)} for i, t in enumerate(tpots)]
    hist = None
    if share_engine is not None:
        n = len(tpots)
        hist = {"count": float(n), "sum": 1.0,
                "buckets": {"0.05": 0.0, "0.1": share_engine * n, "+Inf": float(n)}}
    return {"workload": name, "requests": reqs, "invalid": 0, "hit_rate": 0.9, "preemptions": 0,
            "server_itl_s": 0.07, "worst_send_lag_ms": 1.0, "engine_tpot_hist": hist}


def _reg(p95: float, meets: bool) -> dict:
    return {"profile A": {"tpot_p95_ms": p95, "tpot_p50_ms": 60.0, "itl_mean_ms": 70.0,
                          "ttft_p50_ms": 300.0, "ttft_p95_ms": 300.0, "hit_rate": 0.9,
                          "share_within_limit": 1.0 if meets else 0.5,
                          "meets_per_token_limit": meets, "margin": abs(p95 - 100) / 100}}


def _traffic_prereg(runs: dict) -> dict:
    p = _prereg()
    p["workloads"] = {name: {} for name in runs}
    p["predictions"]["runs"] = runs
    return p


def test_the_measured_run_is_recomputed_from_its_requests():
    m = c4.measured_run(_holdout("w", [50.0] * 90 + [150.0] * 10, share_engine=0.9), 100.0)
    assert (m["tpot_p50_ms"], m["tpot_p95_ms"]) == (50.0, 150.0)
    assert m["client_share_within_limit"] == 0.9 and m["engine_share_within_limit"] == 0.9
    assert m["itl_mean_ms"] == pytest.approx(70.0) and m["with_tpot"] == 100
    # The engine's share is read at a bucket bound, never interpolated between two.
    assert c4.measured_run(_holdout("w", [50.0] * 10, share_engine=1.0), 80.0)[
        "engine_share_within_limit"] is None


def test_t3_tests_numbers_everywhere_and_verdicts_only_away_from_the_limit():
    runs = {"below": _reg(80.0, True), "at": _reg(104.0, False), "above": _reg(250.0, False)}
    data = {"holdout": [_holdout("below", [70.0] * 100, share_engine=1.0),
                        _holdout("at", [95.0] * 100, share_engine=1.0),  # the engine meets it
                        _holdout("above", [230.0] * 100, share_engine=0.3)]}
    t = c4.rule_t3(data, _traffic_prereg(runs), fresh={})
    assert t["t3a"]["holds"]  # 80/70, 104/95, 250/230: all within 1.25x
    # `at` was registered 4% from the limit, too close for its verdict to be tested: it disagrees
    # with the engine's, and that is not counted against the model.
    assert t["t3b"]["workloads_tested"] == ["below", "above"] and t["t3b"]["holds"]
    at = next(r for r in t["runs"] if r["workload"] == "at")
    assert at["verdict_agrees"] is False and at["verdict_tested"] is False


def test_t3_fails_on_a_wrong_tail_and_on_a_wrong_verdict():
    runs = {"below": _reg(80.0, True), "above": _reg(250.0, False)}
    data = {"holdout": [_holdout("below", [70.0] * 90 + [140.0] * 10, share_engine=0.9),
                        _holdout("above", [230.0] * 100, share_engine=0.3)]}
    t = c4.rule_t3(data, _traffic_prereg(runs), fresh={})
    assert t["t3a"]["decided"] and not t["t3a"]["holds"]  # p95 140 against a registered 80
    assert t["t3b"]["decided"] and not t["t3b"]["holds"]  # the model said it meets the limit


def test_t3_uses_the_clients_times_when_the_engine_has_no_histogram_and_says_so():
    runs = {"below": _reg(80.0, True)}
    t = c4.rule_t3({"holdout": [_holdout("below", [70.0] * 100, share_engine=None)]},
                   _traffic_prereg(runs), fresh={})
    row = t["runs"][0]
    assert row["verdict_source"] == "client-side token times" and row["verdict_agrees"] is True
    missing = c4.rule_t3({"holdout": []}, _traffic_prereg(runs), fresh={})
    assert not missing["t3a"]["decided"] and not missing["t3b"]["decided"]


# E and F: the fresh calibrations, in the simulator


def _cell(first_ok: int, both_ok: int, n: int) -> dict:
    reps = range(1, n + 1)
    return {"ttft_p95_ms": [[1000.0] * 2 if first_ok and r >= first_ok else [2500.0] * 2
                            for r in reps],
            "tpot_p95_ms": [[80.0] * 2 if both_ok and r >= both_ok else [130.0] * 2
                            for r in reps]}


def _grid(fits: dict[str, dict[str, tuple[int, int]]], prompts=("1",), loads=("1",),
          n: int = 8) -> dict:
    """fits: {fit id: {policy: (replicas for the first token, replicas for both halves)}}."""
    return {"axes": {"replicas": list(range(1, n + 1)), "prompts": [int(k) for k in prompts]},
            "fits": {f: {k: {ld: {p: _cell(*c, n) for p, c in pols.items()} for ld in loads}
                         for k in prompts} for f, pols in fits.items()}}


def _e_prereg() -> dict:
    return {"config": {"rules": RULES, "slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 100}}}


def test_e_counts_how_often_one_runs_spread_covers_another_runs_count():
    fits = {
        "A": {STICKY: (3, 5)},
        "s0": {STICKY: (3, 5)}, "s0r0": {STICKY: (3, 5)}, "s0r1": {STICKY: (4, 6)},
        "s1": {STICKY: (4, 7)}, "s1r0": {STICKY: (4, 7)}, "s1r1": {STICKY: (4, 7)},
    }
    e = c4.rule_e({"grids": {"firmness": _grid(fits)}}, _e_prereg())
    assert e["runs"] == {"s0": 3, "s1": 3}  # each run: its pooled fit and two repeats
    # First token: s0's range 3-4 covers s1's 4, but s1's range 4-4 does not cover s0's 3.
    # Both halves: s0's range 5-6 does not cover s1's 7, and s1's 7-7 does not cover s0's 5.
    assert (e["cases"], e["covered"]) == (4, 1) and e["coverage"] == 0.25
    assert e["coverage_within_one"] == 0.75  # widened by one: all but 7-7 against 5
    assert e["decided"] and not e["holds"] and len(e["misses"]) == 3
    # Profile A (3, 5): inside s0's ranges on both targets, outside s1's on both.
    assert e["profile_a_coverage"] == 0.5


def test_e_holds_when_the_ranges_cover_and_needs_two_runs():
    same = {f: {STICKY: (3, 5), RR: (5, 0)} for f in ("s0", "s0r0", "s1", "s1r0")}
    e = c4.rule_e({"grids": {"firmness": _grid(same)}}, _e_prereg())
    assert e["decided"] and e["holds"] and e["coverage"] == 1.0 and e["cases"] == 8
    assert e["profile_a_coverage"] is None  # no profile A in this grid
    one = c4.rule_e({"grids": {"firmness": _grid({"s0": {STICKY: (3, 5)},
                                                  "s0r0": {STICKY: (3, 5)}})}}, _e_prereg())
    assert not one["decided"] and not one["holds"]  # one run: nothing to cover


def _f_grid(by_fit: dict[str, dict[str, dict[str, tuple[int, int]]]]) -> dict:
    """by_fit: {fit: {load: {policy: (first token, both halves)}}} at 64 prompts, 8 replicas."""
    return {"axes": {"replicas": list(range(1, 9)), "prompts": [64]},
            "fits": {f: {"64": {ld: {p: _cell(*c, 8) for p, c in pols.items()}
                                for ld, pols in loads.items()}} for f, loads in by_fit.items()}}


def test_f1_and_f2_on_a_grid_where_both_hold():
    loads = {"1": {STICKY: (3, 4), BAL16: (3, 4), BAL1: (2, 0)},
             "2": {STICKY: (6, 7), BAL16: (6, 6), BAL1: (4, 0)},   # balanced wins this load
             "3": {STICKY: (7, 8), BAL16: (8, 0), BAL1: (0, 0)}}   # sticky wins this one
    f = c4.rule_f({"grids": {"prompts64": _f_grid({"s0": loads, "s1": loads})}}, _e_prereg())
    assert f["f1"]["holds"] and f["f1"]["by_fit"]["s0"] == {"sticky": 6, "weight_1": 4,
                                                             "holds": True}
    two = f["f2"]["by_fit"]["s1"]
    assert two["weight_1_holds_nothing"] and two["weight_1_at_load"] == 9  # one past the axis
    assert (two["loads_balanced_wins"], two["loads_sticky_wins"]) == (["2"], ["3"])
    assert f["f2"]["holds"] and f["tables"]["two_part"]["s0"]["2"][STICKY] == 7


def test_f_fails_on_one_fresh_fit_and_says_which():
    ok = {"2": {STICKY: (6, 7), BAL16: (6, 7), BAL1: (4, 0)}}
    tie = {"2": {STICKY: (6, 7), BAL16: (6, 7), BAL1: (6, 5)}}  # weight 1: no saving, holds both
    f = c4.rule_f({"grids": {"prompts64": _f_grid({"s0": ok, "s1": tie})}}, _e_prereg())
    assert not f["f1"]["holds"] and f["f1"]["by_fit"]["s0"]["holds"]
    assert not f["f1"]["by_fit"]["s1"]["holds"]  # 6 against 6 is not "fewer"
    assert not f["f2"]["holds"] and not f["f2"]["by_fit"]["s1"]["weight_1_holds_nothing"]


# The registration check


def test_a_run_that_is_not_the_registered_one_is_refused(tmp_path):
    prereg = {"config": {}, "predictions_sha256": "p", "trace_fingerprints_sha256": {"w": "f"},
              "wheel": {"sha256": "w"}}
    data = {"predictions_sha256": "p", "fingerprints": {"w": "f"}, "wheel_sha256": "w"}
    for key, value in (("predictions_sha256", "x"), ("wheel_sha256", "x"),
                       ("fingerprints", {"w": "other"})):
        with pytest.raises(SystemExit, match="different"):
            c4.analyze({**data, key: value}, prereg, tmp_path)
