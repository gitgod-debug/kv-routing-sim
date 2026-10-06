"""R3's committed result must be what the stored runs say, through the registered analysis.

R3 (configs/r3_bailian.yaml, registered 2026-10-06): the two published routing policies — LMetric,
and SMetric's rule without its global tier — beside this project's own, on Alibaba's public coder
trace, on four calibrations. The trace is other people's data and is not in the repository, so
nothing here replays it: these tests tie the analysis and every quoted number to the stored grid,
and the stored grid to the registered design.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "r3_bailian.yaml").read_text())
RESULTS = REPO / CFG["results_dir"]
STICKY, RR, PURE = "session-sticky", "round-robin", "radix-affinity"
LM, SM, BAL16 = "lmetric", "smetric-local", "radix-affinity-balanced@16"
CALS, SPEEDS = ["A", "s0", "s1", "s2"], ["1", "2", "4"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _load("r3_analyze")
GRID = json.loads((RESULTS / "grid.json").read_text())
PROBE = json.loads((RESULTS / "probe.json").read_text())
ANALYSIS = json.loads((RESULTS / "analysis.json").read_text())


def _counts(target: str, policy: str) -> list[list[int]]:
    """One policy's replica counts: a row per speed, a column per calibration."""
    tab = ANALYSIS["tables"][target]
    return [[tab[c][sp][policy] for c in CALS] for sp in SPEEDS]


def test_the_analysis_is_the_registered_scripts_answer_to_the_stored_grid():
    fresh = json.loads(json.dumps(ra.analyze(GRID, CFG), default=str))
    assert fresh == ANALYSIS
    assert (RESULTS / "report.md").read_text() == ra.markdown(ANALYSIS, GRID, CFG)


def test_the_speeds_are_the_registered_rules_answer_to_the_round_robin_probe():
    assert PROBE["probe"] is True and PROBE["git_dirty"] is False
    axes = PROBE["axes"]
    assert (axes["calibrations"], axes["policies"], axes["seeds"]) == (["A"], [RR], [0])
    reps, slo = axes["replicas"], CFG["slo"]
    cells = PROBE["cells"]["A"]
    first = {sp: ra.t2.required(cells[sp][RR], reps, slo["ttft_p95_ms"], None)
             for sp in axes["speeds"]}
    both = {sp: ra.t2.required(cells[sp][RR], reps, slo["ttft_p95_ms"], slo["tpot_p95_ms"])
            for sp in axes["speeds"]}
    assert first == {"0.25": 1, "0.5": 1, "1": 2, "2": 4, "4": 8}
    assert both == {"0.25": 2, "0.5": 2, "1": 4, "2": 7, "4": 14}
    # The rule: the highest probe speed round-robin holds the target at, and the two below it.
    probed = sorted(CFG["probe_speeds"])
    top = max(s for s in probed if first[f"{s:g}"] <= reps[-1])
    assert probed[: probed.index(top) + 1][-3:] == CFG["speeds"] == [1, 2, 4]
    # The probe ran before the speeds were written down; the grid ran from the commit that did.
    assert PROBE["config"]["speeds"] == [] and GRID["config"]["speeds"] == [1, 2, 4]


def test_the_grid_is_the_registered_design_from_a_clean_tree():
    assert GRID["probe"] is False and GRID["git_dirty"] is False
    assert GRID["config"] == CFG
    assert GRID["trace_sha256"] == CFG["trace_sha256"]
    axes = GRID["axes"]
    assert axes["calibrations"] == CALS and axes["speeds"] == SPEEDS
    assert axes["policies"] == [p["label"] for p in CFG["policies"]] and len(axes["policies"]) == 9
    assert axes["replicas"] == list(range(1, 17)) and axes["seeds"] == [0, 1, 2]
    assert GRID["n_runs"] == 4 * 3 * 9 * 16 * 3 == 5184
    assert ANALYSIS["all_completed"] is True
    assert all(p["calibrated"] is True for p in GRID["profiles"].values())


def test_the_traffic_the_write_up_describes():
    s = ANALYSIS["structure"]
    assert (s["requests"], s["sessions"]) == (20054, 12647)
    assert s["requests_per_session"] == {"p50": 1, "p90": 3, "p99": 8, "max": 28}
    assert s["share_of_requests_in_sessions_of_5_or_more"] == 0.196
    assert (s["prompt_tokens"]["p50"], s["prompt_tokens"]["max"]) == (293, 1611)
    assert s["reusable_from_any_earlier_request"] == 0.6478
    assert s["reusable_from_the_same_session"] == 0.2974
    assert ANALYSIS["adapter_stats"]["skip_reasons"] == {"output_capped": 13}


def test_the_runs_are_tied_to_the_commits_that_registered_them():
    """The probe ran from the commit that registered the design, with no speeds chosen yet; the
    grid ran from the commit that wrote the speeds down."""
    assert PROBE["git_sha"] == "1ca73e9" and GRID["git_sha"] == "d189b1a"
    assert ANALYSIS["grid_header"]["git_sha"] == "d189b1a"
    assert ANALYSIS["grid_header"]["n_runs"] == 5184


def test_two_of_the_four_registered_rules_hold():
    assert ANALYSIS["target"] == "first_token" and ANALYSIS["not_met"] == 17
    assert ANALYSIS["predicted_outcomes"] == {k: "holds" for k in ("r3a", "r3b", "r3c", "r3d")}
    assert {k: ANALYSIS[k]["holds"] for k in ("r3a", "r3b", "r3c", "r3d")} == {
        "r3a": False, "r3b": True, "r3c": True, "r3d": False}
    assert all(ANALYSIS[k]["decided"] for k in ("r3a", "r3b", "r3c", "r3d"))
    # R3a fails for want of a strict win: never more than session-sticky, fewer on s0 at 4x only.
    a = ANALYSIS["r3a"]["by_calibration"]
    assert all(c["more_at"] == [] for c in a.values())
    assert {cal: c["fewer_at"] for cal, c in a.items()} == {
        "A": [], "s0": ["4"], "s1": [], "s2": []}
    # R3d fails outright: pure prefix-affinity is never worse than round-robin.
    assert all(c["not_worse_at"] == SPEEDS for c in ANALYSIS["r3d"]["by_calibration"].values())


def _same(n: int) -> list[int]:
    return [n] * 4


def test_the_replica_counts_the_write_up_quotes_first_token_target():
    """A row per speed (1x, 2x, 4x), a column per calibration (A, s0, s1, s2)."""
    want = {
        RR: [_same(2), _same(4), _same(8)],
        "least-loaded": [_same(2), _same(4), [8, 7, 7, 7]],
        STICKY: [_same(2), _same(4), _same(7)],
        BAL16: [_same(2), _same(4), _same(7)],
        "radix-affinity-balanced@1": [_same(2), _same(4), _same(6)],
        PURE: [_same(2), _same(4), [7, 6, 6, 6]],
        LM: [_same(2), _same(4), [7, 6, 7, 7]],
        SM: [_same(2), _same(4), _same(6)],
        "lmetric@dispatch": [_same(2), _same(4), _same(7)],
    }
    assert {p: _counts("first_token", p) for p in want} == want
    assert set(want) == set(GRID["axes"]["policies"])


def test_the_reported_counts_under_the_two_part_target():
    other = ANALYSIS["reported_other_target"]
    assert other["target"] == "two_part"
    assert {k: other[k]["holds"] for k in ("r3a", "r3b", "r3c", "r3d")} == {
        "r3a": True, "r3b": True, "r3c": False, "r3d": False}
    want = {
        RR: [[4, 4, 4, 4], [7, 7, 7, 7], [14, 13, 14, 14]],
        "least-loaded": [[4, 4, 4, 4], [7, 7, 8, 7], [14, 13, 14, 14]],
        STICKY: [[4, 4, 4, 4], [7, 7, 7, 7], [13, 12, 13, 13]],
        BAL16: [[4, 4, 4, 4], [7, 6, 7, 7], [12, 11, 11, 12]],
        "radix-affinity-balanced@1": [[4, 4, 4, 4], [7, 6, 7, 7], [12, 11, 12, 12]],
        PURE: [[4, 4, 4, 4], [8, 8, 7, 7], [16, 14, 17, 17]],  # 17 = not held on 16
        LM: [[4, 3, 4, 4], [6, 6, 6, 6], [11, 10, 11, 11]],
        SM: [[4, 3, 4, 4], [6, 6, 6, 6], [10, 10, 10, 10]],
        "lmetric@dispatch": [[4, 4, 4, 4], [6, 6, 6, 6], [11, 10, 11, 11]],
    }
    assert {p: _counts("two_part", p) for p in want} == want
    # "two to three replicas fewer than session-sticky at the highest load"
    for i in range(4):
        assert want[STICKY][2][i] - want[LM][2][i] == 2
        assert want[STICKY][2][i] - want[SM][2][i] in (2, 3)


def _at(policy: str, field: str) -> list:
    """One policy's value at the replica count where session-sticky first holds the target, on
    calibration A, at the three speeds."""
    at = ANALYSIS["at_sticky"]["A"]
    return [at[sp]["policies"][policy][field] for sp in SPEEDS]


def test_the_side_numbers_the_write_up_quotes():
    assert [ANALYSIS["at_sticky"]["A"][sp]["replicas"] for sp in SPEEDS] == [2, 4, 7]
    assert [round(x, 3) for x in _at(RR, "hit_rate")] == [0.304, 0.285, 0.232]
    assert [round(x, 3) for x in _at(STICKY, "hit_rate")] == [0.338, 0.359, 0.384]
    assert [round(x, 3) for x in _at(LM, "hit_rate")] == [0.353, 0.392, 0.425]
    assert [round(x, 3) for x in _at(SM, "hit_rate")] == [0.362, 0.411, 0.461]
    assert [round(x, 3) for x in _at(PURE, "hit_rate")] == [0.378, 0.434, 0.488]
    sticky = [round(x) for x in _at(STICKY, "ttft_p95_ms")]
    lmetric = [round(x) for x in _at(LM, "ttft_p95_ms")]
    assert sticky == [1042, 950, 999] and lmetric == [777, 612, 601]
    assert [round(x) for x in _at("lmetric@dispatch", "ttft_p95_ms")] == [901, 714, 685]
    # LMetric's p95 first token is a quarter to two fifths below session-sticky's at the same count.
    assert all(0.59 < a / b < 0.76 for a, b in zip(lmetric, sticky, strict=True))
    # pure prefix-affinity crowds one replica
    assert [round(x, 2) for x in _at(PURE, "busiest_over_mean")] == [1.13, 1.4, 1.87]
    # SMetric's rule at those counts: it kept 7%, 12% and 17% of requests in place and moved nothing
    router = _at(SM, "router")
    assert [round(r["stuck"]) for r in router] == [1428, 2398, 3509]
    assert [round(100 * r["stuck"] / 20054) for r in router] == [7, 12, 17]
    worst_moved = max(ANALYSIS["at_sticky"][c][sp]["policies"][SM]["router"]["migrated"]
                      for c in CALS for sp in SPEEDS)
    assert worst_moved == 0.7
    # ... and it did move sessions where the fleet was too small
    moved = GRID["cells"]["s1"]["4"][SM]["router"]["migrated"]
    assert max(m for c in CALS for sp in SPEEDS
               for m in GRID["cells"][c][sp][SM]["router"]["migrated"]) == max(moved) == 574.0
    assert GRID["axes"]["replicas"][moved.index(574.0)] == 5


def test_session_stickys_known_weakness_does_not_show_on_this_trace():
    """SMetric reports the load 2.17x out of balance under routing by session id. Here the busiest
    replica gets at most 1.14x its share in any cell with two or more replicas."""
    worst = max(b for c in CALS for sp in SPEEDS
                for b in GRID["cells"][c][sp][STICKY]["busiest_over_mean"][1:])
    assert round(worst, 2) == 1.14
    assert [round(x, 2) for x in _at(STICKY, "busiest_over_mean")] == [1.02, 1.04, 1.06]
