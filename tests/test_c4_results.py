"""C4's committed result must come out of the committed raw files, through the registered analysis.

C4 (configs/c4_per_token.yaml, registered 2026-10-05, run on a Kaggle T4 the same day): the
per-token model where it matters. Seven of its nine rules hold; E and F2 do not — both had been
predicted to. The numbers pinned here are the ones the README, CHECKPOINTS and
docs/09-hypotheses.md quote; if a later change to the simulator moves one, this fails first.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import sys
import tempfile

import pytest

from kv_routing_sim.sim.cost_model import load_profile

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results" / "c4_kaggle_t4"
PREREG = json.loads((REPO / "artifacts" / "results" / "c4_prereg" / "predictions.json").read_text())
COMMITTED = json.loads((RESULTS / "analysis.json").read_text())
RULES = ("c4a", "d1a", "d1b", "d2", "t3a", "t3b", "e", "f1", "f2")
EXPECTED_VERDICTS = {"c4a": True, "d1a": True, "d1b": True, "d2": True, "t3a": True, "t3b": True,
                     "e": False, "f1": True, "f2": False}
STICKY, RR = "session-sticky", "round-robin"
BAL16, BAL1 = "radix-affinity-balanced@16", "radix-affinity-balanced@1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c4 = _load("c4_analyze")
c4g = _load("c4_grids")
posthoc = _load("c4_posthoc_fresh_cells")
GRIDS = json.loads((RESULTS / "grids.json").read_text())


@pytest.fixture(scope="module")
def analysis():
    data = json.loads((RESULTS / "c4.json").read_text())
    return c4.analyze(data, PREREG, RESULTS, GRIDS)  # stops unless the run is the registered one


def test_the_run_is_the_registered_one():
    data = json.loads((RESULTS / "c4.json").read_text())
    assert data["predictions_sha256"] == PREREG["predictions_sha256"]
    assert data["wheel_sha256"] == PREREG["wheel"]["sha256"]
    assert PREREG["wheel"]["built_from"] == "c5b5d74"
    assert data["fingerprints"] == PREREG["trace_fingerprints_sha256"]
    # Nothing was invalid, and nothing was preempted.
    for part, n in (("decode_shared", 14), ("decode_control", 4), ("stall", 6)):
        assert len(data[part]) == n and all(m["valid"] for m in data[part])
    assert [(m["invalid"], m["without_tpot"], m["preemptions"]) for m in data["holdout"]] == [
        (0, 0, 0)] * 3
    run = json.loads((RESULTS / "kaggle_run.json").read_text())
    assert set(run) == {"kernel", "status", "failure_message", "notebook", "accelerator", "files"}
    assert run["status"].endswith("COMPLETE") and "/" not in run["kernel"]  # no account name


def test_the_registered_verdicts_reproduce_from_the_raw_files(analysis):
    for rule in RULES:
        assert analysis[rule]["decided"] == COMMITTED[rule]["decided"], rule
        assert analysis[rule]["holds"] == COMMITTED[rule]["holds"], rule
    assert {r: analysis[r]["holds"] for r in RULES} == EXPECTED_VERDICTS
    assert analysis["predicted_outcomes"] == dict.fromkeys(RULES, "holds")


def test_the_command_with_its_repeats_worked_three_times(analysis):
    for seed in ("0", "1", "2"):
        cmd = analysis["commands"][seed]
        assert cmd["holds"] and all(cmd["conditions"].values())
        assert len(cmd["replicates"]) == 3
        assert 355 < cmd["wall_clock_s"] < 362
    # The first run differs from the two after it by more than their repeats differ among
    # themselves: decode cost per context token, ms per 1,000.
    per_1k = [analysis["commands"][s]["fit"]["decode_per_ctx_token_s"] * 1e6 for s in "012"]
    assert [round(x, 2) for x in per_1k] == [3.83, 3.76, 3.76]


def test_the_shared_prefix_cells_the_write_up_quotes(analysis):
    cells = {c["b"]: c for c in analysis["decode"]["cells"]}
    assert analysis["decode"]["fitted_max_summed_context"] == 20_200
    assert {b: round(c["measured_step_s"] * 1e3, 1) for b, c in cells.items()} == {
        8: 81.3, 12: 102.7, 16: 122.2, 32: 207.0, 64: 342.3, 96: 494.1, 128: 634.1}
    # Profile A against the measurement: +3% up to 2.3× the fitted range, +12 to +15% beyond.
    assert {b: round((c["predicted_over_measured"] - 1) * 100) for b, c in cells.items()} == {
        8: 2, 12: 2, 16: 3, 32: 3, 64: 12, 96: 13, 128: 15}
    assert analysis["d1b"]["cells_tested"] == [16, 32, 64, 96, 128]
    worst = max(cells[b]["factor"] for b in analysis["d1b"]["cells_tested"])
    assert round(worst, 2) == 1.15 and worst < analysis["d1b"]["max_factor"] == 1.25
    assert round(cells[128]["summed_context"] / 20_200, 1) == 9.2
    # Sharing a prefix costs nothing by itself: a little LESS than unique prompts.
    controls = {c["b"]: c for c in analysis["decode"]["controls"]}
    assert [round((controls[b]["shared_over_unique"] - 1) * 100) for b in (8, 12)] == [-3, -4]
    assert all(c["factor"] < 1.05 for c in controls.values())


def test_the_stall_cells_the_write_up_quotes(analysis):
    cells = {c["b"]: c for c in analysis["d2"]["cells"]}
    assert [round(cells[b]["measured_extra_s"] * 1e3) for b in (8, 32, 64)] == [179, 154, 172]
    assert all(round(c["predicted_extra_s"] * 1e3) == 184 for c in cells.values())
    assert [round(cells[b]["factor"], 2) for b in (8, 32, 64)] == [1.03, 1.19, 1.07]


def test_the_traffic_numbers_the_write_up_quotes(analysis):
    runs = {r["workload"]: r for r in analysis["traffic"]["runs"]}
    quoted = {"long_below": (82.6, 82.9, 1.00), "short_at": (104.2, 104.3, 1.00),
              "long_above": (279.7, 298.5, 1.07)}
    for name, (measured, registered, factor) in quoted.items():
        r = runs[name]
        assert round(r["measured"]["tpot_p95_ms"], 1) == measured
        assert round(r["registered"]["tpot_p95_ms"], 1) == registered
        assert round(r["factor"], 2) == factor
        # The engine's own histogram and the client-side arithmetic agree exactly.
        assert r["measured"]["engine_share_within_limit"] == pytest.approx(
            r["measured"]["client_share_within_limit"])
        assert r["verdict_agrees"] is True
    assert [round(runs[n]["measured"]["engine_share_within_limit"], 3)
            for n in quoted] == [1.0, 0.906, 0.477]
    assert analysis["t3b"]["workloads_tested"] == ["long_below", "long_above"]
    # Reported: the first-token model is the less accurate half here, and always too fast.
    ttft = [runs[n]["ttft_factor"]["p95"] for n in quoted]
    assert [round(x, 2) for x in ttft] == [1.27, 1.23, 1.30]
    assert all(runs[n]["measured"]["ttft_p95_ms"] > runs[n]["registered"]["ttft_p95_ms"]
               for n in quoted)
    # Reported: the command's own fits are a little worse than A where the batches are largest.
    fresh = [runs["long_above"]["fresh_fits"][f"s{s}"]["factor"] for s in "012"]
    assert [round(x, 2) for x in fresh] == [1.14, 1.16, 1.16]


# ── the two rules decided in the simulator ──────────────────────────────────────────────────────


def test_the_grids_are_the_registered_design_from_a_clean_tree():
    assert GRIDS["git_dirty"] is False and GRIDS["n_runs"] == 22_032
    # Run at a later commit than the registration (c5b5d74): documentation only in between —
    # `git diff c5b5d74 7f95146 -- src scripts configs` is empty.
    assert GRIDS["git_sha"] == "7f95146"
    runs = [f"s{s}" for s in "012"]
    assert GRIDS["fit_ids"] == sorted(["A", *runs, *(f"{r}r{i}" for r in runs for i in range(3))])
    firm, p64 = GRIDS["grids"]["firmness"], GRIDS["grids"]["prompts64"]
    assert firm["axes"] == {"prompts": [1, 64], "loads": [0.5, 1, 2],
                            "replicas": list(range(1, 25)), "seeds": [0, 1, 2],
                            "policies": [STICKY, RR, BAL16]}
    assert sorted(firm["fits"]) == GRIDS["fit_ids"]
    assert p64["axes"]["prompts"] == [64] and p64["axes"]["replicas"] == list(range(1, 33))
    assert p64["axes"]["policies"] == [STICKY, BAL16, BAL1] and sorted(p64["fits"]) == runs


@pytest.mark.parametrize(("name", "fit", "k", "load", "policy", "alpha", "n", "seed"), [
    ("firmness", "s1", 64, 1, STICKY, None, 8, 1),
    ("firmness", "s0r2", 1, 2, "radix-affinity-balanced", 16.0, 9, 0),
    ("prompts64", "s2", 64, 3, "radix-affinity-balanced", 1.0, 20, 2),
])
def test_a_stored_cell_is_what_the_simulator_gives(name, fit, k, load, policy, alpha, n, seed):
    """Ties the grids to the code: one cell of each kind, run again here."""
    cfg = PREREG["config"]
    grid = c4g.grid_design(cfg, cfg["grids"][name])
    profile = c4g.fits(cfg, RESULTS)[fit]
    stored = GRIDS["grids"][name]["fits"][fit][str(k)][str(load)][
        c4g.worker.label(policy, alpha)]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "profile.json"
        path.write_text(json.dumps(dataclasses.asdict(profile)))
        c4g.worker._init(grid, str(path))
        _, ttft, tpot, _ = c4g.worker._job((k, load, policy, alpha, n, seed))
    i, j = grid["replicas"].index(n), grid["seeds"].index(seed)
    assert stored["ttft_p95_ms"][i][j] == ttft and stored["tpot_p95_ms"][i][j] == tpot


def test_one_runs_spread_does_not_cover_another_runs_count(analysis):
    """E, predicted to hold: it does not. 82.4% against the 90% needed."""
    e = analysis["e"]
    assert e["runs"] == {"s0": 4, "s1": 4, "s2": 4}  # each run: its pooled fit and three repeats
    assert (e["covered"], e["cases"]) == (178, 216)
    assert round(e["coverage"], 3) == 0.824 < e["min_coverage"] == 0.9
    assert round(e["coverage_within_one"], 3) == 0.949
    assert round(e["profile_a_coverage"], 3) == 0.796
    misses = e["misses"]
    assert len(misses) == 38
    far = [m for m in misses if not m["range"][0] - 1 <= m["count"] <= m["range"][1] + 1]
    assert len(far) == 11
    # The two runs taken back to back agree with each other; the run an hour earlier does not
    # agree with them — which no run's own repeats can know.
    by_pair = {}
    for m in misses:
        by_pair[(m["range_from"], m["other"])] = by_pair.get((m["range_from"], m["other"]), 0) + 1
    assert by_pair == {("s0", "s1"): 7, ("s0", "s2"): 9, ("s1", "s0"): 10, ("s2", "s0"): 10,
                       ("s1", "s2"): 2}
    late = 72 - by_pair[("s1", "s2")] - by_pair.get(("s2", "s1"), 0)
    assert round(late / 72, 3) == 0.972
    early = 144 - sum(v for k, v in by_pair.items() if "s0" in k)
    assert round(early / 144, 3) == 0.75


def test_the_64_prompt_question_on_the_fresh_calibrations(analysis):
    """F1 holds: the first-token hint repeats. F2, predicted to hold, does not: under both halves
    balanced affinity needs one replica fewer than session-sticky at more than two loads, on two
    of the three fits."""
    f1 = analysis["f1"]["by_fit"]
    assert [(f1[f]["sticky"], f1[f]["weight_1"]) for f in ("s0", "s1", "s2")] == [
        (9, 8), (10, 7), (10, 8)]
    f2 = analysis["f2"]["by_fit"]
    assert all(f2[f]["weight_1_holds_nothing"] for f in f2)  # part (i) holds on all three
    assert {f: f2[f]["loads_balanced_wins"] for f in f2} == {
        "s0": ["4"], "s1": ["0.5", "1", "4"], "s2": ["1", "2", "3", "4"]}
    assert {f: f2[f]["loads_sticky_wins"] for f in f2} == {"s0": [], "s1": ["2"], "s2": []}
    assert [f2[f]["holds"] for f in ("s0", "s1", "s2")] == [True, False, False]
    two = analysis["prompts64"]["tables"]["two_part"]
    assert [two[f]["2"][STICKY] for f in ("s0", "s1", "s2")] == [13, 14, 15]
    assert [two[f]["2"][BAL16] for f in ("s0", "s1", "s2")] == [13, 15, 14]
    # Every difference between the two under both halves is exactly one replica.
    gaps = {abs(two[f][ld][STICKY] - two[f][ld][BAL16]) for f in two for ld in two[f]}
    assert gaps == {0, 1}
    first = analysis["prompts64"]["tables"]["first_token"]
    assert [first[f]["3"][STICKY] for f in ("s0", "s1", "s2")] == [17, 18, 19]
    assert [first[f]["3"][BAL16] for f in ("s0", "s1", "s2")] == [16, 16, 17]


def test_the_post_hoc_table_reproduces_and_starts_from_the_registered_numbers(analysis):
    """Post hoc, and labelled so: each fresh fit against the micro-benchmark cells — a reported
    item the registration listed and the registered analysis computed for profile A only."""
    cfg = PREREG["config"]
    fits = {"A": load_profile(REPO / cfg["profile"])}
    for seed in cfg["calibrate"]["seeds"]:
        fits[f"s{seed}"] = load_profile(RESULTS / f"calibrated_profile_seed{seed}.json")
    out = posthoc.compute(analysis, PREREG, fits)  # stops if A does not give its registration
    committed = json.loads((RESULTS / "posthoc_fresh_cells.json").read_text())
    assert committed["post_hoc"] is True
    assert json.loads(json.dumps(out)) == committed
    at_128 = out["decode_shared"][-1]
    assert at_128["b"] == 128
    assert {n: round((v["over_measured"] - 1) * 100) for n, v in at_128["fits"].items()} == {
        "A": 15, "s0": 20, "s1": 18, "s2": 18}
    assert [round(out["decode_per_1k_context_ms"][n], 2) for n in ("A", "s0", "s1", "s2")] == [
        3.5, 3.83, 3.76, 3.76]
