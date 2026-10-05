"""C3's committed result must come out of the committed raw files — and so must every number the
README quotes from it.

The run is not repeatable (it was a GPU session on Kaggle), but everything after it is: the
registered analysis on the raw files, and the post-hoc comparison of three calibrations. If either
stops reproducing, a quoted number has lost its source.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results" / "c3_kaggle_t4"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c3 = _load("c3_analyze")
posthoc = _load("c3_posthoc_grid")
COMMITTED = json.loads((RESULTS / "analysis.json").read_text())


@pytest.fixture(scope="module")
def analysis():
    data = json.loads((RESULTS / "c3.json").read_text())
    prereg = json.loads((REPO / "artifacts/results/c3_prereg/predictions.json").read_text())
    return c3.analyze(data, prereg, RESULTS)  # refuses a run that is not the registered one


def test_the_registered_verdicts_reproduce_from_the_raw_files(analysis):
    for rule in ("c3a", "c3b", "c3c"):
        assert analysis[rule]["decided"] and analysis[rule]["holds"], rule
        assert analysis[rule] == COMMITTED[rule]
    assert all(analysis["commands"]["0"]["conditions"].values())
    for got, want in zip(analysis["holdout"], COMMITTED["holdout"], strict=True):
        assert got["workload"] == want["workload"]
        assert got["command_factor"] == pytest.approx(want["command_factor"])
        assert got["c2_factor"] == pytest.approx(want["c2_factor"])


def test_the_factors_the_readme_quotes(analysis):
    rows = {r["workload"]: r for r in analysis["holdout"]}
    assert set(rows) == {"K1_half", "K64_quarter", "T6_K8"}
    worst_b = max(v for r in rows.values() for v in r["command_factor"].values())
    worst_c = max(v for r in rows.values() for v in r["c2_factor"].values())
    assert worst_b == pytest.approx(1.41, abs=0.005)  # C3b: "within 1.41×"
    assert worst_c == pytest.approx(1.38, abs=0.005)  # C3c: "within 1.38×"
    # Reported, and uncomfortable: the command's second run, on the workload at the knee.
    second = {w: r["command"]["1"]["factor"] for w, r in rows.items()}
    assert second["K1_half"]["p95"] == pytest.approx(4.17, abs=0.01)
    others = [v for w, f in second.items() for q, v in f.items() if (w, q) != ("K1_half", "p95")]
    assert max(others) < 1.36  # "the other five numbers stayed within 1.35×"
    # Every hit rate equals its registered prediction to four decimals.
    for r in rows.values():
        assert round(r["measured"]["hit_rate"], 4) == round(r["c2_registered"]["hit_rate"], 4)


def test_the_reported_grid_comparison_is_what_the_two_tables_say():
    """`c3_analyze.py --grid` took minutes of CPU; its counts must follow from the two committed
    tables — C2's profile, and the profile the command wrote."""
    a = json.loads(posthoc.TABLES["A"][1].read_text())
    b = json.loads(posthoc.TABLES["B"][1].read_text())
    pairs = [(r, b["required_replicas"][k][ld][p])
             for k, by_load in a["required_replicas"].items()
             for ld, by_pol in by_load.items() for p, r in by_pol.items()]
    g = COMMITTED["grid"]
    assert (len(pairs), g["cells"]) == (144, 144)
    assert sum(1 for x, y in pairs if x == y) == g["equal"] == 122
    assert sum(1 for x, y in pairs if abs(x - y) <= 1) == g["within_one"] == 141
    # The registered test was falsified on C2's profile; the same rules hold on the command's.
    assert (a["verdict"]["holds"], b["verdict"]["holds"]) == (False, True)
    assert (g["verdict_committed"], g["verdict_command"]) == (False, True)


def test_what_survives_three_calibrations_and_what_does_not():
    """POST HOC (scripts/c3_posthoc_grid.py). The README's headline claims only what all three
    calibrations support; these are those claims, and the one that did not survive."""
    tables = {t: json.loads(path.read_text()) for t, (_, path) in posthoc.TABLES.items()}
    a = posthoc.analyze(tables)
    assert a == json.loads((RESULTS / "posthoc_three_calibrations.json").read_text())
    assert (a["cells"], a["cells_identical_on_all"], a["cells_within_one"]) == (144, 102, 138)
    # Does not survive: H2b's verdict.
    assert [a["h2b_verdict"][t]["holds"] for t in "ABC"] == [False, True, True]
    # Survives: sticky best or tied at every load with 1 prompt and with 4 …
    by_k = a["sticky_best_or_tied_on_all_by_K"]
    assert (by_k["1"], by_k["4"]) == ({"points": 6, "of": 6}, {"points": 6, "of": 6})
    assert a["sticky_best_or_tied"] == {"A": 22, "B": 20, "C": 20}
    assert a["sticky_best_or_tied_on_all"] == 19
    # … and never more replicas than round-robin or least-loaded, anywhere, on any calibration.
    never = a["sticky_never_needs_more_than"]
    assert never["round-robin"] == never["least-loaded"] == a["points"] == 24
    # The README's two table rows at 2 sessions/s, on A · B · C.
    row = a["required_replicas"]
    assert row["1@2"]["session-sticky"] == [5, 5, 5] and row["1@2"]["round-robin"] == [7, 7, 7]
    assert row["1@2"]["radix-affinity-balanced@16"] == [10, 17, 17]  # 17 = not held on 16
    assert row["64@2"]["session-sticky"] == [10, 12, 12]
    assert row["64@2"]["radix-affinity-balanced@16"] == [11, 11, 10]
    assert row["64@2"]["radix-affinity-balanced@1"] == [8, 8, 8]  # post hoc weight: not a finding
