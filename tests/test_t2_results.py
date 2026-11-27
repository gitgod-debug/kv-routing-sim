"""Pins T2's registered result (configs/tpot_slo.yaml): the result must come out of the committed
grids, and the grids out of the code.

T2 reruns the replica grid under a two-part SLO on three calibrations of one T4; all three rules
hold. Because T1 failed (the per-token model is 1.5 to 1.7x too slow at the knee), these counts are
the model's answer, not validated guidance.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results" / "t2_two_part_slo"
CFG = yaml.safe_load((REPO / "configs" / "tpot_slo.yaml").read_text())
STICKY, RR, LL = "session-sticky", "round-robin", "least-loaded"
BAL16, BAL1 = "radix-affinity-balanced@16", "radix-affinity-balanced@1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t2 = _load("t2_analyze")
GRIDS = {n: json.loads((RESULTS / f"grid_{n}.json").read_text())
         for n in CFG["t2"]["calibrations"]}
COMMITTED = json.loads((RESULTS / "analysis.json").read_text())


@pytest.fixture(scope="module")
def analysis():
    tables = {n: json.loads((REPO / c["first_token_table"]).read_text())
              for n, c in CFG["t2"]["calibrations"].items()}
    return t2.analyze(CFG, GRIDS, tables)  # stops if a first-token table does not reproduce


def test_the_grids_are_the_registered_design_from_committed_code():
    for name, g in GRIDS.items():
        assert g["calibration"] == name and g["n_runs"] == 13_824
        assert g["git_sha"] == "ce16bee"  # derived again with the fixed engine of release 0.2.0
        assert g["axes"]["replicas"] == list(range(1, 33)) and g["axes"]["seeds"] == [0, 1, 2]
        assert g["axes"]["prompts"] == [1, 4, 16, 64]
        assert g["profile_path"] == CFG["t2"]["calibrations"][name]["profile"]


def test_the_grids_come_from_a_clean_tree_and_their_cells_were_run_again():
    """All three grids were made at a clean tree. (Release 0.1.2's grids of B and C said dirty,
    because files no grid run imports were edited while they ran.) 240 cells of each, rerun from a
    clean checkout (scripts/t2_verify_grid.py), are identical."""
    assert all(g["git_dirty"] is False for g in GRIDS.values())
    v = json.loads((RESULTS / "verification.json").read_text())
    assert v["git_dirty"] is False and v["all_identical"] is True
    for name in "ABC":
        c = v["calibrations"][name]
        assert (c["cells_run_again"], c["identical"], c["different"]) == (240, 240, [])
        assert c["src_identical_to_grid_commit"] is True and c["runs_in_grid"] == 13_824


@pytest.mark.parametrize(("name", "k", "load", "policy", "n", "seed"), [
    ("A", 1, 2, STICKY, 10, 0), ("B", 64, 1, BAL16, 8, 2), ("C", 16, 0.5, RR, 6, 1),
    ("A", 4, 4, LL, 25, 1), ("B", 1, 0.25, BAL1, 3, 0), ("C", 64, 3, STICKY, 21, 2),
])
def test_a_stored_cell_is_what_the_simulator_gives(name, k, load, policy, n, seed):
    """Six cells, two per calibration, rerun from the committed code, match the stored values."""
    grid = GRIDS[name]
    trace, _ = AgenticWorkload(**grid["trace"], n_shared_prefixes=k,
                               arrival_rate=float(load)).generate()
    base, _, alpha = policy.partition("@")
    r = run(trace, policy=base, n_replicas=n, seed=seed, cache_model=grid["cache_model"],
            profile=load_profile(REPO / grid["profile_path"]),
            policy_kwargs={"load_penalty": float(alpha)} if alpha else None)
    cell = grid["cells"][str(k)][str(load)][policy]
    i, j = grid["axes"]["replicas"].index(n), grid["axes"]["seeds"].index(seed)
    assert cell["ttft_p95_ms"][i][j] == r.ttft_percentiles()["p95_ms"]
    assert cell["tpot_p95_ms"][i][j] == r.tpot_percentiles()["p95_ms"]


def test_the_registered_verdicts_reproduce(analysis):
    for rule in ("t2a", "t2b", "t2c"):
        assert analysis[rule]["decided"] and analysis[rule]["holds"], rule
        assert COMMITTED[rule]["holds"] is True and COMMITTED["predicted_outcomes"][rule] == "holds"
    assert analysis["required"] == COMMITTED["required"]
    a, b, c = (analysis[r]["by_calibration"] for r in ("t2a", "t2b", "t2c"))
    for name in "ABC":
        assert (a[name]["sticky_best_or_tied"], a[name]["points"], a[name]["none_holds"]) == (
            12, 12, 0)
        assert b[name]["worse"] == [] and b[name]["points"] == 24
        assert c[name]["points"] == 24
    assert [round(c[n]["median_ratio"], 2) for n in "ABC"] == [2.0, 2.0, 2.0]


def test_the_numbers_the_write_up_quotes(analysis):
    first, both = analysis["required"]["first_token"], analysis["required"]["two_part"]

    def row(tab, k, load, policy):
        return [tab[n][str(k)][str(load)][policy] for n in "ABC"]

    # One prompt, 2 sessions/s: 5 T4s for the first token, 10 or 11 for both halves.
    assert row(first, 1, 2, STICKY) == [5, 5, 5] and row(both, 1, 2, STICKY) == [10, 11, 11]
    assert row(both, 1, 2, RR) == [14, 15, 15] and row(both, 1, 2, LL) == [13, 14, 14]
    # 64 prompts, 2 sessions/s: sticky and balanced affinity tie on all three calibrations,
    assert row(both, 64, 2, STICKY) == [13, 14, 14] and row(both, 64, 2, BAL16) == [13, 14, 14]
    # and load weight 1, the first-token hint (7 or 8 against 10 to 12), holds nothing here.
    assert row(first, 64, 2, BAL1) == [8, 7, 8] and row(both, 64, 2, BAL1) == [33, 33, 33]
    assert analysis["binding"] == {"cells": 432, "per_token_half_adds_replicas": 307,
                                   "same_count_under_both": 23, "held_under_neither": 102}
    two, one = analysis["agreement"]["two_part"], analysis["agreement"]["first_token"]
    assert (two["identical"], two["within_one"], two["cells"]) == (81, 123, 144)
    assert (one["identical"], one["within_one"]) == (95, 122)  # on replicas 1 to 32
    assert two["sticky_best_or_tied"] == {"A": 19, "B": 22, "C": 22}
    assert two["sticky_best_or_tied_on_all"] == 18
    # H2b's rules, exploratory (not registered here): under the two-part target they hold on all
    # three calibrations. With the engine of release 0.1.2 they failed on C.
    assert [analysis["h2b_exploratory"]["two_part"][n]["holds"] for n in "ABC"] == [
        True, True, True]
    # A looser per-token limit shrinks the factor.
    lim = analysis["other_limits"]
    assert [round(lim["150"]["t2c_median_ratio"][n], 2) for n in "ABC"] == [1.5, 1.5, 1.5]
    assert [round(lim["200"]["t2c_median_ratio"][n], 2) for n in "ABC"] == [1.37, 1.37, 1.37]
