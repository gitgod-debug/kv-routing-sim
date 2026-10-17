"""Pins T1's registered result (configs/tpot_slo.yaml) to the committed raw files.

T1 does not hold: the simulator's mean inter-token latency is within 1.07x of the engine's on three
held-out runs and 1.5 to 1.7x too slow on the two at the knee.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "results" / "t1_itl"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t1 = _load("t1_itl_check")
COMMITTED = json.loads((OUT / "t1.json").read_text())


@pytest.fixture(scope="module")
def analysis():
    return t1.analyze(yaml.safe_load((REPO / "configs" / "tpot_slo.yaml").read_text()))


def test_the_registered_verdict_reproduces_and_it_is_a_failure(analysis):
    assert analysis["decided"] and not analysis["holds"]
    assert COMMITTED["holds"] is False and COMMITTED["predicted_outcome"] == "holds"
    assert analysis["worst_factor"] == pytest.approx(COMMITTED["worst_factor"])
    assert analysis["worst_factor"] == pytest.approx(1.71, abs=0.005)
    for got, want in zip(analysis["runs"], COMMITTED["runs"], strict=True):
        assert (got["session"], got["workload"]) == (want["session"], want["workload"])
        for prof in ("A", "B", "C"):
            assert (prof in got) == (prof in want)
            if prof in got:
                for key in ("factor", "predicted_itl_ms", "measured_itl_ms"):
                    assert got[prof][key] == pytest.approx(want[prof][key])


def test_the_two_groups_the_write_up_describes(analysis):
    """Within 1.07x off the knee and 1.5 to 1.7x too slow at it, on every profile and session."""
    rows = {(r["session"], r["workload"]): r for r in analysis["runs"]}
    assert len(rows) == 5
    knee = [r for (_, w), r in rows.items() if w == "K1_half"]
    rest = [r for (_, w), r in rows.items() if w != "K1_half"]
    assert len(knee) == 2 and len(rest) == 3
    for r in rest:
        for prof in ("A", "B", "C"):
            if prof in r:
                assert r[prof]["factor"] < 1.085, (r["workload"], prof)
    assert max(r["A"]["factor"] for r in rest) == pytest.approx(1.07, abs=0.005)
    for r in knee:
        assert 1.5 < r["A"]["factor"] < 1.72
        assert r["A"]["predicted_itl_ms"] > r["A"]["measured_itl_ms"]  # too slow, not too fast
        # The model has the engine preempting there; the engine never did.
        assert r["A"]["preemptions"] == {"measured": 0, "predicted": 6}
    c3_knee = rows[("c3", "K1_half")]
    assert c3_knee["B"]["factor"] == pytest.approx(1.51, abs=0.005)
    assert c3_knee["C"]["factor"] == pytest.approx(1.71, abs=0.005)
    # Every one of the five engine measurements is itself beyond a 100 ms per-token limit.
    assert all(r["A"]["measured_itl_ms"] > 100 for r in rows.values())
    assert sorted(round(r["A"]["measured_itl_ms"]) for r in rows.values()) == [134, 161, 174,
                                                                              470, 521]


def test_the_post_hoc_range_numbers_reproduce():
    """Post hoc (scripts/t1_posthoc_range.py): at the knee the model decodes 97% of its tokens in
    batches whose summed context is far beyond anything a calibration cell measured."""
    post = _load("t1_posthoc_range")
    fitted = post.fitted_range()
    want = json.loads((OUT / "posthoc_range.json").read_text())
    assert fitted == want["fitted"]
    assert fitted["A"]["max_summed_context"] == 20_200  # C2: 10 sequences of 1,984 + 36 tokens
    assert fitted["B"]["max_summed_context"] == fitted["C"]["max_summed_context"] == 16_960
    assert want["kv_cache_tokens"] == 24_592
    by = {(r["session"], r["workload"]): r for r in want["runs"]}
    for session in ("c2", "c3"):
        knee = by[(session, "K1_half")]
        assert knee["share_of_tokens_beyond_fitted_range"] == pytest.approx(0.97, abs=0.005)
        assert knee["summed_context_median"] > 7 * fitted["A"]["max_summed_context"]
    assert by[("c3", "T6_K8")]["summed_context_median"] < fitted["A"]["max_summed_context"]
    assert by[("c3", "K64_quarter")]["summed_context_median"] < fitted["A"]["max_summed_context"]
