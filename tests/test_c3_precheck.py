"""Pins the post hoc pre-check (scripts/c3_precheck.py) on C2's recorded data.

Fitted on cold prompts only, the model misses a held-out p95 by 5.7 to 7.2x; with prefill also
timed after a cached prefix, it stays within 2x. This result is why `kvsim calibrate` measures
both kinds of prefill cell.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_pre_check_that_changed_the_commands_design_reproduces():
    spec = importlib.util.spec_from_file_location("c3_precheck",
                                                  REPO / "scripts" / "c3_precheck.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    committed = json.loads((REPO / "artifacts/results/c3_prereg/precheck.json").read_text())
    assert committed["worst_factor"]["cold_only"] > 2.0 > committed["worst_factor"][
        "with_cached_prefix"]
    data = json.loads((pre.RESULTS / "c2.json").read_text())
    prereg = json.loads(pre.PREREG.read_text())
    for want in (committed["rows"][0], committed["rows"][2]):  # one row of each kind, for speed
        got = pre.one(data, prereg, want["prefill_cells"],
                      want["prefill_cells"] == "with_cached_prefix", want["decode_rep"])
        assert got["fit"] == pytest.approx(want["fit"])
        for name, run_ in want["holdout"].items():
            assert got["holdout"][name]["factor"] == pytest.approx(run_["factor"])
