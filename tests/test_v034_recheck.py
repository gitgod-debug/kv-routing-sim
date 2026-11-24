"""Checks on V1's registered files: Phase 3's test with another engine version, nothing else.

Read-only, and on made-up notebooks. Nothing here says anything about V1's result.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v034 = _load("v034_prepare")


def _notebook(*cells: str) -> str:
    return json.dumps({"cells": [{"cell_type": "code", "source": c} for c in cells]})


def test_the_registered_predictions_are_phase_3s_with_another_engine_version():
    phase3 = json.loads((RESULTS / "phase3_prereg" / "predictions.json").read_text())
    v1 = json.loads((RESULTS / "v034_prereg" / "predictions.json").read_text())
    wanted = str(yaml.safe_load((REPO / "configs" / "v034_recheck.yaml").read_text())
                 ["engine"]["vllm"])
    assert v1["trace_fingerprint_sha256"] == phase3["trace_fingerprint_sha256"]
    assert v1["predictions"]["workloads"] == phase3["predictions"]["workloads"]
    assert v1["analysis_procedure"] == phase3["analysis_procedure"]
    assert v1["decision_rules"] == phase3["decision_rules"]
    differing = [k for k in v1["predictions"] if v1["predictions"][k] != phase3["predictions"][k]]
    assert differing == ["engine"]
    engine, was = v1["predictions"]["engine"], phase3["predictions"]["engine"]
    assert {k for k in engine if engine[k] != was[k]} == {"vllm"}
    assert engine["vllm"] == wanted != was["vllm"]


def test_the_notebook_pins_the_registered_version_and_carries_the_registered_hashes():
    v1 = json.loads((RESULTS / "v034_prereg" / "predictions.json").read_text())
    cells = json.loads((REPO / "notebooks" / "v034_recheck_t4.ipynb").read_text())["cells"]
    text = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                     for c in cells)
    version = v1["predictions"]["engine"]["vllm"]
    assert f'VLLM_PINNED = "{version}"' in text
    assert f"!pip install -q vllm=={version}" in text
    assert f'TRACE_SHA256 = "{v1["trace_fingerprint_sha256"]}"' in text
    assert f'PREDICTIONS_SHA256 = "{v1["predictions_sha256"]}"' in text


def test_two_notebooks_that_differ_only_in_version_data_and_hash_count_as_the_same():
    data_a = '    "' + "A" * 60 + '"\n    "' + "B" * 60 + '"\n'
    data_b = '    "' + "C" * 60 + '"\n'
    old = _notebook("!pip install -q vllm==0.30.0\n", f"BLOB = (\n{data_a})\nSHA = 'aaaa'\n")
    new = _notebook("!pip install -q vllm==0.34.0\n", f"BLOB = (\n{data_b})\nSHA = 'bbbb'\n")
    assert v034.same_but_for_the_version(new, old, ("bbbb", "aaaa"), ("0.30.0", "0.34.0"))


def test_any_other_difference_between_the_notebooks_is_seen():
    old = _notebook("!pip install -q vllm==0.30.0\n", "NUM_GPU_BLOCKS = 512\n")
    for changed in ("NUM_GPU_BLOCKS = 511\n", "NUM_GPU_BLOCKS = 512\nprint('more')\n"):
        new = _notebook("!pip install -q vllm==0.34.0\n", changed)
        assert not v034.same_but_for_the_version(new, old, ("x", "y"), ("0.30.0", "0.34.0"))
    one_cell_more = _notebook("!pip install -q vllm==0.34.0\n", "NUM_GPU_BLOCKS = 512\n", "pass\n")
    assert not v034.same_but_for_the_version(one_cell_more, old, ("x", "y"),
                                             ("0.30.0", "0.34.0"))
