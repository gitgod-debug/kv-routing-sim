"""The Phase 3 pre-registration must stay reproducible from the committed code — and the notebook
must carry exactly what was committed.

If a model change moves any prediction, these fail. That is the point: a pre-registered number may
only change by a deliberate, dated re-registration (re-run scripts/phase3_prepare.py, commit, and
say why in docs/09-hypotheses.md) — never silently, and never after the GPU run has been seen.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import phase3_prepare as prep

PREREG = json.loads(prep.PREREG.read_text())


def test_committed_predictions_reproduce_from_the_committed_code():
    _, _, fp, predictions = prep.make_predictions(PREREG["config"])
    assert fp == PREREG["trace_fingerprint_sha256"]
    assert predictions == PREREG["predictions"]
    assert prep.sha256(predictions) == PREREG["predictions_sha256"]


def test_the_config_file_is_the_one_that_was_registered():
    import yaml

    assert yaml.safe_load(prep.CONFIG.read_text()) == PREREG["config"]


def test_the_notebook_carries_exactly_the_committed_trace_and_predictions():
    nb = json.loads(prep.NOTEBOOK.read_text())
    cell = next(
        "".join(c["source"]) for c in nb["cells"] if "BLOB = (" in "".join(c["source"])
    )
    trace_sha = re.search(r'TRACE_SHA256 = "([0-9a-f]{64})"', cell).group(1)
    pred_sha = re.search(r'PREDICTIONS_SHA256 = "([0-9a-f]{64})"', cell).group(1)
    blob = "".join(re.findall(r'^    "([A-Za-z0-9+/=]+)"$', cell, re.MULTILINE))
    data = json.loads(lzma.decompress(base64.b64decode(blob)))

    assert trace_sha == PREREG["trace_fingerprint_sha256"]
    assert pred_sha == PREREG["predictions_sha256"]
    assert prep.sha256(data["predictions"]) == pred_sha
    sent = {
        "trace": [[r["request_id"], r["arrival_ts"], r["session_id"], r["ids"]]
                  for r in data["trace"]],
        "decode_sweep_prompts": data["decode_sweep_prompts"],
    }
    assert hashlib.sha256(prep.canonical(sent)).hexdigest() == trace_sha


def test_every_id_the_notebook_sends_is_inside_a_real_vocabulary():
    """The generator's raw ids were once far outside any vocabulary; a live engine rejects them."""
    trace, prompts, _, _ = prep.make_predictions(PREREG["config"])
    ids = [t for _, seq in trace for t in seq] + [t for p in prompts for t in p]
    assert 1000 <= min(ids) and max(ids) < 100_000


def test_the_design_discriminates_and_has_a_control():
    """W1 must separate the two models; W2 must not — otherwise the experiment cannot tell which is
    right, or cannot tell a model error from an accounting error."""
    w = PREREG["predictions"]["workloads"]
    w1, w2 = w["W1_long"]["predictions"], w["W2_short"]["predictions"]
    assert abs(w1["vllm"]["hit_rate"] - w1["lru"]["hit_rate"]) > 0.05
    assert abs(w2["vllm"]["hit_rate"] - w2["lru"]["hit_rate"]) < 0.005
    # W2 can also tell the source-reading final-token rule from its off-by-one alternative.
    assert w2["alternative_distinguishable"]
    claims = w["W3_concurrent"]["predictions"]["timing_independent_claims"]
    assert claims["lru"]["preemptions"] == 0 and claims["vllm"]["preemptions"] == "> 0"
