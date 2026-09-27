"""Phase 6's pre-registration must stay reproducible from the committed code — and the notebook must
carry exactly what was committed.

If a model change moves a Phase 6 prediction, these fail. A pre-registered number may only change by
a deliberate, dated re-registration (re-run scripts/phase6_prepare.py, commit, and say why in
docs/09-hypotheses.md) — never silently, and never after the GPU run has been seen.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import yaml

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = json.loads((REPO / "artifacts/results/phase6_prereg/predictions.json").read_text())
NOTEBOOK = json.loads((REPO / "notebooks/phase6_sim2real.ipynb").read_text())
CONFIG = yaml.safe_load((REPO / "configs/phase6_sim2real.yaml").read_text())


def _canonical(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def test_the_notebook_carries_exactly_the_committed_predictions_and_fingerprints():
    src = "".join("".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code")
    sha = re.search(r'PREDICTIONS_SHA256 = "([0-9a-f]{64})"', src).group(1)
    assert sha == PREREG["predictions_sha256"]
    assert hashlib.sha256(_canonical(PREREG["predictions"])).hexdigest() == sha
    for fp in PREREG["trace_fingerprints_sha256"].values():
        assert fp in src


def test_committed_predictions_reproduce_from_the_committed_code():
    profile = load_profile(REPO / CONFIG["profile"])
    for name, spec in CONFIG["workloads"].items():
        trace, _ = AgenticWorkload(**spec).generate()
        for policy in CONFIG["policies"]:
            r = run(trace, policy=policy, n_replicas=CONFIG["replicas"],
                    seed=CONFIG["router_seed"], profile=profile, cache_model="vllm")
            want = PREREG["predictions"]["runs"][name][policy]
            assert round(r.hit_rate, 6) == want["hit_rate"], (name, policy)
            assert r.per_replica_requests == want["requests_per_replica"], (name, policy)
            assert r.preemptions == want["preemptions"], (name, policy)


def test_the_design_can_tell_the_policies_apart():
    """S2 checks only pairs predicted at least s2_min_gap apart; the headline pair must be one."""
    k64 = PREREG["predictions"]["runs"]["K64"]
    gap = k64["radix-affinity"]["hit_rate"] - k64["session-sticky"]["hit_rate"]
    assert gap >= CONFIG["s2_min_gap"]
