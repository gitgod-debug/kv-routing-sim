"""Golden-file test: a small sweep, end to end, against numbers committed in tests/golden/.

Every other test checks one piece. This one pins the whole chain — the generator and its
hashing, the vLLM block-pool model, four policies and their tie-breaking, the event loop, the
calibrated T4 profile, the cost model and the recommendation — to numbers a reviewer can read. A
change anywhere in that chain that moves a result fails here, on purpose: either it is a bug, or
it is deliberate and the file is regenerated in the same commit, where the diff shows exactly
which numbers moved.

    KVSIM_UPDATE_GOLDEN=1 pytest tests/test_golden.py      # regenerate; then read the diff
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.recommend import recommend
from kv_routing_sim.eval.sweep import sweep
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = pathlib.Path(__file__).parent / "golden" / "small_sweep.json"
POLICIES = ("round-robin", "least-loaded", "session-sticky", "radix-affinity-balanced")


def _small_sweep() -> dict[str, Any]:
    trace, _ = AgenticWorkload(n_sessions=30, turns_per_session=3, prefix_sharing_ratio=0.6,
                               arrival_rate=3.0, seed=0).generate()
    # The trace's own fingerprint, so a failure says at once whether the generator moved.
    canon = json.dumps([[q.request_id, round(q.arrival_ts, 9), list(q.block_hashes),
                         q.prompt_tokens, q.output_tokens, q.session_id] for q in trace])
    profile = load_profile(REPO / "configs" / "profiles" / "t4-qwen3-4b-vllm0.30.0.json")
    cost = CostModel.from_yaml(REPO / "configs" / "cost_model.yaml")
    points, _ = sweep(trace, policies=POLICIES, replica_counts=(1, 2, 3), seeds=(0, 1),
                      profile=profile, cost=cost)
    rec = recommend(points)
    return {
        "trace": {"requests": len(trace), "sha256": hashlib.sha256(canon.encode()).hexdigest()},
        "points": [p.as_dict() for p in points],
        "recommended": {
            "policy": rec.chosen.policy if rec.chosen else None,
            "n_replicas": rec.chosen.n_replicas if rec.chosen else None,
            "tied_with": list(rec.tied_with),
            "reason": rec.reason,
        },
    }


def test_a_small_sweep_reproduces_its_golden_file():
    got = _small_sweep()
    if os.environ.get("KVSIM_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got["trace"] == want["trace"], "the generator or its hashing changed"
    for g, w in zip(got["points"], want["points"], strict=True):
        assert g == w, f"{w['policy']} on {w['n_replicas']} replica(s) moved"
    assert got["recommended"] == want["recommended"]
