#!/usr/bin/env python3
"""Prepare Phase 6: trace fingerprints, the simulator's predictions and the notebook.

    python scripts/phase6_prepare.py
        # configs/phase6_sim2real.yaml -> artifacts/results/phase6_prereg/predictions.json
        #                               -> notebooks/phase6_sim2real.ipynb

All three come from one config in one run. The notebook regenerates the traffic on the GPU box
with a stdlib port of the generator, so before writing the notebook this script runs that port
here and requires the same fingerprints and the same block hashes the simulator routed on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _phase6_notebook import DATA_CELL, build_notebook
from _spike_common import results_header

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.hashing import chain_hashes


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def _check_port(workloads: dict[str, Any], fingerprints: dict[str, str],
                traces: dict[str, list]) -> None:
    """Run the notebook's generator and hashing here; require the simulator's traces and hashes."""
    code = DATA_CELL.split("TRACES = {}")[0]  # the definitions, not the notebook's own checks
    code = code.replace('json.loads(r"""@@WORKLOADS@@""")', "WORKLOADS_IN")
    code = code.replace('json.loads(r"""@@FINGERPRINTS@@""")', "{}")
    code = code.replace('json.loads(r"""@@PRED@@""")', "{}")
    ns: dict[str, Any] = {"json": json, "hashlib": hashlib, "BLOCK_SIZE": 16,
                          "WORKLOADS_IN": workloads}
    exec(compile("import random\n" + code, "phase6_data_cell", "exec"), ns)  # noqa: S102
    for name, w in workloads.items():
        port = ns["generate"](w)
        fp = sha256([[q["request_id"], q["arrival_ts"], q["session_id"], q["ids"]] for q in port])
        if fp != fingerprints[name]:
            raise SystemExit(f"error: the notebook's generator port gives {name} a different "
                             "trace from the repo's generator; fix the port before any GPU run")
        for q, (req, _ids) in zip(port, traces[name], strict=True):
            if tuple(ns["chain_hashes"](q["ids"])) != req.block_hashes:
                raise SystemExit(f"error: the notebook's hashing differs at {q['request_id']}")
    print("notebook port ✓  same traces and the same block hashes as the simulator")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path,
                    default=REPO / "configs" / "phase6_sim2real.yaml")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "phase6_prereg" / "predictions.json")
    ap.add_argument("--notebook", type=pathlib.Path,
                    default=REPO / "notebooks" / "phase6_sim2real.ipynb")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    profile = load_profile(REPO / cfg["profile"])
    if profile.kv_cache_blocks != cfg["engine"]["num_gpu_blocks"]:
        raise SystemExit(f"error: the profile has {profile.kv_cache_blocks} blocks, the engine "
                         f"is given {cfg['engine']['num_gpu_blocks']} — they must match")
    # The generator settings exactly as the config states them (YAML merge keys resolved).
    workloads = {name: dict(spec) for name, spec in cfg["workloads"].items()}

    traces, fingerprints = {}, {}
    for name, spec in workloads.items():
        pairs = list(AgenticWorkload(**spec).iter_with_tokens())
        traces[name] = pairs
        fingerprints[name] = sha256([[r.request_id, r.arrival_ts, r.session_id, ids]
                                     for r, ids in pairs])
        assert all(chain_hashes(ids) == r.block_hashes for r, ids in pairs)
    _check_port(workloads, fingerprints, traces)

    pred: dict[str, Any] = {"runs": {}}
    per_request: dict[str, Any] = {}
    for name, pairs in traces.items():
        trace = [r for r, _ in pairs]
        pred["runs"][name], per_request[name] = {}, {}
        for policy in cfg["policies"]:
            r = run(trace, policy=policy, n_replicas=cfg["replicas"], seed=cfg["router_seed"],
                    profile=profile, cache_model="vllm", record_per_request=True)
            pct = r.ttft_percentiles()
            pred["runs"][name][policy] = {
                "hit_rate": round(r.hit_rate, 6),
                "prefix_cache_queries": r.cache_queries,
                "prefix_cache_hits": r.cache_hits,
                "requests_per_replica": r.per_replica_requests,
                "preemptions": r.preemptions,
                "ttft_p50_ms": pct["p50_ms"],
                "ttft_p95_ms": pct["p95_ms"],
            }
            per_request[name][policy] = {rid: {"replica": v["replica"],
                                               "cached_tokens": v["cached_tokens"]}
                                         for rid, v in sorted(r.per_request.items())}
            print(f"  {name:4s} {policy:15s} predicted hit {r.hit_rate:.4f} · requests "
                  f"{r.per_replica_requests} · preemptions {r.preemptions}")
    pred_sha = sha256(pred)

    payload = results_header("scripts/phase6_prepare.py", seed=cfg["router_seed"],
                             hardware_path="cpu-only", extra={"config": cfg})
    payload.update({"profile": profile.provenance(), "workloads": workloads,
                    "trace_fingerprints_sha256": fingerprints, "predictions": pred,
                    "predictions_sha256": pred_sha, "per_request_predictions": per_request})
    args.prereg.parent.mkdir(parents=True, exist_ok=True)
    args.prereg.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(build_notebook(cfg, workloads, fingerprints, pred, pred_sha))
    print(f"wrote {args.prereg}\nwrote {args.notebook}  (predictions sha256 {pred_sha[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
