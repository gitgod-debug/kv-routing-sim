#!/usr/bin/env python3
"""T2's grid: H2b-C2's traffic on one calibration, with every run's two latencies kept.

    python scripts/run_t2_grid.py                 # all three calibrations, ~10 minutes each
    python scripts/run_t2_grid.py --calibration A
        # configs/tpot_slo.yaml -> artifacts/results/t2_two_part_slo/grid_<A|B|C>.json

Registered before it ran (configs/tpot_slo.yaml; docs/09-hypotheses.md). This script only RUNS the
simulator and writes down, for every (prompts, load, policy, replicas, seed), the run's p95 time to
first token and p95 time per output token. It applies no target and decides nothing:
`scripts/t2_analyze.py` does that, from these files — which is why the replica count under any
target, the registered one or a reported one, comes from the same runs.

The grid is H2b-C2's (configs/h2b_c2.yaml: the traffic, loads, policies, load weights, seeds),
read from that file, with the replica axis extended to `t2.replicas_max`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Any

import yaml

from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "tpot_slo.yaml"
OUT = REPO / "artifacts" / "results" / "t2_two_part_slo"
BAL, STICKY = "radix-affinity-balanced", "session-sticky"

_STATE: dict[str, Any] = {}  # per process: the design and the profile, set by `_init`


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


def design(cfg: dict[str, Any]) -> dict[str, Any]:
    """The grid, from H2b-C2's registered config and Phase 5's (which it reads in turn)."""
    h2b = yaml.safe_load((REPO / cfg["t2"]["h2b_config"]).read_text())
    p5 = yaml.safe_load((REPO / h2b["h2b_config"]).read_text())
    policies = ([(STICKY, None), (BAL, h2b["alpha"])] + [(BAL, a) for a in h2b["alphas_reported"]]
                + [(p, None) for p in h2b["policies_reported"]])
    return {
        "prompts": list(p5["h2b"]["n_shared_prefixes"]), "loads": list(h2b["loads"]),
        "replicas": list(range(1, cfg["t2"]["replicas_max"] + 1)), "seeds": list(h2b["seeds"]),
        "policies": [[p, a] for p, a in policies], "cache_model": h2b["cache_model"],
        "trace": {**p5["session_shape"], "prefix_sharing_ratio": p5["h2b"]["sharing"],
                  "prefix_zipf_s": p5["h2b"]["prefix_zipf_s"]},
    }


def _init(grid: dict[str, Any], profile_path: str) -> None:
    _STATE["grid"] = grid
    _STATE["profile"] = load_profile(profile_path)
    _trace.cache_clear()


@cache
def _trace(k: int, load: float):
    spec = {**_STATE["grid"]["trace"], "n_shared_prefixes": k}
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()[0]


def _job(key: tuple) -> tuple:
    k, load, policy, alpha, n, seed = key
    profile: CostProfile = _STATE["profile"]
    r = run(_trace(k, load), policy=policy, n_replicas=n, seed=seed, profile=profile,
            cache_model=_STATE["grid"]["cache_model"],
            policy_kwargs=None if alpha is None else {"load_penalty": float(alpha)})
    # The same rounded percentiles `to_point` decides an SLO on (3 decimals of a millisecond).
    return key, r.ttft_percentiles()["p95_ms"], r.tpot_percentiles()["p95_ms"], r.hit_rate


def run_calibration(name: str, cfg: dict[str, Any], out: pathlib.Path, workers: int) -> None:
    grid = design(cfg)
    profile_path = REPO / cfg["t2"]["calibrations"][name]["profile"]
    profile = load_profile(profile_path)
    if not profile.context_aware or not profile.calibrated:
        raise SystemExit(f"error: calibration {name} must be a calibrated, context-aware profile")
    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken BEFORE the runs: the header must describe the tree that produced the numbers.
    payload = results_header("scripts/run_t2_grid.py", seed=0, hardware_path="cpu-only")
    pols = [(p, a) for p, a in grid["policies"]]
    jobs = [(k, ld, p, a, n, s) for k in grid["prompts"] for ld in grid["loads"] for p, a in pols
            for n in grid["replicas"] for s in grid["seeds"]]
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(grid, str(profile_path))) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    runtime = time.time() - t0

    got = {key: (ttft, tpot, hit) for key, ttft, tpot, hit in results}
    cells: dict[str, Any] = {}
    for k in grid["prompts"]:
        for ld in grid["loads"]:
            for p, a in pols:
                rows = [[got[(k, ld, p, a, n, s)] for s in grid["seeds"]] for n in grid["replicas"]]
                cells.setdefault(str(k), {}).setdefault(str(ld), {})[label(p, a)] = {
                    "ttft_p95_ms": [[x[0] for x in row] for row in rows],
                    "tpot_p95_ms": [[x[1] for x in row] for row in rows],
                    "hit_rate": [round(sum(x[2] for x in row) / len(row), 4) for row in rows],
                }
    payload.update({
        "calibration": name, "profile_path": cfg["t2"]["calibrations"][name]["profile"],
        "profile": profile.provenance(), "runtime_s": round(runtime, 1), "n_runs": len(jobs),
        "axes": {"prompts": grid["prompts"], "loads": grid["loads"],
                 "replicas": grid["replicas"], "seeds": grid["seeds"],
                 "policies": [label(p, a) for p, a in pols]},
        "trace": grid["trace"], "cache_model": grid["cache_model"], "cells": cells,
    })
    out.mkdir(parents=True, exist_ok=True)
    # One line per cell: readable in a diff, and a third of the size of an indented dump.
    head = {k: v for k, v in payload.items() if k != "cells"}
    body = ",\n".join(
        f'  {json.dumps(k)}: {{\n' + ",\n".join(
            f'   {json.dumps(ld)}: {{\n' + ",\n".join(
                f"    {json.dumps(pol)}: {json.dumps(v, separators=(',', ':'))}"
                for pol, v in by_pol.items()) + "\n   }"
            for ld, by_pol in by_load.items()) + "\n  }"
        for k, by_load in cells.items())
    text = json.dumps(head, indent=1, default=str)[:-2] + ',\n "cells": {\n' + body + "\n }\n}\n"
    json.loads(text)  # it must still be JSON
    (out / f"grid_{name}.json").write_text(text)
    print(f"calibration {name}: {len(jobs)} runs in {runtime:.0f}s on {workers} workers -> "
          f"{out / f'grid_{name}.json'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--calibration", action="append",
                    help="A, B or C; repeatable. Default: every calibration in the config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for name in args.calibration or list(cfg["t2"]["calibrations"]):
        run_calibration(name, cfg, args.out, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
