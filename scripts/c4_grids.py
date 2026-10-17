#!/usr/bin/env python3
"""Run the C4 simulator grids on the session's fresh calibrations (CPU only).

    python scripts/c4_grids.py        # artifacts/results/c4_kaggle_t4/ -> grids.json

Reads configs/c4_per_token.yaml. Like scripts/run_t2_grid.py, whose worker code it uses, this only
runs the simulator and records each run's p95 time to first token and p95 time per output token;
scripts/c4_analyze.py reads grids.json and applies rules E, F1 and F2.

Two grids, both on H2b-C2's traffic and seeds (read from that config):
  firmness   every fit from the session: each run of `kvsim calibrate`, as its pooled fit and as
             the fit of each repeat alone, plus profile A for the report.
  prompts64  the pooled fit of each run, at 64 prompts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_t2_grid as worker  # its _init and _job are the grid's worker code
from _spike_common import results_header

from kv_routing_sim.sim.cost_model import CostProfile, load_profile, replicate_profiles

CONFIG = REPO / "configs" / "c4_per_token.yaml"
RESULTS = REPO / "artifacts" / "results" / "c4_kaggle_t4"


def fits(cfg: dict[str, Any], results: pathlib.Path) -> dict[str, CostProfile]:
    """Every fit from the session, keyed by id.

    `s<seed>` is a run's pooled fit, `s<seed>r<rep>` the fit of one of its repeats, and `A` the
    shipped profile the predictions were made with.
    """
    out: dict[str, CostProfile] = {"A": load_profile(REPO / cfg["profile"])}
    for seed in cfg["calibrate"]["seeds"]:
        path = results / f"calibrated_profile_seed{seed}.json"
        if not path.exists():
            continue
        out[f"s{seed}"] = load_profile(path)
        for i, rep in enumerate(replicate_profiles(path)):
            out[f"s{seed}r{i}"] = rep
    return out


def grid_design(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """One of the two grids, in the shape scripts/run_t2_grid.py's workers take."""
    h2b = yaml.safe_load((REPO / cfg["grids"]["h2b_config"]).read_text())
    p5 = yaml.safe_load((REPO / h2b["h2b_config"]).read_text())
    policies = []
    for label in spec["policies"]:
        name, _, alpha = label.partition("@")
        policies.append([name, float(alpha) if alpha else None])
    return {"prompts": list(spec["prompts"]), "loads": list(spec["loads"]),
            "replicas": list(range(1, spec["replicas_max"] + 1)), "seeds": list(h2b["seeds"]),
            "policies": policies, "cache_model": h2b["cache_model"],
            "trace": {**p5["session_shape"], "prefix_sharing_ratio": p5["h2b"]["sharing"],
                      "prefix_zipf_s": p5["h2b"]["prefix_zipf_s"]}}


def run_grid(grid: dict[str, Any], profile: CostProfile, workers: int) -> dict[str, Any]:
    """Every cell of one grid on one fit: {prompts: {load: {policy: {the two p95 lists}}}}."""
    pols = [(p, a) for p, a in grid["policies"]]
    jobs = [(k, ld, p, a, n, s) for k in grid["prompts"] for ld in grid["loads"] for p, a in pols
            for n in grid["replicas"] for s in grid["seeds"]]
    with tempfile.TemporaryDirectory() as tmp:
        # The workers load their profile from a file; a repeat's fit exists only in memory.
        path = pathlib.Path(tmp) / "profile.json"
        path.write_text(json.dumps(dataclasses.asdict(profile)))
        with ProcessPoolExecutor(workers, initializer=worker._init,
                                 initargs=(grid, str(path))) as ex:
            got = {key: (ttft, tpot) for key, ttft, tpot, _hit in
                   ex.map(worker._job, jobs, chunksize=4)}
    cells: dict[str, Any] = {}
    for k in grid["prompts"]:
        for ld in grid["loads"]:
            for p, a in pols:
                rows = [[got[(k, ld, p, a, n, s)] for s in grid["seeds"]] for n in grid["replicas"]]
                cells.setdefault(str(k), {}).setdefault(str(ld), {})[worker.label(p, a)] = {
                    "ttft_p95_ms": [[x[0] for x in row] for row in rows],
                    "tpot_p95_ms": [[x[1] for x in row] for row in rows]}
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # Taken before the runs, so the header describes the tree that produced the numbers.
    payload = results_header("scripts/c4_grids.py", seed=0, hardware_path="cpu-only")
    every = fits(cfg, args.results)
    pooled = {k: v for k, v in every.items() if k != "A" and "r" not in k}
    if not pooled:
        raise SystemExit(f"error: no calibrated_profile_seed*.json in {args.results}")
    plan = {"firmness": every, "prompts64": pooled}
    t0, out, n_runs = time.time(), {}, 0
    for name, which in plan.items():
        grid = grid_design(cfg, cfg["grids"][name])
        out[name] = {"axes": {"prompts": grid["prompts"], "loads": grid["loads"],
                              "replicas": grid["replicas"], "seeds": grid["seeds"],
                              "policies": [worker.label(p, a) for p, a in grid["policies"]]},
                     "fits": {}}
        for fit_id, profile in which.items():
            out[name]["fits"][fit_id] = run_grid(grid, profile, args.workers)
            n_runs += (len(grid["prompts"]) * len(grid["loads"]) * len(grid["policies"])
                       * len(grid["replicas"]) * len(grid["seeds"]))
            print(f"{name}: fit {fit_id} done ({time.time() - t0:.0f} s)", flush=True)
    payload.update({"runtime_s": round(time.time() - t0, 1), "n_runs": n_runs,
                    "fit_ids": sorted(every),
                    "coefficients": {k: {f.name: getattr(v, f.name)
                                         for f in dataclasses.fields(v)
                                         if f.name.endswith("_s")} for k, v in every.items()},
                    "grids": out})
    (args.results / "grids.json").write_text(
        json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    print(f"{n_runs} runs in {time.time() - t0:.0f} s -> {args.results / 'grids.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
