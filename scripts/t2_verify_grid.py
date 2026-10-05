#!/usr/bin/env python3
"""Are T2's stored grids what the committed simulator gives? A random sample, run again.

    python scripts/t2_verify_grid.py      # -> artifacts/results/t2_two_part_slo/verification.json

Why this exists. T2's grids took 35 minutes to compute, one calibration after another in one
process. Grid A's header was written from a clean tree. While B and C were running, unrelated
files were being edited in the working tree (`kvsim calibrate`, the CLI, a new module) — so their
headers, written when each calibration started, say `git_dirty: true`. The simulator's code had
been loaded when the process started, from the clean commit, and the edits touched nothing a grid
run imports; but a header cannot show that, and "it could not have mattered" is an argument.

This is the evidence instead: a seeded random sample of cells from every grid, run again by THIS
checkout — meant to be a clean checkout — and compared with what is stored. The grids are
deterministic, so a cell either reproduces to the last stored digit or the code differs.
Deliberately a sample: the full grids are ~30 minutes of every core, and this ran on battery.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _spike_common import results_header

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

CONFIG = REPO / "configs" / "tpot_slo.yaml"
RESULTS = REPO / "artifacts" / "results" / "t2_two_part_slo"


def _again(job: tuple) -> tuple:
    profile_path, trace_spec, cache_model, k, load, policy, n, seed = job
    trace, _ = AgenticWorkload(**trace_spec, n_shared_prefixes=k,
                               arrival_rate=float(load)).generate()
    base, _, alpha = policy.partition("@")
    r = run(trace, policy=base, n_replicas=n, seed=seed, cache_model=cache_model,
            profile=load_profile(profile_path),
            policy_kwargs={"load_penalty": float(alpha)} if alpha else None)
    return r.ttft_percentiles()["p95_ms"], r.tpot_percentiles()["p95_ms"]


def verify(name: str, grid: dict[str, Any], sample: int, workers: int) -> dict[str, Any]:
    ax = grid["axes"]
    cells = [(k, ld, p, n, s) for k in ax["prompts"] for ld in ax["loads"] for p in ax["policies"]
             for n in ax["replicas"] for s in ax["seeds"]]
    picked = random.Random(f"t2-verify-{name}").sample(cells, sample)
    jobs = [(str(REPO / grid["profile_path"]), grid["trace"], grid["cache_model"], *c)
            for c in picked]
    with ProcessPoolExecutor(workers) as ex:
        got = list(ex.map(_again, jobs, chunksize=4))
    wrong = []
    for (k, ld, p, n, s), (ttft, tpot) in zip(picked, got, strict=True):
        cell = grid["cells"][str(k)][str(ld)][p]
        i, j = ax["replicas"].index(n), ax["seeds"].index(s)
        stored = (cell["ttft_p95_ms"][i][j], cell["tpot_p95_ms"][i][j])
        if stored != (ttft, tpot):
            wrong.append({"cell": [k, ld, p, n, s], "stored": stored, "again": [ttft, tpot]})
    return {"runs_in_grid": len(cells), "cells_run_again": len(picked),
            "identical": len(picked) - len(wrong), "different": wrong,
            "grid_written_at": grid["git_sha"], "grid_git_dirty": grid["git_dirty"],
            "grid_git_dirty_paths": grid.get("git_dirty_paths")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS,
                    help="where the grids are (default: this checkout's)")
    ap.add_argument("--sample", type=int, default=240, help="cells per calibration")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    header = results_header("scripts/t2_verify_grid.py", seed=0, hardware_path="cpu-only")
    out: dict[str, Any] = {**header, "calibrations": {}}
    for name in cfg["t2"]["calibrations"]:
        grid = json.loads((args.results / f"grid_{name}.json").read_text())
        # The simulator this checkout runs, against the one the grid says it was written at.
        same = subprocess.run(["git", "diff", "--quiet", grid["git_sha"], "HEAD", "--", "src"],
                              cwd=REPO, check=False).returncode == 0
        out["calibrations"][name] = {**verify(name, grid, args.sample, args.workers),
                                     "src_identical_to_grid_commit": same}
        v = out["calibrations"][name]
        print(f"calibration {name}: {v['identical']} of {v['cells_run_again']} sampled cells "
              f"identical (grid header: {grid['git_sha']}, dirty {grid['git_dirty']}; this "
              f"checkout: {header['git_sha']}, dirty {header['git_dirty']}; src identical to the "
              f"grid's commit: {same})", flush=True)
    out["all_identical"] = all(not v["different"] for v in out["calibrations"].values())
    (args.results / "verification.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0 if out["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
