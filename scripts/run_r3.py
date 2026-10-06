#!/usr/bin/env python3
"""R3's grid: a public production trace through every policy, on every calibration.

    python scripts/run_r3.py --probe     # round-robin ALONE, one calibration: to choose the speeds
    python scripts/run_r3.py             # configs/r3_bailian.yaml -> artifacts/results/r3_bailian/

Registered before it ran (the config, and a dated entry in docs/09-hypotheses.md). Like
scripts/run_t2_grid.py, this script only RUNS the simulator and writes down what each run measured
— p95 time to first token, p95 time per output token, hit rate, how unevenly the requests were
spread, and the policy's own account of its decisions. It applies no target and decides nothing:
`scripts/r3_analyze.py` does that, from this file.

`--probe` runs round-robin alone — no comparison between policies — and exists only to choose the
replay speeds by the rule in the config, the way R1 and R2 chose theirs.

The trace is other people's data and is never committed (.gitignore: artifacts/traces/). Fetch it
from the source named in the config; the results record its sha256, so a reader can check they
replayed the same bytes.

Tier (ADR-010): T4 timings and a SCALED replay (every token count × `token_scale`) — see the
adapter's docstring for what that keeps and what it does not.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Any

import yaml

from kv_routing_sim.cache.block_pool import max_hit_blocks
from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces import adapters
from kv_routing_sim.traces.schema import NormalizedRequest, TraceStats

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "r3_bailian.yaml"

_STATE: dict[str, Any] = {}  # per process: the config and the adapted trace, set by `_init`


def read_trace(cfg: dict[str, Any]) -> tuple[tuple[NormalizedRequest, ...], TraceStats]:
    """The adapted trace, cut to the window."""
    stats = TraceStats()
    reader = adapters.get(cfg["adapter_name"])
    reqs = reader.read(REPO / cfg["trace"], stats=stats, **cfg["adapter"])
    return tuple(r for r in reqs if r.arrival_ts < cfg["window_s"]), stats


def structure(trace: tuple[NormalizedRequest, ...], block_size: int) -> dict[str, Any]:
    """What the traffic looks like, with an unbounded cache and in arrival order — a description
    of the trace, independent of any policy."""
    seen: set[int] = set()
    by_session: dict[str | None, set[int]] = {}
    sizes: Counter = Counter()
    blocks = any_hit = own_hit = 0
    for r in trace:
        mine = by_session.setdefault(r.session_id, set())
        sizes[r.session_id] += 1
        limit = max_hit_blocks(r.prompt_tokens, block_size)
        hit_any = hit_own = 0
        for h in r.block_hashes[:limit]:
            if h not in seen:
                break
            hit_any += 1
        for h in r.block_hashes[:limit]:
            if h not in mine:
                break
            hit_own += 1
        blocks += len(r.block_hashes)
        any_hit += hit_any
        own_hit += hit_own
        seen.update(r.block_hashes)
        mine.update(r.block_hashes)
    per = sorted(sizes.values())
    prompt = sorted(r.prompt_tokens for r in trace)
    out = sorted(r.output_tokens for r in trace)

    def q(xs: list[int], p: float) -> int:
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0

    return {
        "requests": len(trace), "sessions": len(per),
        "requests_per_session": {"p50": q(per, 0.5), "p90": q(per, 0.9), "p99": q(per, 0.99),
                                 "max": per[-1] if per else 0},
        "share_of_requests_in_sessions_of_5_or_more": (
            round(sum(n for n in per if n >= 5) / len(trace), 4) if trace else 0.0),
        "prompt_tokens": {"p50": q(prompt, 0.5), "p90": q(prompt, 0.9), "max": prompt[-1]},
        "output_tokens": {"p50": q(out, 0.5), "p90": q(out, 0.9), "max": out[-1]},
        "blocks": blocks, "distinct_blocks": len(seen),
        "reusable_from_any_earlier_request": round(any_hit / blocks, 4) if blocks else 0.0,
        "reusable_from_the_same_session": round(own_hit / blocks, 4) if blocks else 0.0,
    }


def _init(config_path: str) -> None:
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    _STATE["cfg"] = cfg
    _STATE["trace"], _ = read_trace(cfg)
    _at_speed.cache_clear()
    _profile.cache_clear()


@cache
def _at_speed(speed: float) -> tuple[tuple[NormalizedRequest, ...], int]:
    """The window at `speed`× the recorded arrival rate, and how many requests are warm-up."""
    base = _STATE["trace"]
    warmup = sum(1 for r in base if r.arrival_ts < _STATE["cfg"]["warmup_s"])
    return tuple(dataclasses.replace(r, arrival_ts=r.arrival_ts / speed) for r in base), warmup


@cache
def _profile(path: str) -> CostProfile:
    return load_profile(REPO / path)


def _job(key: tuple) -> tuple:
    cal, speed, label, n, seed = key
    cfg = _STATE["cfg"]
    spec = next(p for p in cfg["policies"] if p["label"] == label)
    trace, warmup = _at_speed(float(speed))
    r = run(trace, policy=spec["policy"], n_replicas=n, seed=seed,
            profile=_profile(cfg["calibrations"][cal]), cache_model=cfg["cache_model"],
            policy_kwargs=spec.get("kwargs"), warmup_requests=warmup)
    mean = len(trace) / n
    return key, {
        # The same rounded percentiles `to_point` decides an SLO on (3 decimals of a millisecond).
        "ttft": r.ttft_percentiles()["p95_ms"], "tpot": r.tpot_percentiles()["p95_ms"],
        "hit": r.hit_rate, "busiest": max(r.per_replica_requests) / mean if mean else 0.0,
        "preemptions": r.preemptions, "router": r.router,
        "complete": r.completed == r.arrived == len(trace),
    }


def _sweep(jobs: list[tuple], config_path: pathlib.Path, workers: int) -> dict[tuple, dict]:
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(config_path),)) as ex:
        return dict(ex.map(_job, jobs, chunksize=1))


def _cells(got: dict[tuple, dict], cals: list[str], speeds: list[float], labels: list[str],
           replicas: list[int], seeds: list[int]) -> dict[str, Any]:
    """{calibration: {speed: {policy: per-replica lists, each over the seeds}}}."""
    cells: dict[str, Any] = {}
    for cal in cals:
        for speed in speeds:
            for label in labels:
                rows = [[got[(cal, speed, label, n, s)] for s in seeds] for n in replicas]
                router: dict[str, list[float]] = {}
                for row in rows:
                    keys = sorted({k for x in row for k in x["router"]})
                    for k in keys:
                        router.setdefault(k, []).append(
                            round(sum(x["router"].get(k, 0) for x in row) / len(row), 1))
                cell = {
                    "ttft_p95_ms": [[x["ttft"] for x in row] for row in rows],
                    "tpot_p95_ms": [[x["tpot"] for x in row] for row in rows],
                    "hit_rate": [round(sum(x["hit"] for x in row) / len(row), 4) for row in rows],
                    "busiest_over_mean": [round(max(x["busiest"] for x in row), 3)
                                          for row in rows],
                    "preemptions": [round(sum(x["preemptions"] for x in row) / len(row), 1)
                                    for row in rows],
                    "all_completed": all(x["complete"] for row in rows for x in row),
                }
                if router:
                    cell["router"] = router
                cells.setdefault(cal, {}).setdefault(f"{speed:g}", {})[label] = cell
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: the config's results_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--probe", action="store_true",
                    help="round-robin alone, one seed, the first calibration, the probe speeds")
    args = ap.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = args.out or REPO / cfg["results_dir"]
    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken BEFORE the runs: the header must describe the tree that produced the numbers.
    payload = results_header("scripts/run_r3.py", seed=0, hardware_path="cpu-only")
    trace_path = REPO / cfg["trace"]
    sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if cfg.get("trace_sha256") and cfg["trace_sha256"] != sha:
        raise SystemExit(f"error: {cfg['trace']} is not the registered file (sha256 {sha[:16]}…)")
    trace, stats = read_trace(cfg)
    block_size = load_profile(REPO / next(iter(cfg["calibrations"].values()))).block_size
    replicas = list(range(1, cfg["replicas_max"] + 1))
    cals = list(cfg["calibrations"])
    labels = [p["label"] for p in cfg["policies"]]
    if args.probe:
        cals, labels, seeds = cals[:1], ["round-robin"], [cfg["seeds"][0]]
        speeds = [float(s) for s in cfg["probe_speeds"]]
    else:
        seeds, speeds = list(cfg["seeds"]), [float(s) for s in cfg["speeds"]]
    if "round-robin" not in [p["label"] for p in cfg["policies"]]:
        raise SystemExit("error: the config must list round-robin (the probe and the baseline)")

    jobs = [(c, sp, lb, n, s) for c in cals for sp in speeds for lb in labels for n in replicas
            for s in seeds]
    t0 = time.time()
    got = _sweep(jobs, config_path, args.workers)
    runtime = time.time() - t0
    payload.update({
        "config": cfg, "trace_sha256": sha, "adapter_stats": stats.as_dict(),
        "structure": structure(trace, block_size), "runtime_s": round(runtime, 1),
        "n_runs": len(jobs), "probe": bool(args.probe),
        "axes": {"calibrations": cals, "speeds": [f"{s:g}" for s in speeds], "policies": labels,
                 "replicas": replicas, "seeds": seeds},
        "profiles": {c: load_profile(REPO / cfg["calibrations"][c]).provenance() for c in cals},
        "cells": _cells(got, cals, speeds, labels, replicas, seeds),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "probe.json" if args.probe else "grid.json"
    (out_dir / name).write_text(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    print(f"{len(jobs)} runs in {runtime:.0f} s on {args.workers} workers -> {out_dir / name}")
    if args.probe:
        slo = cfg["slo"]
        for sp in payload["axes"]["speeds"]:
            cell = payload["cells"][cals[0]][sp]["round-robin"]
            first = next((n for i, n in enumerate(replicas)
                          if cell["ttft_p95_ms"][i][0] <= slo["ttft_p95_ms"]), None)
            both = next((n for i, n in enumerate(replicas)
                         if cell["ttft_p95_ms"][i][0] <= slo["ttft_p95_ms"]
                         and cell["tpot_p95_ms"][i][0] <= slo["tpot_p95_ms"]), None)
            print(f"  round-robin at {sp}×: first-token target on {first or 'none'} · "
                  f"both halves on {both or 'none'} (of {replicas[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
