#!/usr/bin/env python3
"""Precompute the demo grid: tenant prompts × load × replicas × policy -> one JSON file.

    python scripts/build_grid.py                   # context-free profile -> demo_grid.json
    python scripts/build_grid.py --source h2b_c2   # C2's context-aware one -> demo_grid_ctx.json

⭐ THIS IS THE DEMO'S ENTIRE BACKEND (ADR-011). The static page (`site/`) ships this file and its
selectors look answers up in it: no server, nothing to wake, and every number comes from this
committed script with pinned seeds — auditable exactly like anything else in `artifacts/results/`.

The axes are the finding's (H2b): K distinct tenant prompts × offered load × replicas × policy, on
the calibrated T4 profile and the cited cost model, so each cell says whether it holds the 2 s p95
TTFT target and what it costs. Traffic, prompt counts, loads, replicas, seeds and alpha* are read
from Phase 5's config and results, and a guard stops the build unless the grid reproduces Phase 5's
H2b required-replica table exactly: the demo cannot drift from the experiment it illustrates.

`--source h2b_c2` builds the same grid on C2's context-aware T4 profile, with the loads and replicas
of configs/h2b_c2.yaml, and guards it against that registered re-test's table instead.

⚠️ T4 figures, generated traffic, one model. The direction of the K flip survived H6's hardware
rescales; the replica counts and rupees did not transfer (H6's ranking half). The page says so.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache, lru_cache
from typing import Any

import yaml

from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
P5 = REPO / "configs" / "phase5_crossover.yaml"
P5_RESULTS = REPO / "artifacts" / "results" / "phase5_crossover" / "phase5.json"
H2B_C2 = REPO / "configs" / "h2b_c2.yaml"
H2B_C2_RESULTS = REPO / "artifacts" / "results" / "h2b_c2" / "h2b_c2.json"
FIELDS = ("hit_rate", "p50_ms", "p95_ms", "inr_per_1k", "meets_slo")
_SOURCE = "phase5"  # set in main() and in every worker: which profile, grid and guard


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


@lru_cache(maxsize=1)
def _p5() -> dict[str, Any]:
    return yaml.safe_load(P5.read_text())


@lru_cache(maxsize=1)
def _p5_results() -> dict[str, Any]:
    return json.loads(P5_RESULTS.read_text())


@lru_cache(maxsize=1)
def _src() -> dict[str, Any]:
    """The profile, loads and replicas to build on, and the registered table the grid must match.
    The traffic is always Phase 5's H2b traffic."""
    if _SOURCE == "phase5":
        p5 = _p5()
        return {"profile": p5["profile"], "loads": p5["loads"], "replicas": p5["replicas"],
                "guard": json.loads(P5_RESULTS.read_text())["h2b"]["required_replicas"],
                "out": "demo_grid.json",
                "reproduces": ("artifacts/results/phase5_crossover/phase5.json "
                               "(h2b.required_replicas)")}
    cfg = yaml.safe_load(H2B_C2.read_text())
    return {"profile": cfg["profile"], "loads": cfg["loads"], "replicas": cfg["replicas"],
            "guard": json.loads(H2B_C2_RESULTS.read_text())["required_replicas"],
            "out": "demo_grid_ctx.json",
            "reproduces": "artifacts/results/h2b_c2/h2b_c2.json (required_replicas)"}


def _init_worker(source: str) -> None:
    global _SOURCE
    _SOURCE = source
    _src.cache_clear()
    _profile.cache_clear()


def _policies() -> list[tuple[str, float | None]]:
    alpha_star = float(_p5_results()["a1"]["alpha_star"])
    return [
        ("round-robin", None),
        ("least-loaded", None),
        ("session-sticky", None),
        ("radix-affinity", None),
        ("radix-affinity-balanced", alpha_star),  # Phase 5's A1 choice
        ("radix-affinity-balanced", 1.0),  # the best weight once prompts multiply (post-hoc)
    ]


@cache
def _trace(k: int, load: float):
    p5 = _p5()
    spec = {**p5["session_shape"], "prefix_sharing_ratio": p5["h2b"]["sharing"],
            "n_shared_prefixes": k, "prefix_zipf_s": p5["h2b"]["prefix_zipf_s"]}
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()


@lru_cache(maxsize=1)
def _profile():
    return load_profile(REPO / _src()["profile"])


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _p5()["cost_model"])


def _job(key: tuple) -> tuple[tuple, Point]:
    k, load, policy, alpha, n, seed = key
    trace, _ = _trace(k, load)
    kwargs = None if alpha is None else {"load_penalty": alpha}
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(),
            cache_model=_p5()["cache_model"], policy_kwargs=kwargs)
    return key, to_point(r, _cost())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("phase5", "h2b_c2"), default="phase5")
    args = ap.parse_args()
    _init_worker(args.source)
    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    p5, src = _p5(), _src()
    ks, seeds = p5["h2b"]["n_shared_prefixes"], p5["h2b"]["seeds"]
    loads, replicas, policies = src["loads"], src["replicas"], _policies()
    not_met = max(replicas) + 1  # no replica count in the grid holds the SLO
    payload = results_header("scripts/build_grid.py", seed=0, hardware_path="cpu-only",
                             extra={"source": args.source})
    jobs = [(k, ld, p, al, n, s) for k in ks for ld in loads for p, al in policies
            for n in replicas for s in seeds]
    t0 = time.monotonic()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init_worker,
                             initargs=(args.source,)) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    wall = time.monotonic() - t0
    runs: dict[tuple, list[Point]] = {}
    for (k, ld, p, al, n, _s), point in results:
        runs.setdefault((k, ld, p, al, n), []).append(point)
    agg = {key: average_points(pts) for key, pts in runs.items()}

    # Guard: every cell here is also in the registered experiment's grid — same traces, seeds,
    # profile, cost model and code path. If the fewest replicas holding the SLO differ anywhere,
    # the page would show numbers the experiment did not produce: stop instead of writing.
    want = src["guard"]
    for k in ks:
        for ld in loads:
            for p, al in policies:
                got = next((n for n in replicas if agg[(k, ld, p, al, n)].meets_slo), not_met)
                exp = want[str(k)][str(ld)][label(p, al)]
                if got != exp:
                    raise SystemExit(
                        f"error: K={k} at {ld}/s, {label(p, al)} needs {got} replicas here but "
                        f"{exp} in {src['reproduces']}. The code or the config drifted; not "
                        "writing the grid."
                    )
    print(f"the grid reproduces {src['reproduces']} exactly")

    def cell(k: int, ld: float, p: str, al: float | None, n: int) -> list:
        a = agg[(k, ld, p, al, n)]
        inr = None if a.inr_per_1k == float("inf") else round(a.inr_per_1k, 3)
        return [round(a.hit_rate, 4), round(a.p50_ttft_ms), round(a.p95_ttft_ms), inr,
                1 if a.meets_slo else 0]

    first = {k: _trace(k, loads[0])[1] for k in ks}
    payload.update({
        "axes": {"prompts": ks, "loads": loads, "replicas": replicas,
                 "policies": [label(p, al) for p, al in policies]},
        "cell_fields": list(FIELDS),
        "cell_order": "prompts, load, policy, replicas — nested lists in that order",
        "cells": [[[[cell(k, ld, p, al, n) for n in replicas] for p, al in policies]
                   for ld in loads] for k in ks],
        "alpha_star": _p5_results()["a1"]["alpha_star"],
        "seeds": seeds,
        "working_set_blocks": {
            str(k): first[k].get("shared_working_set_blocks",
                                 first[k]["shared_prefix_tokens"] // _profile().block_size)
            for k in ks
        },
        "usable_blocks_per_replica": _profile().kv_cache_blocks - 1,
        "slo_p95_ttft_ms": _cost().ttft_p95_slo_ms,
        "profile": _profile().provenance(),
        "cost_model": _cost().provenance,
        "reproduces": src["reproduces"],
        "n_simulator_runs": len(jobs),
        "build_wall_clock_s": round(wall, 1),
    })
    out = REPO / "artifacts" / "results" / src["out"]
    out.write_text(json.dumps(payload, separators=(",", ":"), default=str) + "\n",
                   encoding="utf-8")
    raw = out.stat().st_size
    print(f"{len(jobs):,} runs in {wall:.0f}s · wrote {out.relative_to(REPO)} — {raw / 1024:.0f} "
          f"KiB raw, {len(gzip.compress(out.read_bytes())) / 1024:.0f} KiB gzipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
