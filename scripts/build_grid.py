#!/usr/bin/env python3
"""Precompute the demo grid: sharing x load x replicas x policy -> one JSON file.

⭐ THIS IS THE DEMO'S ENTIRE BACKEND (ADR-011). The static page ships this file and the sliders look
answers up in it. No server, nothing to wake, and — the part that matters for this project — every
number on the page comes from a committed script with a pinned seed, so it is auditable exactly like
anything in `artifacts/results/`. A live app computing on demand would be LESS checkable, because
nobody can verify what it computed.

⚠️ The grid inherits the cost profile's status. While that profile is uncalibrated, the page may show
the policy RANKING and the cache numbers but NOT the millisecond or rupee figures as predictions
(ADR-010 tier 1). The emitted JSON carries `calibrated: false` so the page cannot forget.

    python scripts/build_grid.py --sharing-steps 11 --rates 4,8,16,32 --replicas 1-8 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

from kv_routing_sim import policies as policy_pkg
from kv_routing_sim.sim.cost_model import UNCALIBRATED_DEFAULT
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

# The metrics each cell carries. Kept short and fixed-order: the page indexes by position, which is
# what keeps the file small enough to ship.
FIELDS = ("hit_rate", "p50_ms", "p95_ms", "p99_ms", "imbalance")


def _scenario(sharing: float, rate: float, replicas: list[int], names: list[str],
              seeds: tuple[int, ...], n_sessions: int, turns: int) -> dict[str, object]:
    """One (sharing, rate) point: generate the trace once, sweep replicas x policies over it.

    The trace is generated ONCE per scenario rather than per policy — otherwise each policy would be
    measured on a different trace and the comparison would not be paired.
    """
    trace, report = AgenticWorkload(
        n_sessions=n_sessions,
        turns_per_session=turns,
        prefix_sharing_ratio=sharing,
        arrival_rate=rate,
        seed=0,
    ).generate()
    cells: list[list[float]] = []
    for rep in replicas:
        for name in names:
            runs = [
                run(trace, policy=name, n_replicas=rep, seed=s, profile=UNCALIBRATED_DEFAULT)
                for s in seeds
            ]
            n = len(runs)
            pct = [r.ttft_percentiles() for r in runs]
            cells.append([
                round(sum(r.hit_rate for r in runs) / n, 5),
                round(sum(p["p50_ms"] for p in pct) / n, 2),
                round(sum(p["p95_ms"] for p in pct) / n, 2),
                round(sum(p["p99_ms"] for p in pct) / n, 2),
                round(sum(r.load_imbalance() for r in runs) / n, 4),
            ])
    return {
        # The MEASURED axis value, not the requested knob — the page must plot what happened.
        "cross_session_sharing": report["achieved_cross_session_sharing"],
        "requested_sharing": sharing,
        "arrival_rate": rate,
        "requests": report["requests"],
        "mean_prompt_tokens": report["mean_prompt_tokens"],
        "cells": cells,  # ordered replicas-major, then policies
    }


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spike_common import results_header

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sharing-steps", type=int, default=11, help="values from 0.0 to 0.9")
    ap.add_argument("--rates", default="4,8,16,32")
    ap.add_argument("--replicas", default="1-8")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=120)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=-1, help="-1 = all cores")
    ap.add_argument("--out", default="artifacts/results/demo_grid.json")
    args = ap.parse_args()

    lo, hi = args.replicas.split("-") if "-" in args.replicas else (args.replicas, args.replicas)
    replicas = list(range(int(lo), int(hi) + 1))
    rates = [float(x) for x in args.rates.split(",")]
    steps = max(1, args.sharing_steps - 1)
    sharings = [round(i * 0.9 / steps, 4) for i in range(args.sharing_steps)]
    names = list(policy_pkg.ALL)
    seeds = tuple(range(args.seeds))

    scenarios = [(s, r) for s in sharings for r in rates]
    n_runs = len(scenarios) * len(replicas) * len(names) * len(seeds)
    print(f"grid: {len(sharings)} sharing x {len(rates)} rates x {len(replicas)} replicas "
          f"x {len(names)} policies x {len(seeds)} seeds")
    print(f"  = {len(scenarios)} traces, {n_runs:,} simulator runs")

    t0 = time.monotonic()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_scenario)(s, r, replicas, names, seeds, args.sessions, args.turns)
        for s, r in scenarios
    )
    wall = time.monotonic() - t0

    payload = results_header(
        "scripts/build_grid.py",
        seed=0,
        hardware_path="cpu-only",
        extra={
            "axes": {
                "sharing_requested": sharings,
                "arrival_rates": rates,
                "replicas": replicas,
                "policies": names,
                "seeds": list(seeds),
            },
            "cell_fields": list(FIELDS),
            "cell_order": "replicas-major, then policies",
            "n_simulator_runs": n_runs,
            "build_wall_clock_s": round(wall, 1),
            "profile": UNCALIBRATED_DEFAULT.provenance(),
        },
    )
    payload["scenarios"] = results

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    size = out.stat().st_size

    import gzip
    gz = len(gzip.compress(out.read_bytes()))
    print(f"\nwrote {out}")
    print(f"  wall clock : {wall:.1f}s on {args.jobs} jobs")
    print(f"  raw size   : {size / 1024:.0f} KiB")
    print(f"  gzipped    : {gz / 1024:.0f} KiB   <- what a visitor actually downloads")
    print(f"  per cell   : {size / max(1, n_runs // max(1, len(seeds))):.0f} B")
    print("\n⚠️  UNCALIBRATED: the page may show the ranking and the cache numbers. "
          "The millisecond figures are not predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
