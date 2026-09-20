#!/usr/bin/env python3
"""Regenerate the policy x replica sweep. Every README table comes from a script like this one.

`run_*.py` scripts are held to a stricter standard than `spike_*.py` (docs/05-codebase-map.md):
seeded, config-driven, and byte-identical on re-run apart from the timestamp. If re-running this
changes a number, something is reading unseeded randomness and must be found.

⚠️ THIS PRODUCES NO PUBLISHABLE LATENCY OR COST NUMBER. The cost profile is uncalibrated, so what
comes out is the policy RANKING and the cache behaviour (ADR-010 tier 1, pending H6). The
millisecond figures are shapes. Phase 3 replaces the profile; until then this script exists to make
the pipeline reproducible, not to make claims.

    python scripts/run_sweep.py --sharing 0.6 --replicas 1,2,4,8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The package is imported normally — it is installed (`pip install -e ".[dev]"`), so no sys.path
# surgery is needed for it. Only `_spike_common` lives beside this file rather than in the package,
# and it is imported inside main() so every top-level import here stays ordinary.
from kv_routing_sim import policies
from kv_routing_sim.eval.recommend import recommend
from kv_routing_sim.eval.sweep import sweep
from kv_routing_sim.sim.cost_model import UNCALIBRATED_DEFAULT
from kv_routing_sim.traces.generator import AgenticWorkload


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spike_common import results_header, write_results

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--sharing", type=float, default=0.6)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--replicas", default="1,2,4,8")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweep.json")
    args = ap.parse_args()

    replica_counts = [int(x) for x in args.replicas.split(",")]
    wl = AgenticWorkload(
        n_sessions=args.sessions,
        prefix_sharing_ratio=args.sharing,
        arrival_rate=args.rate,
        seed=args.seed,
    )
    trace, report = wl.generate()
    print(f"trace: {report['requests']} requests, "
          f"cross-session sharing {report['achieved_cross_session_sharing']:.3f}")

    points, _ = sweep(
        trace,
        policies=list(policies.ALL),
        replica_counts=replica_counts,
        seeds=tuple(range(args.seeds)),
        trace_info=report,
    )
    rec = recommend(points)

    print(f"\n{'policy':26s} {'rep':>4} {'hit_rate':>9} {'p50_ms':>9} {'p95_ms':>10} {'imbal':>7}")
    for p in points:
        print(
            f"{p.policy:26s} {p.n_replicas:4d} {p.hit_rate:9.4f} "
            f"{p.p50_ttft_ms:9.1f} {p.p95_ttft_ms:10.1f} {p.load_imbalance:7.3f}"
        )
    print(f"\nrecommended: {rec.chosen.policy} on {rec.chosen.n_replicas} replicas")
    print(f"reason: {rec.reason}")

    payload = results_header(
        "scripts/run_sweep.py",
        seed=args.seed,
        hardware_path="cpu-only",
        extra={"workload": report, "profile": UNCALIBRATED_DEFAULT.provenance()},
    )
    payload["points"] = [p.as_dict() for p in points]
    payload["recommendation"] = rec.as_dict()
    print(f"\nwrote {write_results(args.out, payload)}")
    print(
        "\n⚠️  UNCALIBRATED: the ranking and the cache numbers are the output. "
        "The millisecond figures are not predictions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
