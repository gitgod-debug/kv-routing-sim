#!/usr/bin/env python3
"""What breaks at 10× and 100×? Measured, for the README's named section.

    python scripts/measure_scale.py                # -> artifacts/results/scale/
    python scripts/measure_scale.py --report-only  # re-render report.md from scale.json, no runs

Each case runs in a FRESH process, so its peak RSS is its own. Base = Phase 4's session shape
(200 sessions × 4 turns, sharing 0.8) on the calibrated T4 profile. Scaling the trace keeps the
arrival rate, so a 10× trace simply lasts 10× longer; scaling the fleet scales the arrival rate with
it, so each replica sees the same load.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "results" / "scale"

CASE = r"""
import json, resource, sys, time
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
sessions, rate = int(sys.argv[1]), float(sys.argv[2])
replicas, policy = int(sys.argv[3]), sys.argv[4]
profile = load_profile("configs/profiles/t4-qwen3-4b-vllm0.30.0.json")
t0 = time.perf_counter()
trace, rep = AgenticWorkload(n_sessions=sessions, turns_per_session=4, session_context_tokens=150,
                             user_turn_tokens=40, assistant_turn_tokens=60, output_tokens=64,
                             think_time_s=6.0, seed=0, prefix_sharing_ratio=0.8,
                             arrival_rate=rate).generate()
t1 = time.perf_counter()
rss_trace = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
r = run(trace, policy=policy, n_replicas=replicas, seed=0, profile=profile)
t2 = time.perf_counter()
n_hashes = sum(len(q.block_hashes) for q in trace)
rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"requests": rep["requests"], "block_hashes": n_hashes,
                  "generate_s": round(t1 - t0, 2), "simulate_s": round(t2 - t1, 2),
                  "peak_rss_mib_after_trace": round(rss_trace / 1024, 1),
                  "peak_rss_mib": round(rss_peak / 1024, 1),
                  "hit_rate": round(r.hit_rate, 4), "completed": r.completed}))
"""

# (label, sessions, sessions/s, replicas)
CASES = [
    ("1× trace", 200, 4.0, 8),
    ("10× trace", 2_000, 4.0, 8),
    ("100× trace", 20_000, 4.0, 8),
    ("10× replicas (and 10× load)", 2_000, 40.0, 80),
]
POLICIES = ("round-robin", "radix-affinity-balanced")


def _report(payload: dict[str, Any]) -> str:
    cpus = payload["hardware"].get("usable_cpus", "?")
    where = (f"`scripts/measure_scale.py` at `{payload['git_sha']}`"
             f"{' (dirty tree)' if payload['git_dirty'] else ''}; each case is one "
             f"single-threaded process on a machine with {cpus} usable CPUs.")
    lines = ["# What breaks at 10× and 100× — measured", "", where, "",
             "| case | policy | requests | block hashes | generate | simulate | peak RSS |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in payload["cases"]:
        lines.append(f"| {r['case']} | {r['policy']} | {r['requests']:,} | {r['block_hashes']:,} | "
                     f"{r['generate_s']:.1f} s | {r['simulate_s']:.1f} s | "
                     f"{r['peak_rss_mib']:,.0f} MiB |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-only", action="store_true",
                    help="re-render report.md from the existing scale.json; run nothing")
    args = ap.parse_args()
    if args.report_only:
        (OUT / "report.md").write_text(_report(json.loads((OUT / "scale.json").read_text())))
        return 0

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    payload = results_header("scripts/measure_scale.py", seed=0, hardware_path="cpu-only")
    rows = []
    for label, sessions, rate, replicas in CASES:
        for policy in POLICIES:
            out = subprocess.run([sys.executable, "-c", CASE, str(sessions), str(rate),
                                  str(replicas), policy], cwd=REPO, capture_output=True,
                                 text=True, check=True)
            row = {"case": label, "sessions": sessions, "rate": rate, "replicas": replicas,
                   "policy": policy, **json.loads(out.stdout.strip().splitlines()[-1])}
            rows.append(row)
            print(f"{label:28s} {policy:24s} {row['requests']:>7,} requests · generate "
                  f"{row['generate_s']:6.1f}s · simulate {row['simulate_s']:6.1f}s · peak RSS "
                  f"{row['peak_rss_mib']:7.1f} MiB")
    payload["cases"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "report.md").write_text(_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
