#!/usr/bin/env python3
"""T1 — is the simulator's per-token latency right? Checked on the held-out runs already on disk.

    python scripts/t1_itl_check.py      # configs/tpot_slo.yaml -> artifacts/results/t1_itl/

Registered before it ran (configs/tpot_slo.yaml, and a dated entry in docs/09-hypotheses.md). The
procedure, exactly:
  1. Take the five held-out traffic runs of C2 (part D) and C3 (part 2): one live T4 each. For each
     the notebook recorded the engine's own mean inter-token latency over the run — Δsum ÷ Δcount of
     vllm:inter_token_latency_seconds, one observation per token after a request's first.
  2. Replay each run through the simulator on one replica, every request at the time it was really
     sent — C2c's replay — with profile A, the shipped C2-fitted one. Read the same quantity off the
     simulator: Σ inter-token time ÷ Σ gaps, over every request (`RunResult.engine_itl_mean_ms`).
  3. T1 holds only if all five predictions are within `max_factor` of the measurements (either
     direction). A run without a measurement leaves it UNDECIDED, and undecided does not hold.

⚠️ The measurements were on disk, and three were read, before this was registered. The simulator's
side had never been computed: it could not measure time per token until commit 534f5b9.

⚠️ What it cannot show: C3's three runs are themselves beyond a 100 ms per-token limit, and C2's two
are the same workloads. T1 checks the model where the engine was loaded more heavily than any
configuration T2 will call acceptable. The report counts how many runs are beyond the limit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c2_analyze as c2  # the registered replay's percentile and factor: imported, never copied
from _spike_common import results_header

from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.sim.metrics import RunResult
from kv_routing_sim.traces.generator import AgenticWorkload

CONFIG = REPO / "configs" / "tpot_slo.yaml"
OUT = REPO / "artifacts" / "results" / "t1_itl"


def replay(spec: dict, measured: dict, profile: CostProfile) -> RunResult:
    """C2c's replay — scripts/c2_analyze.py `replay`, step for step — returning the whole run, so
    the per-token side can be read off it. `one()` checks it against the registered function."""
    sent = {r["request_id"]: max(0.0, r["send_offset_s"]) for r in measured["requests"]}
    trace, _ = AgenticWorkload(**spec).generate()
    trace = sorted((dataclasses.replace(q, arrival_ts=sent[q.request_id]) for q in trace),
                   key=lambda q: q.arrival_ts)
    return run(trace, policy="round-robin", n_replicas=1, seed=0, profile=profile,
               cache_model="vllm", record_per_request=True)


def one(name: str, spec: dict, measured: dict, profile: CostProfile) -> dict[str, Any]:
    """One held-out run against one profile: the engine's mean inter-token latency and the
    simulator's, and the first-token numbers beside them."""
    r = replay(spec, measured, profile)
    valid = {x["request_id"] for x in measured["requests"] if x["valid"]}
    ttfts = sorted(v["ttft_ms"] for k, v in r.per_request.items() if k in valid)
    pred = {"ttft_p50_ms": c2.nearest_rank(ttfts, 0.50),
            "ttft_p95_ms": c2.nearest_rank(ttfts, 0.95)}
    ref = c2.replay(name, spec, measured, profile)
    if any(pred[k] != ref[k] for k in pred):
        raise SystemExit(f"error: {name}: this replay is not C2c's — {pred} against {ref}")
    seen = sorted(x["ttft_ms"] for x in measured["requests"] if x["valid"] and x["ttft_ms"])
    meas = {"ttft_p50_ms": c2.nearest_rank(seen, 0.50), "ttft_p95_ms": c2.nearest_rank(seen, 0.95)}
    itl = measured.get("server_itl_s")
    meas_itl_ms = None if itl is None else itl * 1e3
    tpot = r.tpot_percentiles()
    return {
        "measured_itl_ms": meas_itl_ms,
        "predicted_itl_ms": r.engine_itl_mean_ms(),
        "factor": c2.factor(r.engine_itl_mean_ms(), meas_itl_ms),
        "predicted_tpot_p50_ms": tpot["p50_ms"], "predicted_tpot_p95_ms": tpot["p95_ms"],
        "gaps": r.itl_count,
        "preemptions": {"measured": measured["preemptions"], "predicted": r.preemptions},
        "ttft": {"measured": meas, "predicted": pred,
                 "factor": {q: c2.factor(pred[f"ttft_{q}_ms"], meas[f"ttft_{q}_ms"])
                            for q in ("p50", "p95")}},
    }


def analyze(cfg: dict[str, Any]) -> dict[str, Any]:
    t1 = cfg["t1"]
    tested = load_profile(REPO / t1["profile"])
    others = {k: load_profile(REPO / v) for k, v in t1["reported_profiles"].items()}
    rows = []
    for src in t1["runs"]:
        data = json.loads((REPO / src["results"]).read_text())
        specs = json.loads((REPO / src["prereg"]).read_text())["workloads"]
        by_name = {m["workload"]: m for m in data.get("holdout", [])}
        for name in src["workloads"]:
            m = by_name.get(name)
            row: dict[str, Any] = {"session": src["session"], "workload": name}
            if m is None:
                rows.append({**row, "A": None})
                continue
            row["A"] = one(name, specs[name], m, tested)
            # Reported: the profiles `kvsim calibrate` wrote in C3, on the traffic of that session.
            if src["session"] == "c3":
                row.update({k: one(name, specs[name], m, p) for k, p in others.items()})
            rows.append(row)
    factors = [r["A"]["factor"] if r["A"] else None for r in rows]
    decided = bool(rows) and all(f is not None for f in factors)
    return {
        "script": "scripts/t1_itl_check.py", "max_factor": t1["max_factor"], "runs": rows,
        "worst_factor": max(factors) if decided else None, "decided": decided,
        "holds": decided and all(f <= t1["max_factor"] for f in factors),
        "predicted_outcome": cfg["predictions"]["t1"],
        "tpot_limit_ms": cfg["slo"]["tpot_p95_ms"],
    }


def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f} ms"


def _x(f: float | None) -> str:
    return "—" if f is None else f"{f:.2f}×"


def markdown(a: dict[str, Any], header: dict[str, Any]) -> str:
    verdict = ("HOLDS" if a["holds"] else "DOES NOT HOLD") if a["decided"] else \
        "UNDECIDED (a measurement is missing)"
    lines = [
        "# T1 — the simulator's per-token latency against the engine's",
        "",
        (f"Registered `configs/tpot_slo.yaml` · five held-out runs on one live T4 each (C2 part D, "
         f"2026-09-30; C3 part 2, 2026-10-05), replayed at the times their requests were sent · "
         f"generated `{header['date_utc']}` at `{header['git_sha']}`"
         f"{' (dirty tree)' if header['git_dirty'] else ''} by `scripts/t1_itl_check.py`."),
        "",
        (f"**T1: {verdict}** (registered prediction: {a['predicted_outcome']}). It holds only if "
         f"every factor in the table is ≤ {a['max_factor']:g}"
         + (f"; the worst is {a['worst_factor']:.2f}×." if a["decided"] else ".")),
        "",
        ("*Mean inter-token latency* = Σ time between a request's consecutive tokens ÷ the number "
         "of such gaps, over every request of the run. Engine: Δsum ÷ Δcount of "
         "`vllm:inter_token_latency_seconds`. Simulator: the same sum and count."),
        "",
        ("| session · workload | engine | simulator, profile A | factor | preemptions engine · "
         "simulator | first-token factor p50 · p95 |"),
        "|---|---:|---:|---:|---|---|",
    ]
    for r in a["runs"]:
        x = r["A"]
        if x is None:
            lines.append(f"| {r['session']} · {r['workload']} | not on disk | | | | |")
            continue
        t = x["ttft"]["factor"]
        lines.append(
            f"| {r['session']} · {r['workload']} | {_ms(x['measured_itl_ms'])} | "
            f"{_ms(x['predicted_itl_ms'])} | {_x(x['factor'])} | "
            f"{x['preemptions']['measured']} · {x['preemptions']['predicted']} | "
            f"{_x(t['p50'])} · {_x(t['p95'])} |")
    seen = [r["A"]["measured_itl_ms"] for r in a["runs"]
            if r["A"] and r["A"]["measured_itl_ms"] is not None]
    beyond = sum(1 for x in seen if x > a["tpot_limit_ms"])
    lines += [
        "",
        (f"⚠️ **What this cannot show.** A per-token limit of {a['tpot_limit_ms']:g} ms is the "
         f"target T2 sizes on, and {beyond} of the {len(seen)} engine numbers above are beyond "
         "it. So this checks the model where the engine was loaded more heavily than a "
         "configuration T2 calls acceptable; below the limit it is untested on traffic."),
        "",
        "## Reported, not tested",
        "",
        ("The simulator's per-request time per output token on the same replays (profile A). No "
         "engine measurement of it exists yet — the notebooks kept only the mean above:"),
        "",
        "| session · workload | p50 | p95 |",
        "|---|---:|---:|",
    ]
    for r in a["runs"]:
        if r["A"]:
            lines.append(f"| {r['session']} · {r['workload']} | "
                         f"{_ms(r['A']['predicted_tpot_p50_ms'])} | "
                         f"{_ms(r['A']['predicted_tpot_p95_ms'])} |")
    extra = [r for r in a["runs"] if "B" in r]
    if extra:
        lines += [
            "",
            ("The same check with the two profiles `kvsim calibrate` wrote in C3 (B: its first "
             "run; C: its second), on that session's traffic:"),
            "",
            "| workload | engine | B | factor | C | factor |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in extra:
            lines.append(
                f"| {r['workload']} | {_ms(r['A']['measured_itl_ms'])} | "
                f"{_ms(r['B']['predicted_itl_ms'])} | {_x(r['B']['factor'])} | "
                f"{_ms(r['C']['predicted_itl_ms'])} | {_x(r['C']['factor'])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    header = results_header("scripts/t1_itl_check.py", seed=0, hardware_path="cpu-only",
                            extra={"config": {k: cfg[k] for k in ("slo", "t1", "predictions")}})
    a = analyze(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "t1.json").write_text(json.dumps({**header, **a}, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(markdown(a, header))
    print((args.out / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
