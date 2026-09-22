#!/usr/bin/env python3
"""Phase 4's first CALIBRATED sweep — and its pre-registered tests of H1 and H5.

    python scripts/run_phase4_sweep.py        # configs/phase4_sweep.yaml
                                              #   -> artifacts/results/phase4_sweep_t4/

The design and every decision rule were committed before this ran (configs/phase4_sweep.yaml, plus a
dated amendment in docs/09-hypotheses.md). This script IS that procedure: it chooses nothing after
seeing a result.

Tier (ADR-010): the profile is calibrated — Phase 3's fitted T4 timings — so milliseconds and rupees
here are **T4 figures**, and are not predictions for any other GPU. Whether the policy RANKING
transfers to other hardware is H6's question, tested separately.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache, lru_cache
from itertools import pairwise
from typing import Any

import yaml

from kv_routing_sim import policies as policy_pkg
from kv_routing_sim.eval.bootstrap import paired_bootstrap_delta
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.recommend import recommend
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "configs" / "phase4_sweep.yaml"


# ── one run, in a worker process ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@cache
def _trace(name: str, load: float):
    spec = {k: v for k, v in _cfg()["traces"][name].items()}
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()


@lru_cache(maxsize=1)
def _profile():
    return load_profile(REPO / _cfg()["profile"])


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _cfg()["cost_model"])


def _job(key: tuple) -> tuple:
    name, load, policy, n, seed, keep = key
    trace, _ = _trace(name, load)
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(),
            cache_model=_cfg()["cache_model"], record_per_request=keep)
    per_request = None
    if keep:  # (session, cached tokens, prompt tokens) — the H1 bootstrap's raw material
        meta = {q.request_id: (q.session_id, q.prompt_tokens) for q in trace}
        per_request = [(meta[rid][0], v["cached_tokens"], meta[rid][1])
                       for rid, v in sorted(r.per_request.items())]
    extra = {"preemptions": r.preemptions, "p99_ms": r.ttft_percentiles()["p99_ms"],
             "completed": r.completed}
    return key, to_point(r, _cost()), extra, per_request


# ── the decision rules ────────────────────────────────────────────────────────────────────────


def _ratio(items: list[tuple[int, int]]) -> float:
    return sum(c for c, _ in items) / sum(p for _, p in items)


def _by_session(rows: list[tuple[str, int, int]]) -> list[list[tuple[int, int]]]:
    g: dict[str, list[tuple[int, int]]] = {}
    for sid, cached, prompt in rows:
        g.setdefault(sid, []).append((cached, prompt))
    return [g[k] for k in sorted(g)]


def _gt(a: int | None, b: int | None) -> bool:
    """a > b, where None means "not met at 8" and counts as more than any number."""
    if a is None:
        return b is not None
    return b is not None and a > b


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "phase4_sweep_t4")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _cfg.cache_clear()
    cfg = _cfg()

    pols = policy_pkg.resolve(cfg["policies"])
    replicas, seeds, loads = cfg["replicas"], cfg["seeds"], cfg["loads"]
    h1, h5 = cfg["h1"], cfg["h5"]
    h1_pols = (*h1["policies"], h1["versus"])

    def keep(name: str, policy: str, n: int, seed: int) -> bool:
        return name == h1["trace"] and seed == 0 and policy in h1_pols and n in h1["replicas"]

    jobs = [(name, load, p, n, s, keep(name, p, n, s))
            for name in cfg["traces"] for load in loads for p in pols
            for n in replicas for s in seeds]
    t0 = time.time()
    # Workers re-read the config by path, so they must be handed the same one the parent uses.
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    runtime = time.time() - t0
    print(f"{len(jobs)} runs in {runtime:.0f}s on {args.workers} workers")

    cells: dict[tuple, list[Point]] = {}
    extras: dict[tuple, list[dict]] = {}
    per_request: dict[tuple, list] = {}
    for (name, load, p, n, _s, _k), point, extra, pr in results:
        cells.setdefault((name, load, p, n), []).append(point)
        extras.setdefault((name, load, p, n), []).append(extra)
        if pr is not None:
            per_request[(name, load, p, n)] = pr
    agg = {k: average_points(v) for k, v in cells.items()}

    def required(name: str, load: float, policy: str) -> int | None:
        return next((n for n in sorted(replicas) if agg[(name, load, policy, n)].meets_slo), None)

    req = {name: {str(load): {p: required(name, load, p) for p in pols} for load in loads}
           for name in cfg["traces"]}

    # H1 — high sharing: affinity vs least-loaded, paired by session, at every load.
    h1_out: dict[str, Any] = {}
    for pol in h1["policies"]:
        per_load = {}
        for load in loads:
            rows = []
            for n in h1["replicas"]:
                a = _by_session(per_request[(h1["trace"], load, pol, n)])
                b = _by_session(per_request[(h1["trace"], load, h1["versus"], n)])
                ci = paired_bootstrap_delta(a, b, _ratio, n_resamples=h1["bootstrap_resamples"],
                                            seed=0)
                rows.append({"replicas": n, **ci.as_dict()})
            deltas = [r["point"] for r in rows]
            strictly = all(r["ci_lo"] > 0 for r in rows)
            widens = deltas[-1] > deltas[0] and all(
                b - a >= -h1["widen_tolerance"] for a, b in pairwise(deltas))
            per_load[str(load)] = {"strictly_higher": strictly, "widens": widens, "rows": rows}
        h1_out[pol] = {"holds": all(v["strictly_higher"] and v["widens"]
                                    for v in per_load.values()),
                       "per_load": per_load}

    # H5 — per trace: fewer replicas somewhere, more nowhere.
    h5_out: dict[str, Any] = {}
    for name in cfg["traces"]:
        h5_out[name] = {}
        for pol in h5["policies"]:
            pairs = {str(load): [req[name][str(load)][h5["versus"]], req[name][str(load)][pol]]
                     for load in loads}
            fewer = [ld for ld, (rr, p) in pairs.items() if _gt(rr, p)]
            more = [ld for ld, (rr, p) in pairs.items() if _gt(p, rr)]
            h5_out[name][pol] = {"holds": bool(fewer) and not more,
                                 "loads_where_it_needs_fewer": fewer,
                                 "loads_where_it_needs_more": more,
                                 "required_replicas_rr_vs_policy": pairs}

    recs = {}
    for name in cfg["traces"]:
        for load in loads:
            rec = recommend([agg[(name, load, p, n)] for p in pols for n in replicas])
            recs[f"{name}@{load}"] = rec.as_dict()

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    payload = results_header("scripts/run_phase4_sweep.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg, "runtime_s": round(runtime, 1)})
    payload["profile"] = _profile().provenance()
    payload["cost_model"] = _cost().provenance
    payload["trace_reports"] = {f"{n}@{ld}": _trace(n, ld)[1] for n in cfg["traces"]
                                for ld in loads}
    payload["cells"] = [
        {"trace": k[0], "load": k[1], **agg[k].as_dict(),
         "preemptions_mean": sum(e["preemptions"] for e in extras[k]) / len(extras[k]),
         "p99_ms_mean": sum(e["p99_ms"] for e in extras[k]) / len(extras[k])}
        for k in sorted(agg, key=lambda k: (k[0], k[1], pols.index(k[2]), k[3]))
    ]
    payload["required_replicas"] = req
    payload["h1"] = h1_out
    payload["h5"] = h5_out
    payload["recommendations"] = recs
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(_markdown(payload, pols, loads))
    print((args.out / "report.md").read_text())
    return 0


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()


def _markdown(pl: dict[str, Any], pols: list[str], loads: list[float]) -> str:
    cfg = pl["config"]
    lines = [
        "# Phase 4 — the first calibrated sweep",
        "",
        (
            f"Profile `{pl['profile']['name']}` ({pl['profile']['kv_cache_blocks']:,} blocks, "
            f"fitted T4 timings) · cost `{pl['cost_model']['compute']['sku']}` at "
            f"${pl['cost_model']['compute']['price_per_hour_usd']}/h · SLO p95 TTFT ≤ "
            f"{pl['cost_model']['slo']['ttft_p95_ms']:,} ms · {len(cfg['seeds'])} seeds · "
            f"generated `{pl['date_utc']}` at `{pl['git_sha']}`"
            f"{' (dirty tree)' if pl['git_dirty'] else ''}."
        ),
        "",
        (
            "⚠️ **T4 figures.** Milliseconds and rupees are measured-profile numbers for one "
            "Tesla T4 per replica; whether the *ranking* transfers is H6's question."
        ),
        "",
        "## Replicas each policy needs to hold the SLO",
        "",
    ]
    for name in cfg["traces"]:
        lines += [f"### {name}", "",
                  "| policy | " + " | ".join(f"{ld}/s" for ld in loads) + " |",
                  "|---|" + "---|" * len(loads)]
        for p in pols:
            cells = [pl["required_replicas"][name][str(ld)][p] for ld in loads]
            lines.append(f"| {p} | " + " | ".join("—" if c is None else str(c)
                                                   for c in cells) + " |")
        lines += ["", "— = not met at 8 replicas.", ""]
    lines += ["## Why — how evenly each policy spreads load, and how often it preempts", "",
              ("*Descriptive, not a registered test.* Each cell: load imbalance (max ÷ mean "
               "requests per replica) · preemptions per run, at the replica count round-robin "
               "needs at that load (8 where it never meets the SLO)."), ""]
    cell_by = {(c["trace"], c["load"], c["policy"], c["n_replicas"]): c for c in pl["cells"]}
    for name in cfg["traces"]:
        lines += [f"### {name}", "",
                  "| policy | " + " | ".join(f"{ld}/s" for ld in loads) + " |",
                  "|---|" + "---|" * len(loads)]
        for p in pols:
            cells = []
            for ld in loads:
                n = pl["required_replicas"][name][str(ld)]["round-robin"] or max(cfg["replicas"])
                c = cell_by[(name, ld, p, n)]
                cells.append(f"{c['load_imbalance']:.2f} · {c['preemptions_mean']:.0f}")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Recommendation at each load (cheapest configuration meeting the SLO)", "",
              "| trace @ load | recommended | ₹ / 1k requests | vs round-robin |",
              "|---|---|---|---|"]
    for key, r in pl["recommendations"].items():
        c = r["recommended"]
        inr = "—" if c["inr_per_1k_requests"] is None else f"{c['inr_per_1k_requests']:.2f}"
        vs = "—"
        if r["inr_delta_pct_vs_baseline"] is not None:
            vs = (f"{r['inr_delta_pct_vs_baseline']:+.1f}% · "
                  f"{r['replica_delta_vs_baseline']:+d} replicas")
            if not r["baseline"]["meets_slo"]:
                vs += " (round-robin misses the SLO even at 8 — vs its best attempt)"
        ok = "" if c["meets_slo"] else " ⚠️ nothing met the SLO"
        lines.append(f"| {key} | {c['policy']} × {c['n_replicas']}{ok} | {inr} | {vs} |")
    lines += ["", "## H1 — prefix-affinity vs least-loaded on hit rate, high sharing", ""]
    for pol, v in pl["h1"].items():
        lines.append(f"**{pol}: {'HOLDS' if v['holds'] else 'FALSIFIED'}**")
        lines += ["", "| load | " + " | ".join(f"{r['replicas']} rep" for r in
                                                 next(iter(v['per_load'].values()))['rows'])
                  + " | strictly higher | widens |",
                  "|---|" + "---|" * (len(next(iter(v['per_load'].values()))['rows']) + 2)]
        for ld, pv in v["per_load"].items():
            cells = [f"{r['point']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                     for r in pv["rows"]]
            lines.append(f"| {ld}/s | " + " | ".join(cells) + f" | {pv['strictly_higher']} | "
                         f"{pv['widens']} |")
        lines.append("")
    lines += ["Δ hit rate (policy − least-loaded), paired session bootstrap, 95% CI.", "",
              "## H5 — does prefix-affinity hold the SLO with fewer replicas?", ""]
    for name, per in pl["h5"].items():
        for pol, v in per.items():
            lines.append(f"- **{name} · {pol}: {'HOLDS' if v['holds'] else 'FALSIFIED'}** — "
                         f"fewer at {v['loads_where_it_needs_fewer'] or 'no load'}; "
                         f"more at {v['loads_where_it_needs_more'] or 'no load'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
