#!/usr/bin/env python3
"""Phase 5 — the crossover: A1 (the alpha sweep), H2 (as registered) and H2b (new), in that order.

    python scripts/run_phase5.py
        # configs/phase5_crossover.yaml -> artifacts/results/phase5_crossover/

The design and every decision rule were committed before any of this ran (the config, plus a dated
entry in docs/09-hypotheses.md). A1 runs first and picks alpha* by the registered rule; alpha* is
then FIXED for H2 and H2b. This script chooses nothing after seeing a result.

Tier (ADR-010): the profile is calibrated — Phase 3's fitted T4 timings — so milliseconds and rupees
are T4 figures. The traffic is generated, not real, and there is one model.
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
from itertools import combinations
from typing import Any

import yaml

from kv_routing_sim import policies as policy_pkg
from kv_routing_sim.eval.bootstrap import p95, paired_bootstrap_delta
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "configs" / "phase5_crossover.yaml"
NOT_MET = 9  # no replica count from 1 to 8 holds the SLO
BAL = "radix-affinity-balanced"
LL, P2C, STICKY = "least-loaded", "power-of-two", "session-sticky"


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


# ── one run, in a worker process ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@lru_cache(maxsize=1)
def _phase4() -> dict[str, Any]:
    return yaml.safe_load((REPO / _cfg()["alpha"]["base_sweep"]).read_text())


def _spec(tid: tuple) -> dict[str, Any]:
    """Generator arguments for one trace id. Every trace here is (config + seed), never data."""
    kind, value = tid
    if kind == "phase4":
        return dict(_phase4()["traces"][value])
    shape = dict(_cfg()["session_shape"])
    if kind == "h2":
        return {**shape, "prefix_sharing_ratio": value}
    h2b = _cfg()["h2b"]
    return {**shape, "prefix_sharing_ratio": h2b["sharing"], "n_shared_prefixes": value,
            "prefix_zipf_s": h2b["prefix_zipf_s"]}


@cache
def _trace(tid: tuple, load: float):
    return AgenticWorkload(**_spec(tid), arrival_rate=float(load)).generate()


@lru_cache(maxsize=1)
def _profile() -> CostProfile:
    return load_profile(REPO / _cfg()["profile"])


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _cfg()["cost_model"])


def _job(key: tuple) -> tuple:
    tid, load, policy, alpha, n, seed, keep = key
    trace, _ = _trace(tid, load)
    kwargs = None if alpha is None else {"load_penalty": float(alpha)}
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(),
            cache_model=_cfg()["cache_model"], policy_kwargs=kwargs, record_per_request=keep)
    by_session = None
    if keep:  # per-session TTFTs: the paired bootstrap's raw material (sessions are the unit)
        sid = {q.request_id: q.session_id for q in trace}
        by_session = {}
        for rid, v in r.per_request.items():
            by_session.setdefault(sid[rid], []).append(v["ttft_ms"])
    return key, to_point(r, _cost()), r.preemptions, by_session


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()
    _phase4.cache_clear()


def _sweep(jobs: list[tuple], workers: int) -> tuple[dict, dict, dict]:
    """(cell -> seed-averaged Point, cell -> mean preemptions, cell -> per-session TTFTs)."""
    # Workers re-read the config by path, so they must be handed the same one the parent uses.
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    points: dict[tuple, list[Point]] = {}
    preempt: dict[tuple, list[int]] = {}
    sessions: dict[tuple, dict] = {}
    for (tid, load, policy, alpha, n, _seed, _keep), point, pre, by_session in results:
        cell = (tid, load, policy, alpha, n)
        points.setdefault(cell, []).append(point)
        preempt.setdefault(cell, []).append(pre)
        if by_session is not None:
            sessions[cell] = by_session
    agg = {k: average_points(v) for k, v in points.items()}
    return agg, {k: sum(v) / len(v) for k, v in preempt.items()}, sessions


def _required(agg: dict, tid: tuple, load: float, policy: str, alpha: float | None,
              replicas: list[int]) -> int:
    met = (n for n in sorted(replicas) if agg[(tid, load, policy, alpha, n)].meets_slo)
    return next(met, NOT_MET)


# ── the decision rules ────────────────────────────────────────────────────────────────────────


def _h2_map(sessions: dict, pol: tuple, n: int, sharing: list[float], loads: list[float],
            resamples: int) -> dict[str, Any]:
    """H2 at one replica count for one affinity policy P: the win map and the four checks."""
    cells: dict[str, Any] = {}
    win: dict[str, dict[tuple, str]] = {LL: {}, P2C: {}}
    for r in sharing:
        for ld in loads:
            tid = ("h2", r)
            gp = sessions[(tid, ld, *pol, n)]
            keys = sorted(gp)
            cell = {}
            for x in (LL, P2C):
                gx = sessions[(tid, ld, x, None, n)]
                ci = paired_bootstrap_delta([gx[k] for k in keys], [gp[k] for k in keys], p95,
                                            n_resamples=resamples, seed=0)
                w = "P" if ci.lo > 0 else ("X" if ci.hi < 0 else "tie")
                win[x][(r, ld)] = w
                cell[x] = {**ci.as_dict(), "winner": w}
            cells[f"{r}@{ld}"] = cell
    ll = win[LL]
    crossover = "P" in ll.values() and "X" in ll.values()
    along_sharing = [[ld, r1, r2] for ld in loads
                     for r1, r2 in combinations(sharing, 2)
                     if ll[(r1, ld)] == "P" and ll[(r2, ld)] == "X"]
    along_load = [[r, l1, l2] for r in sharing
                  for l1, l2 in combinations(loads, 2)
                  if ll[(r, l1)] == "X" and ll[(r, l2)] == "P"]
    p2c_order = []
    for r in sharing:
        first_ll = next((ld for ld in loads if ll[(r, ld)] == "X"), None)
        first_p2c = next((ld for ld in loads if win[P2C][(r, ld)] == "X"), None)
        if first_p2c is not None and (first_ll is None or first_p2c < first_ll):
            p2c_order.append([r, first_ll, first_p2c])
    return {
        "cells": cells,
        "crossover": crossover,
        "violations_along_sharing": along_sharing,
        "violations_along_load": along_load,
        "violations_power_of_two": p2c_order,
        "holds": crossover and not along_sharing and not along_load and not p2c_order,
    }


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "phase5_crossover")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _cfg.cache_clear()
    _phase4.cache_clear()
    cfg = _cfg()
    replicas, loads = cfg["replicas"], cfg["loads"]
    a, h2, h2b = cfg["alpha"], cfg["h2"], cfg["h2b"]
    reported = policy_pkg.resolve(cfg["policies_reported"])

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken BEFORE the runs: the header must describe the tree that produced the numbers.
    payload = results_header("scripts/run_phase5.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg})
    t0 = time.time()

    # ── A1: the alpha sweep on Phase 4's traces ──
    names = list(_phase4()["traces"])
    a_jobs = [(("phase4", nm), ld, BAL, al, n, s, False) for nm in names for ld in loads
              for al in a["values"] for n in replicas for s in a["seeds"]]
    agg_a, _, _ = _sweep(a_jobs, args.workers)
    r_alpha = {al: {nm: {str(ld): _required(agg_a, ("phase4", nm), ld, BAL, al, replicas)
                         for ld in loads} for nm in names} for al in a["values"]}
    p4 = None
    if a.get("phase4_results"):
        p4 = json.loads((REPO / a["phase4_results"]).read_text())["required_replicas"]
        if 1 in a["values"]:  # Guard (registered): alpha = 1 is Phase 4's policy, exactly.
            for nm in names:
                for ld in loads:
                    want = p4[nm][str(ld)][BAL]
                    want = NOT_MET if want is None else want
                    if r_alpha[1][nm][str(ld)] != want:
                        raise SystemExit(
                            f"error: alpha = 1 does not reproduce Phase 4 at {nm}@{ld}/s: "
                            f"{r_alpha[1][nm][str(ld)]} vs {want}. The code has drifted; stopping."
                        )
            print("alpha = 1 reproduces Phase 4's radix-affinity-balanced row exactly")
    totals = {al: sum(r_alpha[al][nm][str(ld)] for nm in names for ld in loads)
              for al in a["values"]}
    alpha_star = min(a["values"], key=lambda al: (totals[al], al))
    print(f"A1: alpha* = {alpha_star:g} (total replicas {totals[alpha_star]})")

    # ── H2 and H2b, with alpha* fixed ──
    variants = [(p, None) for p in reported] + [(BAL, alpha_star)]
    tested = {(LL, None), (P2C, None), ("radix-affinity", None), (BAL, alpha_star)}

    def keep(pol: tuple, n: int, seed: int) -> bool:
        return seed == 0 and n in h2["test_replicas"] and pol in tested

    h2_jobs = [(("h2", r), ld, p, al, n, s, keep((p, al), n, s)) for r in h2["sharing"]
               for ld in loads for p, al in variants for n in replicas for s in h2["seeds"]]
    oracle = [(BAL, al) for al in a["values"] if al != alpha_star]
    ks = h2b["n_shared_prefixes"]
    h2b_jobs = [(("h2b", k), ld, p, al, n, s, False) for k in ks for ld in loads
                for p, al in variants + oracle for n in replicas for s in h2b["seeds"]]
    agg, preempt, sessions = _sweep(h2_jobs + h2b_jobs, args.workers)
    runtime = time.time() - t0
    n_runs = len(a_jobs) + len(h2_jobs) + len(h2b_jobs)
    print(f"{n_runs} runs in {runtime:.0f}s on {args.workers} workers")

    # H2 — the win maps, per affinity policy and replica count
    h2_out: dict[str, Any] = {}
    for pol in (("radix-affinity", None), (BAL, alpha_star)):
        per_n = {str(n): _h2_map(sessions, pol, n, h2["sharing"], loads,
                                 h2["bootstrap_resamples"]) for n in h2["test_replicas"]}
        h2_out[label(*pol)] = {"per_replicas": per_n,
                               "holds": all(v["holds"] for v in per_n.values())}
    def p95_ms(p: str, al: float | None, n: int, r: float, ld: float) -> float:
        return round(agg[(("h2", r), ld, p, al, n)].p95_ttft_ms, 1)

    h2_p95 = {label(p, al): {str(n): {f"{r}@{ld}": p95_ms(p, al, n, r, ld)
                                      for r in h2["sharing"] for ld in loads}
                             for n in h2["test_replicas"]}
              for p, al in variants}
    h2_req = {label(p, al): {f"{r}@{ld}": _required(agg, ("h2", r), ld, p, al, replicas)
                             for r in h2["sharing"] for ld in loads} for p, al in variants}

    # H2b — the working set
    usable = _profile().kv_cache_blocks - 1
    rb = {str(k): {str(ld): {label(p, al): _required(agg, ("h2b", k), ld, p, al, replicas)
                             for p, al in variants + oracle} for ld in loads} for k in ks}
    star = label(BAL, alpha_star)
    win_b = {}
    for k in ks:
        for ld in loads:
            ra, rs = rb[str(k)][str(ld)][star], rb[str(k)][str(ld)][STICKY]
            win_b[(k, ld)] = "affinity" if ra < rs else ("sticky" if rs < ra else "tie")
    cond_a = all(win_b[(ks[0], ld)] != "affinity" for ld in loads)
    cond_b = any(win_b[(ks[-1], ld)] == "affinity" for ld in loads)
    mono = [[ld, k1, k2] for ld in loads for k1, k2 in combinations(ks, 2)
            if win_b[(k1, ld)] == "affinity" and win_b[(k2, ld)] == "sticky"]
    best_alpha = {str(k): {str(ld): min(a["values"], key=lambda al, k=k, ld=ld: (
        rb[str(k)][str(ld)][label(BAL, al)], al)) for ld in loads} for k in ks}
    n_hit = 4 if 4 in replicas else max(replicas)  # the registered config sweeps 1-8, so: 4

    def hit(k: int, ld: float, p: str, al: float | None) -> float:
        return round(agg[(("h2b", k), ld, p, al, n_hit)].hit_rate, 4)

    hit_rates = {str(k): {str(ld): {label(p, al): hit(k, ld, p, al) for p, al in variants}
                          for ld in loads} for k in ks}
    reports_b = {str(k): _trace(("h2b", k), loads[0])[1] for k in ks}

    payload["runtime_s"] = round(runtime, 1)
    payload["n_runs"] = n_runs
    payload["profile"] = _profile().provenance()
    payload["cost_model"] = _cost().provenance
    payload["a1"] = {
        "required_replicas": r_alpha,
        "total_replicas": totals,
        "alpha_star": alpha_star,
        "phase4_required_replicas": p4,
    }
    payload["h2"] = {
        "tests": h2_out,
        "holds_radix_affinity": h2_out["radix-affinity"]["holds"],
        "p95_ms_by_policy": h2_p95,
        "required_replicas": h2_req,
        "trace_reports": {str(r): _trace(("h2", r), loads[0])[1] for r in h2["sharing"]},
    }
    payload["h2b"] = {
        "required_replicas": rb,
        "winner": {f"{k}@{ld}": w for (k, ld), w in win_b.items()},
        "cond_a_no_affinity_win_at_smallest_k": cond_a,
        "cond_b_affinity_win_at_largest_k": cond_b,
        "violations_monotone_in_k": mono,
        "holds": cond_a and cond_b and not mono,
        "best_alpha_per_cell": best_alpha,
        "hit_rate_replicas": n_hit,
        "hit_rate": hit_rates,
        "trace_reports": reports_b,
        "usable_blocks_per_replica": usable,
    }
    payload["mean_preemptions"] = {f"{label(c[2], c[3])} {c[0]}@{c[1]} n={c[4]}": v
                                   for c, v in preempt.items() if c[4] == 4}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase5.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(_markdown(payload))
    print((args.out / "report.md").read_text())
    return 0


def _row(values: list[int]) -> str:
    return " ".join("—" if v == NOT_MET else str(v) for v in values)


def _markdown(pl: dict[str, Any]) -> str:
    cfg = pl["config"]
    loads, a, h2, h2b = cfg["loads"], cfg["alpha"], cfg["h2"], cfg["h2b"]
    a1, star = pl["a1"], pl["a1"]["alpha_star"]
    lines = [
        "# Phase 5 — the crossover",
        "",
        (
            f"Profile `{pl['profile']['name']}` (T4 figures) · SLO p95 TTFT ≤ "
            f"{pl['cost_model']['slo']['ttft_p95_ms']:,} ms · {pl['n_runs']:,} runs in "
            f"{pl['runtime_s']:.0f} s · generated `{pl['date_utc']}` at `{pl['git_sha']}`"
            f"{' (dirty tree)' if pl['git_dirty'] else ''}. Generated traffic, one model."
        ),
        "",
        "## A1 — the load weight α of radix-affinity-balanced (Phase 4's two traces)",
        "",
        (
            f"Replicas needed to hold the SLO at loads {', '.join(f'{ld}/s' for ld in loads)} "
            "(— = not met at 8)."
        ),
        "",
        "| α | low_sharing | high_sharing | total |",
        "|---:|---|---|---:|",
    ]
    for al in a["values"]:
        r = a1["required_replicas"][al]
        lines.append(f"| {al:g} | {_row([r['low_sharing'][str(ld)] for ld in loads])} | "
                     f"{_row([r['high_sharing'][str(ld)] for ld in loads])} | "
                     f"{a1['total_replicas'][al]} |")
    p4 = a1["phase4_required_replicas"]
    if p4:
        for nm in ("low_sharing", "high_sharing"):
            for pol in ("round-robin", STICKY):
                row = [NOT_MET if p4[nm][str(ld)][pol] is None else p4[nm][str(ld)][pol]
                       for ld in loads]
                lines.append(f"| *Phase 4 {pol}, {nm}* | {_row(row)} | | |")
    lines += ["", f"**α\\* = {star:g}** — the fewest replicas in total; ties go to the smaller α.",
              ""]

    lines += ["## H2 — where does least-loaded beat prefix-affinity on p95 TTFT?", "",
              ("Each cell: the winner against least-loaded, from the paired session-bootstrap "
               "95% CI on p95(least-loaded) − p95(P) (seed 0). **P** = P faster, **LL** = "
               "least-loaded faster, · = inside the CI. Rows: load (sessions/s); columns: "
               "requested sharing (achieved cross-session sharing in brackets)."), ""]
    rep = pl["h2"]["trace_reports"]
    header = " | ".join(f"{r} ({rep[str(r)]['achieved_cross_session_sharing']:.2f})"
                        for r in h2["sharing"])
    for pol, t in pl["h2"]["tests"].items():
        lines.append(f"### P = {pol} — **{'HOLDS' if t['holds'] else 'FALSIFIED'}**")
        lines.append("")
        for n, m in t["per_replicas"].items():
            lines += [(f"{n} replicas — crossover: {m['crossover']} · violations along sharing: "
                       f"{m['violations_along_sharing'] or 'none'} · along load: "
                       f"{m['violations_along_load'] or 'none'} · power-of-two order: "
                       f"{m['violations_power_of_two'] or 'none'}"), "",
                      f"| load | {header} |", "|---|" + "---|" * len(h2["sharing"])]
            for ld in loads:
                cells = []
                for r in h2["sharing"]:
                    c = m["cells"][f"{r}@{ld}"][LL]
                    mark = {"P": "**P**", "X": "**LL**", "tie": "·"}[c["winner"]]
                    cells.append(f"{mark} {c['point']:+.0f} ms")
                lines.append(f"| {ld}/s | " + " | ".join(cells) + " |")
            lines.append("")

    lines += ["## H2b — does affinity pay once the prompts outgrow one replica's cache?", "",
              (f"Sharing {h2b['sharing']}, K distinct system prompts with Zipf popularity (s = "
               f"{h2b['prefix_zipf_s']}). Each cell: replicas needed to hold the SLO — "
               f"radix-affinity-balanced@{star:g} vs session-sticky."), ""]
    b = pl["h2b"]
    lines.append(f"**H2b: {'HOLDS' if b['holds'] else 'FALSIFIED'}** — (a) no affinity win at "
                 f"K = {h2b['n_shared_prefixes'][0]}: {b['cond_a_no_affinity_win_at_smallest_k']}"
                 f" · (b) an affinity win at K = {h2b['n_shared_prefixes'][-1]}: "
                 f"{b['cond_b_affinity_win_at_largest_k']} · (c) monotone in K: "
                 f"{'yes' if not b['violations_monotone_in_k'] else b['violations_monotone_in_k']}")
    lines.append("")
    ks = h2b["n_shared_prefixes"]
    head = []
    for k in ks:
        tr = b["trace_reports"][str(k)]
        ws = tr.get("shared_working_set_blocks", tr["shared_prefix_tokens"] // 16)
        head.append(f"K={k} (working set {ws / b['usable_blocks_per_replica']:.2f}× a replica)")
    lines += ["| load | " + " | ".join(head) + " |", "|---|" + "---|" * len(ks)]
    for ld in loads:
        cells = []
        for k in ks:
            r = b["required_replicas"][str(k)][str(ld)]
            w = b["winner"][f"{k}@{ld}"]
            ra, rs = r[label(BAL, star)], r[STICKY]
            cells.append(f"{_row([ra])} vs {_row([rs])} → {w}")
        lines.append(f"| {ld}/s | " + " | ".join(cells) + " |")
    lines += ["", "Every policy (replicas needed, loads left to right; reported, not tested):", ""]
    lines += ["| policy | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    pols = [*cfg["policies_reported"], label(BAL, star)]
    for p in pols:
        cells = [_row([b["required_replicas"][str(k)][str(ld)][p] for ld in loads]) for k in ks]
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines.append("")
    oracle_cells = [_row([min(b["required_replicas"][str(k)][str(ld)][label(BAL, al)]
                              for al in a["values"]) for ld in loads]) for k in ks]
    lines.append("| *best α per cell (oracle)* | " + " | ".join(oracle_cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
