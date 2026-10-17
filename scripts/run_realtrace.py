#!/usr/bin/env python3
"""Replay real traffic through every policy on the T4 profile: R1 (Mooncake) and R2 (AgentX).

    python scripts/run_realtrace.py            # configs/realtrace_mooncake.yaml (R1)
    python scripts/run_realtrace.py --config configs/realtrace_agentx.yaml          # R2
    python scripts/run_realtrace.py --probe    # round-robin only: how many replicas at each speed

Writes realtrace.json and report.md to the config's results_dir. The decision rules come from the
config; this script applies them and tunes nothing. `--probe` runs round-robin alone, with no
comparison between policies, to choose the replay speeds.

The trace files are third-party data and are not committed (artifacts/traces/ is in .gitignore).
Fetch them from the source named in the config; the results file records each file's sha256 so a
reader can check that the same bytes were replayed.

ADR-010 tier: T4 timings and a scaled replay (every token count times `token_scale`); the
adapter's docstring says what the scaling preserves and what it does not.
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
from concurrent.futures import ProcessPoolExecutor
from functools import cache, lru_cache
from typing import Any

import yaml

from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces import adapters
from kv_routing_sim.traces.hashing import cross_session_sharing_ratio, sharing_ratio

REPO = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "configs" / "realtrace_mooncake.yaml"
NOT_MET = 9
BAL, STICKY = "radix-affinity-balanced", "session-sticky"


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@cache
def _base_trace(name: str) -> tuple:
    """The adapted trace, cut to the window. Read once per worker."""
    cfg = _cfg()
    reader = adapters.get(cfg.get("adapter_name", "mooncake"))
    reqs = reader.read(REPO / cfg["traces"][name], **cfg["adapter"])
    return tuple(r for r in reqs if r.arrival_ts < cfg["window_s"])


@cache
def _trace(name: str, speed: float) -> tuple[tuple, int]:
    """The window at `speed` times the recorded rate, and the number of warm-up requests."""
    base = _base_trace(name)
    trace = tuple(dataclasses.replace(r, arrival_ts=r.arrival_ts / speed) for r in base)
    warmup = sum(1 for r in base if r.arrival_ts < _cfg()["warmup_s"])
    return trace, warmup


@lru_cache(maxsize=1)
def _profile() -> CostProfile:
    return load_profile(REPO / _cfg()["profile"])


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _cfg()["cost_model"])


def _job(key: tuple) -> tuple:
    name, speed, policy, alpha, n, seed = key
    trace, warmup = _trace(name, speed)
    kwargs = None if alpha is None else {"load_penalty": float(alpha)}
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(),
            cache_model=_cfg()["cache_model"], policy_kwargs=kwargs, warmup_requests=warmup)
    return key, to_point(r, _cost())


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()


def _sweep(jobs: list[tuple], workers: int) -> dict[tuple, Point]:
    # Workers re-read the config by path, so they must be handed the same one the parent uses.
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=1))
    cells: dict[tuple, list[Point]] = {}
    for (name, speed, p, al, n, _s), point in results:
        cells.setdefault((name, speed, p, al, n), []).append(point)
    return {c: average_points(v) for c, v in cells.items()}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: the config's results_dir (R1's: realtrace_mooncake)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report-only", action="store_true",
                    help="re-render report.md from the results_dir's realtrace.json; run nothing")
    ap.add_argument("--probe", action="store_true",
                    help="round-robin alone, one seed, at the probe speeds: to choose the speeds")
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _cfg.cache_clear()
    cfg = _cfg()
    names, replicas = list(cfg["traces"]), cfg["replicas"]
    out_dir = args.out or REPO / cfg.get("results_dir", "artifacts/results/realtrace_mooncake")
    if args.report_only:
        pl = json.loads((out_dir / "realtrace.json").read_text())
        run_cfg = pl["config"]
        pols = ([(p, None) for p in run_cfg["policies"]]
                + [(BAL, al) for al in run_cfg["alphas"]])
        (out_dir / "report.md").write_text(_markdown(pl, pols))
        return 0

    if args.probe:
        jobs = [(nm, sp, "round-robin", None, n, 0) for nm in names for sp in cfg["probe_speeds"]
                for n in replicas]
        agg = _sweep(jobs, args.workers)
        for nm in names:
            for sp in cfg["probe_speeds"]:
                met = (n for n in replicas if agg[(nm, sp, "round-robin", None, n)].meets_slo)
                need = next(met, None)
                print(f"probe {nm} speed {sp}: round-robin needs {need or 'more than 8'}")
        return 0

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken before the runs, so the header describes the tree that produced the numbers.
    payload = results_header("scripts/run_realtrace.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg})
    payload["trace_sha256"] = {nm: _sha256(REPO / cfg["traces"][nm]) for nm in names}
    pols = [(p, None) for p in cfg["policies"]] + [(BAL, al) for al in cfg["alphas"]]
    jobs = [(nm, sp, p, al, n, s) for nm in names for sp in cfg["speeds"] for p, al in pols
            for n in replicas for s in cfg["seeds"]]
    t0 = time.time()
    agg = _sweep(jobs, args.workers)
    runtime = time.time() - t0
    print(f"{len(jobs)} runs in {runtime:.0f}s on {args.workers} workers")

    def required(nm: str, sp: float, p: str, al: float | None) -> int:
        met = (n for n in replicas if agg[(nm, sp, p, al, n)].meets_slo)
        return next(met, NOT_MET)

    req = {nm: {str(sp): {label(p, al): required(nm, sp, p, al) for p, al in pols}
                for sp in cfg["speeds"]} for nm in names}
    hit4 = {nm: {str(sp): {label(p, al): round(agg[(nm, sp, p, al, 4)].hit_rate, 4)
                           for p, al in pols} for sp in cfg["speeds"]} for nm in names}
    star = label(BAL, cfg["alpha_star"])
    verdicts = {}
    for nm, rules in cfg["rules"].items():
        wins = {str(sp): ("affinity" if req[nm][str(sp)][star] < req[nm][str(sp)][STICKY] else
                          "sticky" if req[nm][str(sp)][STICKY] < req[nm][str(sp)][star] else "tie")
                for sp in cfg["speeds"]}
        results = {rule: _rule(rule, req[nm], wins, star)
                   for rule in ([rules] if isinstance(rules, str) else rules)}
        verdicts[nm] = {"rule": " + ".join(results), "winner_by_speed": wins,
                        "rules": results, "holds": all(results.values())}

    structure = {}
    for nm in names:
        base = _base_trace(nm)
        structure[nm] = {
            "requests_in_window": len(base),
            "sessions": len({r.session_id for r in base}),
            "cross_session_sharing": round(cross_session_sharing_ratio(
                (r.block_hashes, r.session_id) for r in base), 4),
            "total_block_reuse": round(sharing_ratio(r.block_hashes for r in base), 4),
        }
    payload.update({"runtime_s": round(runtime, 1), "n_runs": len(jobs), "profile":
                    _profile().provenance(), "required_replicas": req, "hit_rate_at_4": hit4,
                    "verdicts": verdicts, "structure": structure,
                    "r1_holds": all(v["holds"] for v in verdicts.values())})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "realtrace.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (out_dir / "report.md").write_text(_markdown(payload, pols))
    print((out_dir / "report.md").read_text())
    return 0


def _rule(rule: str, req: dict[str, dict[str, int]], wins: dict[str, str], star: str) -> bool:
    """One decision rule on one trace; `req[speed][policy]` is replicas needed, or NOT_MET.

    affinity_wins / affinity_never_wins (R1) compare balanced affinity at alpha* with
    session-sticky. tie_within_one / pure_affinity_ties_sticky (R2) require that, at every speed
    where either policy holds the SLO on 8 replicas, the two need replicas within one of each
    other; at least one speed must be testable, or the rule does not hold.
    """
    aff = [sp for sp, w in wins.items() if w == "affinity"]
    stk = [sp for sp, w in wins.items() if w == "sticky"]
    if rule == "affinity_wins":
        return bool(aff) and not stk
    if rule == "affinity_never_wins":
        return not aff
    other = {"tie_within_one": star, "pure_affinity_ties_sticky": "radix-affinity"}.get(rule)
    if other is None:
        raise SystemExit(f"error: unknown rule {rule!r}")
    pairs = [(r[other], r[STICKY]) for r in req.values() if min(r[other], r[STICKY]) < NOT_MET]
    return bool(pairs) and all(abs(a - b) <= 1 for a, b in pairs)


def _row(values: list[int]) -> str:
    return " ".join("—" if v == NOT_MET else str(v) for v in values)


def _markdown(pl: dict[str, Any], pols: list[tuple]) -> str:
    cfg = pl["config"]
    speeds = cfg["speeds"]
    name = cfg.get("name", "R1")
    title = cfg.get("title", "R1 — real traffic: the Mooncake FAST'25 traces")
    sessions = cfg.get("sessions_note", "sessions inferred (adapter)")
    lines = [
        f"# {title}",
        "",
        (
            f"T4 profile · token counts × {cfg['adapter']['token_scale']} · first "
            f"{cfg['window_s'] // 60} min, the first {cfg['warmup_s'] // 60} min warm-up · "
            f"{sessions} · SLO p95 TTFT ≤ 2,000 ms · "
            f"{pl['n_runs']:,} runs in "
            f"{pl['runtime_s']:.0f} s · generated `{pl['date_utc']}` at `{pl['git_sha']}`"
            f"{' (dirty tree)' if pl['git_dirty'] else ''}."
        ),
        "",
        f"**{name}: {'HOLDS' if pl['r1_holds'] else 'FALSIFIED'}**",
        "",
        "| trace | prediction | winner at each speed | verdict |",
        "|---|---|---|---|",
    ]
    for nm, v in pl["verdicts"].items():
        w = " · ".join(f"{sp}×: {x}" for sp, x in v["winner_by_speed"].items())
        lines.append(f"| {nm} | {v['rule']} | {w} | {'HOLDS' if v['holds'] else 'FALSIFIED'} |")
    column = "sessions (inferred)" if "inferred" in sessions else "sessions"
    lines += ["", "## Structure of the window (reported)", "",
              f"| trace | requests | {column} | cross-session sharing | total reuse |",
              "|---|---:|---:|---:|---:|"]
    for nm, s in pl["structure"].items():
        lines.append(f"| {nm} | {s['requests_in_window']:,} | {s['sessions']:,} | "
                     f"{s['cross_session_sharing']:.3f} | {s['total_block_reuse']:.3f} |")
    for nm in pl["required_replicas"]:
        heading = (f"## {nm} — replicas needed to hold the SLO, speeds "
                   f"{', '.join(f'{sp}×' for sp in speeds)} (— = not met at 8)")
        lines += ["", heading, "", "| policy | replicas | hit rate at 4 replicas |",
                  "|---|---|---|"]
        for p, al in pols:
            lab = label(p, al)
            reqs = [pl["required_replicas"][nm][str(sp)][lab] for sp in speeds]
            hits = " ".join(f"{pl['hit_rate_at_4'][nm][str(sp)][lab]:.3f}" for sp in speeds)
            lines.append(f"| {lab} | {_row(reqs)} | {hits} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
