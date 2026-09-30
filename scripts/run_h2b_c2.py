#!/usr/bin/env python3
"""H2b under C2's timing model: does the finding survive a timing model that matches the engine?

    python scripts/run_h2b_c2.py        # configs/h2b_c2.yaml -> artifacts/results/h2b_c2/

Registered before it ran (the config, plus a dated entry in docs/09-hypotheses.md). It re-runs
Phase 5's H2b grid — K tenant prompts × load × replicas — on C2's fitted, context-aware T4 profile,
and applies H2b's three rules unchanged. This script chooses nothing after seeing a result.

Tier (ADR-010): a measured profile for one T4 and one model. Its timing was validated on held-out
single-engine traffic (C2c) and, post hoc, on Phase 6's two replicas — so its replica counts are the
best T4 figures this project has, and still T4 figures.
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

from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "configs" / "h2b_c2.yaml"
BAL, STICKY = "radix-affinity-balanced", "session-sticky"


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@lru_cache(maxsize=1)
def _p5() -> dict[str, Any]:
    """Phase 5's config: H2b's session shape and prompts, read from there so they cannot drift."""
    return yaml.safe_load((REPO / _cfg()["h2b_config"]).read_text())


@cache
def _trace(k: int, load: float):
    p5 = _p5()
    spec = {**p5["session_shape"], "prefix_sharing_ratio": p5["h2b"]["sharing"],
            "n_shared_prefixes": k, "prefix_zipf_s": p5["h2b"]["prefix_zipf_s"]}
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()


@lru_cache(maxsize=1)
def _profile() -> CostProfile:
    return load_profile(REPO / _cfg()["profile"])


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _p5()["cost_model"])


def _job(key: tuple) -> tuple:
    k, load, policy, alpha, n, seed = key
    trace, _ = _trace(k, load)
    kwargs = None if alpha is None else {"load_penalty": float(alpha)}
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(),
            cache_model=_cfg()["cache_model"], policy_kwargs=kwargs)
    return key, to_point(r, _cost())


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()
    _p5.cache_clear()
    _profile.cache_clear()


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "artifacts" / "results" / "h2b_c2")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _init_worker(str(_CONFIG_PATH))
    cfg, p5 = _cfg(), _p5()
    ks, loads, replicas = p5["h2b"]["n_shared_prefixes"], cfg["loads"], cfg["replicas"]
    not_met = max(replicas) + 1
    alpha = cfg["alpha"]
    if not _profile().context_aware or not _profile().calibrated:
        raise SystemExit("error: the profile must be C2's calibrated, context-aware one")

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken BEFORE the runs: the header must describe the tree that produced the numbers.
    payload = results_header("scripts/run_h2b_c2.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg})
    pols = ([(STICKY, None), (BAL, alpha)] + [(BAL, a) for a in cfg["alphas_reported"]]
            + [(p, None) for p in cfg["policies_reported"]])
    jobs = [(k, ld, p, al, n, s) for k in ks for ld in loads for p, al in pols
            for n in replicas for s in cfg["seeds"]]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=2))
    runtime = time.time() - t0
    print(f"{len(jobs)} runs in {runtime:.0f}s on {args.workers} workers")

    cells: dict[tuple, list[Point]] = {}
    for (k, ld, p, al, n, _s), point in results:
        cells.setdefault((k, ld, p, al, n), []).append(point)
    agg = {c: average_points(v) for c, v in cells.items()}

    def required(k: int, ld: float, p: str, al: float | None) -> int:
        return next((n for n in sorted(replicas) if agg[(k, ld, p, al, n)].meets_slo), not_met)

    req = {str(k): {str(ld): {label(p, al): required(k, ld, p, al) for p, al in pols}
                    for ld in loads} for k in ks}
    n_hit = 8 if 8 in replicas else max(replicas)  # hit rates on 8 replicas, as in Phase 5
    hit = {str(k): {str(ld): {label(p, al): round(agg[(k, ld, p, al, n_hit)].hit_rate, 4)
                              for p, al in pols} for ld in loads} for k in ks}

    star = label(BAL, alpha)
    win = {}
    for k in ks:
        for ld in loads:
            ra, rs = req[str(k)][str(ld)][star], req[str(k)][str(ld)][STICKY]
            win[(k, ld)] = "affinity" if ra < rs else ("sticky" if rs < ra else "tie")
    cond_a = [ld for ld in loads if win[(ks[0], ld)] == "affinity"]
    cond_b = [ld for ld in loads if win[(ks[-1], ld)] == "affinity"]
    mono = [[ld, k1, k2] for ld in loads for k1, k2 in combinations(ks, 2)
            if win[(k1, ld)] == "affinity" and win[(k2, ld)] == "sticky"]
    verdict = {"a_no_affinity_win_at_k1": not cond_a, "affinity_wins_at_k1": cond_a,
               "b_affinity_win_at_k64": bool(cond_b), "affinity_wins_at_k64": cond_b,
               "c_no_reversal": not mono, "reversals": mono}
    verdict["holds"] = verdict["a_no_affinity_win_at_k1"] and verdict["b_affinity_win_at_k64"] \
        and verdict["c_no_reversal"]

    payload.update({
        "runtime_s": round(runtime, 1), "n_runs": len(jobs), "profile": _profile().provenance(),
        "cost_model": _cost().provenance, "not_met": not_met, "required_replicas": req,
        "hit_rate_replicas": n_hit, "hit_rate": hit,
        "winner": {f"{k}@{ld}": w for (k, ld), w in win.items()},
        "verdict": verdict,
        "trace_reports": {str(k): _trace(k, loads[0])[1] for k in ks},
    })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "h2b_c2.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(_markdown(payload, ks, loads, pols))
    print((args.out / "report.md").read_text())
    return 0


def _markdown(pl: dict[str, Any], ks: list[int], loads: list[float], pols: list[tuple]) -> str:
    cfg, v, nm = pl["config"], pl["verdict"], pl["not_met"]
    star = label(BAL, cfg["alpha"])

    def row(values: list[int]) -> str:
        return " ".join("—" if x == nm else str(x) for x in values)

    a = "✅" if v["a_no_affinity_win_at_k1"] else f"⛔ wins at {v['affinity_wins_at_k1']}"
    b = (f"✅ at {v['affinity_wins_at_k64']}" if v["b_affinity_win_at_k64"] else "⛔ nowhere")
    c = "✅" if v["c_no_reversal"] else f"⛔ {v['reversals']}"

    lines = [
        "# H2b under C2's timing model",
        "",
        (f"Phase 5's H2b traffic on the context-aware T4 profile `{pl['profile']['name']}` · loads "
         f"{', '.join(f'{x:g}' for x in loads)} sessions/s · replicas 1–{nm - 1} · "
         f"{len(cfg['seeds'])} seeds · {pl['n_runs']:,} runs in {pl['runtime_s']:.0f} s · "
         f"generated `{pl['date_utc']}` at `{pl['git_sha']}`"
         f"{' (dirty tree)' if pl['git_dirty'] else ''}."),
        "",
        (f"**H2b under C2's timing: {'HOLDS' if v['holds'] else 'FALSIFIED'}** "
         f"(registered prediction: {cfg['prediction']})."),
        "",
        "| rule | result |",
        "|---|---|",
        f"| (a) at K = 1, balanced affinity wins at no load | {a} |",
        f"| (b) at K = 64, balanced affinity wins at one load or more | {b} |",
        f"| (c) no reversal along K | {c} |",
        "",
        (f"## {star} against session-sticky — replicas needed, and the winner "
         f"(— = not held on {nm - 1})"),
        "",
        "| load | " + " | ".join(f"K={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for ld in loads:
        cells = []
        for k in ks:
            r = pl["required_replicas"][str(k)][str(ld)]
            cells.append(f"{row([r[star]])} vs {row([r[STICKY]])} → {pl['winner'][f'{k}@{ld}']}")
        lines.append(f"| {ld:g}/s | " + " | ".join(cells) + " |")
    lines += ["", "## Every policy — replicas needed at each load, left to right", "",
              "| policy | " + " | ".join(f"K={k}" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
    for p, al in pols:
        lab = label(p, al)
        cells = [row([pl["required_replicas"][str(k)][str(ld)][lab] for ld in loads]) for k in ks]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", f"## Hit rate on {pl['hit_rate_replicas']} replicas (loads left to right)", "",
              "| policy | " + " | ".join(f"K={k}" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
    for p, al in pols:
        lab = label(p, al)
        cells = [" ".join(f"{pl['hit_rate'][str(k)][str(ld)][lab]:.2f}" for ld in loads)
                 for k in ks]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
