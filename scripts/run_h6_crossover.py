#!/usr/bin/env python3
"""H6, crossover half: whether H2b's flip holds when the hardware changes.

    python scripts/run_h6_crossover.py
        # configs/h6_crossover.yaml -> artifacts/results/h6_crossover/

Re-runs Phase 5's H2b grid (K tenant prompts × load × replicas) under the cost regimes of H6's
ranking half and checks, regime by regime, that the flip from session-sticky to prefix-affinity
may shift but does not invert. The decision rules come from the config; this script applies
them and tunes nothing.

ADR-010 tier: only the t4 regime is a measured profile; the rescaled ones describe no real GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
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
_CONFIG_PATH = REPO / "configs" / "h6_crossover.yaml"
NOT_MET = 9  # no replica count from 1 to 8 holds the SLO
BAL, STICKY = "radix-affinity-balanced", "session-sticky"


def label(policy: str, alpha: float | None) -> str:
    return policy if alpha is None else f"{policy}@{alpha:g}"


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@lru_cache(maxsize=1)
def _p5() -> dict[str, Any]:
    """Phase 5's config: the H2b grid and session shape, read from there so they cannot drift."""
    return yaml.safe_load((REPO / _cfg()["h2b_config"]).read_text())


@lru_cache(maxsize=1)
def _regimes() -> dict[str, Any]:
    return yaml.safe_load((REPO / _cfg()["regimes_config"]).read_text())


@cache
def _trace(k: int, load: float):
    p5 = _p5()
    spec = {**p5["session_shape"], "prefix_sharing_ratio": p5["h2b"]["sharing"],
            "n_shared_prefixes": k, "prefix_zipf_s": p5["h2b"]["prefix_zipf_s"]}
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()


@cache
def _profile(regime: str) -> CostProfile:
    """The calibrated profile with its timing coefficients multiplied, as in H6's ranking half."""
    base = load_profile(REPO / _p5()["profile"])
    r = _regimes()["regimes"][regime]
    if r["prefill_x"] == 1.0 and r["decode_x"] == 1.0:
        return base
    return dataclasses.replace(
        base,
        name=f"{base.name}+{regime}",
        prefill_base_s=base.prefill_base_s * r["prefill_x"],
        prefill_per_token_s=base.prefill_per_token_s * r["prefill_x"],
        decode_base_s=base.decode_base_s * r["decode_x"],
        decode_per_seq_s=base.decode_per_seq_s * r["decode_x"],
        calibrated=False,  # a rescaled profile describes no real GPU
        calibration_error=None,
    )


@lru_cache(maxsize=1)
def _cost() -> CostModel:
    return CostModel.from_yaml(REPO / _p5()["cost_model"])


def _job(key: tuple) -> tuple:
    regime, k, load, policy, alpha, n, seed = key
    trace, _ = _trace(k, load)
    kwargs = None if alpha is None else {"load_penalty": float(alpha)}
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(regime),
            cache_model=_p5()["cache_model"], policy_kwargs=kwargs)
    return key, to_point(r, _cost())


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()
    _p5.cache_clear()
    _regimes.cache_clear()


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "h6_crossover")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _init_worker(str(_CONFIG_PATH))
    cfg, p5, reg = _cfg(), _p5(), _regimes()
    regimes, baseline = list(reg["regimes"]), reg["baseline"]
    ks, loads, replicas = p5["h2b"]["n_shared_prefixes"], p5["loads"], p5["replicas"]
    alpha = cfg["alpha"]

    p5_results = None
    if cfg.get("phase5_results"):
        p5_results = json.loads((REPO / cfg["phase5_results"]).read_text())
        if p5_results["a1"]["alpha_star"] != alpha:
            raise SystemExit(f"error: alpha {alpha} is not Phase 5's alpha* "
                             f"{p5_results['a1']['alpha_star']}; stopping.")

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken before the runs, so the header describes the tree that produced the numbers.
    payload = results_header("scripts/run_h6_crossover.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg})

    pols = [(STICKY, None), (BAL, alpha)] + [(p, None) for p in cfg["policies_reported"]]
    jobs = [(g, k, ld, p, al, n, s) for g in regimes for k in ks for ld in loads
            for p, al in pols for n in replicas for s in cfg["seeds"]]
    t0 = time.time()
    # Workers re-read the config by path, so they must be handed the same one the parent uses.
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    runtime = time.time() - t0
    print(f"{len(jobs)} runs in {runtime:.0f}s on {args.workers} workers")

    cells: dict[tuple, list[Point]] = {}
    for (g, k, ld, p, al, n, _s), point in results:
        cells.setdefault((g, k, ld, p, al, n), []).append(point)
    agg = {c: average_points(v) for c, v in cells.items()}

    def required(g: str, k: int, ld: float, p: str, al: float | None) -> int:
        met = (n for n in sorted(replicas) if agg[(g, k, ld, p, al, n)].meets_slo)
        return next(met, NOT_MET)

    req = {g: {str(k): {str(ld): {label(p, al): required(g, k, ld, p, al) for p, al in pols}
                        for ld in loads} for k in ks} for g in regimes}

    # Guard: the t4 regime must reproduce Phase 5's H2b table exactly.
    if p5_results is not None:
        want = p5_results["h2b"]["required_replicas"]
        for k in ks:
            for ld in loads:
                for p, al in pols:
                    lab = label(p, al)
                    if req[baseline][str(k)][str(ld)][lab] != want[str(k)][str(ld)][lab]:
                        raise SystemExit(
                            f"error: t4 does not reproduce Phase 5 at K={k}, {ld}/s, {lab}: "
                            f"{req[baseline][str(k)][str(ld)][lab]} vs "
                            f"{want[str(k)][str(ld)][lab]}. The code has drifted; stopping."
                        )
        print("the t4 regime reproduces Phase 5's H2b table exactly")

    star = label(BAL, alpha)
    winners: dict[str, dict[str, str]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for g in regimes:
        win = {}
        for k in ks:
            for ld in loads:
                ra, rs = req[g][str(k)][str(ld)][star], req[g][str(k)][str(ld)][STICKY]
                win[(k, ld)] = "affinity" if ra < rs else ("sticky" if rs < ra else "tie")
        winners[g] = {f"{k}@{ld}": w for (k, ld), w in win.items()}
        small = [ld for ld in loads if win[(ks[0], ld)] == "affinity"]
        large = [ld for ld in loads if win[(ks[-1], ld)] == "sticky"]
        mono = [[ld, k1, k2] for ld in loads for k1, k2 in combinations(ks, 2)
                if win[(k1, ld)] == "affinity" and win[(k2, ld)] == "sticky"]
        verdicts[g] = {
            "affinity_wins_at_smallest_k": small,
            "sticky_wins_at_largest_k": large,
            "violations_monotone_in_k": mono,
            "holds": not small and not large and not mono,
            "flip_visible": any(win[(ks[-1], ld)] == "affinity" for ld in loads),
        }

    payload["runtime_s"] = round(runtime, 1)
    payload["n_runs"] = len(jobs)
    payload["profiles"] = {g: _profile(g).provenance() for g in regimes}
    payload["required_replicas"] = req
    payload["winner"] = winners
    payload["verdicts"] = verdicts
    payload["h6_crossover_holds"] = all(v["holds"] for g, v in verdicts.items() if g != baseline)
    payload["trace_reports"] = {str(k): _trace(k, loads[0])[1] for k in ks}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "h6_crossover.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(_markdown(payload, regimes, baseline, ks, loads, pols))
    print((args.out / "report.md").read_text())
    return 0


def _row(values: list[int]) -> str:
    return " ".join("—" if v == NOT_MET else str(v) for v in values)


def _markdown(pl: dict[str, Any], regimes: list[str], baseline: str, ks: list[int],
              loads: list[float], pols: list[tuple]) -> str:
    star = label(BAL, pl["config"]["alpha"])
    lines = [
        "# H6, crossover half — does H2b's flip survive a change of hardware?",
        "",
        (
            f"Phase 5's H2b grid under {len(regimes)} cost regimes · {pl['n_runs']:,} runs in "
            f"{pl['runtime_s']:.0f} s · generated `{pl['date_utc']}` at `{pl['git_sha']}`"
            f"{' (dirty tree)' if pl['git_dirty'] else ''}."
        ),
        "",
        (
            f"⚠️ **Only `{baseline}` is a measured profile.** The other regimes rescale its timings "
            "and describe no real GPU; only which policy needs fewer replicas is under test."
        ),
        "",
        "## Verdict",
        "",
        (
            "| regime | affinity wins at K = 1 | sticky wins at K = 64 | reversals along K | "
            "flip still visible | verdict |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for g, v in pl["verdicts"].items():
        tag = "(baseline)" if g == baseline else ("HOLDS" if v["holds"] else "FALSIFIED")
        lines.append(
            f"| {g} | {v['affinity_wins_at_smallest_k'] or 'none'} | "
            f"{v['sticky_wins_at_largest_k'] or 'none'} | "
            f"{v['violations_monotone_in_k'] or 'none'} | {v['flip_visible']} | {tag} |"
        )
    verdict = "HOLDS" if pl["h6_crossover_holds"] else "DOES NOT HOLD"
    lines += ["", f"**H6, crossover half: {verdict}.**", ""]
    for g in regimes:
        lines += [f"## {g}", "",
                  f"Replicas needed to hold the SLO: {star} vs session-sticky → the winner.", "",
                  "| load | " + " | ".join(f"K={k}" for k in ks) + " |",
                  "|---|" + "---|" * len(ks)]
        for ld in loads:
            cells = []
            for k in ks:
                r = pl["required_replicas"][g][str(k)][str(ld)]
                cells.append(f"{_row([r[star]])} vs {_row([r[STICKY]])} → "
                             f"{pl['winner'][g][f'{k}@{ld}']}")
            lines.append(f"| {ld}/s | " + " | ".join(cells) + " |")
        lines += ["", "Every policy (replicas needed at each load, left to right):", "",
                  "| policy | " + " | ".join(f"K={k}" for k in ks) + " |",
                  "|---|" + "---|" * len(ks)]
        for p, al in pols:
            lab = label(p, al)
            cells = [_row([pl["required_replicas"][g][str(k)][str(ld)][lab] for ld in loads])
                     for k in ks]
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
