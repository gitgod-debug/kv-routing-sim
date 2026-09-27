#!/usr/bin/env python3
"""H6's ranking half: does the ORDER of the routing policies survive a change of hardware?

    python scripts/run_h6.py        # configs/h6_ranking.yaml -> artifacts/results/h6_ranking/

The design and every decision rule were committed before this ran (configs/h6_ranking.yaml, plus a
dated entry in docs/09-hypotheses.md). This script IS that procedure: it chooses nothing after
seeing a result.

What it does: re-runs the Phase-4 design once per REGIME. A regime multiplies the calibrated T4
profile's timing coefficients — a faster GPU, or a different prefill/decode balance — and changes
nothing else: KV capacity, max_num_seqs, the SLO and the price stay fixed. Then, at every
(trace, load) point, it compares each rescaled regime with the calibrated baseline.

Tier (ADR-010): a rescaled profile describes no real GPU, and that is the point — H6 asks whether
the ranking depends on the hardware at all. No millisecond or rupee figure from a rescaled regime
is a prediction for anything, and the results file says so.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache, lru_cache
from typing import Any

import yaml

from kv_routing_sim import policies as policy_pkg
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.eval.pareto import Point, to_point
from kv_routing_sim.eval.rank import kendall_tau_b
from kv_routing_sim.eval.recommend import Recommendation, recommend
from kv_routing_sim.eval.sweep import average_points
from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = REPO / "configs" / "h6_ranking.yaml"
NOT_MET = 9  # no replica count from 1 to 8 holds the SLO: ranks below every count that did


# ── one run, in a worker process ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text())


@lru_cache(maxsize=1)
def _base() -> dict[str, Any]:
    """The Phase-4 design, read from its own config so the two can never drift apart."""
    return yaml.safe_load((REPO / _cfg()["base_sweep"]).read_text())


@cache
def _trace(name: str, load: float):
    spec = dict(_base()["traces"][name])
    return AgenticWorkload(**spec, arrival_rate=float(load)).generate()


@cache
def _profile(regime: str) -> CostProfile:
    """The calibrated profile with its timing coefficients multiplied. Nothing else changes."""
    base = load_profile(REPO / _base()["profile"])
    r = _cfg()["regimes"][regime]
    if r["prefill_x"] == 1.0 and r["decode_x"] == 1.0:
        return base  # the baseline IS the calibrated profile, unchanged
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
    return CostModel.from_yaml(REPO / _base()["cost_model"])


def _job(key: tuple) -> tuple:
    regime, name, load, policy, n, seed = key
    trace, _ = _trace(name, load)
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=_profile(regime),
            cache_model=_base()["cache_model"])
    return key, to_point(r, _cost())


def _init_worker(config_path: str) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = pathlib.Path(config_path)
    _cfg.cache_clear()
    _base.cache_clear()


# ── the decision rules ────────────────────────────────────────────────────────────────────────


def _best_set(required: dict[str, int]) -> list[str]:
    """The policies needing the fewest replicas at one point; empty if none holds the SLO."""
    best = min(required.values())
    return sorted(p for p, n in required.items() if n == best) if best < NOT_MET else []


def _compare(regime: str, name: str, load: float, pols: list[str],
             r_base: dict[str, int], r_reg: dict[str, int],
             rec_base: Recommendation, rec_reg: Recommendation) -> dict[str, Any]:
    """One point, one rescaled regime against the baseline: the registered rules, verbatim."""
    best_base, best_reg = _best_set(r_base), _best_set(r_reg)
    ours = rec_base.chosen.policy if best_base and rec_base.chosen else None
    theirs = rec_reg.chosen.policy if best_reg and rec_reg.chosen else None
    # FLIP: our pick is not among the regime's best policies, so following our advice on that
    # hardware would cost at least one more replica. n/a where nothing holds the SLO.
    flip = None if ours is None or not best_reg else ours not in best_reg
    tau = kendall_tau_b([r_base[p] for p in pols], [r_reg[p] for p in pols])
    return {
        "regime": regime,
        "trace": name,
        "load": load,
        "required_baseline": r_base,
        "required_regime": r_reg,
        "our_pick": ours,
        "regime_pick": theirs,
        "best_set_baseline": best_base,
        "best_set_regime": best_reg,
        "tau_b": None if tau is None else round(tau, 4),
        "flip": flip,
        # Reported, not tested (see the config): the literal reading, and a kinder one.
        "strict_flip": None if ours is None or theirs is None else ours != theirs,
        "lenient_flip": (
            None if not best_base or not best_reg else not set(best_base) & set(best_reg)
        ),
    }


def main() -> int:
    global _CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=_CONFIG_PATH)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "h6_ranking")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    _CONFIG_PATH = args.config.resolve()
    _cfg.cache_clear()
    _base.cache_clear()
    cfg, base = _cfg(), _base()

    regimes, baseline = list(cfg["regimes"]), cfg["baseline"]
    if baseline not in regimes:
        raise SystemExit(f"error: baseline {baseline!r} is not one of the regimes {regimes}")
    pols = policy_pkg.resolve(base["policies"])
    replicas, seeds, loads = base["replicas"], base["seeds"], base["loads"]
    traces = list(base["traces"])

    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    # Taken BEFORE the runs: the header must describe the tree that produced the numbers, not
    # whatever the tree looks like by the time they finish.
    payload = results_header("scripts/run_h6.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg, "base_config": base})

    jobs = [(g, name, load, p, n, s) for g in regimes for name in traces for load in loads
            for p in pols for n in replicas for s in seeds]
    t0 = time.time()
    # Workers re-read the config by path, so they must be handed the same one the parent uses.
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(_CONFIG_PATH),)) as ex:
        results = list(ex.map(_job, jobs, chunksize=4))
    runtime = time.time() - t0
    print(f"{len(jobs)} runs in {runtime:.0f}s on {args.workers} workers")

    cells: dict[tuple, list[Point]] = {}
    for (g, name, load, p, n, _s), point in results:
        cells.setdefault((g, name, load, p, n), []).append(point)
    agg = {k: average_points(v) for k, v in cells.items()}

    def required(g: str, name: str, load: float, policy: str) -> int:
        met = (n for n in sorted(replicas) if agg[(g, name, load, policy, n)].meets_slo)
        return next(met, NOT_MET)

    def rec(g: str, name: str, load: float) -> Recommendation:
        return recommend([agg[(g, name, load, p, n)] for p in pols for n in replicas])

    req = {g: {name: {str(ld): {p: required(g, name, ld, p) for p in pols} for ld in loads}
               for name in traces} for g in regimes}
    recs = {g: {name: {str(ld): rec(g, name, ld) for ld in loads} for name in traces}
            for g in regimes}

    # Guard (registered): the baseline must reproduce Phase 4's table exactly. If it does not, the
    # code has changed since Phase 4 and nothing below is comparable — stop, do not report.
    if cfg.get("phase4_results"):
        p4 = json.loads((REPO / cfg["phase4_results"]).read_text())["required_replicas"]
        for name in traces:
            for ld in loads:
                for p in pols:
                    want = p4[name][str(ld)][p]
                    want = NOT_MET if want is None else want
                    got = req[baseline][name][str(ld)][p]
                    if got != want:
                        raise SystemExit(
                            f"error: the baseline does not reproduce Phase 4 at {name}@{ld}/s, "
                            f"{p}: {got} replicas vs {want}. The code has drifted; stopping."
                        )
        print("the baseline reproduces Phase 4's required-replica table exactly")

    points = [
        _compare(g, name, ld, pols, req[baseline][name][str(ld)], req[g][name][str(ld)],
                 recs[baseline][name][str(ld)], recs[g][name][str(ld)])
        for g in regimes if g != baseline for name in traces for ld in loads
    ]

    verdicts: dict[str, dict[str, Any]] = {}
    for g in regimes:
        if g == baseline:
            continue
        pts = [c for c in points if c["regime"] == g]
        taus = [c["tau_b"] for c in pts if c["tau_b"] is not None]
        flips = sum(1 for c in pts if c["flip"] is True)
        # A regime with no defined tau-b anywhere is UNTESTABLE, and does not count as holding.
        holds = (flips == 0 and all(t >= cfg["tau_threshold"] for t in taus)) if taus else None
        verdicts[g] = {
            "points": len(pts),
            "flips": flips,
            "flips_not_applicable": sum(1 for c in pts if c["flip"] is None),
            "strict_flips": sum(1 for c in pts if c["strict_flip"] is True),
            "lenient_flips": sum(1 for c in pts if c["lenient_flip"] is True),
            "points_with_tau": len(taus),
            "min_tau_b": min(taus) if taus else None,
            "median_tau_b": statistics.median(taus) if taus else None,
            "holds": holds,
        }

    payload["runtime_s"] = round(runtime, 1)
    payload["policies"] = pols
    payload["profiles"] = {g: _profile(g).provenance() for g in regimes}
    payload["cost_model"] = _cost().provenance
    payload["required_replicas"] = req
    payload["recommendations"] = {
        g: {name: {ld: r.as_dict() for ld, r in by_load.items()}
            for name, by_load in by_trace.items()}
        for g, by_trace in recs.items()
    }
    payload["points"] = points
    payload["verdicts"] = verdicts
    payload["h6_ranking_holds"] = all(v["holds"] is True for v in verdicts.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "h6.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    (args.out / "report.md").write_text(_markdown(payload))
    print((args.out / "report.md").read_text())
    return 0


def _markdown(pl: dict[str, Any]) -> str:
    cfg, base = pl["config"], pl["base_config"]
    loads, pols, traces = base["loads"], pl["policies"], list(base["traces"])
    req, baseline = pl["required_replicas"], cfg["baseline"]

    def fmt(t: float | None) -> str:
        return "—" if t is None else f"{t:.3f}"

    lines = [
        "# H6, ranking half — does the order of the policies survive a change of hardware?",
        "",
        (
            f"Phase 4's design ({len(traces)} traces × {len(loads)} loads × {len(pols)} policies × "
            f"replicas {min(base['replicas'])}–{max(base['replicas'])} × {len(base['seeds'])} "
            f"seeds) under {len(cfg['regimes'])} regimes · SLO p95 TTFT ≤ "
            f"{pl['cost_model']['slo']['ttft_p95_ms']:,} ms · generated `{pl['date_utc']}` at "
            f"`{pl['git_sha']}`{' (dirty tree)' if pl['git_dirty'] else ''}."
        ),
        "",
        (
            f"⚠️ **Only `{baseline}` is a measured profile.** Every other regime rescales its "
            "timings and describes no real GPU, so its milliseconds and rupees are not "
            "predictions. Only the ORDER of the policies is under test."
        ),
        "",
        "## Verdict",
        "",
        "| regime | prefill × | decode × | flips | min τ-b | points with a τ-b | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for g, v in pl["verdicts"].items():
        r = cfg["regimes"][g]
        verdict = {True: "HOLDS", False: "FALSIFIED", None: "UNTESTABLE"}[v["holds"]]
        lines.append(
            f"| {g} | {r['prefill_x']} | {r['decode_x']} | {v['flips']} | "
            f"{fmt(v['min_tau_b'])} | {v['points_with_tau']} of {v['points']} | {verdict} |"
        )
    lines += [
        "",
        (
            f"**H6, ranking half: {'HOLDS' if pl['h6_ranking_holds'] else 'DOES NOT HOLD'}.** "
            "It holds only if every rescaled regime has zero flips and every τ-b is at least "
            f"{cfg['tau_threshold']}."
        ),
        "",
        (
            "*Flip:* the baseline's recommendation is not among the regime's lowest-replica "
            "policies, so following it on that hardware would cost a replica. *τ-b:* Kendall's "
            "rank agreement between the replica counts each policy needs, baseline vs regime "
            "(— where either side is all ties)."
        ),
        "",
        (
            "Reported, not tested: the literal reading (the regime's own pick names a different "
            "policy) and a kinder one (the two best sets share no policy)."
        ),
        "",
        "| regime | strict flips | lenient flips |",
        "|---|---:|---:|",
    ]
    for g, v in pl["verdicts"].items():
        lines.append(f"| {g} | {v['strict_flips']} | {v['lenient_flips']} |")

    lines += ["", "## Replicas each policy needs to hold the SLO", ""]
    for g in cfg["regimes"]:
        for name in traces:
            lines += [f"### {g} · {name}", "",
                      "| policy | " + " | ".join(f"{ld}/s" for ld in loads) + " |",
                      "|---|" + "---|" * len(loads)]
            for p in pols:
                row = [req[g][name][str(ld)][p] for ld in loads]
                cells = " | ".join("—" if n == NOT_MET else str(n) for n in row)
                lines.append(f"| {p} | {cells} |")
            lines += ["", "— = not met at 8 replicas.", ""]

    lines += ["## Every point", "",
              "| regime | trace @ load | our pick | best there | τ-b | flip |",
              "|---|---|---|---|---:|:---:|"]
    for c in pl["points"]:
        best = ", ".join(c["best_set_regime"]) or "nothing holds the SLO"
        flip = {True: "⛔ yes", False: "no", None: "n/a"}[c["flip"]]
        lines.append(f"| {c['regime']} | {c['trace']} @ {c['load']}/s | {c['our_pick']} | "
                     f"{best} | {fmt(c['tau_b'])} | {flip} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
