#!/usr/bin/env python3
"""What one agent session costs under each routing policy (S1 in docs/09-hypotheses.md).

    python scripts/session_cost.py               # every cell -> artifacts/results/session_cost/
    python scripts/session_cost.py --only 4 0.5  # one cell (prompts, load): printed, not written

Every cell is a cell of H2b on the validated timing model (configs/h2b_c2.yaml, calibration A):
the traces, seeds, profile, policies and cost model of `scripts/build_grid.py --source h2b_c2`,
whose helpers are imported so that the code path is the same. Each policy runs at the fewest
replicas that meet the target. For the first-token target that count is read from
artifacts/results/h2b_c2/h2b_c2.json, for the two-part target from
artifacts/results/t2_two_part_slo/analysis.json (calibration A).

Per session it reports prompt tokens, prefilled tokens, new tokens, re-prefilled tokens with a
bootstrap interval over sessions, and rupees. Every run is checked against the stored grids and
against its own counters before anything is written; a mismatch stops the script.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from typing import Any

import yaml

from kv_routing_sim.eval.bootstrap import (
    DEFAULT_RESAMPLES,
    HEADLINE_RESAMPLES,
    bootstrap_ci_grouped,
    mean,
)
from kv_routing_sim.eval.cost import CostModel, successful_requests
from kv_routing_sim.eval.pareto import to_point
from kv_routing_sim.sim.engine import run

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results"
OUT = RESULTS / "session_cost"
TPOT_CONFIG = REPO / "configs" / "tpot_slo.yaml"
TARGETS = ("first_token", "two_part")
HEADLINE = (1, 2)  # the cell the README shows: one shared prompt, 2 sessions per second
REGISTERED = "docs/09-hypotheses.md, 2026-11-24, S1"


class CheckFailed(RuntimeError):
    """A run disagrees with a stored grid or with its own counters."""


@lru_cache(maxsize=1)
def _grid():
    """Return scripts/build_grid.py set to the validated timing model: the stored grid's code."""
    sys.path.insert(0, str(REPO / "scripts"))
    import build_grid

    build_grid._init_worker("h2b_c2")
    return build_grid


@lru_cache(maxsize=1)
def _stored() -> dict[str, Any]:
    """The stored tables this script reads its replica counts from and checks itself against."""
    return {
        "first_token": json.loads((RESULTS / "h2b_c2" / "h2b_c2.json").read_text()),
        "two_part": json.loads((RESULTS / "t2_two_part_slo" / "analysis.json").read_text()),
        "grid_first_token": json.loads((RESULTS / "demo_grid_ctx.json").read_text()),
        "grid_two_part": json.loads((RESULTS / "t2_two_part_slo" / "grid_A.json").read_text()),
    }


def required(target: str, k: int, load: float, policy_label: str) -> int | None:
    """Fewest replicas at which every seed met the target, or None if no count in the grid did."""
    s = _stored()
    if target == "first_token":
        table, not_met = s["first_token"]["required_replicas"], s["first_token"]["not_met"]
    else:
        table, not_met = s["two_part"]["required"]["two_part"]["A"], s["two_part"]["not_met"]
    n = table[str(k)][str(load)][policy_label]
    return None if n >= not_met else n


@lru_cache(maxsize=2)
def cost_model(target: str) -> CostModel:
    """The grid's cost model; for the two-part target, with T2's per-token limit added."""
    base = _grid()._cost()
    if target == "first_token":
        return base
    slo = yaml.safe_load(TPOT_CONFIG.read_text())["slo"]
    if float(slo["ttft_p95_ms"]) != base.ttft_p95_slo_ms:
        raise CheckFailed("the two-part target's first-token limit differs from the cost model's")
    return replace(base, tpot_p95_slo_ms=float(slo["tpot_p95_ms"]))


def new_tokens(trace, block_size: int) -> list[int]:
    """Per request, in trace order: the prompt tokens no earlier request sent as a full block.

    Block hashes are chained, so the blocks a request shares with earlier ones are a leading run.
    The unfinished last block is always new: it is never cached. Summed over a trace this is the
    least any routing could compute: each distinct full block once, plus every unfinished block.
    """
    seen: set[int] = set()
    out = []
    for req in trace:
        known = 0
        for block in req.block_hashes:
            if block not in seen:
                break
            known += 1
        seen.update(req.block_hashes)
        out.append(req.prompt_tokens - known * block_size)
    return out


def measure(trace, *, policy: str, alpha: float | None, n: int, seed: int, profile,
            cache_model: str, cost: CostModel) -> dict[str, Any]:
    """Run one configuration and return its counters, latencies, cost and per-request cache use."""
    r = run(trace, policy=policy, n_replicas=n, seed=seed, profile=profile,
            cache_model=cache_model, record_per_request=True,
            policy_kwargs=None if alpha is None else {"load_penalty": alpha})
    if len(r.per_request) != len(trace):
        raise CheckFailed(f"{len(trace) - len(r.per_request)} requests did not complete")
    point = to_point(r, cost)
    ok = successful_requests(r.ttft_ms, cost.ttft_p95_slo_ms, r.tpot_ms, cost.tpot_p95_slo_ms)
    return {
        "hit_rate": r.hit_rate,
        "queries": r.cache_queries,
        "hits": r.cache_hits,
        "preemptions": r.preemptions,
        "preempted_recompute": r.preempted_queries - r.preempted_hits,
        "p95_ttft_ms": point.p95_ttft_ms,
        "p95_tpot_ms": point.p95_tpot_ms,
        "inr_per_1k": point.inr_per_1k,
        "meets": bool(point.meets_slo),
        "makespan_s": r.makespan_s,
        "successful_share": ok / len(r.ttft_ms),
        "cached": [r.per_request[q.request_id]["cached_tokens"] for q in trace],
    }


def check_run(trace, m: dict[str, Any], new_total: int) -> int:
    """Check a run's counters against the trace and return its re-prefilled tokens."""
    sent = sum(q.prompt_tokens for q in trace)
    if m["queries"] != sent:
        raise CheckFailed(f"cache_queries {m['queries']:,} differs from prompt tokens {sent:,}")
    by_request = sent - sum(m["cached"])
    if m["preemptions"] == 0 and by_request != m["queries"] - m["hits"]:
        raise CheckFailed("per-request records and counters disagree on prefilled tokens")
    again = m["queries"] - m["hits"] + m["preempted_recompute"] - new_total
    if again < 0:
        raise CheckFailed(f"re-prefilled tokens are negative ({again:,})")
    return again


def _job(key: tuple) -> tuple[tuple, dict[str, Any]]:
    target, k, load, policy, alpha, n, seed = key
    g = _grid()
    trace, _ = g._trace(k, load)
    return key, measure(trace, policy=policy, alpha=alpha, n=n, seed=seed, profile=g._profile(),
                        cache_model=g._p5()["cache_model"], cost=cost_model(target))


def _check_stored(target: str, k: int, load: float, label: str, n: int, runs: list[dict]) -> None:
    """Reproduction rule 1: the re-run cell equals the stored cell."""
    s = _stored()
    hit = round(mean([m["hit_rate"] for m in runs]), 4)
    where = f"{target}, K={k}, {load}/s, {label}, {n} replicas"
    if target == "first_token":
        g = s["grid_first_token"]
        ax = g["axes"]
        want = g["cells"][ax["prompts"].index(k)][ax["loads"].index(load)][
            ax["policies"].index(label)][ax["replicas"].index(n)]
        got = [hit, None, None, round(mean([m["inr_per_1k"] for m in runs]), 3), 1]
        if (got[0], got[3], got[4]) != (want[0], want[3], want[4]):
            raise CheckFailed(f"{where}: got hit rate and rupees {got[0]}, {got[3]}; "
                              f"stored {want[0]}, {want[3]} (meets {want[4]})")
        return
    cell = s["grid_two_part"]["cells"][str(k)][str(load)][label]
    i = s["grid_two_part"]["axes"]["replicas"].index(n)
    for field in ("p95_ttft_ms", "p95_tpot_ms"):
        got_p = [round(m[field], 3) for m in runs]
        want_p = cell[{"p95_ttft_ms": "ttft_p95_ms", "p95_tpot_ms": "tpot_p95_ms"}[field]][i]
        if got_p != want_p:
            raise CheckFailed(f"{where}: {field} {got_p}, stored {want_p}")
    if hit != cell["hit_rate"][i]:
        raise CheckFailed(f"{where}: hit rate {hit}, stored {cell['hit_rate'][i]}")


def _row(target: str, k: int, load: float, label: str, n: int, runs: list[dict], trace,
         new: list[int], headline: bool) -> dict[str, Any]:
    """Fold one cell's seeds into a table row. Every run must meet the target it was sized for."""
    sessions: dict[str, list[int]] = {}
    for i, q in enumerate(trace):
        sessions.setdefault(q.session_id, []).append(i)
    n_sessions, new_total = len(sessions), sum(new)
    again = [check_run(trace, m, new_total) for m in runs]
    if not all(m["meets"] for m in runs):
        raise CheckFailed(f"{target}, K={k}, {load}/s, {label}: a seed misses the target at {n}")
    # Each session's re-prefilled tokens, averaged over the seeds. A new block is charged to the
    # session that sends it first. Tokens computed again after a preemption are not in a request's
    # record, so they are reported beside the interval, not inside it.
    per_session = [
        [mean([float(sum(trace[i].prompt_tokens - m["cached"][i] - new[i] for i in idx))
               for m in runs])]
        for idx in sessions.values()
    ]
    ci = bootstrap_ci_grouped(per_session, mean, seed=0,
                              n_resamples=HEADLINE_RESAMPLES if headline else DEFAULT_RESAMPLES)
    sent = sum(q.prompt_tokens for q in trace)
    inr_1k = mean([m["inr_per_1k"] for m in runs])
    return {
        "target": target, "prompts": k, "load": load, "policy": label, "replicas": n,
        "hit_rate": round(mean([m["hit_rate"] for m in runs]), 4),
        "prompt_tokens_per_session": round(sent / n_sessions, 1),
        "new_tokens_per_session": round(new_total / n_sessions, 1),
        "prefilled_tokens_per_session": round(
            mean([float(a) for a in again]) / n_sessions + new_total / n_sessions, 1),
        "re_prefilled_tokens_per_session": round(mean([float(a) for a in again]) / n_sessions, 1),
        "re_prefilled_ci": ci.as_dict(),
        "recomputed_after_preemption_per_session": round(
            mean([float(m["preempted_recompute"]) for m in runs]) / n_sessions, 1),
        "inr_per_1k_requests": round(inr_1k, 3),
        "inr_per_session": round(inr_1k * len(trace) / n_sessions / 1000.0, 5),
        "successful_share": round(mean([m["successful_share"] for m in runs]), 4),
        "run_length_s": round(mean([m["makespan_s"] for m in runs]), 1),
    }


def predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score S1a, S1b and S1c on the headline cell, by the rules written before the run."""
    k, load = HEADLINE
    at = {(r["target"], r["policy"]): r for r in rows if (r["prompts"], r["load"]) == (k, load)}
    new = next(iter(at.values()))["new_tokens_per_session"]
    sticky = at[("first_token", "session-sticky")]["re_prefilled_tokens_per_session"]
    rr = at[("first_token", "round-robin")]["re_prefilled_tokens_per_session"]
    ratios = {}
    for (target, policy), r in at.items():
        if target == "two_part" and ("first_token", policy) in at:
            first = at[("first_token", policy)]
            ratios[policy] = round((r["inr_per_session"] / first["inr_per_session"])
                                   / (r["replicas"] / first["replicas"]), 4)
    return {
        "S1a": {"new_tokens_per_session": new, "range": [500, 540], "holds": 500 <= new <= 540},
        "S1b": {"session_sticky": sticky, "below": 100, "round_robin": rr, "above": 500,
                "holds": sticky < 100 and rr > 500},
        "S1c": {"rupee_ratio_over_replica_ratio": ratios, "within": 0.10,
                "holds": bool(ratios) and all(abs(v - 1.0) <= 0.10 for v in ratios.values())},
    }


def _table(rows: list[dict[str, Any]], target: str, k: int, load: float) -> list[str]:
    lines = [
        "| Policy | T4s | Hit rate | Prefilled | Re-prefilled (95% interval) | ₹ per session |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    picked = [r for r in rows if (r["target"], r["prompts"], r["load"]) == (target, k, load)]
    for r in sorted(picked, key=lambda r: (r["inr_per_session"], r["policy"])):
        ci = r["re_prefilled_ci"]
        lines.append(
            f"| {r['policy']} | {r['replicas']} | {r['hit_rate']:.1%} | "
            f"{r['prefilled_tokens_per_session']:,.0f} | "
            f"{r['re_prefilled_tokens_per_session']:,.0f} ({ci['ci_lo']:,.0f} to "
            f"{ci['ci_hi']:,.0f}) | {r['inr_per_session']:.4f} |")
    return lines


def report(payload: dict[str, Any]) -> str:
    rows, (k, load) = payload["rows"], HEADLINE
    first = next(r for r in rows if (r["prompts"], r["load"]) == (k, load))
    p = payload["predictions"]
    mark = {True: "held", False: "did not hold"}
    out = [
        "# What one agent session costs under each routing policy (S1)",
        "",
        f"Registered before it ran: {REGISTERED}. Made by `scripts/session_cost.py`; the numbers",
        "are in `session_cost.json` beside this file. A derived table: its cells are cells of",
        "H2b on the validated timing model (calibration A), run again and checked against the",
        "stored grids. One GPU type (T4), one model, generated traffic.",
        "",
        (f"A session sends {first['prompt_tokens_per_session']:,.0f} prompt tokens in "
         f"{payload['requests_per_session']:g} requests. "
         f"{first['new_tokens_per_session']:,.0f} of them are new in this replay: text that no "
         "earlier request sent, which no routing can avoid computing. The replay is open-loop, so "
         "the model's own earlier replies count as new text; a live engine could still hold them."),
        "Prefilled is what the GPUs computed. Re-prefilled is prefilled minus new: work a shared",
        "cache that never forgot would have saved. Each policy is shown at the fewest T4s that",
        "meet the target. Session-sticky's re-prefill is a one-time cost, the shared prompt",
        "computed once on each further GPU, so its figure per session falls as a trace gets",
        "longer.",
        "",
        f"## One shared prompt, {load:g} sessions per second",
        "",
        "First-token target (p95 time to first token within 2 s):",
        "",
        *_table(rows, "first_token", k, load),
        "",
        "Two-part target (also p95 time per output token within 100 ms):",
        "",
        *_table(rows, "two_part", k, load),
        "",
        "A policy with no row met the target at no replica count of the grid. Intervals are",
        "percentile bootstrap intervals over sessions. Rupees are counted over a trace's whole",
        "run, which includes the time after the last arrival.",
        "",
        "## The three predictions",
        "",
        (f"- **S1a {mark[p['S1a']['holds']]}.** New tokens per session: "
         f"{p['S1a']['new_tokens_per_session']:,.1f} (predicted 500 to 540)."),
        (f"- **S1b {mark[p['S1b']['holds']]}.** Re-prefilled per session, first-token target: "
         f"session-sticky {p['S1b']['session_sticky']:,.1f} (predicted below 100), round-robin "
         f"{p['S1b']['round_robin']:,.1f} (predicted above 500)."),
        (f"- **S1c {mark[p['S1c']['holds']]}.** Ratio of rupees divided by ratio of replicas, "
         "two-part against first-token (predicted within 10% of 1): "
         + ", ".join(f"{name} {v:.3f}"
                     for name, v in p["S1c"]["rupee_ratio_over_replica_ratio"].items()) + "."),
        "",
        (f"All {payload['n_cells']} rows, for every prompt count and load, are in "
         "`session_cost.json`."),
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs=2, metavar=("PROMPTS", "LOAD"),
                    help="run one cell and print its rows; nothing is written")
    args = ap.parse_args()
    g = _grid()
    src, p5 = g._src(), g._p5()
    ks, seeds = p5["h2b"]["n_shared_prefixes"], p5["h2b"]["seeds"]
    loads, policies = src["loads"], g._policies()
    if args.only:
        ks = [k for k in ks if k == int(args.only[0])]
        loads = [ld for ld in loads if float(ld) == float(args.only[1])]
        if not ks or not loads:
            raise SystemExit("error: that cell is not in the grid")

    cells = [(t, k, ld, p, al, required(t, k, ld, g.label(p, al)))
             for t in TARGETS for k in ks for ld in loads for p, al in policies]
    cells = [c for c in cells if c[5] is not None]
    jobs = [(*c, s) for c in cells for s in seeds]
    t0 = time.monotonic()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = dict(ex.map(_job, jobs, chunksize=4))

    block_size = g._profile().block_size
    rows = []
    for t, k, ld, p, al, n in cells:
        trace, _ = g._trace(k, ld)
        runs = [results[(t, k, ld, p, al, n, s)] for s in seeds]
        label = g.label(p, al)
        _check_stored(t, k, ld, label, n, runs)
        rows.append(_row(t, k, ld, label, n, runs, trace, new_tokens(trace, block_size),
                         headline=(k, ld) == HEADLINE))
    wall = time.monotonic() - t0
    print(f"{len(jobs):,} runs in {wall:.0f} s · {len(rows)} rows · every run equals its stored "
          "cell and passes its own checks")
    if args.only:
        for t in TARGETS:
            print("\n".join([f"\n{t}:"] + _table(rows, t, ks[0], loads[0])))
        return 0

    from _spike_common import results_header

    trace, _ = g._trace(*HEADLINE)
    payload = results_header("scripts/session_cost.py", seed=0, hardware_path="cpu-only")
    payload.update({
        "registered": REGISTERED,
        "reads": {
            "traffic, seeds, cost model": "configs/phase5_crossover.yaml (h2b)",
            "profile, loads, policies": "configs/h2b_c2.yaml, as scripts/build_grid.py reads it",
            "replicas, first-token target": "artifacts/results/h2b_c2/h2b_c2.json",
            "replicas, two-part target": "artifacts/results/t2_two_part_slo/analysis.json (A)",
            "checked against": ["artifacts/results/demo_grid_ctx.json",
                                "artifacts/results/t2_two_part_slo/grid_A.json"],
        },
        "profile": g._profile().provenance(),
        "block_size": block_size,
        "targets": {t: {"ttft_p95_ms": cost_model(t).ttft_p95_slo_ms,
                        "tpot_p95_ms": cost_model(t).tpot_p95_slo_ms} for t in TARGETS},
        "seeds": seeds,
        "requests_per_session": len(trace) / len({q.session_id for q in trace}),
        "headline_cell": {"prompts": HEADLINE[0], "load": HEADLINE[1]},
        "interval": {"method": "percentile bootstrap", "unit": "session", "confidence": 0.95,
                     "resamples": DEFAULT_RESAMPLES, "resamples_headline": HEADLINE_RESAMPLES},
        "rows": rows,
        "predictions": predictions(rows),
        "n_cells": len(rows),
        "n_simulator_runs": len(jobs),
        "runtime_s": round(wall, 1),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "session_cost.json").write_text(json.dumps(payload, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    (OUT / "report.md").write_text(report(payload), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}/session_cost.json and report.md")
    for name, verdict in payload["predictions"].items():
        print(f"  {name}: {'held' if verdict['holds'] else 'did not hold'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as e:
        raise SystemExit(f"error: {e}. Nothing was written.") from None
