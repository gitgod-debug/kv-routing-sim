#!/usr/bin/env python3
"""Analyse the C4 run against the rules in configs/c4_per_token.yaml.

    python scripts/c4_analyze.py
        # artifacts/results/c4_kaggle_t4/ -> analysis.json, report.md
        # rules E, F1 and F2 also need grids.json: run scripts/c4_grids.py first (CPU only)

"Within f×" means in either direction. Medians are over repeats, of valid cells only. A rule whose
data is missing is undecided, and an undecided rule does not hold. Every predicted number is
profile A's, read from artifacts/results/c4_prereg/predictions.json rather than recomputed. The
decision rules are read from the config; this script applies them and tunes nothing.

  C4a  the command's first run meets C3a's five conditions, recomputed from the raw files, and its
       profile carries the fit of each of its repeats.
  D1a  at each control's batch size, the shared-prefix cell's step is within d1a_max_factor of the
       unique-prompt control's.
  D1b  for every shared cell whose summed context is beyond the largest C2 fitted, the registered
       step is within d1b_max_factor of the measured one.
  D2   for every stall cell, (gap at the arrival − the usual gap) is within d2_max_factor of the
       registered extra gap. A measured extra of zero or less is a miss, not a missing value.
  T3a  for every workload, the registered p95 time per output token is within t3a_max_factor of
       the p95 measured client-side, over the requests whose every token came as its own chunk.
  T3b  for the workloads registered at least t3b_min_margin away from the limit, the model's
       verdict (p95 ≤ limit) equals the engine's: at least 95% of requests at or under the limit,
       from the engine's histogram, or from the client-side times if it has none.
  E    over every ordered pair of runs (i, j), every cell of the firmness grid and both targets,
       the count from run j's pooled fit lies between the smallest and largest count over run i's
       fits (pooled and repeats) in at least e_min_coverage of cases.
  F1   under the first-token target at 64 prompts and f1_load sessions/s, balanced affinity with
       load weight 1 needs fewer replicas than session-sticky on every fresh pooled fit.
  F2   under the two-part target, on every fresh pooled fit: (i) that policy does not hold the
       target at f1_load on the replicas swept; (ii) balanced affinity (α 16) needs fewer replicas
       than session-sticky at no more than f2_max_loads_won loads.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c2_analyze as c2  # nearest-rank percentile and the factor
import c3_analyze as c3  # the command's five conditions, and the engine log's capacity
import t1_itl_check as t1  # C2c's replay, returning the whole run
import t2_analyze as t2  # replica counts from stored percentiles

from kv_routing_sim.sim.cost_model import load_profile

RESULTS = REPO / "artifacts" / "results" / "c4_kaggle_t4"
PREREG = REPO / "artifacts" / "results" / "c4_prereg" / "predictions.json"
STICKY, BAL16, BAL1 = "session-sticky", "radix-affinity-balanced@16", "radix-affinity-balanced@1"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _valid(rows: list[dict], key: str, **match: Any) -> list[float]:
    return [r[key] for r in rows if r.get("valid") and r.get(key) is not None
            and all(r.get(k) == v for k, v in match.items())]


def _all_within(rows: list[dict], limit: float) -> tuple[bool, bool]:
    """(decided, holds) for a rule over rows that each carry a `factor`."""
    decided = bool(rows) and all(r["factor"] is not None for r in rows)
    return decided, decided and all(r["factor"] <= limit for r in rows)


# C4a: the command's runs.


def rule_c4a(data: dict, prereg: dict, results: pathlib.Path) -> dict[str, Any]:
    cfg = prereg["config"]
    want = cfg["rules"]["c4a_repeats"]
    blocks = c3.log_blocks(results, cfg["engine"]["block_size"])
    ran = {r["seed"]: r for r in data.get("calibrate", [])}
    commands = {}
    for seed in cfg["calibrate"]["seeds"]:
        path = results / f"calibrated_profile_seed{seed}.json"
        raw = json.loads(path.read_text()) if path.exists() else None
        facts = c3.command_facts(ran.get(seed), raw, prereg["predictions"]["check"][str(seed)],
                                 blocks)
        reps = ((raw or {}).get("_source") or {}).get("replicates") or []
        facts["conditions"][f"the profile carries the fit of each of its {want} repeats"] = (
            len(reps) == want)
        facts["holds"] = facts["decided"] and all(facts["conditions"].values())
        facts["replicates"] = reps
        commands[str(seed)] = facts
    first = commands[str(cfg["calibrate"]["seeds"][0])]
    return {"commands": commands, "decided": first["decided"], "holds": first["holds"]}


# D1 and D2: the micro-benchmarks.


def rule_d1(data: dict, prereg: dict) -> dict[str, Any]:
    cfg, pred = prereg["config"], prereg["predictions"]
    s, rules = cfg["decode_shared"], cfg["rules"]
    shared, control = data.get("decode_shared", []), data.get("decode_control", [])
    top = prereg["fitted_max_summed_context"]
    cells = []
    for b in s["batches"]:
        p = pred["decode_shared"][str(b)]
        steps = _valid(shared, "step_s", b=b)
        meas = _median(steps)
        cells.append({"b": b, "summed_context": p["summed_context"],
                      "beyond_fitted": p["summed_context"] > top, "reps_valid": len(steps),
                      "measured_step_s": meas, "predicted_step_s": p["step_s"],
                      "factor": c2.factor(p["step_s"], meas),
                      # Signed: above 1 the model is too slow, below 1 too fast.
                      "predicted_over_measured": p["step_s"] / meas if meas else None})
    controls = []
    for length, b in s["controls"]:
        uniq = _median(_valid(control, "step_s", L=length, b=b))
        same = next(c["measured_step_s"] for c in cells if c["b"] == b)
        controls.append({"L": length, "b": b, "unique_step_s": uniq, "shared_step_s": same,
                         "factor": c2.factor(same, uniq),
                         "shared_over_unique": same / uniq if same and uniq else None})
    beyond = [c for c in cells if c["beyond_fitted"]]
    da, ha = _all_within(controls, rules["d1a_max_factor"])
    db, hb = _all_within(beyond, rules["d1b_max_factor"])
    return {"cells": cells, "controls": controls, "fitted_max_summed_context": top,
            "d1a": {"decided": da, "holds": ha, "max_factor": rules["d1a_max_factor"]},
            "d1b": {"decided": db, "holds": hb, "max_factor": rules["d1b_max_factor"],
                    "cells_tested": [c["b"] for c in beyond]}}


def rule_d2(data: dict, prereg: dict) -> dict[str, Any]:
    cfg, pred = prereg["config"], prereg["predictions"]
    limit = cfg["rules"]["d2_max_factor"]
    cells = []
    for b, new, delay in cfg["stall"]["cells"]:
        p = pred["stall"][f"{b}+{new}"]
        rows = [r for r in data.get("stall", []) if r.get("valid") and r.get("b") == b
                and r.get("new_tokens") == new and r.get("gap_at_arrival_s") is not None]
        extra = _median([r["gap_at_arrival_s"] - r["usual_gap_s"] for r in rows])
        if extra is None:
            factor = None
        elif extra <= 0:
            factor = float("inf")  # no stall at all where one was predicted: a miss, not a gap
        else:
            factor = c2.factor(p["extra_gap_s"], extra)
        cells.append({"b": b, "new_tokens": new, "delay_s": delay, "reps_valid": len(rows),
                      "usual_gap_s": _median([r["usual_gap_s"] for r in rows]),
                      "predicted_usual_gap_s": p["usual_gap_s"],
                      "measured_extra_s": extra, "predicted_extra_s": p["extra_gap_s"],
                      "late_ttft_s": _median([r["late_ttft_s"] for r in rows
                                              if r.get("late_ttft_s") is not None]),
                      "factor": factor,
                      "predicted_over_measured": (p["extra_gap_s"] / extra
                                                  if extra and extra > 0 else None)})
    decided, holds = _all_within(cells, limit)
    return {"cells": cells, "decided": decided, "holds": holds, "max_factor": limit}


# T3: traffic around the per-token limit.


def measured_run(m: dict, limit_ms: float) -> dict[str, Any]:
    """Summarise one held-out run from its per-request records, not the notebook's printout."""
    reqs = m["requests"]
    ttft = sorted(r["ttft_ms"] for r in reqs if r["valid"] and r["ttft_ms"])
    tpot = sorted(r["tpot_ms"] for r in reqs if r["valid"] and r.get("tpot_ms") is not None)
    hist = m.get("engine_tpot_hist")
    share = None
    if hist and hist.get("count"):
        share = next((v / hist["count"] for le, v in hist["buckets"].items()
                      if le != "+Inf" and abs(float(le) - limit_ms / 1e3) < 1e-12), None)
    return {
        "requests": len(reqs), "invalid": m["invalid"], "with_tpot": len(tpot),
        "hit_rate": m["hit_rate"], "preemptions": m["preemptions"],
        "ttft_p50_ms": c2.nearest_rank(ttft, 0.50), "ttft_p95_ms": c2.nearest_rank(ttft, 0.95),
        "tpot_p50_ms": c2.nearest_rank(tpot, 0.50), "tpot_p95_ms": c2.nearest_rank(tpot, 0.95),
        "client_share_within_limit": (sum(1 for x in tpot if x <= limit_ms) / len(tpot)
                                      if tpot else None),
        "engine_share_within_limit": share,
        "engine_requests_in_histogram": hist and hist.get("count"),
        "itl_mean_ms": None if m.get("server_itl_s") is None else m["server_itl_s"] * 1e3,
        "worst_send_lag_ms": m["worst_send_lag_ms"],
    }


def rule_t3(data: dict, prereg: dict, fresh: dict[str, Any]) -> dict[str, Any]:
    cfg = prereg["config"]
    rules, limit = cfg["rules"], cfg["slo"]["tpot_p95_ms"]
    by_name = {m["workload"]: m for m in data.get("holdout", [])}
    rows = []
    for name, spec in prereg["workloads"].items():
        reg = prereg["predictions"]["runs"][name]["profile A"]
        m = by_name.get(name)
        row: dict[str, Any] = {"workload": name, "registered": reg,
                               "verdict_tested": reg["margin"] >= rules["t3b_min_margin"],
                               "measured": None, "factor": None, "verdict_agrees": None}
        if m is not None:
            meas = measured_run(m, limit)
            share = meas["engine_share_within_limit"]
            source = "the engine's histogram"
            if share is None:
                share, source = meas["client_share_within_limit"], "client-side token times"
            engine_meets = None if share is None else share >= 0.95
            row.update({
                "measured": meas, "factor": c2.factor(reg["tpot_p95_ms"], meas["tpot_p95_ms"]),
                "predicted_over_measured": (reg["tpot_p95_ms"] / meas["tpot_p95_ms"]
                                            if meas["tpot_p95_ms"] else None),
                "engine_meets_limit": engine_meets, "verdict_source": source,
                "verdict_agrees": (None if engine_meets is None
                                   else engine_meets == reg["meets_per_token_limit"]),
                # Reported.
                "itl_factor": c2.factor(reg["itl_mean_ms"], meas["itl_mean_ms"]),
                "tpot_p50_factor": c2.factor(reg["tpot_p50_ms"], meas["tpot_p50_ms"]),
                "ttft_factor": {q: c2.factor(reg[f"ttft_{q}_ms"], meas[f"ttft_{q}_ms"])
                                for q in ("p50", "p95")},
                "fresh_fits": {},
            })
            for fit_id, profile in fresh.items():  # reported: each run's own fit, at real times
                r = t1.replay(spec, m, profile)
                tp = sorted(v["tpot_ms"] for v in r.per_request.values()
                            if v["tpot_ms"] is not None)
                p95 = c2.nearest_rank(tp, 0.95)
                row["fresh_fits"][fit_id] = {
                    "tpot_p95_ms": p95, "factor": c2.factor(p95, meas["tpot_p95_ms"]),
                    "itl_mean_ms": r.engine_itl_mean_ms(),
                    "itl_factor": c2.factor(r.engine_itl_mean_ms(), meas["itl_mean_ms"])}
        rows.append(row)
    da, ha = _all_within(rows, rules["t3a_max_factor"])
    tested = [r for r in rows if r["verdict_tested"]]
    db = bool(tested) and all(r["verdict_agrees"] is not None for r in tested)
    return {"runs": rows, "limit_ms": limit,
            "t3a": {"decided": da, "holds": ha, "max_factor": rules["t3a_max_factor"]},
            "t3b": {"decided": db, "holds": db and all(r["verdict_agrees"] for r in tested),
                    "workloads_tested": [r["workload"] for r in tested],
                    "min_margin": rules["t3b_min_margin"]}}


# E and F: the fresh calibrations in the simulator.


def _tables(grid: dict, slo: dict) -> dict[str, dict[str, Any]]:
    """{target: {fit: {prompts: {load: {policy: R}}}}} for one of the two grids."""
    out: dict[str, dict[str, Any]] = {"first_token": {}, "two_part": {}}
    for fit_id, cells in grid["fits"].items():
        g = {"axes": grid["axes"], "cells": cells}
        out["first_token"][fit_id] = t2.table(g, slo["ttft_p95_ms"], None)
        out["two_part"][fit_id] = t2.table(g, slo["ttft_p95_ms"], slo["tpot_p95_ms"])
    return out


def _runs_of(fit_ids: list[str]) -> dict[str, list[str]]:
    """{run: its fits, pooled first and then each repeat's}, from ids like s0, s0r1."""
    runs: dict[str, list[str]] = {}
    for f in sorted(fit_ids):
        if f.startswith("s"):
            runs.setdefault(f.split("r")[0], []).append(f)
    return runs


def rule_e(grids: dict, prereg: dict) -> dict[str, Any]:
    cfg = prereg["config"]
    need = cfg["rules"]["e_min_coverage"]
    grid = grids["grids"]["firmness"]
    tabs = _tables(grid, cfg["slo"])
    runs = _runs_of(list(grid["fits"]))
    cases, covered, covered_wide, a_cases, a_covered = 0, 0, 0, 0, 0
    misses = []
    for target, by_fit in tabs.items():
        for i, fits_i in runs.items():
            for k, by_load in by_fit[i].items():
                for ld, by_pol in by_load.items():
                    for pol in by_pol:
                        vals = [by_fit[f][k][ld][pol] for f in fits_i]
                        lo, hi = min(vals), max(vals)
                        if "A" in by_fit:
                            a_cases += 1
                            a_covered += lo <= by_fit["A"][k][ld][pol] <= hi
                        for j in runs:
                            if j == i:
                                continue
                            other = by_fit[j][k][ld][pol]
                            cases += 1
                            covered += lo <= other <= hi
                            covered_wide += lo - 1 <= other <= hi + 1
                            if not lo <= other <= hi:
                                misses.append({"target": target, "range_from": i, "other": j,
                                               "K": k, "load": ld, "policy": pol,
                                               "range": [lo, hi], "count": other})
    decided = len(runs) >= 2 and cases > 0
    coverage = covered / cases if cases else None
    return {"runs": {r: len(f) for r, f in runs.items()}, "cases": cases, "covered": covered,
            "coverage": coverage, "coverage_within_one": covered_wide / cases if cases else None,
            "profile_a_coverage": a_covered / a_cases if a_cases else None,
            "misses": misses, "min_coverage": need, "decided": decided,
            "holds": decided and coverage >= need}


def rule_f(grids: dict, prereg: dict) -> dict[str, Any]:
    cfg = prereg["config"]
    rules = cfg["rules"]
    grid = grids["grids"]["prompts64"]
    tabs = _tables(grid, cfg["slo"])
    not_met = grid["axes"]["replicas"][-1] + 1
    k, load = str(grid["axes"]["prompts"][0]), str(rules["f1_load"])
    fits = sorted(grid["fits"])
    f1 = {f: {"sticky": tabs["first_token"][f][k][load][STICKY],
              "weight_1": tabs["first_token"][f][k][load][BAL1]} for f in fits}
    for v in f1.values():
        v["holds"] = v["weight_1"] < v["sticky"]
    f2 = {}
    for f in fits:
        two = tabs["two_part"][f][k]
        won = [ld for ld, r in two.items() if r[BAL16] < r[STICKY]]
        lost = [ld for ld, r in two.items() if r[STICKY] < r[BAL16]]
        f2[f] = {"weight_1_at_load": two[load][BAL1], "weight_1_holds_nothing":
                 two[load][BAL1] == not_met, "loads_balanced_wins": won,
                 "loads_sticky_wins": lost,
                 "holds": two[load][BAL1] == not_met and len(won) <= rules["f2_max_loads_won"]}
    decided = bool(fits)
    return {"fits": fits, "not_met": not_met, "tables": {t: {f: tabs[t][f][k] for f in fits}
                                                         for t in tabs},
            "f1": {"by_fit": f1, "decided": decided,
                   "holds": decided and all(v["holds"] for v in f1.values())},
            "f2": {"by_fit": f2, "decided": decided,
                   "holds": decided and all(v["holds"] for v in f2.values())}}


# All rules together.


def analyze(data: dict, prereg: dict, results: pathlib.Path,
            grids: dict | None = None) -> dict[str, Any]:
    cfg = prereg["config"]
    if data["predictions_sha256"] != prereg["predictions_sha256"]:
        raise SystemExit("error: the run carried different predictions from the registered ones")
    if data["fingerprints"] != prereg["trace_fingerprints_sha256"]:
        raise SystemExit("error: the run replayed different traffic from the registered traces")
    if data["wheel_sha256"] != prereg["wheel"]["sha256"]:
        raise SystemExit("error: the run carried a different wheel from the registered one")
    out: dict[str, Any] = {"script": "scripts/c4_analyze.py"}
    a = rule_c4a(data, prereg, results)
    out["commands"] = a.pop("commands")
    out["c4a"] = a
    d1 = rule_d1(data, prereg)
    out["d1a"], out["d1b"] = d1.pop("d1a"), d1.pop("d1b")
    out["decode"] = d1
    out["d2"] = rule_d2(data, prereg)
    fresh = {f"s{seed}": load_profile(p) for seed in cfg["calibrate"]["seeds"]
             if (p := results / f"calibrated_profile_seed{seed}.json").exists()}
    t3 = rule_t3(data, prereg, fresh)
    out["t3a"], out["t3b"] = t3.pop("t3a"), t3.pop("t3b")
    out["traffic"] = t3
    undecided = {"decided": False, "holds": False, "why": "scripts/c4_grids.py has not been run"}
    if grids is None:
        out["e"], out["f1"], out["f2"] = undecided, undecided, undecided
    else:
        out["e"] = rule_e(grids, prereg)
        f = rule_f(grids, prereg)
        out["f1"], out["f2"] = f.pop("f1"), f.pop("f2")
        out["prompts64"] = f
        out["grids_header"] = {k: grids.get(k) for k in ("git_sha", "git_dirty", "date_utc",
                                                         "n_runs", "runtime_s")}
    out["software"] = c3.software(data)
    out["predicted_outcomes"] = cfg["predictions"]
    return out


# The report.

RULES = (("c4a", "**C4a** — the command, with its repeats, works on a real engine"),
         ("d1a", "**D1a** — sharing a prefix costs nothing by itself"),
         ("d1b", "**D1b** — the straight line continues past the cache"),
         ("d2", "**D2** — prefill and decode add up"),
         ("t3a", "**T3a** — p95 time per output token on traffic"),
         ("t3b", "**T3b** — the verdict on the per-token limit"),
         ("e", "**E** — one run's spread covers another run's count"),
         ("f1", "**F1** — the first-token hint at 64 prompts repeats"),
         ("f2", "**F2** — and there is no saving with both halves"))


def _verdict(rule: dict) -> str:
    if not rule["decided"]:
        return "UNDECIDED" + (f" ({rule['why']})" if rule.get("why") else " (data missing)")
    return "HOLDS" if rule["holds"] else "DOES NOT HOLD"


def _ms(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 1e3:.{digits}f} ms"


def _x(f: float | None) -> str:
    if f is None:
        return "—"
    return "∞" if f == float("inf") else f"{f:.2f}×"


def _signed(f: float | None) -> str:
    """How far the model is from the measurement, and which way."""
    if f is None:
        return "—"
    pct = round((f - 1) * 100)
    return f"{pct:+d}%" if pct else "0%"


def markdown(a: dict, data: dict, prereg: dict) -> str:
    cfg = prereg["config"]
    pred = a["predicted_outcomes"]
    env = data.get("environment", {})
    lines = [
        "# C4 — the per-token model where it matters, on a real T4",
        "",
        (f"Registered `configs/c4_per_token.yaml` · predictions "
         f"`{prereg['predictions_sha256'][:16]}…` and wheel `{prereg['wheel']['sha256'][:16]}…` "
         f"(from `{prereg['wheel']['built_from']}`) match the run · environment "
         f"{env.get('date_utc', '?')} · generated by `scripts/c4_analyze.py`, committed before "
         "the run. Every predicted number is profile A's, registered before the run."),
        "",
        "| rule | registered prediction | result |",
        "|---|---|---|",
        *(f"| {text} | {pred[key]} | **{_verdict(a[key])}** |" for key, text in RULES),
        "",
    ]
    sw = c3._software_line(a["software"])
    if sw:
        lines += [f"Software: {sw}.", ""]
    for r in a["software"]["repairs"]:
        lines += [(f"Pre-flight repair, before anything was measured: `{r['package']}` "
                   f"{r['was'] or '?'} could not be imported → `pip {r['did']}` → {r['now']}."),
                  ""]

    # the command
    seeds = [str(s) for s in cfg["calibrate"]["seeds"]]
    cmds = a["commands"]
    lines += [f"## The command — seed {seeds[0]} is the tested run", "",
              "| condition | " + " | ".join(f"seed {s}" for s in seeds) + " |",
              "|---|" + "---|" * len(seeds)]
    for cond in cmds[seeds[0]]["conditions"]:
        lines.append(f"| {cond} | " + " | ".join(
            ("✅" if cmds[s]["conditions"][cond] else "⛔") if cmds[s]["decided"] else "not run"
            for s in seeds) + " |")
    lines += ["| wall clock | " + " | ".join(
        f"{cmds[s]['wall_clock_s']:.0f} s" if cmds[s]["decided"] else "—" for s in seeds) + " |",
        "", "The fitted coefficients, and how far each run's own repeats spread:", "",
        "| | " + " | ".join(f"seed {s}" for s in seeds) + " | profile A |",
        "|---|" + "---:|" * (len(seeds) + 1)]
    for label, key, scale, unit, digits in c3.COEFFICIENTS:
        cells = []
        for s in seeds:
            fit, reps = cmds[s]["fit"], cmds[s]["replicates"]
            if not fit:
                cells.append("—")
                continue
            text = f"{fit[key] * scale:.{digits}f}"
            if reps:
                vals = [r[key] * scale for r in reps]
                text += f" ({min(vals):.{digits}f}–{max(vals):.{digits}f})"
            cells.append(f"{text} {unit}")
        lines.append(f"| {label} | {' | '.join(cells)} | "
                     f"{getattr(load_profile(REPO / cfg['profile']), key) * scale:.{digits}f} "
                     f"{unit} |")

    # decode beyond the cache
    d = a["decode"]
    lines += ["", "## Part 2 — decode steps of batches that share a cached prefix", "",
              (f"Summed context = b × (prompt + the middle of the timed tokens). The largest any "
               f"calibration cell had measured: {d['fitted_max_summed_context']:,.0f}."), "",
              ("| b | summed context | × the largest fitted | measured step | profile A | "
               "model against measured | reps valid |"),
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in d["cells"]:
        lines.append(
            f"| {c['b']} | {c['summed_context']:,.0f} | "
            f"{c['summed_context'] / d['fitted_max_summed_context']:.1f} | "
            f"{_ms(c['measured_step_s'])} | {_ms(c['predicted_step_s'])} | "
            f"{_signed(c['predicted_over_measured'])} | {c['reps_valid']} |")
    lines += ["", (f"**D1b** is decided on the cells past the largest fitted (b = "
                   f"{', '.join(str(b) for b in a['d1b']['cells_tested'])}): each within "
                   f"{a['d1b']['max_factor']:g}×, i.e. the model between "
                   f"{(1 / a['d1b']['max_factor'] - 1) * 100:+.0f}% and "
                   f"{(a['d1b']['max_factor'] - 1) * 100:+.0f}% of the measurement."), "",
              "| control | unique prompts | shared prefix | shared against unique |",
              "|---|---:|---:|---:|"]
    for c in d["controls"]:
        lines.append(f"| L {c['L']}, b {c['b']} | {_ms(c['unique_step_s'])} | "
                     f"{_ms(c['shared_step_s'])} | {_signed(c['shared_over_unique'])} |")

    # stalls
    lines += ["", "## Part 2b — what one arriving request costs the sequences already decoding",
              "",
              ("| b | usual gap: measured · profile A | extra gap at the arrival: measured · "
               "profile A | model against measured | the late request's TTFT | reps valid |"),
              "|---:|---|---|---:|---:|---:|"]
    for c in a["d2"]["cells"]:
        lines.append(
            f"| {c['b']} | {_ms(c['usual_gap_s'])} · {_ms(c['predicted_usual_gap_s'])} | "
            f"{_ms(c['measured_extra_s'])} · {_ms(c['predicted_extra_s'])} | "
            f"{_signed(c['predicted_over_measured'])} | {_ms(c['late_ttft_s'])} | "
            f"{c['reps_valid']} |")

    # traffic
    t = a["traffic"]
    lines += ["", f"## Part 3 — held-out traffic around the {t['limit_ms']:g} ms per-token limit",
              "",
              ("| workload | p95 ms per token: measured · registered | factor | share of "
               "requests within the limit: engine · client · registered | engine meets it · "
               "model says | mean gap: measured · registered | p95 TTFT: measured · registered |"),
              "|---|---|---:|---|---|---|---|"]
    for r in t["runs"]:
        m, reg = r["measured"], r["registered"]
        if m is None:
            lines.append(f"| {r['workload']} | not run | | | | | |")
            continue

        def pct(x: float | None) -> str:
            return "—" if x is None else f"{x:.1%}"

        meets = {True: "yes", False: "no", None: "—"}
        tested = "" if r["verdict_tested"] else " *(on the limit: not tested)*"
        lines.append(
            f"| {r['workload']} | {m['tpot_p95_ms']:.1f} · {reg['tpot_p95_ms']:.1f} | "
            f"{_x(r['factor'])} | {pct(m['engine_share_within_limit'])} · "
            f"{pct(m['client_share_within_limit'])} · {pct(reg['share_within_limit'])} | "
            f"{meets[r['engine_meets_limit']]} · {meets[reg['meets_per_token_limit']]}{tested} | "
            f"{m['itl_mean_ms']:.1f} · {reg['itl_mean_ms']:.1f} ms ({_x(r['itl_factor'])}) | "
            f"{m['ttft_p95_ms']:.0f} · {reg['ttft_p95_ms']:.0f} ms ({_x(r['ttft_factor']['p95'])})"
            " |")
    lines += ["", ("Reported: each run of the command's own pooled fit, replayed at the real "
                   "send times — p95 ms per token (factor) · mean gap (factor):"), "",
              "| workload | " + " | ".join(f"seed {s}" for s in seeds) + " |",
              "|---|" + "---|" * len(seeds)]
    for r in t["runs"]:
        if r["measured"] is None:
            continue
        cells = []
        for s in seeds:
            f = r["fresh_fits"].get(f"s{s}")
            cells.append("—" if not f else f"{f['tpot_p95_ms']:.1f} ({_x(f['factor'])}) · "
                                           f"{f['itl_mean_ms']:.1f} ({_x(f['itl_factor'])})")
        lines.append(f"| {r['workload']} | " + " | ".join(cells) + " |")
    lines += ["", ("Requests · invalid · with a time per token · worst send lag: "
                   + "; ".join(f"`{r['workload']}` {r['measured']['requests']} · "
                               f"{r['measured']['invalid']} · {r['measured']['with_tpot']} · "
                               f"{r['measured']['worst_send_lag_ms']:.0f} ms"
                               for r in t["runs"] if r["measured"]) + ".")]

    # E, F
    e = a["e"]
    if e["decided"]:
        lines += ["", "## E — does one run's spread cover another run's count?", "",
                  (f"{len(e['runs'])} runs of the command ({', '.join(e['runs'])}), each with "
                   f"its pooled fit and its repeats; firmness grid, both targets: the other "
                   f"run's count is inside the range in **{e['covered']} of {e['cases']}** cases "
                   f"(**{e['coverage']:.1%}**; needed ≥ {e['min_coverage']:.0%}). Widened by one "
                   f"replica each way: {e['coverage_within_one']:.1%}. Profile A's count is "
                   f"inside a fresh run's range in {e['profile_a_coverage']:.1%} of cases.")]
        if e["misses"]:
            lines += ["", ("| target | range from | other run | K | load | policy | range | "
                           "its count |"), "|---|---|---|---|---|---|---|---:|"]
            lines += [f"| {m['target']} | {m['range_from']} | {m['other']} | {m['K']} | "
                      f"{m['load']} | {m['policy']} | {m['range'][0]}–{m['range'][1]} | "
                      f"{m['count']} |" for m in e["misses"][:40]]
    if a["f1"]["decided"]:
        p64 = a["prompts64"]
        nm = p64["not_met"]
        lines += ["", "## F — the 64-prompt question on the fresh calibrations", "",
                  ("| fit | F1: first token, sticky · load weight 1 | F2: both halves — weight "
                   "1 · loads where balanced α16 needs fewer · where sticky does |"),
                  "|---|---|---|"]
        for f in p64["fits"]:
            one, two = a["f1"]["by_fit"][f], a["f2"]["by_fit"][f]
            lines.append(
                f"| {f} | {t2._r(one['sticky'], nm)} · {t2._r(one['weight_1'], nm)} "
                f"{'✅' if one['holds'] else '⛔'} | {t2._r(two['weight_1_at_load'], nm)} · "
                f"{two['loads_balanced_wins'] or 'none'} · {two['loads_sticky_wins'] or 'none'} "
                f"{'✅' if two['holds'] else '⛔'} |")
        for target, title in (("two_part", "both halves"), ("first_token", "the first token")):
            lines += ["", f"Replicas at 64 prompts, target on {title} — "
                          + " / ".join(p64["fits"]) + f" (— = not held on {nm - 1}):", "",
                      "| sessions/s | sticky | balanced α16 | balanced α1 |", "|---|---|---|---|"]
            tabs = p64["tables"][target]
            for ld in tabs[p64["fits"][0]]:
                lines.append(f"| {ld} | " + " | ".join(
                    " / ".join(t2._r(tabs[f][ld][pol], nm) for f in p64["fits"])
                    for pol in (STICKY, BAL16, BAL1)) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    ap.add_argument("--prereg", type=pathlib.Path, default=PREREG)
    args = ap.parse_args()
    data = json.loads((args.results / "c4.json").read_text())
    prereg = json.loads(args.prereg.read_text())
    grids = (json.loads(g.read_text()) if (g := args.results / "grids.json").exists() else None)
    a = analyze(data, prereg, args.results, grids)
    (args.results / "analysis.json").write_text(json.dumps(a, indent=1, default=str) + "\n")
    (args.results / "report.md").write_text(markdown(a, data, prereg))
    print((args.results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
