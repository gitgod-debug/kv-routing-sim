#!/usr/bin/env python3
"""T2's analysis — committed BEFORE the grid runs, so nothing in it is chosen after a result.

    python scripts/t2_analyze.py
        # artifacts/results/t2_two_part_slo/grid_{A,B,C}.json -> analysis.json, report.md

The procedure, exactly as configs/tpot_slo.yaml registers it. For each calibration the grid file
holds every run's p95 time to first token and p95 time per output token; from those:

  R(policy) = the fewest replicas whose cell meets the target on EVERY seed; `not met` if none.
    first-token target:  p95 TTFT ≤ ttft_p95_ms                       (every result so far)
    two-part target:     that AND p95 TPOT ≤ tpot_p95_ms

  GUARD — under the first-token target, on replicas 1–16, every calibration must reproduce its
          committed table exactly, or this stops: the code drifted and nothing is comparable.
  T2a   — at every load with the prompts in `t2a_prompts`, no policy needs fewer replicas than
          session-sticky under the two-part target.
  T2b   — at every (prompts, load) point, session-sticky needs no more replicas than each policy
          in `t2b_versus`.
  T2c   — over the points where session-sticky's count is defined under both targets, the median
          of R(two-part) ÷ R(first token) is at least `t2c_min_median_ratio`; fewer than
          `t2c_min_points` such points leave it UNDECIDED.
Each rule must hold on ALL calibrations. A point where no policy holds the target is a tie, as in
the claims these rules restate; the report counts such points beside each verdict.

Reported, not tested: every count under both targets, which half binds, how far the calibrations
agree, H2b's rules under the two-part target (exploratory), and the tables at the per-token limits
in `reported_tpot_limits_ms`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from itertools import combinations
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "tpot_slo.yaml"
RESULTS = REPO / "artifacts" / "results" / "t2_two_part_slo"
STICKY, BAL = "session-sticky", "radix-affinity-balanced@16"
SHORT = {"session-sticky": "sticky", "radix-affinity-balanced@16": "balanced α16",
         "radix-affinity-balanced@1": "balanced α1", "round-robin": "round-robin",
         "least-loaded": "least-loaded", "radix-affinity": "pure affinity"}


# ── replica counts from a grid ──────────────────────────────────────────────────────────────────


def required(cell: dict[str, Any], replicas: list[int], ttft_ms: float,
             tpot_ms: float | None, max_replicas: int | None = None) -> int:
    """The fewest replicas whose cell meets the target on every seed; one past the axis if none.
    `tpot_ms=None` is the first-token target alone. `max_replicas` cuts the axis short."""
    for i, n in enumerate(replicas):
        if max_replicas is not None and n > max_replicas:
            return max_replicas + 1
        if all(t <= ttft_ms for t in cell["ttft_p95_ms"][i]) and (
                tpot_ms is None or all(t <= tpot_ms for t in cell["tpot_p95_ms"][i])):
            return n
    return (max_replicas if max_replicas is not None else replicas[-1]) + 1


def table(grid: dict[str, Any], ttft_ms: float, tpot_ms: float | None,
          max_replicas: int | None = None) -> dict[str, dict[str, dict[str, int]]]:
    """{prompts: {load: {policy: R}}} for one calibration under one target."""
    reps = grid["axes"]["replicas"]
    return {k: {ld: {p: required(c, reps, ttft_ms, tpot_ms, max_replicas)
                     for p, c in by_pol.items()} for ld, by_pol in by_load.items()}
            for k, by_load in grid["cells"].items()}


def guard(grid: dict[str, Any], committed: dict[str, Any], ttft_ms: float) -> list[str]:
    """Where the first-token counts on the committed table's own replica range differ from it."""
    top = committed["not_met"] - 1
    mine = table(grid, ttft_ms, None, max_replicas=top)
    return [f"K={k} load={ld} {p}: {mine[k][ld][p]} here, {r} committed"
            for k, by_load in committed["required_replicas"].items()
            for ld, by_pol in by_load.items() for p, r in by_pol.items() if mine[k][ld][p] != r]


# ── the rules ───────────────────────────────────────────────────────────────────────────────────


def _points(tab: dict) -> list[tuple[str, str]]:
    return [(k, ld) for k, by_load in tab.items() for ld in by_load]


def rule_a(tab: dict, prompts: list[int], not_met: int) -> dict[str, Any]:
    """No policy needs fewer replicas than session-sticky, at every load with these prompts."""
    pts = [(k, ld) for k, ld in _points(tab) if int(k) in prompts]
    beaten = [{"K": k, "load": ld, "sticky": tab[k][ld][STICKY],
               "fewer": {p: r for p, r in tab[k][ld].items() if r < tab[k][ld][STICKY]}}
              for k, ld in pts if min(tab[k][ld].values()) < tab[k][ld][STICKY]]
    return {"points": len(pts), "sticky_best_or_tied": len(pts) - len(beaten), "beaten": beaten,
            "none_holds": sum(1 for k, ld in pts if min(tab[k][ld].values()) == not_met),
            "holds": bool(pts) and not beaten}


def rule_b(tab: dict, versus: list[str]) -> dict[str, Any]:
    """Session-sticky needs no more replicas than each of `versus`, at every point."""
    pts = _points(tab)
    worse = [{"K": k, "load": ld, "sticky": tab[k][ld][STICKY], "versus": q, "needs": tab[k][ld][q]}
             for k, ld in pts for q in versus if tab[k][ld][STICKY] > tab[k][ld][q]]
    return {"points": len(pts), "worse": worse, "holds": bool(pts) and not worse}


def rule_c(first: dict, both: dict, not_met: int, min_ratio: float,
           min_points: int) -> dict[str, Any]:
    """How optimistic the first-token counts were, for session-sticky."""
    ratios = {f"{k}@{ld}": both[k][ld][STICKY] / first[k][ld][STICKY]
              for k, ld in _points(both)
              if both[k][ld][STICKY] < not_met and first[k][ld][STICKY] < not_met}
    decided = len(ratios) >= min_points
    med = statistics.median(ratios.values()) if ratios else None
    return {"points": len(ratios), "median_ratio": med, "ratios": ratios, "decided": decided,
            "min_ratio": min_ratio, "holds": decided and med >= min_ratio}


def h2b_rules(tab: dict) -> dict[str, Any]:
    """H2b's three registered rules, unchanged — here applied for the record only."""
    ks = sorted(tab, key=int)
    win = {(k, ld): ("affinity" if r[BAL] < r[STICKY] else "sticky" if r[STICKY] < r[BAL]
                     else "tie") for k in ks for ld, r in tab[k].items()}
    loads = list(tab[ks[0]])
    a = [ld for ld in loads if win[(ks[0], ld)] == "affinity"]
    b = [ld for ld in loads if win[(ks[-1], ld)] == "affinity"]
    rev = [[ld, k1, k2] for ld in loads for k1, k2 in combinations(ks, 2)
           if win[(k1, ld)] == "affinity" and win[(k2, ld)] == "sticky"]
    return {"affinity_wins_at_k_min": a, "affinity_wins_at_k_max": b, "reversals": rev,
            "holds": not a and bool(b) and not rev}


def agreement(tabs: dict[str, dict], not_met: int) -> dict[str, Any]:
    """How far the calibrations agree on a count, cell by cell."""
    names = list(tabs)
    cells = [[tabs[t][k][ld][p] for t in names]
             for k, by_load in tabs[names[0]].items() for ld, by_pol in by_load.items()
             for p in by_pol]
    held = [c for c in cells if all(x < not_met for x in c)]
    pts = _points(tabs[names[0]])
    best = {t: sum(1 for k, ld in pts if tabs[t][k][ld][STICKY] == min(tabs[t][k][ld].values()))
            for t in names}
    return {
        "cells": len(cells), "identical": sum(1 for c in cells if len(set(c)) == 1),
        "within_one": sum(1 for c in cells if max(c) - min(c) <= 1),
        "disagree_on_whether_held": sum(1 for c in cells if 0 < sum(x == not_met for x in c)
                                        < len(c)),
        "largest_spread_where_all_hold": max((max(c) - min(c) for c in held), default=None),
        "sticky_best_or_tied": best, "points": len(pts),
        "sticky_best_or_tied_on_all": sum(
            1 for k, ld in pts
            if all(tabs[t][k][ld][STICKY] == min(tabs[t][k][ld].values()) for t in names)),
    }


def analyze(cfg: dict[str, Any], grids: dict[str, dict], committed: dict[str, dict]) -> dict:
    slo, t2 = cfg["slo"], cfg["t2"]
    ttft, tpot, rules = slo["ttft_p95_ms"], slo["tpot_p95_ms"], t2["rules"]
    names = list(t2["calibrations"])
    missing = [n for n in names if n not in grids]
    if missing:
        raise SystemExit(f"error: no grid for calibration(s) {missing} — run "
                         "scripts/run_t2_grid.py")
    for n in names:
        diff = guard(grids[n], committed[n], ttft)
        if diff:
            raise SystemExit(f"error: calibration {n} does not reproduce its committed "
                             f"first-token table at {len(diff)} cell(s), e.g. {diff[0]}. The "
                             "code has drifted; stopping.")
    not_met = t2["replicas_max"] + 1
    first = {n: table(grids[n], ttft, None) for n in names}
    both = {n: table(grids[n], ttft, tpot) for n in names}
    out: dict[str, Any] = {
        "script": "scripts/t2_analyze.py", "slo": slo, "not_met": not_met, "calibrations": names,
        "guard": {n: (f"reproduces the committed first-token table on replicas "
                      f"1–{committed[n]['not_met'] - 1}") for n in names},
        "required": {"first_token": first, "two_part": both},
    }
    a = {n: rule_a(both[n], rules["t2a_prompts"], not_met) for n in names}
    b = {n: rule_b(both[n], rules["t2b_versus"]) for n in names}
    c = {n: rule_c(first[n], both[n], not_met, rules["t2c_min_median_ratio"],
                   rules["t2c_min_points"]) for n in names}
    out["t2a"] = {"prompts": rules["t2a_prompts"], "by_calibration": a, "decided": True,
                  "holds": all(x["holds"] for x in a.values())}
    out["t2b"] = {"versus": rules["t2b_versus"], "by_calibration": b, "decided": True,
                  "holds": all(x["holds"] for x in b.values())}
    out["t2c"] = {"by_calibration": c, "decided": all(x["decided"] for x in c.values()),
                  "holds": all(x["holds"] for x in c.values())}
    # Reported.
    cells = [(n, k, ld, p) for n in names for k, by_load in both[n].items()
             for ld, by_pol in by_load.items() for p in by_pol]
    out["binding"] = {
        "cells": len(cells),
        "per_token_half_adds_replicas": sum(
            1 for n, k, ld, p in cells if both[n][k][ld][p] > first[n][k][ld][p]),
        "same_count_under_both": sum(
            1 for n, k, ld, p in cells
            if both[n][k][ld][p] == first[n][k][ld][p] < not_met),
        "held_under_neither": sum(1 for n, k, ld, p in cells if first[n][k][ld][p] == not_met),
    }
    out["agreement"] = {"first_token": agreement(first, not_met),
                        "two_part": agreement(both, not_met)}
    out["h2b_exploratory"] = {"two_part": {n: h2b_rules(both[n]) for n in names},
                              "first_token": {n: h2b_rules(first[n]) for n in names}}
    out["other_limits"] = {}
    for limit in t2.get("reported_tpot_limits_ms", []):
        tabs = {n: table(grids[n], ttft, limit) for n in names}
        out["other_limits"][str(limit)] = {
            "required": tabs,
            "t2c_median_ratio": {n: rule_c(first[n], tabs[n], not_met, 0.0, 1)["median_ratio"]
                                 for n in names},
            "sticky_best_or_tied": agreement(tabs, not_met)["sticky_best_or_tied"],
            "h2b": {n: h2b_rules(tabs[n])["holds"] for n in names},
        }
    out["hit_rate_at_8"] = {
        n: {k: {ld: {p: c["hit_rate"][grids[n]["axes"]["replicas"].index(8)]
                     for p, c in by_pol.items()} for ld, by_pol in by_load.items()}
            for k, by_load in grids[n]["cells"].items()}
        for n in names if 8 in grids[n]["axes"]["replicas"]}
    out["predicted_outcomes"] = {k: v for k, v in cfg["predictions"].items() if k.startswith("t2")}
    return out


# ── the report ──────────────────────────────────────────────────────────────────────────────────


def _r(x: int, not_met: int) -> str:
    return "—" if x == not_met else str(x)


def _triple(tabs: dict[str, dict], k: str, ld: str, p: str, not_met: int) -> str:
    vals = [tabs[n][k][ld][p] for n in tabs]
    text = " / ".join(_r(v, not_met) for v in vals)
    return f"**{text}**" if len(set(vals)) > 1 else text


def _verdict(rule: dict[str, Any]) -> str:
    if not rule["decided"]:
        return "UNDECIDED"
    return "HOLDS" if rule["holds"] else "DOES NOT HOLD"


def markdown(a: dict[str, Any], grids: dict[str, dict]) -> str:
    slo, nm, names = a["slo"], a["not_met"], a["calibrations"]
    first, both = a["required"]["first_token"], a["required"]["two_part"]
    g0 = grids[names[0]]
    pols = g0["axes"]["policies"]
    pred = a["predicted_outcomes"]
    abc = " / ".join(names)
    lines = [
        "# T2 — the finding's grid under a target with both halves",
        "",
        (f"Registered `configs/tpot_slo.yaml`. H2b-C2's traffic, loads, policies and seeds on "
         f"{len(names)} calibrations of the same T4 ({abc}), replicas 1–{nm - 1} · "
         f"{sum(g['n_runs'] for g in grids.values()):,} simulator runs · grids generated at "
         f"`{'`, `'.join(sorted({g['git_sha'] for g in grids.values()}))}`"
         f"{' (a dirty tree)' if any(g['git_dirty'] for g in grids.values()) else ''}. "
         "Generated by `scripts/t2_analyze.py`, committed before the grid ran."),
        "",
        (f"**First-token target** (every result so far): p95 TTFT ≤ {slo['ttft_p95_ms']:,} ms. "
         f"**Two-part target:** that, and p95 time per output token ≤ {slo['tpot_p95_ms']:g} ms. "
         "Guard: under the first-token target each calibration reproduces its committed table "
         "exactly."),
        "",
        "| rule (each must hold on every calibration) | registered prediction | result |",
        "|---|---|---|",
        (f"| **T2a** — no policy needs fewer replicas than session-sticky, at every load with "
         f"{' and '.join(str(k) for k in a['t2a']['prompts'])} prompts | {pred['t2a']} | "
         f"**{_verdict(a['t2a'])}** |"),
        (f"| **T2b** — session-sticky never needs more than "
         f"{' or '.join(SHORT[q] for q in a['t2b']['versus'])} | {pred['t2b']} | "
         f"**{_verdict(a['t2b'])}** |"),
        (f"| **T2c** — for session-sticky, the two-part count is at least "
         f"{a['t2c']['by_calibration'][names[0]]['min_ratio']:g}× the first-token count (median) "
         f"| {pred['t2c']} | **{_verdict(a['t2c'])}** |"),
        "",
        f"| | {' | '.join(names)} |",
        "|---|" + "---|" * len(names),
    ]
    ra, rb, rc = (a[x]["by_calibration"] for x in ("t2a", "t2b", "t2c"))
    lines += [
        "| T2a: points where sticky is best or tied | " + " | ".join(
            f"{ra[n]['sticky_best_or_tied']} of {ra[n]['points']}"
            + (f" ({ra[n]['none_holds']} with no policy holding)" if ra[n]["none_holds"] else "")
            for n in names) + " |",
        "| T2b: comparisons where sticky needs more | " + " | ".join(
            str(len(rb[n]["worse"])) for n in names) + " |",
        "| T2c: median of two-part ÷ first-token, sticky | " + " | ".join(
            "—" if rc[n]["median_ratio"] is None else
            f"{rc[n]['median_ratio']:.2f}× ({rc[n]['points']} points)" for n in names) + " |",
        "",
    ]
    for n in names:
        for x in ra[n]["beaten"]:
            fewer = ", ".join(f"{SHORT[p]} {r}" for p, r in x["fewer"].items())
            lines.append(f"- T2a, calibration {n}: at K {x['K']}, {x['load']} sessions/s "
                         f"session-sticky needs {_r(x['sticky'], nm)}; fewer: {fewer}.")
        for x in rb[n]["worse"]:
            lines.append(f"- T2b, calibration {n}: at K {x['K']}, {x['load']} sessions/s "
                         f"session-sticky needs {_r(x['sticky'], nm)}, {SHORT[x['versus']]} "
                         f"{x['needs']}.")
    if lines[-1].startswith("- "):
        lines.append("")

    lines += [(f"## Replicas needed under the two-part target — {abc} (— = not held on "
               f"{nm - 1}; bold = the calibrations differ)"), ""]
    for k in both[names[0]]:
        lines += [f"### K = {k}", "",
                  "| sessions/s | " + " | ".join(SHORT[p] for p in pols) + " |",
                  "|---|" + "---|" * len(pols)]
        for ld in both[names[0]][k]:
            lines.append(f"| {ld} | " + " | ".join(_triple(both, k, ld, p, nm) for p in pols)
                         + " |")
        lines.append("")
    lines += [f"## The same, under the first-token target alone — {abc}", ""]
    for k in first[names[0]]:
        lines += [f"### K = {k}", "",
                  "| sessions/s | " + " | ".join(SHORT[p] for p in pols) + " |",
                  "|---|" + "---|" * len(pols)]
        for ld in first[names[0]][k]:
            lines.append(f"| {ld} | " + " | ".join(_triple(first, k, ld, p, nm) for p in pols)
                         + " |")
        lines.append("")

    b, ag = a["binding"], a["agreement"]
    lines += [
        "## Reported, not tested",
        "",
        (f"- **Which half binds.** Of {b['cells']} (calibration, prompts, load, policy) cells, the "
         f"per-token half raises the replica count at {b['per_token_half_adds_replicas']}; "
         f"{b['same_count_under_both']} need the same number under both targets; "
         f"{b['held_under_neither']} are held under neither on {nm - 1} replicas."),
        (f"- **How far the calibrations agree** (cells identical on all · within one, of "
         f"{ag['two_part']['cells']}): two-part target {ag['two_part']['identical']} · "
         f"{ag['two_part']['within_one']}; first-token target {ag['first_token']['identical']} · "
         f"{ag['first_token']['within_one']}. They disagree on whether the target can be held at "
         f"all at {ag['two_part']['disagree_on_whether_held']} cells (two-part) and "
         f"{ag['first_token']['disagree_on_whether_held']} (first token)."),
        ("- **Session-sticky best or tied, over all 24 points** (two-part target): "
         + " · ".join(f"{n} {v}" for n, v in ag["two_part"]["sticky_best_or_tied"].items())
         + "; on all calibrations at the same point: "
           f"{ag['two_part']['sticky_best_or_tied_on_all']} of {ag['two_part']['points']}."),
        "",
        ("**H2b's rules under the two-part target — EXPLORATORY.** The 64-prompt question gets its "
         "own registered test on fresh calibrations; this is what these three say meanwhile."),
        "",
        f"| | {' | '.join(names)} |",
        "|---|" + "---|" * len(names),
    ]
    h = a["h2b_exploratory"]["two_part"]
    lines += [
        "| (a) at K = 1, balanced affinity (α 16) wins at no load | " + " | ".join(
            "✅" if not h[n]["affinity_wins_at_k_min"] else
            f"⛔ wins at {h[n]['affinity_wins_at_k_min']}" for n in names) + " |",
        "| (b) at K = 64, it wins at one load or more | " + " | ".join(
            f"✅ at {h[n]['affinity_wins_at_k_max']}" if h[n]["affinity_wins_at_k_max"]
            else "⛔ nowhere" for n in names) + " |",
        "| (c) no reversal along K | " + " | ".join(
            "✅" if not h[n]["reversals"] else f"⛔ {h[n]['reversals']}" for n in names) + " |",
        "| H2b | " + " | ".join("holds" if h[n]["holds"] else "falsified" for n in names) + " |",
        "",
    ]
    for limit, o in a["other_limits"].items():
        tabs = o["required"]
        lines += [
            f"### With the per-token limit at {limit} ms instead — {abc}",
            "",
            ("Median of two-part ÷ first-token for session-sticky: "
             + " · ".join("—" if v is None else f"{n} {v:.2f}×"
                          for n, v in o["t2c_median_ratio"].items())
             + ". Session-sticky best or tied (of 24): "
             + " · ".join(f"{n} {v}" for n, v in o["sticky_best_or_tied"].items())
             + ". H2b: " + " · ".join(f"{n} {'holds' if v else 'falsified'}"
                                      for n, v in o["h2b"].items()) + "."),
            "",
        ]
        for k in tabs[names[0]]:
            lines += [f"K = {k}", "",
                      "| sessions/s | " + " | ".join(SHORT[p] for p in pols) + " |",
                      "|---|" + "---|" * len(pols)]
            for ld in tabs[names[0]][k]:
                lines.append(f"| {ld} | " + " | ".join(_triple(tabs, k, ld, p, nm) for p in pols)
                             + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    grids = {n: json.loads(p.read_text()) for n in cfg["t2"]["calibrations"]
             if (p := args.results / f"grid_{n}.json").exists()}
    committed = {n: json.loads((REPO / c["first_token_table"]).read_text())
                 for n, c in cfg["t2"]["calibrations"].items()}
    a = analyze(cfg, grids, committed)
    (args.results / "analysis.json").write_text(json.dumps(a, indent=1, default=str) + "\n")
    (args.results / "report.md").write_text(markdown(a, grids))
    print((args.results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
