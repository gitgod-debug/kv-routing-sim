#!/usr/bin/env python3
"""Post hoc: compare H2b-C2's grid across three calibrations of the same T4.

    python scripts/c3_posthoc_grid.py
        # -> artifacts/results/c3_kaggle_t4/posthoc_three_calibrations.{json,md}

C3 reported one comparison (`scripts/c3_analyze.py --grid`): the grid re-run on the profile from
`kvsim calibrate`'s first run, against the committed table. That comparison reversed H2b's
verdict, so this post hoc script adds the profile from the command's second run and lays the three
tables side by side to show how much of the grid a calibration can move. The three profiles share
the GPU type, model and engine version:
  A: C2's fit (101 measured points, scripts/c2_analyze.py); its table, artifacts/results/h2b_c2/,
     is the committed one behind the README's finding.
  B: `kvsim calibrate` in C3 part 1, the run C3a and C3b were decided on.
  C: `kvsim calibrate` in C3 part 3, later on the same engine.
Tables B and C come from scripts/run_h2b_c2.py with its config unchanged except for the profile.
This script only reads the three results.
"""

from __future__ import annotations

import json
import pathlib
from itertools import combinations

REPO = pathlib.Path(__file__).resolve().parents[1]
C3 = REPO / "artifacts" / "results" / "c3_kaggle_t4"
TABLES = {
    "A": ("C2's fit, 2026-09-30", REPO / "artifacts" / "results" / "h2b_c2" / "h2b_c2.json"),
    "B": ("`kvsim calibrate`, first run", C3 / "h2b_grid" / "h2b_c2.json"),
    "C": ("`kvsim calibrate`, second run", C3 / "h2b_grid_seed1" / "h2b_c2.json"),
}
SHORT = {"session-sticky": "sticky", "radix-affinity-balanced@16": "balanced α16",
         "radix-affinity-balanced@1": "balanced α1", "round-robin": "round-robin",
         "least-loaded": "least-loaded", "radix-affinity": "pure affinity"}
STICKY, BAL = "session-sticky", "radix-affinity-balanced@16"
COEFFS = (("decode base, ms", "decode_base_s", 1e3),
          ("decode per sequence, ms", "decode_per_seq_s", 1e3),
          ("decode per 1K context tokens, ms", "decode_per_ctx_token_s", 1e6),
          ("prefill base, ms", "prefill_base_s", 1e3),
          ("prefill per new token, ms", "prefill_per_token_s", 1e3),
          ("prefill per new × 1K attended tokens, ms", "prefill_attn_s", 1e6))


def sign(a: int, b: int) -> int:
    """-1 if the first needs fewer replicas, 0 for a tie (also when neither holds the SLO), +1."""
    return (a > b) - (a < b)


def analyze(tables: dict[str, dict]) -> dict:
    names = list(tables)
    req = {t: tables[t]["required_replicas"] for t in names}
    nm = tables[names[0]]["not_met"]
    points = [(k, ld) for k, by_load in req[names[0]].items() for ld in by_load]
    pols = list(req[names[0]][points[0][0]][points[0][1]])
    cells = {(k, ld, p): [req[t][k][ld][p] for t in names] for k, ld in points for p in pols}
    spread = {c: max(v) - min(v) for c, v in cells.items()}

    pairs = {}
    for p, q in combinations(pols, 2):
        rows = []
        for k, ld in points:
            s = [sign(req[t][k][ld][p], req[t][k][ld][q]) for t in names]
            rows.append({"K": k, "load": ld, "signs": s, "agree": len(set(s)) == 1})
        agree = [r for r in rows if r["agree"]]
        pairs[f"{p} vs {q}"] = {
            "points_agreeing": len(agree),
            "first_fewer_on_all": sum(1 for r in agree if r["signs"][0] == -1),
            "tie_on_all": sum(1 for r in agree if r["signs"][0] == 0),
            "second_fewer_on_all": sum(1 for r in agree if r["signs"][0] == 1),
            "disagree": [{"K": r["K"], "load": r["load"],
                          "replicas": {t: [req[t][r["K"]][r["load"]][x] for x in (p, q)]
                                       for t in names}} for r in rows if not r["agree"]],
        }
    best = {t: sum(1 for k, ld in points
                   if req[t][k][ld][STICKY] == min(req[t][k][ld].values())) for t in names}
    best_all = [(k, ld) for k, ld in points
                if all(req[t][k][ld][STICKY] == min(req[t][k][ld].values()) for t in names)]
    by_k: dict[str, list[int]] = {}
    for k, _ld in points:
        by_k.setdefault(k, [0, 0])[1] += 1
    for k, _ld in best_all:
        by_k[k][0] += 1
    # Points where sticky needs no more replicas than the other policy on every calibration.
    never_more = {q: sum(1 for k, ld in points
                         if all(req[t][k][ld][STICKY] <= req[t][k][ld][q] for t in names))
                  for q in pols if q != STICKY}
    return {
        "script": "scripts/c3_posthoc_grid.py", "post_hoc": True, "tables": names,
        "not_met": nm, "policies": pols, "points": len(points), "cells": len(cells),
        "cells_identical_on_all": sum(1 for v in spread.values() if v == 0),
        "cells_within_one": sum(1 for v in spread.values() if v <= 1),
        "cells_where_holding_the_slo_differs": sum(
            1 for v in cells.values() if len({x == nm for x in v}) > 1),
        "largest_spread_where_all_hold": max(
            (spread[c] for c, v in cells.items() if nm not in v), default=None),
        "h2b_verdict": {t: tables[t]["verdict"] for t in names},
        "h2b_winner": {f"{k}@{ld}": [tables[t]["winner"][f"{k}@{ld}"] for t in names]
                       for k, ld in points},
        "sticky_best_or_tied": best,
        "sticky_best_or_tied_on_all": len(best_all),
        "sticky_best_or_tied_on_all_by_K": {k: {"points": v[0], "of": v[1]}
                                            for k, v in by_k.items()},
        "sticky_never_needs_more_than": never_more,
        "pairs": pairs,
        "required_replicas": {f"{k}@{ld}": {p: cells[(k, ld, p)] for p in pols}
                              for k, ld in points},
        "profiles": {t: tables[t]["profile"] for t in names},
    }


def markdown(a: dict) -> str:
    names, nm, pols = a["tables"], a["not_met"], a["policies"]

    def cell(v: list[int]) -> str:
        text = " / ".join("—" if x == nm else str(x) for x in v)
        return f"**{text}**" if len(set(v)) > 1 else text

    lines = [
        "# POST HOC — the same grid on three calibrations of one T4",
        "",
        ("⚠️ Post hoc. C3 registered one comparison as a reported item: H2b-C2's grid on the "
         "profile `kvsim calibrate` wrote in its first run, against the committed table. That "
         "comparison reversed H2b's verdict, so the command's second profile was added after "
         "seeing it. Nothing here is a registered test. Generated by "
         "`scripts/c3_posthoc_grid.py`."),
        "",
        "| | profile | fitted |",
        "|---|---|---|",
    ]
    lines += [f"| **{t}** | {TABLES[t][0]} | `{a['profiles'][t]['name']}` |" for t in names]
    lines += ["", "Same GPU type, model and engine version. Their coefficients:", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label, key, scale in COEFFS:
        lines.append(f"| {label} | " + " | ".join(
            f"{a['profiles'][t][key] * scale:.3f}" for t in names) + " |")

    v = a["h2b_verdict"]
    lines += ["", "## H2b's registered rules, on each calibration", "",
              "| rule | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
              "| (a) at K = 1, balanced affinity wins at no load | " + " | ".join(
                  "✅" if v[t]["a_no_affinity_win_at_k1"] else f"⛔ {v[t]['affinity_wins_at_k1']}"
                  for t in names) + " |",
              "| (b) at K = 64, balanced affinity wins at one load or more | " + " | ".join(
                  f"✅ at {v[t]['affinity_wins_at_k64']}" if v[t]["b_affinity_win_at_k64"]
                  else "⛔ nowhere" for t in names) + " |",
              "| (c) no reversal along K | " + " | ".join(
                  "✅" if v[t]["c_no_reversal"] else f"⛔ {v[t]['reversals']}"
                  for t in names) + " |",
              "| **H2b** | " + " | ".join(
                  f"**{'HOLDS' if v[t]['holds'] else 'FALSIFIED'}**" for t in names) + " |",
              "",
              (f"## Replicas needed to hold the SLO — {' / '.join(names)} (— = not held on "
               f"{nm - 1}; bold = the three differ)"), ""]
    by_k: dict[str, list[str]] = {}
    for point in a["required_replicas"]:
        by_k.setdefault(point.split("@")[0], []).append(point)
    for k, pts in by_k.items():
        lines += [f"### K = {k}", "", "| sessions/s | " + " | ".join(SHORT[p] for p in pols) + " |",
                  "|---|" + "---|" * len(pols)]
        lines += [f"| {pt.split('@')[1]} | " + " | ".join(
            cell(a["required_replicas"][pt][p]) for p in pols) + " |" for pt in pts]
        lines.append("")

    b = a["sticky_best_or_tied"]
    lines += [
        "## How much does a calibration move the counts?", "",
        (f"- **{a['cells_identical_on_all']} of {a['cells']}** (K, load, policy) cells need the "
         f"same number of replicas on all three; **{a['cells_within_one']}** differ by at most "
         f"one. Where all three hold the SLO, the largest spread is "
         f"{a['largest_spread_where_all_hold']} replicas."),
        (f"- At **{a['cells_where_holding_the_slo_differs']}** cells the calibrations disagree on "
         f"whether the SLO can be held on {nm - 1} replicas at all."),
        ("- Session-sticky needs the fewest replicas, or ties for them: " + " · ".join(
            f"{t} {b[t]} of {a['points']}" for t in names)
         + f"; on all three at the same point: **{a['sticky_best_or_tied_on_all']} of "
         f"{a['points']}**."),
        ("- By number of prompts, the points where sticky is best or tied on all three: "
         + " · ".join(f"K {k}: {v['points']} of {v['of']}"
                      for k, v in a["sticky_best_or_tied_on_all_by_K"].items()) + "."),
        ("- Points where sticky needs **no more** replicas than the other policy on every "
         "calibration: " + " · ".join(f"{SHORT[q]} {n} of {a['points']}"
                                      for q, n in a["sticky_never_needs_more_than"].items())
         + "."),
        "", "## Which comparisons come out the same on all three", "",
        ("For each pair, the (K, load) points where all three calibrations agree on who needs "
         "fewer replicas — and the points where they do not."), "",
        "| pair | first fewer | tie | second fewer | calibrations disagree |",
        "|---|---:|---:|---:|---|",
    ]
    for name, p in a["pairs"].items():
        first, second = name.split(" vs ")
        if STICKY not in (first, second):
            continue
        where = "; ".join(
            f"K {d['K']} @ {d['load']}/s (" + ", ".join(
                f"{t} {'—' if r[0] == nm else r[0]} v {'—' if r[1] == nm else r[1]}"
                for t, r in d["replicas"].items()) + ")" for d in p["disagree"]) or "—"
        lines.append(f"| {SHORT[first]} vs {SHORT[second]} | {p['first_fewer_on_all']} | "
                     f"{p['tie_on_all']} | {p['second_fewer_on_all']} | {where} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    tables = {t: json.loads(path.read_text()) for t, (_, path) in TABLES.items()}
    a = analyze(tables)
    (C3 / "posthoc_three_calibrations.json").write_text(json.dumps(a, indent=1) + "\n")
    (C3 / "posthoc_three_calibrations.md").write_text(markdown(a))
    print((C3 / "posthoc_three_calibrations.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
