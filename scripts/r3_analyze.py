#!/usr/bin/env python3
"""R3's analysis — committed BEFORE the run, so nothing in it is chosen after seeing a result.

    python scripts/r3_analyze.py
        # artifacts/results/r3_bailian/grid.json -> analysis.json, report.md

The procedure, exactly as configs/r3_bailian.yaml registers it.

R(policy) is the fewest replicas, of those swept, whose cell meets the target on every seed; a
policy that holds it on none counts as one more than the most swept. A speed is TESTABLE on a
calibration if some policy holds the target there. Each rule is decided under the registered
target, on EVERY calibration: one calibration on which it fails, fails it. A rule with no testable
speed on some calibration is UNDECIDED, and an undecided rule does not hold.

  R3a  the published product needs no more GPUs than keeping sessions in place: R(lmetric) <=
       R(session-sticky) at every testable speed, and fewer at one or more.
  R3b  moving sessions adds nothing without a shared cache store: R(smetric-local) within
       `within` of R(lmetric) at every testable speed.
  R3c  the old finding's two policies tie on real traffic: R(balanced prefix-affinity, the
       registered weight) within `within` of R(session-sticky) at every testable speed.
  R3d  pure prefix-affinity fails: at every testable speed it needs more replicas than
       round-robin, or holds the target on none.

Reported, not tested: the same four rules under the other target; every policy's count on every
calibration under both targets; hit rates, how unevenly requests were spread, and each policy's
own account of its decisions, at the replica count where session-sticky first holds the target.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import t2_analyze as t2  # `required`: replica counts from stored percentiles — imported, not copied

CONFIG = REPO / "configs" / "r3_bailian.yaml"
RULES = (("r3a", "**R3a** — LMetric needs no more GPUs than session-sticky, and fewer somewhere"),
         ("r3b", "**R3b** — moving sessions adds nothing without a shared cache store"),
         ("r3c", "**R3c** — balanced prefix-affinity and session-sticky tie"),
         ("r3d", "**R3d** — pure prefix-affinity fails"))
TARGETS = ("first_token", "two_part")


def tables(grid: dict[str, Any], slo: dict[str, float]) -> dict[str, Any]:
    """{target: {calibration: {speed: {policy: R}}}}."""
    reps = grid["axes"]["replicas"]
    out: dict[str, Any] = {t: {} for t in TARGETS}
    for cal, by_speed in grid["cells"].items():
        for speed, by_pol in by_speed.items():
            for pol, cell in by_pol.items():
                out["first_token"].setdefault(cal, {}).setdefault(speed, {})[pol] = t2.required(
                    cell, reps, slo["ttft_p95_ms"], None)
                out["two_part"].setdefault(cal, {}).setdefault(speed, {})[pol] = t2.required(
                    cell, reps, slo["ttft_p95_ms"], slo["tpot_p95_ms"])
    return out


def testable(tab: dict[str, dict[str, int]], not_met: int) -> list[str]:
    """The speeds of one calibration at which some policy holds the target."""
    return [sp for sp, by_pol in tab.items() if min(by_pol.values()) < not_met]


def decide(tab: dict[str, Any], roles: dict[str, str], within: int,
           not_met: int) -> dict[str, Any]:
    """The four rules under one target: {rule: {decided, holds, by_calibration}}."""
    sticky, lm, sm = roles["sticky"], roles["lmetric"], roles["smetric"]
    bal, pure, rr = roles["balanced"], roles["pure"], roles["baseline"]
    out: dict[str, Any] = {k: {"by_calibration": {}} for k, _ in RULES}
    for cal, by_speed in tab.items():
        speeds = testable(by_speed, not_met)
        rows = {sp: by_speed[sp] for sp in speeds}
        a_le = all(r[lm] <= r[sticky] for r in rows.values())
        a_lt = [sp for sp, r in rows.items() if r[lm] < r[sticky]]
        out["r3a"]["by_calibration"][cal] = {
            "testable": speeds, "holds": bool(speeds) and a_le and bool(a_lt),
            "fewer_at": a_lt, "more_at": [sp for sp, r in rows.items() if r[lm] > r[sticky]],
            "counts": {sp: [r[lm], r[sticky]] for sp, r in rows.items()}}
        for key, x, y in (("r3b", sm, lm), ("r3c", bal, sticky)):
            apart = [sp for sp, r in rows.items() if abs(r[x] - r[y]) > within]
            out[key]["by_calibration"][cal] = {
                "testable": speeds, "holds": bool(speeds) and not apart, "apart_at": apart,
                "counts": {sp: [r[x], r[y]] for sp, r in rows.items()}}
        ok = [sp for sp, r in rows.items() if r[pure] > r[rr] or r[pure] == not_met]
        out["r3d"]["by_calibration"][cal] = {
            "testable": speeds, "holds": bool(speeds) and len(ok) == len(speeds),
            "not_worse_at": [sp for sp in speeds if sp not in ok],
            "counts": {sp: [r[pure], r[rr]] for sp, r in rows.items()}}
    for rule in out.values():
        cals = rule["by_calibration"].values()
        rule["decided"] = bool(rule["by_calibration"]) and all(c["testable"] for c in cals)
        rule["holds"] = rule["decided"] and all(c["holds"] for c in cals)
    return out


def at_sticky(grid: dict[str, Any], tab: dict[str, Any], roles: dict[str, str]) -> dict[str, Any]:
    """Reported: every policy's hit rate, spread and decisions at the replica count where
    session-sticky first holds the registered target (the most swept, if it never does)."""
    reps = grid["axes"]["replicas"]
    out: dict[str, Any] = {}
    for cal, by_speed in grid["cells"].items():
        for speed, by_pol in by_speed.items():
            n = min(tab[cal][speed][roles["sticky"]], reps[-1])
            i = reps.index(n)
            out.setdefault(cal, {})[speed] = {"replicas": n, "policies": {
                pol: {"hit_rate": cell["hit_rate"][i],
                      "busiest_over_mean": cell["busiest_over_mean"][i],
                      "ttft_p95_ms": max(cell["ttft_p95_ms"][i]),
                      "tpot_p95_ms": max(cell["tpot_p95_ms"][i]),
                      "preemptions": cell["preemptions"][i],
                      **({"router": {k: v[i] for k, v in cell["router"].items()}}
                         if "router" in cell else {})}
                for pol, cell in by_pol.items()}}
    return out


def analyze(grid: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if grid.get("probe"):
        raise SystemExit("error: that is the probe's file, not the grid's")
    for key in ("rules", "slo", "policies", "calibrations", "speeds", "replicas_max", "seeds",
                "adapter", "window_s", "warmup_s", "trace_sha256"):
        if grid["config"].get(key) != cfg.get(key):
            raise SystemExit(f"error: the grid was run with a different `{key}` from the "
                             "registered config")
    rules, roles = cfg["rules"], cfg["rules"]["roles"]
    not_met = grid["axes"]["replicas"][-1] + 1
    tabs = tables(grid, cfg["slo"])
    target = rules["target"]
    other = next(t for t in TARGETS if t != target)
    out: dict[str, Any] = {"script": "scripts/r3_analyze.py", "target": target,
                           "not_met": not_met, "tables": tabs}
    out.update(decide(tabs[target], roles, rules["within"], not_met))
    out["reported_other_target"] = {"target": other,
                                    **decide(tabs[other], roles, rules["within"], not_met)}
    out["at_sticky"] = at_sticky(grid, tabs[target], roles)
    out["all_completed"] = all(cell["all_completed"] for by_speed in grid["cells"].values()
                               for by_pol in by_speed.values() for cell in by_pol.values())
    out["structure"] = grid["structure"]
    out["adapter_stats"] = grid["adapter_stats"]
    out["predicted_outcomes"] = cfg["predictions"]
    out["grid_header"] = {k: grid.get(k) for k in ("git_sha", "git_dirty", "date_utc", "n_runs",
                                                   "runtime_s", "trace_sha256")}
    return out


# ── the report ──────────────────────────────────────────────────────────────────────────────────


def _verdict(rule: dict[str, Any]) -> str:
    if not rule["decided"]:
        return "UNDECIDED (no testable speed on some calibration)"
    return "HOLDS" if rule["holds"] else "DOES NOT HOLD"


def _r(n: int, not_met: int) -> str:
    return "—" if n >= not_met else str(n)


def markdown(a: dict[str, Any], grid: dict[str, Any], cfg: dict[str, Any]) -> str:
    nm, pred = a["not_met"], a["predicted_outcomes"]
    cals, speeds = grid["axes"]["calibrations"], grid["axes"]["speeds"]
    pols = grid["axes"]["policies"]
    name = {"first_token": f"p95 TTFT ≤ {cfg['slo']['ttft_p95_ms']:,} ms",
            "two_part": (f"p95 TTFT ≤ {cfg['slo']['ttft_p95_ms']:,} ms and p95 time per output "
                         f"token ≤ {cfg['slo']['tpot_p95_ms']:g} ms")}
    s, h = a["structure"], a["grid_header"]
    lines = [
        "# R3 — the two published policies, on a public production trace",
        "",
        (f"Registered `configs/r3_bailian.yaml` · trace `{cfg['trace']}` (sha256 "
         f"`{h['trace_sha256'][:16]}…`), token scale {cfg['adapter']['token_scale']:g}, the first "
         f"{cfg['window_s']:,} s · {h['n_runs']:,} runs at `{h['git_sha']}`"
         f"{' (dirty tree)' if h['git_dirty'] else ''} · generated by `scripts/r3_analyze.py`, "
         "committed before the run."),
        "",
        (f"The traffic: {s['requests']:,} requests in {s['sessions']:,} sessions (requests per "
         f"session: median {s['requests_per_session']['p50']}, 99th percentile "
         f"{s['requests_per_session']['p99']}, most {s['requests_per_session']['max']}); prompts "
         f"of {s['prompt_tokens']['p50']:,} tokens at the median, {s['prompt_tokens']['max']:,} "
         f"at most. With an unbounded cache {s['reusable_from_any_earlier_request']:.1%} of "
         f"prompt blocks could be reused, {s['reusable_from_the_same_session']:.1%} from the "
         "request's own session."),
        "",
        (f"**The registered target: {name[a['target']]}.** Each rule must hold on every "
         "calibration."),
        "",
        "| rule | registered prediction | result |",
        "|---|---|---|",
        *(f"| {text} | {pred[key]} | **{_verdict(a[key])}** |" for key, text in RULES),
        "",
    ]
    if not a["all_completed"]:
        lines += ["⛔ Some run did not complete every request.", ""]
    o = a["reported_other_target"]
    lines += [f"Reported, not tested — the same rules under {name[o['target']]}: "
              + " · ".join(f"{k.upper()} {_verdict(o[k]).lower()}" for k, _ in RULES) + ".", ""]

    for target in (a["target"], o["target"]):
        tab = a["tables"][target]
        lines += [f"## Replicas needed — {name[target]}", "",
                  f"Each cell: {' / '.join(cals)} (— = not held on {nm - 1}).", "",
                  "| policy | " + " | ".join(f"{sp}×" for sp in speeds) + " |",
                  "|---|" + "---|" * len(speeds)]
        for pol in pols:
            lines.append(f"| {pol} | " + " | ".join(
                " / ".join(_r(tab[c][sp][pol], nm) for c in cals) for sp in speeds) + " |")
        lines.append("")

    lines += ["## At the replica count where session-sticky first holds the registered target",
              "",
              (f"Calibration {cals[0]}. Hit rate · the busiest replica's requests over the mean "
               "· worst seed's p95 TTFT · worst seed's p95 time per token."), "",
              "| policy | " + " | ".join(
                  f"{sp}× (on {a['at_sticky'][cals[0]][sp]['replicas']})" for sp in speeds) + " |",
              "|---|" + "---|" * len(speeds)]
    for pol in pols:
        cells = []
        for sp in speeds:
            v = a["at_sticky"][cals[0]][sp]["policies"][pol]
            cells.append(f"{v['hit_rate']:.3f} · {v['busiest_over_mean']:.2f}× · "
                         f"{v['ttft_p95_ms']:,.0f} ms · {v['tpot_p95_ms']:,.0f} ms")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    with_router = [p for p in pols
                   if "router" in a["at_sticky"][cals[0]][speeds[0]]["policies"][p]]
    for pol in with_router:
        lines += ["", f"`{pol}`'s own account, mean over seeds, at those replica counts: "
                  + " · ".join(
                      f"{sp}×: " + ", ".join(
                          f"{k} {v:,.0f}" for k, v in
                          a["at_sticky"][cals[0]][sp]["policies"][pol]["router"].items())
                      for sp in speeds) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--results", type=pathlib.Path, default=None,
                    help="default: the config's results_dir")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    results = args.results or REPO / cfg["results_dir"]
    grid = json.loads((results / "grid.json").read_text())
    a = analyze(grid, cfg)
    (results / "analysis.json").write_text(json.dumps(a, indent=1, default=str) + "\n")
    (results / "report.md").write_text(markdown(a, grid, cfg))
    print((results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
