#!/usr/bin/env python3
"""R1 and R2 re-run on the validated timing model, with only the profile and speeds changed.

    python scripts/run_realtrace_ctx.py --probe   # round-robin alone on calibration A: the speeds
    python scripts/run_realtrace_ctx.py           # configs/realtrace_ctx.yaml ->
                                                  # artifacts/results/realtrace_ctx/

For every (test, calibration) pair this writes a copy of the test's config
(configs/realtrace_mooncake.yaml or configs/realtrace_agentx.yaml) with `profile` and `speeds`
replaced, and runs `scripts/run_realtrace.py` on it, which applies R1's and R2's own rules
unchanged. It then puts the verdicts side by side and applies one more rule, R1w: on Mooncake,
balanced affinity and session-sticky within `r1w_within` replicas. It decides nothing about a
policy itself.

`--probe` runs that script's `--probe` (round-robin alone) on the first calibration and prints the
speeds the config's rule gives, so no other policy runs before the speeds are fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "realtrace_ctx.yaml"
RUNNER = REPO / "scripts" / "run_realtrace.py"
NOT_MET = 9  # run_realtrace.py's: no replica count from 1 to 8 holds the SLO
STICKY, STAR = "session-sticky", "radix-affinity-balanced@16"
_PROBE = re.compile(r"^probe (\S+) speed (\S+): round-robin needs (\d+|more than 8)$")


def derive(base: dict[str, Any], test: str, cal: str, profile: str, speeds: list[float],
           out_dir: str) -> dict[str, Any]:
    """The test's config with only the profile, speeds, output directory and titles replaced."""
    cfg = dict(base)
    cfg.update({"profile": profile, "speeds": list(speeds), "results_dir": out_dir,
                "name": test,
                "title": (f"{base.get('title') or 'R1 — real traffic: the Mooncake FAST25 traces'}"
                          f" — on the validated timing model, calibration {cal}")})
    changed = {k for k in cfg if cfg[k] != base.get(k)}
    assert changed <= {"profile", "speeds", "results_dir", "name", "title"}, changed
    return cfg


def _rel(path: pathlib.Path) -> str:
    """Repo-relative where it can be, so a derived config reads the same on another machine."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def parse_probe(stdout: str) -> dict[str, dict[str, int | None]]:
    """{trace: {speed: replicas round-robin needs, None if more than 8}} from the probe's lines."""
    needs: dict[str, dict[str, int | None]] = {}
    for line in stdout.splitlines():
        m = _PROBE.match(line.strip())
        if m:
            trace, speed, n = m.groups()
            needs.setdefault(trace, {})[speed] = None if n.startswith("more") else int(n)
    return needs


def choose_speeds(probe_speeds: list[float], needs: dict[str, dict[str, int | None]]) -> list:
    """Apply the speed rule to the probe's results.

    The rule picks the highest probe speed at which round-robin holds the SLO on 8 or fewer
    replicas on every trace, and the two probe speeds below it; [] if there is no such speed.
    """
    ordered = sorted(probe_speeds)
    held = [sp for sp in ordered if needs and all(by.get(f"{sp:g}") is not None
                                                  for by in needs.values())]
    if not held:
        return []
    top = ordered.index(held[-1])
    return ordered[max(0, top - 2):top + 1]


def within(req: dict[str, dict[str, dict[str, int]]], limit: int) -> dict[str, Any]:
    """R1w on one calibration: balanced affinity and session-sticky within `limit` replicas.

    Checked at every testable speed of every trace; `req` is indexed [trace][speed][policy].
    """
    pairs = [(tr, sp, r[STAR], r[STICKY]) for tr, by in req.items() for sp, r in by.items()
             if min(r[STAR], r[STICKY]) < NOT_MET]
    apart = [{"trace": tr, "speed": sp, "balanced": a, "sticky": b}
             for tr, sp, a, b in pairs if abs(a - b) > limit]
    return {"testable": len(pairs), "apart": apart, "decided": bool(pairs),
            "holds": bool(pairs) and not apart}


def verdict(by_cal: dict[str, bool | None]) -> str:
    """HOLDS on every calibration, FALSIFIED on every one, or CALIBRATION-DEPENDENT.

    A None (untestable) on any calibration makes the result UNTESTABLE.
    """
    values = list(by_cal.values())
    if not values or any(v is None for v in values):
        return "UNTESTABLE"
    if all(values):
        return "HOLDS"
    return "FALSIFIED" if not any(values) else "CALIBRATION-DEPENDENT"


def _run(cfg_path: pathlib.Path, out_dir: pathlib.Path, workers: int, *extra: str) -> str:
    out = subprocess.run([sys.executable, str(RUNNER), "--config", str(cfg_path), "--out",
                          str(out_dir), "--workers", str(workers), *extra],
                         check=True, capture_output=True, text=True, cwd=REPO)
    return out.stdout


def summarize(cfg: dict[str, Any], results: pathlib.Path) -> dict[str, Any]:
    out: dict[str, Any] = {"script": "scripts/run_realtrace_ctx.py", "tests": {},
                           "predicted_outcomes": cfg["predictions"], "speeds": cfg["speeds"]}
    for test, base_path in cfg["tests"].items():
        first = json.loads((REPO / yaml.safe_load((REPO / base_path).read_text()).get(
            "results_dir", "artifacts/results/realtrace_mooncake") / "realtrace.json").read_text())
        entry: dict[str, Any] = {"first_model": {
            "holds": first["r1_holds"], "speeds": first["config"]["speeds"],
            "required_replicas": first["required_replicas"],
            "winner_by_speed": {t: v["winner_by_speed"] for t, v in first["verdicts"].items()}},
            "calibrations": {}}
        holds: dict[str, bool | None] = {}
        for cal in cfg["calibrations"]:
            path = results / test / cal / "realtrace.json"
            if not cfg["speeds"][test] or not path.exists():
                holds[cal] = None
                continue
            run = json.loads(path.read_text())
            holds[cal] = bool(run["r1_holds"])
            entry["calibrations"][cal] = {
                "holds": holds[cal], "git_sha": run["git_sha"], "git_dirty": run["git_dirty"],
                "n_runs": run["n_runs"], "required_replicas": run["required_replicas"],
                "hit_rate_at_4": run["hit_rate_at_4"],
                "winner_by_speed": {t: v["winner_by_speed"] for t, v in run["verdicts"].items()},
                "rules": {t: v["rules"] for t, v in run["verdicts"].items()},
                "within": within(run["required_replicas"], cfg["rules"]["r1w_within"])}
        entry["verdict"] = verdict(holds)
        out["tests"][test] = entry
    r1 = out["tests"].get("R1", {}).get("calibrations", {})
    w = {cal: (v["within"]["holds"] if v["within"]["decided"] else None) for cal, v in r1.items()}
    out["R1w"] = {"by_calibration": w,
                  "verdict": verdict(w) if len(w) == len(cfg["calibrations"]) else "UNTESTABLE"}
    return out


def _cells(req: dict[str, dict[str, int]], speeds: list, policy: str) -> str:
    return " ".join("—" if req[f"{sp:g}"][policy] >= NOT_MET else str(req[f"{sp:g}"][policy])
                    for sp in speeds)


def markdown(s: dict[str, Any], cfg: dict[str, Any]) -> str:
    pred = s["predicted_outcomes"]
    lines = [
        "# R1 and R2 again — on the validated timing model",
        "",
        ("Registered `configs/realtrace_ctx.yaml`. The same traces, adapters, windows, policies, "
         "seeds and rules as R1 and R2; only the timing profile changed, and the speeds, re-chosen "
         "by the registered rule from a round-robin-only probe. Each cell below was decided by "
         "`scripts/run_realtrace.py`, R1's and R2's own procedure."),
        "",
        "| test | registered prediction | result | on each calibration | on the first model |",
        "|---|---|---|---|---|",
    ]
    for test, e in s["tests"].items():
        per = " · ".join(f"{c}: {'holds' if v['holds'] else 'falsified'}"
                         for c, v in e["calibrations"].items()) or "not run"
        lines.append(f"| **{test}** — its own rules, unchanged | {pred[test]} | "
                     f"**{e['verdict']}** | {per} | "
                     f"{'held' if e['first_model']['holds'] else 'falsified'} |")
    w = s["R1w"]
    per = " · ".join(f"{c}: {'holds' if v else 'undecided' if v is None else 'does not hold'}"
                     for c, v in w["by_calibration"].items()) or "not run"
    lines += [("| **R1w** — on Mooncake, balanced affinity and session-sticky within one replica "
               f"| {pred['R1w']} | **{w['verdict']}** | {per} | not a registered rule then |"), ""]
    for test, e in s["tests"].items():
        speeds = s["speeds"][test]
        if not e["calibrations"]:
            lines += [f"## {test} — untestable as designed on this model", ""]
            continue
        cals = list(e["calibrations"])
        was = e["first_model"]
        now = ", ".join(f"{sp:g}×" for sp in speeds)
        then = ", ".join(f"{sp:g}×" for sp in was["speeds"])
        note = ("The speeds are the first model's." if list(was["speeds"]) == list(speeds) else
                f"The first model's speeds were {then}: the two tables are not at the same load.")
        lines += [f"## {test} — replicas needed at speeds {now} (— = not met at 8)", "", note, ""]
        for trace in e["calibrations"][cals[0]]["required_replicas"]:
            lines += [f"### {trace}", "",
                      "| policy | " + " | ".join(cals) + " | first model | hit rate at 4, "
                      + cals[0] + " |", "|---|" + "---|" * (len(cals) + 2)]
            for policy in e["calibrations"][cals[0]]["required_replicas"][trace][
                    f"{speeds[0]:g}"]:
                row = [_cells(e["calibrations"][c]["required_replicas"][trace], speeds, policy)
                       for c in cals]
                first = _cells(was["required_replicas"][trace], was["speeds"], policy)
                hit4 = e["calibrations"][cals[0]]["hit_rate_at_4"][trace]
                hits = " ".join(f"{hit4[f'{sp:g}'][policy]:.3f}" for sp in speeds)
                lines.append(f"| {policy} | " + " | ".join(row) + f" | {first} | {hits} |")
            win = " · ".join(f"{c}: " + ", ".join(
                f"{sp}× {x}" for sp, x in e["calibrations"][c]["winner_by_speed"][trace].items())
                for c in cals)
            lines += ["", f"Balanced affinity (α 16) against session-sticky — {win}.", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=CONFIG)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--probe", action="store_true",
                    help="round-robin alone on the first calibration: prints the speeds")
    ap.add_argument("--report-only", action="store_true", help="re-read the runs; run nothing")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    results = REPO / cfg["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    cals = cfg["calibrations"]

    if args.probe:
        cal, profile = next(iter(cals.items()))
        probe: dict[str, Any] = {"calibration": cal, "policy": "round-robin", "tests": {}}
        for test, base_path in cfg["tests"].items():
            base = yaml.safe_load((REPO / base_path).read_text())
            out_dir = results / test / "probe"
            out_dir.mkdir(parents=True, exist_ok=True)
            derived = derive(base, test, cal, profile, base["speeds"], _rel(out_dir))
            (out_dir / "config.yaml").write_text(yaml.safe_dump(derived, sort_keys=False))
            needs = parse_probe(_run(out_dir / "config.yaml", out_dir, args.workers, "--probe"))
            speeds = choose_speeds(base["probe_speeds"], needs)
            probe["tests"][test] = {"probe_speeds": base["probe_speeds"], "needs": needs,
                                    "speeds_by_the_rule": speeds}
            for trace, by in needs.items():
                print(f"{test} {trace}: round-robin needs "
                      + " · ".join(f"{sp}× {n or '—'}" for sp, n in by.items()))
            print(f"{test}: speeds by the rule -> {speeds or 'none: untestable as designed'}")
        (results / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
        return 0

    if not args.report_only:
        for test, base_path in cfg["tests"].items():
            speeds = cfg["speeds"][test]
            if not speeds:
                print(f"{test}: no speeds registered — untestable as designed, not run")
                continue
            base = yaml.safe_load((REPO / base_path).read_text())
            for cal, profile in cals.items():
                out_dir = results / test / cal
                out_dir.mkdir(parents=True, exist_ok=True)
                derived = derive(base, test, cal, profile, speeds, _rel(out_dir))
                (out_dir / "config.yaml").write_text(yaml.safe_dump(derived, sort_keys=False))
                _run(out_dir / "config.yaml", out_dir, args.workers)
                print(f"{test} on calibration {cal}: done", flush=True)
    summary = summarize(cfg, results)
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (results / "report.md").write_text(markdown(summary, cfg))
    print((results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
