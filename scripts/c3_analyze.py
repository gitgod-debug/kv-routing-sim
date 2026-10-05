#!/usr/bin/env python3
"""C3's analysis — committed BEFORE the run, so nothing in it is chosen after seeing a result.

    python scripts/c3_analyze.py [--grid]
        # artifacts/results/c3_kaggle_t4/ -> report.md, analysis.json

The procedure, exactly as configs/c3_calibrate.yaml registers it:
  C3a — the command's first run (the first seed): it exits 0; the profile is marked calibrated; the
        hits and queries its check measured equal the registered prediction; every check request's
        cached tokens equal the model's; the capacity it read equals the engine log's
        `GPU KV cache size` ÷ block size. All five, recomputed here from the raw files — not copied
        from what the notebook printed.
  C3b — that profile replays every part-2 workload on one replica at the times its requests were
        really sent — C2c's replay, imported from scripts/c2_analyze.py, not copied. Its p50 and
        p95 TTFT must be within c3b_max_factor of the measured ones, for every workload.
  C3c — the p50 and p95 registered for C2's profile (computed before the run, at the scheduled
        times) must be within c3c_max_factor of the measured ones, for every workload.
Percentiles on both sides by the notebook's nearest-rank rule; measured ones over valid requests.
A rule whose data is missing is UNDECIDED, and an undecided rule does not hold.

Reported, not tested (all listed in the config before the run): the command's second run and what
its profile predicts; the coefficients against C2's fit; C2's profile at the real send times; C2's
own session's measurements of the two workloads it shares; and, with --grid, H2b-C2's registered
grid re-run on the command's profile against the committed table (a few minutes of CPU).

Amended 2026-10-05, before any measurement existed (docs/09): the report also states the software
the run used, beside C2's, and what the notebook's pre-flight repaired — Kaggle's image changed
between the two sessions, and C3c compares across them. A line of reporting; no rule is touched.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c2_analyze as c2  # the registered replay, percentile and factor: imported, never copied

from kv_routing_sim.sim.cost_model import load_profile

COEFFICIENTS = (
    ("decode base", "decode_base_s", 1e3, "ms", 1),
    ("decode per sequence", "decode_per_seq_s", 1e3, "ms", 2),
    ("decode per 1K context tokens (batch sum)", "decode_per_ctx_token_s", 1e6, "ms", 2),
    ("prefill base", "prefill_base_s", 1e3, "ms", 1),
    ("prefill per new token", "prefill_per_token_s", 1e3, "ms", 3),
    ("prefill attention, per new × 1K attended tokens", "prefill_attn_s", 1e6, "ms", 3),
)
C2_SESSION = REPO / "artifacts" / "results" / "c2_kaggle_t4" / "analysis.json"
C2_RUN = REPO / "artifacts" / "results" / "c2_kaggle_t4" / "c2.json"
H2B = REPO / "artifacts" / "results" / "h2b_c2" / "h2b_c2.json"


# ── C3a: one run of the command, from the raw files ─────────────────────────────────────────────


def log_blocks(results: pathlib.Path, block_size: int) -> int | None:
    """The capacity the ENGINE logged at start-up, in blocks. None if the line is not there."""
    log = results / "server_0.log"
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens",
                  log.read_text(errors="replace") if log.exists() else "")
    return int(m.group(1).replace(",", "")) // block_size if m else None


def command_facts(run: dict | None, raw: dict | None, pred: dict, blocks: int | None) -> dict:
    """C3a's five conditions for one run of the command. `raw` is the profile file it wrote."""
    chk = ((raw or {}).get("_source") or {}).get("check") or {}
    meas = chk.get("measured") or {}
    cond = {
        "exits 0": bool(run) and run["returncode"] == 0,
        "the profile is marked calibrated": bool(raw) and raw.get("calibrated") is True,
        "check counters equal the registered prediction": bool(meas) and (
            meas.get("hits"), meas.get("queries")) == (pred["hits"], pred["queries"]),
        "every check request's cached tokens equal the model's": bool(chk) and (
            chk.get("cached_tokens_equal") == chk.get("cached_tokens_compared")
            == pred["requests"]),
        "capacity read equals the engine log": bool(raw) and blocks is not None
        and raw.get("kv_cache_blocks") == blocks,
    }
    return {
        "decided": run is not None, "holds": run is not None and all(cond.values()),
        "conditions": cond, "returncode": run and run["returncode"],
        "wall_clock_s": run and run["wall_clock_s"], "check_measured": meas or None,
        "cached_tokens_equal": chk.get("cached_tokens_equal"),
        "cached_tokens_compared": chk.get("cached_tokens_compared"),
        "read_blocks": raw and raw.get("kv_cache_blocks"), "log_blocks": blocks,
        "fit": raw and {key: raw.get(key) for _, key, *_ in COEFFICIENTS},
        "fit_quality": raw and {k: {"points": raw["_source"][k]["points"],
                                    "r2": raw["_source"][k]["r2"]}
                                for k in ("decode_fit", "prefill_fit")},
        "notes": raw and raw["_source"].get("notes"),
    }


def software(data: dict) -> dict[str, Any]:
    """Reported: what the run's box had installed, and what the pre-flight had to repair."""
    env, pre = data.get("environment") or {}, data.get("preflight") or {}
    return {
        "python": env.get("python"), "packages": env.get("packages") or {},
        "torch_cuda_build": (pre.get("torch") or {}).get("cuda_build"),
        "torch_family": {k: v.get("version") for k, v in (pre.get("packages") or {}).items()
                         if v.get("installed")},
        "repairs": [{"package": r["package"], "was": r["was"].get("version"),
                     "error": r["was"].get("error"), "did": r["did"],
                     "now": r["now"].get("version") if r["now"].get("installed") else "removed",
                     "imports": bool(r["now"].get("ok"))}
                    for r in pre.get("repairs") or []],
    }


def _software_line(sw: dict) -> str | None:
    pk = sw["packages"]
    if not sw["python"] and not pk:
        return None
    parts = [f"Python {sw['python'] or '?'}"]
    parts += [f"{name} {pk[name]}" for name in ("vllm", "torch", "transformers") if pk.get(name)]
    if sw["torch_cuda_build"]:
        parts.append(f"torch built for CUDA {sw['torch_cuda_build']}")
    return " · ".join(parts)


# ── C3b and C3c: the held-out traffic ───────────────────────────────────────────────────────────


def measured(m: dict) -> dict[str, Any]:
    valid = sorted(r["ttft_ms"] for r in m["requests"] if r["valid"] and r["ttft_ms"])
    return {"hit_rate": m["hit_rate"], "preemptions": m["preemptions"],
            "ttft_p50_ms": c2.nearest_rank(valid, 0.50),
            "ttft_p95_ms": c2.nearest_rank(valid, 0.95), "requests": len(m["requests"]),
            "invalid": m["invalid"], "worst_send_lag_ms": m["worst_send_lag_ms"]}


def factors(pred: dict, meas: dict) -> dict[str, float | None]:
    return {q: c2.factor(pred[f"ttft_{q}_ms"], meas[f"ttft_{q}_ms"]) for q in ("p50", "p95")}


def _within(rows: list[dict], key: str, limit: float) -> tuple[bool, bool]:
    """(decided, holds) for a rule over every workload's two factors."""
    vals = [v for r in rows for v in (r.get(key) or {"p50": None, "p95": None}).values()]
    decided = bool(rows) and all(v is not None for v in vals)
    return decided, decided and all(v <= limit for v in vals)


def analyze(data: dict, prereg: dict, results: pathlib.Path) -> dict[str, Any]:
    cfg = prereg["config"]
    rules, seeds = cfg["rules"], cfg["calibrate"]["seeds"]
    if data["predictions_sha256"] != prereg["predictions_sha256"]:
        raise SystemExit("error: the run carried different predictions from the registered ones")
    if data["fingerprints"] != prereg["trace_fingerprints_sha256"]:
        raise SystemExit("error: the run replayed different traffic from the registered traces")
    if data["wheel_sha256"] != prereg["wheel"]["sha256"]:
        raise SystemExit("error: the run carried a different wheel from the registered one")

    out: dict[str, Any] = {"script": "scripts/c3_analyze.py"}
    blocks = log_blocks(results, cfg["engine"]["block_size"])
    ran = {r["seed"]: r for r in data.get("calibrate", [])}
    profiles, commands = {}, {}
    for seed in seeds:
        path = results / f"calibrated_profile_seed{seed}.json"
        raw = json.loads(path.read_text()) if path.exists() else None
        profiles[seed] = load_profile(path) if raw else None
        commands[str(seed)] = command_facts(ran.get(seed), raw,
                                            prereg["predictions"]["check"][str(seed)], blocks)
    tested = str(seeds[0])
    out["commands"] = commands
    out["c3a"] = {"seed": seeds[0], "decided": commands[tested]["decided"],
                  "holds": commands[tested]["holds"]}

    c2_profile = load_profile(REPO / cfg["profile"])
    by_name = {m["workload"]: m for m in data.get("holdout", [])}
    rows = []
    for name, spec in prereg["workloads"].items():
        m = by_name.get(name)
        if m is None:
            rows.append({"workload": name, "measured": None})
            continue
        meas = measured(m)
        reg = prereg["predictions"]["runs"][name]["C2 profile"]
        at_send = c2.replay(name, spec, m, c2_profile)
        row = {"workload": name, "measured": meas, "c2_registered": reg,
               "c2_factor": factors(reg, meas), "c2_at_send_times": at_send,
               "c2_at_send_times_factor": factors(at_send, meas),
               "hit_rate_error": (abs(reg["hit_rate"] - meas["hit_rate"]) / meas["hit_rate"]
                                  if meas["hit_rate"] else None),
               "command": {}}
        for seed in seeds:
            if profiles[seed] is not None:
                sim = c2.replay(name, spec, m, profiles[seed])
                row["command"][str(seed)] = {"prediction": sim, "factor": factors(sim, meas)}
        row["command_factor"] = (row["command"].get(tested) or {}).get("factor")
        rows.append(row)
    out["holdout"] = rows

    complete = all(r["measured"] is not None for r in rows)
    decided_b, holds_b = _within(rows, "command_factor", rules["c3b_max_factor"])
    decided_c, holds_c = _within(rows, "c2_factor", rules["c3c_max_factor"])
    out["c3b"] = {"seed": seeds[0], "max_factor": rules["c3b_max_factor"],
                  "decided": complete and decided_b, "holds": complete and holds_b}
    out["c3c"] = {"max_factor": rules["c3c_max_factor"], "decided": complete and decided_c,
                  "holds": complete and holds_c}
    out["c2_fit"] = {key: getattr(c2_profile, key) for _, key, *_ in COEFFICIENTS}
    out["predicted_outcomes"] = cfg["predictions"]
    out["software"] = software(data)
    if C2_RUN.exists():  # reported: the box C2's profile was fitted on — C3c compares across them
        out["c2_software"] = software(json.loads(C2_RUN.read_text()))
    if C2_SESSION.exists():  # reported: what C2's own session measured on the shared workloads
        out["c2_session"] = {r["workload"]: {q: r["measured"][f"ttft_{q}_ms"]
                                             for q in ("p50", "p95")}
                             for r in json.loads(C2_SESSION.read_text())["c2c"]["runs"]}
    return out


# ── reported with --grid: does a second calibration give the same GPU counts? ───────────────────


def _sticky_best(req: dict, not_met: int) -> int:
    """(K, load) points where session-sticky needs the fewest replicas or ties for them."""
    return sum(1 for by_load in req.values() for r in by_load.values()
               if r["session-sticky"] == min(r.values()))


def grid(results: pathlib.Path, seed: int) -> dict[str, Any]:
    """H2b-C2's registered grid — scripts/run_h2b_c2.py, its config unchanged but for the profile —
    on the profile the command wrote, compared with the committed table (C2's profile)."""
    profile = results / f"calibrated_profile_seed{seed}.json"
    out = results / "h2b_grid"
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load((REPO / "configs" / "h2b_c2.yaml").read_text())
    cfg["profile"] = str(profile.resolve())
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "run_h2b_c2.py"), "--config",
                           str(out / "config.yaml"), "--out", str(out)], capture_output=True,
                          text=True, cwd=REPO, check=False)
    if proc.returncode != 0:
        why = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return {"ran": False, "why": why}
    new, old = json.loads((out / "h2b_c2.json").read_text()), json.loads(H2B.read_text())
    cells = [(k, ld, pol, r, new["required_replicas"][k][ld][pol])
             for k, by_load in old["required_replicas"].items()
             for ld, by_pol in by_load.items() for pol, r in by_pol.items()]
    nm = old["not_met"]
    both = [(o, n) for *_, o, n in cells if o != nm and n != nm]
    return {
        "ran": True, "seed": seed, "cells": len(cells),
        "equal": sum(1 for *_, o, n in cells if o == n),
        "within_one": sum(1 for *_, o, n in cells if abs(o - n) <= 1),
        "held_in_both": len(both),
        "largest_difference": max((abs(o - n) for o, n in both), default=None),
        "sticky_best_or_tied": {"committed": _sticky_best(old["required_replicas"], nm),
                                "command": _sticky_best(new["required_replicas"], nm),
                                "points": sum(len(v) for v in old["required_replicas"].values())},
        "verdict_committed": old["verdict"]["holds"], "verdict_command": new["verdict"]["holds"],
        "differences": [{"K": k, "load": ld, "policy": pol, "committed": o, "command": n}
                        for k, ld, pol, o, n in cells if o != n],
    }


# ── the report ──────────────────────────────────────────────────────────────────────────────────


def _pair(d: dict | None) -> str:
    if not d or d.get("ttft_p50_ms") is None:
        return "—"
    return f"{d['ttft_p50_ms']:,.0f} · {d['ttft_p95_ms']:,.0f} ms"


def _fac(f: dict | None) -> str:
    if not f:
        return "—"
    return " · ".join("—" if f[q] is None else f"{f[q]:.2f}×" for q in ("p50", "p95"))


def markdown(a: dict, data: dict, prereg: dict) -> str:
    cfg = prereg["config"]
    seeds = [str(s) for s in cfg["calibrate"]["seeds"]]
    verdict = {True: "HOLDS", False: "DOES NOT HOLD"}

    def v(rule: str) -> str:
        return verdict[a[rule]["holds"]] if a[rule]["decided"] else "UNDECIDED (data missing)"

    env = data.get("environment", {})
    lines = [
        "# C3 — the context-aware `kvsim calibrate` on a real T4",
        "",
        (f"Registered `configs/c3_calibrate.yaml` · predictions "
         f"`{prereg['predictions_sha256'][:16]}…` and wheel `{prereg['wheel']['sha256'][:16]}…` "
         f"(from `{prereg['wheel']['built_from']}`) match the run · environment "
         f"{env.get('date_utc', '?')} · generated by `scripts/c3_analyze.py`, committed before "
         "the run."),
        "",
        "| rule | registered prediction | result |",
        "|---|---|---|",
        (f"| **C3a** — the command works on a real engine | {cfg['predictions']['c3a']} | "
         f"**{v('c3a')}** |"),
        (f"| **C3b** — its profile predicts held-out traffic within {a['c3b']['max_factor']:g}× | "
         f"{cfg['predictions']['c3b']} | **{v('c3b')}** |"),
        (f"| **C3c** — C2's profile predicts another day within {a['c3c']['max_factor']:g}× | "
         f"{cfg['predictions']['c3c']} | **{v('c3c')}** |"),
        "",
    ]
    here = _software_line(a["software"])
    if here:
        then = _software_line(a["c2_software"]) if a.get("c2_software") else None
        lines += [f"Software: {here}." + (f" C2's session, where the profile C3c tests was "
                                          f"fitted: {then}." if then else ""), ""]
    for r in a["software"]["repairs"]:
        lines += [(f"Pre-flight repair, before anything was measured: `{r['package']}` "
                   f"{r['was'] or '?'} could not be imported ({(r['error'] or '')[:160]}) → "
                   f"`pip {r['did']}` → {r['now']}"
                   f"{', which imports' if r['imports'] else ''}."), ""]
    lines += [
        f"## The command — seed {seeds[0]} is the tested run, seed {seeds[1]} is reported",
        "",
        "| condition | " + " | ".join(f"seed {s}" for s in seeds) + " |",
        "|---|" + "---|" * len(seeds),
    ]
    cmds = a["commands"]
    for cond in cmds[seeds[0]]["conditions"]:
        lines.append(f"| {cond} | " + " | ".join(
            ("✅" if cmds[s]["conditions"][cond] else "⛔") if cmds[s]["decided"] else "not run"
            for s in seeds) + " |")

    def detail(s: str) -> str:
        c = cmds[s]
        if not c["decided"]:
            return "not run"
        m = c["check_measured"] or {}
        return (f"exit {c['returncode']} · {c['wall_clock_s']:.0f} s · check "
                f"{m.get('hits', 0):,.0f} of {m.get('queries', 0):,.0f} tokens, "
                f"{c['cached_tokens_equal']} of {c['cached_tokens_compared']} requests equal · "
                f"{c['read_blocks']} blocks read, {c['log_blocks']} in the log")

    lines += [f"| *in numbers* | {' | '.join(detail(s) for s in seeds)} |", "",
              "## The fitted timing model", "",
              "| | " + " | ".join(f"command, seed {s}" for s in seeds) + " | C2's fit |",
              "|---|" + "---:|" * (len(seeds) + 1)]
    for label, key, scale, unit, digits in COEFFICIENTS:
        cells = [("—" if not cmds[s]["fit"] else f"{cmds[s]['fit'][key] * scale:.{digits}f} {unit}")
                 for s in seeds]
        lines.append(f"| {label} | {' | '.join(cells)} | "
                     f"{a['c2_fit'][key] * scale:.{digits}f} {unit} |")
    for label, key in (("decode fit", "decode_fit"), ("prefill fit", "prefill_fit")):
        cells = [("—" if not cmds[s]["fit_quality"] else
                  f"{cmds[s]['fit_quality'][key]['points']} points, R² "
                  f"{cmds[s]['fit_quality'][key]['r2']:.3f}") for s in seeds]
        lines.append(f"| {label} | {' | '.join(cells)} | |")
    for s in seeds:
        for note in cmds[s]["notes"] or []:
            lines += ["", f"Seed {s} noted: {note}."]

    lines += ["", "## Part 2 — held-out traffic on one engine", "",
              (f"| workload | measured p50 · p95 | the command's profile (seed {seeds[0]}) | "
               "factor | C2's profile, registered | factor | hit rate measured · registered | "
               "preemptions measured · command · C2 |"),
              "|---|---|---|---|---|---|---|---|"]
    for r in a["holdout"]:
        m = r["measured"]
        if m is None:
            lines.append(f"| {r['workload']} | not run | | | | | | |")
            continue
        cmd = (r["command"].get(seeds[0]) or {}).get("prediction")
        lines.append(
            f"| {r['workload']} | {_pair(m)} | {_pair(cmd)} | {_fac(r['command_factor'])} | "
            f"{_pair(r['c2_registered'])} | {_fac(r['c2_factor'])} | "
            f"{m['hit_rate']:.4f} · {r['c2_registered']['hit_rate']:.4f} | {m['preemptions']} · "
            f"{'—' if cmd is None else cmd['preemptions']} · "
            f"{r['c2_registered']['preemptions']} |")
    lines += ["", (f"**C3b** holds only if every factor in the first factor column is ≤ "
                   f"{a['c3b']['max_factor']:g}; **C3c**, every factor in the second, ≤ "
                   f"{a['c3c']['max_factor']:g}. The command's profile is replayed at the times "
                   "requests were really sent; C2's numbers are the ones registered before the "
                   "run, at the scheduled times."), "",
              "## Reported, not tested", "",
              (f"| workload | command, seed {seeds[1]} | factor | C2's profile at the real send "
               "times | factor | C2's own session measured | requests · invalid · worst send "
               "lag |"),
              "|---|---|---|---|---|---|---|"]
    for r in a["holdout"]:
        m = r["measured"]
        if m is None:
            continue
        other = r["command"].get(seeds[1]) or {}
        then = (a.get("c2_session") or {}).get(r["workload"])
        earlier = "—" if not then else f"{then['p50']:,.0f} · {then['p95']:,.0f} ms"
        lines.append(
            f"| {r['workload']} | {_pair(other.get('prediction'))} | {_fac(other.get('factor'))} "
            f"| {_pair(r['c2_at_send_times'])} | {_fac(r['c2_at_send_times_factor'])} | "
            f"{earlier} | {m['requests']} · {m['invalid']} · {m['worst_send_lag_ms']:.0f} ms |")
    errs = [f"{r['workload']} {r['hit_rate_error']:.1%}" for r in a["holdout"]
            if r["measured"] is not None and r["hit_rate_error"] is not None]
    if errs:
        lines += ["", (f"Hit rate against the registered prediction: {', '.join(errs)} (reported "
                       f"limit {cfg['rules']['hit_rate_max_relative_error']:.0%}).")]
    g = a.get("grid")
    if g and g["ran"]:
        s = g["sticky_best_or_tied"]
        lines += [
            "", "### The same GPU counts from a second calibration?", "",
            (f"H2b-C2's registered grid, re-run on the profile the command wrote (seed "
             f"{g['seed']}), against the committed table from C2's profile: **{g['equal']} of "
             f"{g['cells']}** (K, load, policy) cells need the same number of replicas, "
             f"{g['within_one']} are within one; the largest difference where both hold the SLO "
             f"is {g['largest_difference']}. Session-sticky needs the fewest replicas or ties at "
             f"{s['command']} of {s['points']} points ({s['committed']} in the committed table). "
             f"H2b's verdict: {'holds' if g['verdict_command'] else 'falsified'} "
             f"({'holds' if g['verdict_committed'] else 'falsified'} in the committed run)."),
        ]
        if g["differences"]:
            lines += ["", "| K | load | policy | committed | command |", "|---|---|---|---:|---:|"]
            lines += [f"| {d['K']} | {d['load']} | {d['policy']} | {d['committed']} | "
                      f"{d['command']} |" for d in g["differences"]]
    elif g:
        lines += ["", f"The grid comparison did not run: {g['why']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c3_kaggle_t4")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c3_prereg" / "predictions.json")
    ap.add_argument("--grid", action="store_true",
                    help="also re-run H2b-C2's grid on the command's profile (reported; minutes)")
    args = ap.parse_args()
    data = json.loads((args.results / "c3.json").read_text())
    prereg = json.loads(args.prereg.read_text())
    a = analyze(data, prereg, args.results)
    if args.grid:
        a["grid"] = grid(args.results, prereg["config"]["calibrate"]["seeds"][0])
    (args.results / "analysis.json").write_text(json.dumps(a, indent=1, default=str) + "\n")
    (args.results / "report.md").write_text(markdown(a, data, prereg))
    print((args.results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
