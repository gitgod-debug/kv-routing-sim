#!/usr/bin/env python3
"""Phase 6 analysis: two live T4 replicas against the predictions made before the run.

    python scripts/phase6_analyze.py      # artifacts/results/phase6_kaggle_t4/ -> report.md

Reports both runs of the notebook. Run 1's per-run results are parsed from the printed output of
the executed notebook (phase6_sim2real_EXECUTED_run1.ipynb), because its JSON was not kept. Run 2's
come from phase6.json, with S1 and S2 recomputed from its raw per-replica counters rather than
copied from the notebook's verdict; its two server logs feed a post-hoc decode-step estimate. Also
writes analysis.json.

The decision rules in configs/phase6_sim2real.yaml apply to hit rates only. S1: every run is within
s1_max_relative_error of its prediction. S2: every pair of policies predicted at least s2_min_gap
apart comes out in the same order. Latency and preemptions are reported beside them but are not
part of the test.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "results" / "phase6_kaggle_t4"
PREREG = REPO / "artifacts" / "results" / "phase6_prereg" / "predictions.json"
CONFIG = REPO / "configs" / "phase6_sim2real.yaml"

# One printed line per run in the notebook's RUN cell, such as the one below, where a pass/fail
# mark follows the percentage:
#   K64  session-sticky  hit 0.8099 (predicted 0.8214, -1.4%) · requests [220, 260] (...) ·
#   preemptions 71 (predicted 0) · p95 TTFT 71525 ms · 0 invalid · send lag ≤ 2 ms
RUN_LINE = re.compile(
    r"(K\d+)\s+(\S+)\s+hit ([\d.]+) \(predicted [\d.]+, [-+][\d.]+%\) \S+ · "
    r"requests \[(\d+), (\d+)\].*?preemptions (\d+) \(predicted \d+\) · "
    r"p95 TTFT (\d+) ms · (\d+) invalid"
)
# vLLM's periodic stats line (loggers.py), one per engine every 10 s
STATS = re.compile(
    r"(\d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%"
)


def run1() -> dict[tuple[str, str], dict[str, Any]]:
    nb = json.loads((OUT / "phase6_sim2real_EXECUTED_run1.ipynb").read_text())
    text = ""
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            t = o.get("text", "")
            text += "".join(t) if isinstance(t, list) else t
    out = {}
    for w, p, hit, r0, r1, pre, p95, bad in RUN_LINE.findall(text):
        out[(w, p)] = {"hit_rate": float(hit), "requests_per_replica": [int(r0), int(r1)],
                       "preemptions": int(pre), "ttft_p95_ms": float(p95), "invalid": int(bad)}
    return out


def run2(d: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for r in d["runs"]:
        c = r["per_replica_counters"]
        q = sum(x["prefix_cache_queries"] for x in c)
        h = sum(x["prefix_cache_hits"] for x in c)
        out[(r["workload"], r["policy"])] = {
            "hit_rate": h / q, "requests_per_replica": r["requests_per_replica"],
            "preemptions": sum(x["preemptions"] for x in c), "ttft_p50_ms": r["ttft_p50_ms"],
            "ttft_p95_ms": r["ttft_p95_ms"], "invalid": len(r["invalid_requests"]),
            "worst_send_lag_ms": r["worst_send_lag_ms"],
        }
    return out


def verdict(runs: dict, pred: dict, cfg: dict) -> dict[str, Any]:
    """Apply the S1 and S2 rules from the config to one run's measured hit rates."""
    s1 = {f"{w}/{p}": abs(m["hit_rate"] - pred[w][p]["hit_rate"]) / pred[w][p]["hit_rate"]
          for (w, p), m in runs.items()}
    pairs = []
    pols = cfg["policies"]
    for w in cfg["workloads"]:
        for i, a in enumerate(pols):
            for b in pols[i + 1:]:
                pg = pred[w][a]["hit_rate"] - pred[w][b]["hit_rate"]
                if abs(pg) < cfg["s2_min_gap"]:
                    continue  # predicted gap within the noise: not tested
                mg = runs[(w, a)]["hit_rate"] - runs[(w, b)]["hit_rate"]
                pairs.append({"workload": w, "pair": [a, b], "predicted_gap": round(pg, 4),
                              "measured_gap": round(mg, 4), "same_order": (pg > 0) == (mg > 0)})
    s1_ok = all(e <= cfg["s1_max_relative_error"] for e in s1.values())
    s2_ok = all(x["same_order"] for x in pairs)
    return {"s1_relative_errors": {k: round(v, 5) for k, v in s1.items()},
            "s1_max": round(max(s1.values()), 5), "s1_holds": s1_ok, "s2_pairs": pairs,
            "s2_holds": s2_ok, "holds": s1_ok and s2_ok}


def decode_windows() -> list[dict[str, float]]:
    """Post hoc: the 10-second stats windows in which an engine decoded without any prefill.

    In those windows (prompt throughput 0), running requests / generation throughput estimates the
    time of one decode step.
    """
    rows = []
    for name in ("server_0.log", "server_1.log"):
        for line in (OUT / name).read_text(errors="replace").splitlines():
            m = STATS.search(line)
            if not m:
                continue
            _, prompt, gen, running, _, kv = m.groups()
            if float(prompt) == 0 and float(gen) > 0 and int(running) > 0:
                rows.append({"running": int(running), "gen_tps": float(gen),
                             "step_ms": 1000 * int(running) / float(gen), "kv_pct": float(kv)})
    return rows


def main() -> int:
    d = json.loads((OUT / "phase6.json").read_text())
    prereg = json.loads(PREREG.read_text())
    cfg = yaml.safe_load(CONFIG.read_text())
    pred = prereg["predictions"]["runs"]
    if d["predictions_sha256"] != prereg["predictions_sha256"]:
        raise SystemExit("error: run 2 carried different predictions from the registered ones")
    if d["fingerprints"] != prereg["trace_fingerprints_sha256"]:
        raise SystemExit("error: run 2 replayed different traffic from the registered traces")
    r1, r2 = run1(), run2(d)
    if len(r1) != 6:
        raise SystemExit(f"error: expected 6 printed runs in run 1's notebook, found {len(r1)}")
    v1, v2 = verdict(r1, pred, cfg), verdict(r2, pred, cfg)
    win = decode_windows()
    prof = json.loads((REPO / cfg["profile"]).read_text())
    # Medians by batch size, not a line: a 10-second average is noisy (a stall, or a run ending
    # mid-window, gives one wild point), and a median per band is robust to that.
    bands = []
    for lo, hi in ((1, 30), (30, 60), (60, 100), (100, 10_000)):
        seg = [w for w in win if lo <= w["running"] < hi]
        if not seg:
            continue
        running = statistics.median(w["running"] for w in seg)
        model = 1e3 * (prof["decode_base_s"] + prof["decode_per_seq_s"] * running)
        bands.append({"running": f"{lo}–{hi - 1}" if hi < 10_000 else f"{lo}+", "windows": len(seg),
                      "median_running": running,
                      "median_step_ms": round(statistics.median(w["step_ms"] for w in seg)),
                      "profile_step_ms": round(model)})
    ratios = [m["ttft_p95_ms"] / pred[w][p]["ttft_p95_ms"]
              for run in (r1, r2) for (w, p), m in run.items()]
    unpredicted = [m["preemptions"] for run in (r1, r2) for (w, p), m in run.items()
                   if pred[w][p]["preemptions"] == 0]

    payload = {
        "script": "scripts/phase6_analyze.py",
        "run1": {f"{w}/{p}": v for (w, p), v in r1.items()},
        "run2": {f"{w}/{p}": v for (w, p), v in r2.items()},
        "verdict_run1": v1, "verdict_run2": v2,
        "phase6_holds": v1["holds"] and v2["holds"],
        "p95_ratio_measured_over_predicted": [round(min(ratios), 1), round(max(ratios), 1)],
        "preemptions_where_none_predicted": unpredicted,
        "posthoc_decode_steps": {"windows": len(win), "by_batch": bands},
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "report.md").write_text(_markdown(payload, d, pred, cfg, r1, r2, win))
    print((OUT / "report.md").read_text())
    return 0


def _markdown(pl, d, pred, cfg, r1, r2, win) -> str:
    v1, v2 = pl["verdict_run1"], pl["verdict_run2"]
    ok = {True: "HOLDS", False: "FALSIFIED"}
    steps = pl["posthoc_decode_steps"]
    x = [b["median_step_ms"] / b["profile_step_ms"] for b in steps["by_batch"]]
    lo_x, hi_x = min(x), max(x)
    unpred = pl["preemptions_where_none_predicted"]
    env = d["environment"]
    lines = [
        "# Phase 6 — two live T4 replicas against the registered predictions",
        "",
        (f"vLLM {env['packages']['vllm']} · Qwen/Qwen3-4B fp16 · two Tesla T4s on one Kaggle box "
         f"({env['cpu_count']} vCPUs) · 1,537 blocks per replica · 120 sessions × 4 turns at 1 "
         f"session/s · predictions `{d['predictions_sha256'][:16]}…` and traffic fingerprints "
         "match the registration. Generated by `scripts/phase6_analyze.py`."),
        "",
        (f"**Phase 6: {ok[pl['phase6_holds']]}** — in both runs. Run 1: S1 {ok[v1['s1_holds']]} "
         f"(largest error {v1['s1_max']:.2%}), S2 {ok[v1['s2_holds']]}. Run 2: S1 "
         f"{ok[v2['s1_holds']]} (largest error {v2['s1_max']:.2%}), S2 {ok[v2['s2_holds']]}. "
         f"Limit {cfg['s1_max_relative_error']:.0%}."),
        "",
        "## Hit rate — the registered test",
        "",
        "| workload | policy | predicted | run 1 | run 2 |",
        "|---|---|---:|---:|---:|",
    ]
    for w in cfg["workloads"]:
        for p in cfg["policies"]:
            lines.append(f"| {w} | {p} | {pred[w][p]['hit_rate']:.4f} | "
                         f"{r1[(w, p)]['hit_rate']:.4f} | {r2[(w, p)]['hit_rate']:.4f} |")
    lines += ["", (f"S2, every pair predicted at least {cfg['s2_min_gap']} apart (measured gap, "
                   "run 1 · run 2):"), ""]
    for a, b in zip(v1["s2_pairs"], v2["s2_pairs"], strict=True):
        mark = "✅" if a["same_order"] and b["same_order"] else "⛔"
        lines.append(f"- {a['workload']} {a['pair'][0]} − {a['pair'][1]}: predicted "
                     f"{a['predicted_gap']:+.4f} · measured {a['measured_gap']:+.4f} · "
                     f"{b['measured_gap']:+.4f} {mark}")
    lines += [
        "",
        "## Latency and preemptions — predicted, NOT a registered test",
        "",
        ("| workload | policy | p95 TTFT predicted | run 1 | run 2 | preemptions predicted | "
         "run 1 | run 2 |"),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for w in cfg["workloads"]:
        for p in cfg["policies"]:
            a, b, pr = r1[(w, p)], r2[(w, p)], pred[w][p]
            lines.append(f"| {w} | {p} | {pr['ttft_p95_ms'] / 1e3:.2f} s | "
                         f"{a['ttft_p95_ms'] / 1e3:.1f} s | {b['ttft_p95_ms'] / 1e3:.1f} s | "
                         f"{pr['preemptions']} | {a['preemptions']} | {b['preemptions']} |")
    lines += [
        "",
        (f"The simulator's timing model **under-predicted p95 TTFT by "
         f"{pl['p95_ratio_measured_over_predicted'][0]:.0f}–"
         f"{pl['p95_ratio_measured_over_predicted'][1]:.0f}×** at this load. Where it predicted "
         f"no preemptions, the engines had {', '.join(map(str, unpred))}. The *order* of the three "
         "policies by p95 TTFT matched the prediction in both workloads and both runs; only "
         "K1 session-sticky held the 2 s target live (1.8–1.9 s, predicted 0.35 s). No request "
         "failed, and the client sent every request within 10 ms of its schedule, so the delay was "
         "the engines'."),
        "",
        "## Why — post hoc, a lead and not a finding",
        "",
        (f"From vLLM's own 10-second stats lines, in the {steps['windows']} windows where an "
         "engine decoded without any prefill: one decode step ≈ running requests ÷ generation "
         "throughput. Coarse, so medians by batch size:"),
        "",
        "| running requests | windows | median step | the profile predicts |",
        "|---|---:|---:|---:|",
        *[f"| {b['running']} (median {b['median_running']:g}) | {b['windows']} | "
          f"{b['median_step_ms']} ms | {b['profile_step_ms']} ms |" for b in steps["by_batch"]],
        "",
        (f"Decode steps took **{lo_x:.1f}–{hi_x:.1f} times** what the profile predicts, at every "
         "batch size. At two requests per second per engine, each with 64 output tokens, that is "
         "enough to tip an engine into overload — the queue grows, the cache fills, requests are "
         "preempted and recomputed — which is how a few-times-slower step becomes a "
         f"{pl['p95_ratio_measured_over_predicted'][0]:.0f}–"
         f"{pl['p95_ratio_measured_over_predicted'][1]:.0f}× longer wait. Two "
         "candidate causes, neither tested: (1) **context length** — these prompts are ~1,700 "
         "tokens, and the per-request decode cost rises with context across three measurements "
         "(about 0.2 ms per request at ~100 tokens in C1, about 2 ms at 448 in Phase 3's sweep), "
         "while the profile has no context term; (2) **CPU** — two engines and a client streaming "
         f"~200 responses shared {env['cpu_count']} vCPUs. A single-engine decode sweep across "
         "context lengths separates the two."),
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
