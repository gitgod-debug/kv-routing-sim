#!/usr/bin/env python3
"""POST HOC: does the edge check see the knee? T1's five held-out runs, on timings 5% slower.

    python scripts/edge_posthoc_t1.py
        # artifacts/results/c2_kaggle_t4, c3_kaggle_t4, t1_itl -> t1_itl/posthoc_edge.{json,md}

`kvsim replay` warns when its recommended configuration sits at the knee of the fleet's load curve
(src/kv_routing_sim/eval/edge.py): it runs that configuration again with every timing coefficient
5% larger and looks at how far the p95 latencies move. This script asks whether that movement
tells apart the runs on which the simulator was wrong from the runs on which it was right.

T1 (registered, 2026-10-05) replayed five held-out runs from a live T4 on one simulated replica.
On two of them — both at the knee — the simulator's mean inter-token latency was 1.54× and 1.71×
the engine's. On the other three it was within 1.07×. Here the same five replays are repeated on
timings 5% slower, with T1's own replay function.

⚠️ WRITTEN AFTER T1's RESULT WAS KNOWN, AND THE CHECK'S THRESHOLD (1.5×) WAS CHOSEN AFTER SEEING
THESE NUMBERS. Nothing here was registered and it decides nothing. Five runs of three workloads
on one GPU: an indication, not a validation.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import t1_itl_check as t1  # T1's replay, imported — never copied
from _spike_common import results_header

from kv_routing_sim.eval.edge import KNEE_FACTOR, SLOWDOWN, slowed
from kv_routing_sim.sim.cost_model import load_profile

OUT = REPO / "artifacts" / "results" / "t1_itl"


def analyze() -> dict[str, Any]:
    cfg = yaml.safe_load((REPO / "configs" / "tpot_slo.yaml").read_text())["t1"]
    profile = load_profile(REPO / cfg["profile"])
    stored = {(r["session"], r["workload"]): r["A"]
              for r in json.loads((OUT / "t1.json").read_text())["runs"]}
    rows = []
    for src in cfg["runs"]:
        data = json.loads((REPO / src["results"]).read_text())
        specs = json.loads((REPO / src["prereg"]).read_text())["workloads"]
        by_name = {m["workload"]: m for m in data.get("holdout", [])}
        for name in src["workloads"]:
            base = t1.replay(specs[name], by_name[name], profile)
            slow = t1.replay(specs[name], by_name[name], slowed(profile, SLOWDOWN))
            t1_row = stored[(src["session"], name)]
            # The base replay must be T1's own: the same per-token p95, to the rounded figure.
            if round(base.tpot_percentiles()["p95_ms"], 3) != t1_row["predicted_tpot_p95_ms"]:
                raise SystemExit(f"error: {src['session']} {name}: this replay is not T1's")
            ttft = (base.ttft_percentiles()["p95_ms"], slow.ttft_percentiles()["p95_ms"])
            tpot = (base.tpot_percentiles()["p95_ms"], slow.tpot_percentiles()["p95_ms"])
            worst = max(ttft[1] / ttft[0], tpot[1] / tpot[0])
            rows.append({
                "session": src["session"], "workload": name,
                "t1_factor": round(t1_row["factor"], 3),
                "p95_ttft_ms": [round(x, 1) for x in ttft],
                "ttft_factor": round(ttft[1] / ttft[0], 3),
                "p95_tpot_ms": [round(x, 1) for x in tpot],
                "tpot_factor": round(tpot[1] / tpot[0], 3),
                "preemptions": [base.preemptions, slow.preemptions],
                "flagged_as_knee": worst > KNEE_FACTOR,
            })
    wrong = [r for r in rows if r["t1_factor"] > 1.5]
    right = [r for r in rows if r["t1_factor"] <= 1.5]
    return {
        "script": "scripts/edge_posthoc_t1.py", "post_hoc": True,
        "slowdown": SLOWDOWN, "knee_factor": KNEE_FACTOR, "runs": rows,
        "flags_every_run_the_model_got_wrong": all(r["flagged_as_knee"] for r in wrong),
        "flags_no_run_the_model_got_right": not any(r["flagged_as_knee"] for r in right),
        # Any threshold strictly between these two separates the five runs.
        "separating_range": [max(max(r["ttft_factor"], r["tpot_factor"]) for r in right),
                             min(max(r["ttft_factor"], r["tpot_factor"]) for r in wrong)],
    }


def _moved(pair: list[float], factor: float) -> str:
    return f"{pair[0]:,.0f} → {pair[1]:,.0f} ms (×{factor:.2f})"


def markdown(a: dict[str, Any], header: dict[str, Any]) -> str:
    pct = round((a["slowdown"] - 1) * 100)
    lo, hi = a["separating_range"]
    lines = [
        "# Post hoc — does the edge check see the knee?",
        "",
        (f"⚠️ **Written after T1's result was known; the check's threshold ({a['knee_factor']}×) "
         "was chosen after seeing these numbers.** Nothing here was registered. Five runs of "
         "three workloads on one GPU: an indication, not a validation."),
        "",
        (f"T1's five held-out runs, replayed with T1's own function on profile A and again with "
         f"every timing coefficient {pct}% larger · generated `{header['date_utc']}` at "
         f"`{header['git_sha']}`{' (dirty tree)' if header['git_dirty'] else ''} by "
         "`scripts/edge_posthoc_t1.py`."),
        "",
        (f"| session · workload | simulator ÷ engine, per token (T1) | p95 TTFT, base → {pct}% "
         f"slower | p95 time per token, base → {pct}% slower | preemptions | flagged |"),
        "|---|---:|---|---|---|:---:|",
    ]
    for r in a["runs"]:
        lines.append(
            f"| {r['session']} · {r['workload']} | {r['t1_factor']:.2f}× | "
            f"{_moved(r['p95_ttft_ms'], r['ttft_factor'])} | "
            f"{_moved(r['p95_tpot_ms'], r['tpot_factor'])} | "
            f"{r['preemptions'][0]} → {r['preemptions'][1]} | "
            f"{'⛔ knee' if r['flagged_as_knee'] else '—'} |")
    lines += [
        "",
        (f"The check flags a configuration as *at the knee* when a p95 moves by more than "
         f"{a['knee_factor']}× under the {pct}% slowdown. "
         + ("It flags both runs on which the simulator was more than 1.5× off, and none of the "
            "three on which it was within 1.07×."
            if a["flags_every_run_the_model_got_wrong"] and a["flags_no_run_the_model_got_right"]
            else "It does NOT separate the two groups.")
         + f" Any threshold between {lo:.2f}× and {hi:.2f}× would do the same on these five."),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    header = results_header("scripts/edge_posthoc_t1.py", seed=0, hardware_path="cpu-only")
    a = analyze()
    (OUT / "posthoc_edge.json").write_text(json.dumps({**header, **a}, indent=1) + "\n")
    (OUT / "posthoc_edge.md").write_text(markdown(a, header))
    print((OUT / "posthoc_edge.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
