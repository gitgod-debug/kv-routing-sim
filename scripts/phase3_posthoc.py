#!/usr/bin/env python3
"""Phase 3 POST-HOC diagnostics — written after the results were seen. NOT pre-registered.

    python scripts/phase3_posthoc.py [RESULTS_DIR]    # default artifacts/results/phase3_kaggle_t4

`phase3_analyze.py` is the pre-registered analysis and its numbers are the result. This script only
explains the four things that analysis surfaced, each with a measurement rather than a story — and
it changes nothing the analysis reported:

  1. The decode step is NOT linear in batch size at batch 1 on this engine: one sequence decodes
     slower than two. The pre-registered straight-line fit cannot express that (R² 0.58).
  2. Why W3 was nonetheless predicted well: it ran almost entirely at large batches.
  3. W1's histogram residual — the shortest-lived blocks — comes from that timing fit, not the
     cache model: re-run with the MEASURED batch-1 step it mostly closes, and cache counts hold.
  4. Why the fitted prefill_base_s came out negative (and was clamped to 0).

Writes posthoc.md and posthoc.json into RESULTS_DIR.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from dataclasses import replace
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import phase3_analyze as analysis

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run


def decode_by_batch(res: dict[str, Any], prof: CostProfile) -> dict[str, Any]:
    rows = [{"batch": s["batch"], "measured_ms": s["median_decode_interval_ms"],
             "fitted_ms": (prof.decode_base_s + prof.decode_per_seq_s * s["batch"]) * 1e3,
             "source": "decode sweep"} for s in res["decode_sweep"] if s["valid"]]
    w1 = [q["decode_interval_ms"] for r in res["runs"] if r["workload"] == "W1_long" and r["valid"]
          for q in r["requests"] if q["decode_interval_ms"]]
    rows.insert(0, {"batch": 1, "measured_ms": statistics.median(w1),
                    "fitted_ms": (prof.decode_base_s + prof.decode_per_seq_s) * 1e3,
                    "source": f"W1, median of {len(w1)} requests"})
    return {"rows": rows, "w1_batch1_ms": statistics.median(w1), "w1_batch1_min_ms": min(w1)}


def w3_concurrency(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Time-weighted number of requests between first and last token, from client timestamps.
    An UPPER bound on the engine's decode batch: a preempted request still counts here."""
    out = []
    for r in (x for x in res["runs"] if x["workload"] == "W3_concurrent" and x["valid"]):
        ev = []
        for q in r["requests"]:
            ev += [(q["send_offset_s"] + q["ttft_ms"] / 1e3, 1),
                   (q["send_offset_s"] + q["total_ms"] / 1e3, -1)]
        ev.sort()
        t_prev, n, weight = ev[0][0], 0, {}
        for t, step in ev:
            weight[n] = weight.get(n, 0.0) + t - t_prev
            n, t_prev = n + step, t
        total = sum(weight.values())
        out.append({"run": r["run"], "span_s": round(total, 1),
                    "mean": round(sum(k * v for k, v in weight.items()) / total, 1),
                    "max": max(weight),
                    "share_of_time_at_8_or_more": round(
                        sum(v for k, v in weight.items() if k >= 8) / total, 3)})
    return out


def w1_histogram_attribution(d, prereg, res, prof: CostProfile, w1_b1_s: float) -> list[dict]:
    """The W1 lifetime histogram against the model, under the pre-registered fit and under the
    same fit with only the batch-1 decode step replaced by its measured value."""
    variants = {
        "pre-registered fit": prof,
        "batch-1 step = measured (post-hoc)": replace(
            prof, decode_base_s=w1_b1_s - prof.decode_per_seq_s, name="posthoc"),
    }
    fam = "vllm:kv_block_lifetime_seconds"
    out = []
    for r in (x for x in res["runs"] if x["workload"] == "W1_long" and x["valid"]):
        tag = f"W1_long_r{r['run']}"
        b = analysis._buckets((d / f"metrics_{tag}_before.txt").read_text(), fam)
        a = analysis._buckets((d / f"metrics_{tag}_after.txt").read_text(), fam)
        eng = {le: a[le] - b.get(le, 0.0) for le in sorted(a)}
        for label, p in variants.items():
            sim = run(analysis._requests_at(prereg, r, r["output_tokens"]), policy="round-robin",
                      n_replicas=1, profile=p, cache_model="vllm", track_block_metrics=True)
            vals = [e.lifetime_s for e in sim.eviction_events]
            cum = {le: sum(1 for v in vals if v <= le) for le in eng}
            out.append({
                "run": tag, "profile": label, "cache_hits": sim.cache_hits,
                "max_cdf_distance": round(max(abs(eng[le] / eng[math.inf] - cum[le] / len(vals))
                                              for le in eng), 4),
                "under_2s": {"engine": [eng[le] for le in eng if le <= 2],
                             "model": [cum[le] for le in eng if le <= 2],
                             "edges": [le for le in eng if le <= 2]},
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="?", type=pathlib.Path, default=analysis.DEFAULT_RESULTS)
    args = ap.parse_args()
    d = args.results
    prereg = json.loads(analysis.PREREG.read_text())
    res = json.loads((d / "phase3.json").read_text())
    rep = json.loads((d / "calibration_report.json").read_text())
    prof = CostProfile(**json.loads((d / "fitted_profile.json").read_text()))

    dec = decode_by_batch(res, prof)
    conc = w3_concurrency(res)
    hist = w1_histogram_attribution(d, prereg, res, prof, dec["w1_batch1_ms"] / 1e3)
    ttft = rep["timing_fit"]["ttft_vs_uncached_tokens"]
    fixed = {"ttft_intercept_ms": ttft["intercept"] * 1e3,
             "fitted_decode_step_batch1_ms": (prof.decode_base_s + prof.decode_per_seq_s) * 1e3,
             "measured_decode_step_batch1_ms": dec["w1_batch1_ms"]}
    out = {"NOT_PRE_REGISTERED": True, "decode_by_batch": dec, "w3_concurrency": conc,
           "w1_histogram_attribution": hist, "prefill_fixed_cost": fixed}
    (d / "posthoc.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = [
        "# Phase 3 — post-hoc diagnostics",
        "",
        "> ⚠️ **Written after the results were seen. NOT pre-registered.** The pre-registered",
        "> result is [`calibration_report.md`](calibration_report.md), and nothing here changes",
        "> it. This explains what that report surfaced, each point with a measurement.",
        "> Generated by `scripts/phase3_posthoc.py`.",
        "",
        "## 1 · The decode step is not linear at batch 1",
        "",
        "| batch | measured (ms) | pre-registered linear fit (ms) | measured by |",
        "|---|---|---|---|",
    ]
    for row in dec["rows"]:
        lines.append(f"| {row['batch']} | {row['measured_ms']:.1f} | {row['fitted_ms']:.1f} | "
                     f"{row['source']} |")
    lines += [
        "",
        (
            f"One sequence decodes at **{dec['w1_batch1_ms']:.1f} ms** per token (W1, never below "
            f"{dec['w1_batch1_min_ms']:.1f}) — *slower* than two to eight sequences. A line "
            "through the sweep cannot express that; hence R² 0.58. The cause is not established "
            "here (CUDA graphs were captured for batch 1). What is measured: a lone prefill pass "
            "has at most ~33 ms of fixed cost (§4, HTTP round trip included), and it reads every "
            "weight once, as a decode step does — so the extra ~30 ms per batch-1 decode step is "
            "not memory bandwidth."
        ),
        "",
        "## 2 · Why W3 was predicted well anyway",
        "",
        "| run | span (s) | mean decoding at once | max | share of time at ≥ 8 |",
        "|---|---|---|---|---|",
        *[f"| W3 r{c['run']} | {c['span_s']} | {c['mean']} | {c['max']} | "
          f"{c['share_of_time_at_8_or_more']:.0%} |" for c in conc],
        "",
        (
            "W3 spent nearly all its time at large batches, where the line is within a few "
            "percent of the sweep. (Client-side count: an upper bound on the engine's batch, since "
            "a preempted request still counts.)"
        ),
        "",
        "## 3 · W1's histogram residual is the timing fit, not the cache model",
        "",
        "| run | model timing | cache hits | max CDF distance | ≤ 2 s cumulative, engine/model |",
        "|---|---|---|---|---|",
    ]
    for h in hist:
        u = h["under_2s"]
        pairs = " · ".join(f"≤{e:g}s {en:.0f}/{m}" for e, en, m in
                           zip(u["edges"], u["engine"], u["model"], strict=True) if en or m)
        lines.append(f"| {h['run']} | {h['profile']} | {h['cache_hits']:,} | "
                     f"{h['max_cdf_distance']:.3f} | {pairs} |")
    lines += [
        "",
        (
            "The residual sits entirely in the ~119 shortest-lived events: each request's last "
            "decode blocks, recycled by the next request. The fitted batch-1 step is too short, so "
            "in the simulation every request finishes early and idles until the next real send — "
            "ageing exactly those blocks. Replacing that one number with its measured value moves "
            "them back; **the cache hits do not change under either timing**, as the "
            "pre-registration claimed."
        ),
        "",
        "## 4 · Why prefill_base_s came out negative",
        "",
        (
            f"TTFT for a lone request = **{fixed['ttft_intercept_ms']:.1f} ms** + "
            f"{ttft['slope'] * 1e3:.3f} ms × uncached token (R² {ttft['r2']:.3f}). The simulator "
            "emits the first token at the end of a step costing prefill(uncached) + decode(1), so "
            "the procedure subtracts the fitted decode(1) = "
            f"{fixed['fitted_decode_step_batch1_ms']:.1f} ms — more than the whole intercept. In "
            "the engine the first token comes out of the prefill pass itself; there is no separate "
            "decode step to pay. So the simulator's step model **double-charges the fixed "
            "per-pass cost on every step that prefills**. Harmless where queueing dominates (W3's "
            "TTFTs are 15–41 s); a known over-estimate of ~20 ms on an idle replica. Recorded as a "
            "limit; not re-tuned."
        ),
    ]
    (d / "posthoc.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
