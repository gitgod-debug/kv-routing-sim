#!/usr/bin/env python3
"""Phase 3's pre-registered analysis: the Real-Engine Calibration Report, from a GPU run.

    python scripts/phase3_analyze.py [RESULTS_DIR]    # default artifacts/results/phase3_kaggle_t4

Follows `analysis_procedure` in artifacts/results/phase3_prereg/predictions.json, step by step, and
nothing else. The ONLY thing fitted is the timing model — and never against W3, which it then
predicts. The cache models are not touched: their predictions were committed before the run.

Writes into RESULTS_DIR: calibration_report.json, calibration_report.md, fitted_profile.json.
Refuses to run if the results were not produced by a notebook carrying the committed predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import asdict, replace
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase3_prepare import build_trace, fingerprint, sweep_prompts

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.sim.metrics import percentile

PREREG = REPO / "artifacts" / "results" / "phase3_prereg" / "predictions.json"
DEFAULT_RESULTS = REPO / "artifacts" / "results" / "phase3_kaggle_t4"
HIST_FAMILIES = {
    "lifetime": "vllm:kv_block_lifetime_seconds",
    "idle": "vllm:kv_block_idle_before_evict_seconds",
    "reuse_gap": "vllm:kv_block_reuse_gap_seconds",
}


# ── loading and provenance (procedure step 1) ─────────────────────────────────────────────────


def load_results(d: pathlib.Path) -> dict[str, Any]:
    agg = d / "phase3.json"
    if agg.exists():
        return json.loads(agg.read_text())
    # A run that died before the verdict cell: assemble what exists, and say so.
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"))]
    if not runs:
        raise SystemExit(f"⛔ no phase3.json and no run_*.json in {d}")
    sweep = d / "decode_sweep.json"
    print("⚠️  no phase3.json: the notebook did not reach its verdict cell; using per-run files")
    return {"runs": runs, "decode_sweep": json.loads(sweep.read_text()) if sweep.exists() else [],
            "partial": True}


def verify(prereg: dict[str, Any], res: dict[str, Any]) -> list[str]:
    """The run must have compared against exactly what was committed."""
    notes = []
    for key in ("trace_fingerprint_sha256", "predictions_sha256"):
        if key in res and res[key] != prereg[key]:
            raise SystemExit(f"⛔ {key} differs from the committed prereg: these results were "
                             "produced against different predictions")
        if key not in res:
            notes.append(f"{key} not in the results (partial run) — provenance is per-run only")
    # The trace is regenerated here from the committed config; it must be the one that was sent.
    cfg = prereg["config"]
    if fingerprint(build_trace(cfg), sweep_prompts(cfg)) != prereg["trace_fingerprint_sha256"]:
        raise SystemExit("⛔ this checkout regenerates a different trace than the one committed")
    cap = res.get("capacity", {})
    if cap and cap.get("kv_cache_size_tokens_logged") != cap.get("expected_tokens"):
        raise SystemExit(f"⛔ the engine's capacity was not the predicted one: {cap}")
    return notes


# ── step 2 + 3: sequential counts, exact ──────────────────────────────────────────────────────


def sequential(prereg: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = prereg["predictions"]["compared_counters"]
    out: dict[str, Any] = {}
    for name, w in prereg["predictions"]["workloads"].items():
        if w["mode"] != "sequential":
            continue
        rs = [r for r in runs if r["workload"] == name]
        ok = [r for r in rs if r["valid"]]
        entry: dict[str, Any] = {"runs": len(rs), "valid_runs": len(ok), "models": {}}
        for model in ("vllm", "lru", "vllm_if_one_more_token_committed"):
            p = w["predictions"][model]
            per_run = []
            for r in ok:
                m = r["measured"]
                cached = {x["request_id"]: x["cached_tokens"] for x in r["requests"]}
                diffs = {k: None if p.get(k) is None else m[k] - p[k] for k in keys}
                req = p["per_request_cached_tokens"]
                hr = m["prefix_cache_hits"] / m["prefix_cache_queries"]
                phr = p["prefix_cache_hits"] / p["prefix_cache_queries"]
                per_run.append({
                    "run": r["run"],
                    "diff_measured_minus_predicted": diffs,
                    "counters_exact": all(v in (None, 0) for v in diffs.values()),
                    "requests_matching": sum(cached.get(k) == v for k, v in req.items()),
                    "requests": len(req),
                    "hit_rate_measured": round(hr, 6),
                    "hit_rate_predicted": round(phr, 6),
                    "hit_rate_error_pts": round((hr - phr) * 100, 4),
                    "local_compute_error_tokens": m["local_compute"] - p["local_compute"],
                })
            exact = [x["counters_exact"] and x["requests_matching"] == x["requests"]
                     for x in per_run]
            entry["models"][model] = {"exact_runs": sum(exact), "per_run": per_run}
        # Determinism: a deterministic engine gives the same counters, and the same per-request
        # cached tokens, on every run.
        sig = {json.dumps([r["measured"], [x["cached_tokens"] for x in r["requests"]]])
               for r in ok}
        entry["runs_identical"] = len(sig) == 1 if ok else None
        out[name] = entry
    return out


# ── step 4: the timing fit — never against W3 ─────────────────────────────────────────────────


def ols(x: list[float], y: list[float]) -> dict[str, float]:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = intercept + slope * xa
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    return {"intercept": float(intercept), "slope": float(slope), "n": len(xa),
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "residual_p95_abs": float(np.percentile(np.abs(ya - pred), 95))}


def fit_timing(res: dict[str, Any], prereg: dict[str, Any]) -> tuple[CostProfile, dict[str, Any]]:
    sweep = [s for s in res.get("decode_sweep", []) if s.get("valid")]
    if len(sweep) < 2:
        raise SystemExit("⛔ fewer than two valid decode-sweep batches: no decode fit is possible")
    dec = ols([s["batch"] for s in sweep], [s["median_decode_interval_ms"] / 1e3 for s in sweep])
    decode_base, decode_per_seq = dec["intercept"], dec["slope"]

    xs, ys = [], []
    for r in res["runs"]:
        if r["valid"] and r["workload"] in ("W1_long", "W2_short"):
            for q in r["requests"]:
                xs.append(q["prompt_tokens"] - q["cached_tokens"])
                ys.append(q["ttft_ms"] / 1e3)
    pre = ols(xs, ys)
    # The simulator emits the first token at the END of the step that prefills — so a lone
    # request's TTFT is prefill(uncached) + decode(1). The intercept holds both fixed costs.
    prefill_base_raw = pre["intercept"] - (decode_base + decode_per_seq)
    notes = []
    prefill_base = prefill_base_raw
    if prefill_base < 0:
        notes.append(f"fitted prefill_base_s was {prefill_base_raw * 1e3:.2f} ms (< 0, not "
                     "physical); clamped to 0 — the decode fit absorbs the whole fixed cost")
        prefill_base = 0.0

    e, env = prereg["config"]["engine"], res.get("environment", {})
    gpu_line = (env.get("gpus") or "").splitlines()
    gpu = gpu_line[0].split(",")[1].strip() if gpu_line and "," in gpu_line[0] else "unknown"
    prof = CostProfile(
        name=f"{gpu.lower().replace(' ', '-')}-{e['model'].split('/')[-1].lower()}-vllm{e['vllm']}",
        prefill_base_s=prefill_base,
        prefill_per_token_s=pre["slope"],
        decode_base_s=decode_base,
        decode_per_seq_s=decode_per_seq,
        kv_cache_blocks=e["num_gpu_blocks"],
        block_size=e["block_size"],
        max_num_seqs=e["max_num_seqs"],
        gpu=gpu,
        model=e["model"],
        engine_version=e["vllm"],
        calibrated=True,
    )
    fit = {"decode_step_vs_batch": dec, "ttft_vs_uncached_tokens": pre,
           "prefill_base_s_raw": prefill_base_raw, "notes": notes,
           "decode_sweep_points": [[s["batch"], s["median_decode_interval_ms"]] for s in sweep]}
    return prof, fit


# ── step 5: W3, predicted by the fitted profile ───────────────────────────────────────────────


def _requests_at(prereg, run_record: dict[str, Any], out: int):
    """The trace, with each request arriving exactly when the notebook really sent it."""
    sent = {q["request_id"]: max(0.0, q["send_offset_s"]) for q in run_record["requests"]}
    reqs = [r for r, _ in build_trace(prereg["config"])]
    return sorted((replace(r, arrival_ts=sent[r.request_id], output_tokens=out) for r in reqs),
                  key=lambda r: r.arrival_ts)


def _pcts(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    return {f"p{round(q * 100)}_ms": round(percentile(s, q), 1) for q in (0.5, 0.95, 0.99)}


def w3(prereg, res, prof: CostProfile) -> dict[str, Any]:
    name = next(n for n, w in prereg["predictions"]["workloads"].items()
                if w["mode"] == "concurrent")
    out = prereg["predictions"]["workloads"][name]["output_tokens"]
    rows = []
    for r in (x for x in res["runs"] if x["workload"] == name and x["valid"]):
        m = r["measured"]
        measured = {
            "hit_rate": round(m["prefix_cache_hits"] / m["prefix_cache_queries"], 6),
            "local_compute": m["local_compute"],
            "preemptions": m["preemptions"],
            **_pcts([q["ttft_ms"] for q in r["requests"]]),
        }
        sims = {}
        for model in ("vllm", "lru"):
            s = run(_requests_at(prereg, r, out), policy="round-robin", n_replicas=1,
                    profile=prof, cache_model=model)
            sims[model] = {"hit_rate": round(s.hit_rate, 6),
                           "local_compute": s.cache_queries - s.cache_hits,
                           "preemptions": s.preemptions, **_pcts(s.ttft_ms)}
        rows.append({"run": r["run"], "measured": measured, "simulated": sims})
    summary: dict[str, Any] = {}
    for k in ("hit_rate", "local_compute", "preemptions", "p50_ms", "p95_ms", "p99_ms"):
        vals = [x["measured"][k] for x in rows]
        if not vals:
            continue
        summary[k] = {"measured_mean": float(np.mean(vals)), "measured_min": min(vals),
                      "measured_max": max(vals)}
        for model in ("vllm", "lru"):
            sv = float(np.mean([x["simulated"][model][k] for x in rows]))
            mean = summary[k]["measured_mean"]
            summary[k][model] = {
                "simulated_mean": sv,
                "error": sv - mean,
                "relative_error": (sv - mean) / mean if mean else None,
                "within_run_range": summary[k]["measured_min"] <= sv
                <= summary[k]["measured_max"],
            }
    return {"workload": name, "per_run": rows, "summary": summary}


# ── step 6: H4, the histograms ────────────────────────────────────────────────────────────────


def _buckets(text: str, family: str) -> dict[float, float]:
    pat = re.compile(r"^" + re.escape(family) + r'_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+(\S+)',
                     re.MULTILINE)
    out: dict[float, float] = {}
    for le, v in pat.findall(text):
        edge = math.inf if le == "+Inf" else float(le)
        out[edge] = out.get(edge, 0.0) + float(v)
    return out


def histograms(d: pathlib.Path, prereg, res, prof: CostProfile) -> dict[str, Any]:
    """Engine bucket deltas vs the vllm model's events, the model re-run at the real send times
    with the fitted profile (sequential counts do not depend on timing; lifetimes do)."""
    out = {}
    for r in res["runs"]:
        if not r["valid"] or r["mode"] != "sequential":
            continue
        tag = f"{r['workload']}_r{r['run']}"
        before, after = d / f"metrics_{tag}_before.txt", d / f"metrics_{tag}_after.txt"
        if not (before.exists() and after.exists()):
            continue
        sim = run(_requests_at(prereg, r, r["output_tokens"]), policy="round-robin",
                  n_replicas=1, profile=prof, cache_model="vllm", track_block_metrics=True)
        values = {
            "lifetime": [e.lifetime_s for e in sim.eviction_events],
            "idle": [e.idle_s for e in sim.eviction_events],
            "reuse_gap": [g for e in sim.eviction_events for g in e.reuse_gaps_s],
        }
        entry = {"sim_counts_equal_prereg": sim.cache_hits
                 == prereg["predictions"]["workloads"][r["workload"]]["predictions"]["vllm"][
                     "prefix_cache_hits"]}
        for key, fam in HIST_FAMILIES.items():
            b, a = _buckets(before.read_text(), fam), _buckets(after.read_text(), fam)
            eng = {le: a[le] - b.get(le, 0.0) for le in sorted(a)}
            if not eng:
                entry[key] = {"engine": None}
                continue
            n_eng, vals = eng[math.inf], values[key]
            sim_cum = {le: sum(1 for v in vals if v <= le) for le in eng}
            # A two-sample Kolmogorov-Smirnov-style distance on the engine's own bucket edges.
            dist = max(abs(eng[le] / n_eng - sim_cum[le] / len(vals)) for le in eng) \
                if n_eng and vals else None
            entry[key] = {"engine_count": n_eng, "model_count": len(vals),
                          "max_cdf_distance": dist,
                          "engine_cumulative": {str(k): v for k, v in eng.items()},
                          "model_cumulative": {str(k): v for k, v in sim_cum.items()}}
        out[tag] = entry
    return out


# ── the report ────────────────────────────────────────────────────────────────────────────────


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


MODEL_LABEL = {"vllm": "vllm", "lru": "lru", "vllm_if_one_more_token_committed": "vllm, +1 token ¹"}


def markdown(rep: dict[str, Any]) -> str:
    p, e = rep["profile"], rep["engine"]
    lines = [
        "# Phase 3 — Real-Engine Calibration Report",
        "",
        (
            f"**Engine:** vLLM {e['vllm']} · `{e['model']}` · fp16 · {p['gpu']} · block size "
            f"{e['block_size']} · `--num-gpu-blocks-override {e['num_gpu_blocks']}` · "
            f"max_num_seqs {e['max_num_seqs']} · async scheduling: {rep.get('async_scheduling')}"
        ),
        (
            f"**Pre-registration:** trace `{rep['trace_fingerprint_sha256'][:16]}…`, "
            f"predictions `{rep['predictions_sha256'][:16]}…` — committed before the run."
        ),
        "",
        "## 1 · Sequential workloads — exact predictions",
        "",
        (
            "| workload | model | exact runs | hit-rate error (pts) | computed-token error | "
            "requests matching |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for name, s in rep["sequential"].items():
        for model, m in s["models"].items():
            pr = m["per_run"]
            lines.append(
                f"| {name} | {MODEL_LABEL[model]} | {m['exact_runs']}/{s['valid_runs']} | "
                + ", ".join(f"{x['hit_rate_error_pts']:+.2f}" for x in pr) + " | "
                + ", ".join(f"{x['local_compute_error_tokens']:+,}" for x in pr) + " | "
                + ", ".join(f"{x['requests_matching']}/{x['requests']}" for x in pr) + " |"
            )
        lines.append(f"| {name} | *all runs identical* | {s['runs_identical']} | | | |")
    lines += [
        "",
        (
            "¹ The pre-registered alternative: the engine commits one token more per request "
            "than its source says. On W1 it predicts exactly what the vllm model does (no request "
            "ends on a block boundary), so only W2 can tell them apart."
        ),
    ]
    f = rep["timing_fit"]
    d, t = f["decode_step_vs_batch"], f["ttft_vs_uncached_tokens"]
    lines += [
        "",
        f"## 2 · Timing model — fitted, {p['gpu']} figures",
        "",
        (
            f"- decode step = **{p['decode_base_s'] * 1e3:.2f} ms + "
            f"{p['decode_per_seq_s'] * 1e3:.3f} ms × batch**  (R² {d['r2']:.3f}, "
            f"{d['n']} batch sizes)"
        ),
        (
            f"- prefill = **{p['prefill_base_s'] * 1e3:.2f} ms + "
            f"{p['prefill_per_token_s'] * 1e3:.4f} ms × uncached token**  (R² {t['r2']:.3f}, "
            f"{t['n']} requests)"
        ),
        *[f"- ⚠️ {n}" for n in f["notes"]],
        "",
        f"## 3 · {rep['w3']['workload']} — predicted with the fitted profile, never tuned on it",
        "",
        "| metric | engine mean [min, max] | vllm model | lru model |",
        "|---|---|---|---|",
    ]
    for k, v in rep["w3"]["summary"].items():
        cells = []
        for model in ("vllm", "lru"):
            x = v[model]
            rel = "" if x["relative_error"] is None else f" ({x['relative_error'] * 100:+.1f}%)"
            cells.append(f"{_num(x['simulated_mean'])}{rel}{' ✓' if x['within_run_range'] else ''}")
        lines.append(f"| {k} | {_num(v['measured_mean'])} [{_num(v['measured_min'])}, "
                     f"{_num(v['measured_max'])}] | {cells[0]} | {cells[1]} |")
    lines += ["", "✓ = inside the engine's run-to-run range.", "",
              "## 4 · H4 — eviction histograms (engine vs vllm model)", "",
              "| run | family | engine count | model count | max CDF distance |",
              "|---|---|---|---|---|"]
    for tag, h in rep["histograms"].items():
        for key in HIST_FAMILIES:
            x = h.get(key, {})
            if x.get("engine_count") is None:
                continue
            dist = "—" if x["max_cdf_distance"] is None else f"{x['max_cdf_distance']:.3f}"
            lines.append(f"| {tag} | {key} | {x['engine_count']:,.0f} | {x['model_count']:,} | "
                         f"{dist} |")
    if rep["notes"]:
        lines += ["", "## Notes", "", *[f"- {n}" for n in rep["notes"]]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="?", type=pathlib.Path, default=DEFAULT_RESULTS)
    ap.add_argument("--prereg", type=pathlib.Path, default=PREREG)
    args = ap.parse_args()
    prereg = json.loads(args.prereg.read_text())
    res = load_results(args.results)
    notes = verify(prereg, res)
    invalid = [f"{r['workload']} r{r['run']}" for r in res["runs"] if not r["valid"]]
    if invalid:
        notes.append(f"INVALID runs, reported and excluded: {invalid}")

    seq = sequential(prereg, res["runs"])
    prof, fit = fit_timing(res, prereg)
    w3_rep = w3(prereg, res, prof)
    hist = histograms(args.results, prereg, res, prof)

    errs = [abs(x["hit_rate_error_pts"]) / 100 / x["hit_rate_predicted"]
            for s in seq.values() for x in s["models"]["vllm"]["per_run"]]
    hr = w3_rep["summary"].get("hit_rate", {}).get("vllm", {}).get("relative_error")
    if hr is not None:
        errs.append(abs(hr))
    prof = replace(prof, calibration_error=round(max(errs), 6) if errs else None)

    rep = {
        "engine": prereg["config"]["engine"],
        "trace_fingerprint_sha256": prereg["trace_fingerprint_sha256"],
        "predictions_sha256": prereg["predictions_sha256"],
        "async_scheduling": res.get("async_scheduling"),
        "sequential": seq,
        "timing_fit": fit,
        "profile": asdict(prof),
        "w3": w3_rep,
        "histograms": hist,
        "notes": notes,
    }
    d = args.results
    (d / "calibration_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (d / "fitted_profile.json").write_text(json.dumps(asdict(prof), indent=1) + "\n")
    md = markdown(rep)
    (d / "calibration_report.md").write_text(md)
    print(md)
    print(f"wrote calibration_report.json, calibration_report.md, fitted_profile.json in {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
