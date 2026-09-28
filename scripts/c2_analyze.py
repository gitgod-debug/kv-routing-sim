#!/usr/bin/env python3
"""C2's analysis — committed BEFORE the run, so nothing in it is chosen after seeing a result.

    python scripts/c2_analyze.py      # artifacts/results/c2_kaggle_t4/c2.json -> report.md,
                                      #   analysis.json, fitted_profile.json

The procedure, exactly as configs/c2_timing.yaml registers it:
  1. Decode fit, on every valid part-A rep:   step = d0 + d1·b + d2·S,   S = b·(L + (skip + out)/2)
  2. Prefill fit, on every valid part-B rep:  TTFT − decode(1, c + n) = p0 + p1·n + p2·n·(c + n/2),
     with c the cached tokens the engine reported and n the rest of the prompt.
  Least squares with every coefficient ≥ 0 (all subsets solved; the feasible one with the least
  squared error wins). The context-free fit (d2 = p2 = 0) on the same data is reported beside it.
  3. C2a — step(L 1536, b 8) ÷ step(L 128, b 8), medians over reps, ≥ c2a_min_ratio.
  4. C2b — median over part C's cells of contended ÷ alone ENGINE-SIDE inter-token latency (medians
     over reps) ≥ c2b_min_ratio.
  5. C2c — the fitted model replays each part-D workload on one replica at the times its requests
     were really sent; its p50 and p95 TTFT must be within c2c_max_factor of the measured ones, for
     both workloads. Percentiles on both sides by the notebook's nearest-rank rule, over valid
     requests.
A rule whose data is missing is UNDECIDED, and an undecided rule does not hold.

Standard library only: CI can run its tests without the [analysis] extra.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import statistics
from typing import Any

from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]


# ── least squares, coefficients ≥ 0 ─────────────────────────────────────────────────────────────


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def nnls(rows: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    """Least squares with every coefficient ≥ 0, by solving every subset of free coefficients.
    Exhaustive and deterministic: ties go to the subset enumerated first."""
    k = len(rows[0])
    best: tuple[float, list[float]] | None = None
    for mask in range(1 << k):
        cols = [j for j in range(k) if mask >> j & 1]
        coef = [0.0] * k
        if cols:
            ata = [[sum(r[i] * r[j] for r in rows) for j in cols] for i in cols]
            aty = [sum(r[i] * v for r, v in zip(rows, y, strict=True)) for i in cols]
            sol = _solve(ata, aty)
            if sol is None or any(s < 0 for s in sol):
                continue
            for j, s in zip(cols, sol, strict=True):
                coef[j] = s
        sse = sum((sum(c * x for c, x in zip(coef, r, strict=True)) - v) ** 2
                  for r, v in zip(rows, y, strict=True))
        if best is None or sse < best[0]:
            best = (sse, coef)
    assert best is not None
    return best[1], best[0]


def r_squared(rows: list[list[float]], y: list[float], coef: list[float]) -> float:
    mean = statistics.fmean(y)
    tot = sum((v - mean) ** 2 for v in y)
    res = sum((sum(c * x for c, x in zip(coef, r, strict=True)) - v) ** 2
              for r, v in zip(rows, y, strict=True))
    return 1.0 - res / tot if tot else 1.0


# ── the fits ────────────────────────────────────────────────────────────────────────────────────


def fit_decode(decode: list[dict], design: dict, with_context: bool) -> dict[str, Any]:
    d = design["decode"]
    mid = (d["skip_tokens"] + d["output_tokens"]) / 2.0
    pts = [m for m in decode if m["valid"] and m["step_s"] is not None]
    rows = [[1.0, float(m["b"]), m["b"] * (m["L"] + mid)] for m in pts]
    if not with_context:
        rows = [r[:2] for r in rows]
    y = [m["step_s"] for m in pts]
    coef, _ = nnls(rows, y)
    coef = coef + [0.0] * (3 - len(coef))
    return {"base_s": coef[0], "per_seq_s": coef[1], "per_ctx_token_s": coef[2],
            "points": len(pts), "r2": r_squared([r + [0.0] * (3 - len(r)) for r in rows], y,
                                                coef)}


def fit_prefill(prefill: list[dict], dec: dict, with_context: bool) -> dict[str, Any]:
    pts = [m for m in prefill if m["valid"]]
    rows, y = [], []
    for m in pts:
        c = m["cached_tokens"]
        n = m["prompt_tokens"] - c
        one_step = dec["base_s"] + dec["per_seq_s"] + dec["per_ctx_token_s"] * (c + n)
        rows.append([1.0, float(n), n * (c + n / 2.0)] if with_context else [1.0, float(n)])
        y.append(m["ttft_s"] - one_step)
    coef, _ = nnls(rows, y)
    coef = coef + [0.0] * (3 - len(coef))
    return {"base_s": coef[0], "per_token_s": coef[1], "attn_s": coef[2], "points": len(pts),
            "r2": r_squared([r + [0.0] * (3 - len(r)) for r in rows], y, coef)}


def profile_from(name: str, dec: dict, pre: dict, e: dict, gpu: str) -> CostProfile:
    return CostProfile(
        name=name, prefill_base_s=pre["base_s"], prefill_per_token_s=pre["per_token_s"],
        decode_base_s=dec["base_s"], decode_per_seq_s=dec["per_seq_s"],
        kv_cache_blocks=e["num_gpu_blocks"], block_size=e["block_size"],
        max_num_seqs=e["max_num_seqs"], gpu=gpu, model=e["model"], engine_version=e["vllm"],
        prefill_attn_s=pre["attn_s"], decode_per_ctx_token_s=dec["per_ctx_token_s"],
    )


# ── the rules ───────────────────────────────────────────────────────────────────────────────────


def _median_step(decode: list[dict], L: int, b: int, key: str = "step_s",
                 condition: str | None = None) -> float | None:
    vals = [m[key] for m in decode if m["L"] == L and m["b"] == b and m["valid"]
            and m.get(key) is not None and (condition is None or m["condition"] == condition)]
    return statistics.median(vals) if vals else None


def nearest_rank(sorted_vals: list[float], q: float) -> float | None:
    """The notebook's percentile rule, used on both sides of every comparison."""
    if not sorted_vals:
        return None
    return sorted_vals[max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1))))]


def replay(name: str, spec: dict, measured: dict, profile: CostProfile) -> dict[str, Any]:
    """Part D's workload through the simulator, each request at the time it was really sent."""
    sent = {r["request_id"]: max(0.0, r["send_offset_s"]) for r in measured["requests"]}
    trace, _ = AgenticWorkload(**spec).generate()
    trace = sorted((dataclasses.replace(q, arrival_ts=sent[q.request_id]) for q in trace),
                   key=lambda q: q.arrival_ts)
    r = run(trace, policy="round-robin", n_replicas=1, seed=0, profile=profile,
            cache_model="vllm", record_per_request=True)
    valid = {x["request_id"] for x in measured["requests"] if x["valid"]}
    ttfts = sorted(v["ttft_ms"] for k, v in r.per_request.items() if k in valid)
    return {"hit_rate": r.hit_rate, "preemptions": r.preemptions,
            "ttft_p50_ms": nearest_rank(ttfts, 0.50), "ttft_p95_ms": nearest_rank(ttfts, 0.95)}


def factor(pred: float | None, meas: float | None) -> float | None:
    if not pred or not meas:
        return None
    return max(pred / meas, meas / pred)


def analyze(data: dict, prereg: dict) -> dict[str, Any]:
    cfg = prereg["config"]
    rules, e = cfg["rules"], cfg["engine"]
    design = {k: cfg[k] for k in ("decode", "prefill", "cpu")}
    if data["predictions_sha256"] != prereg["predictions_sha256"]:
        raise SystemExit("error: the run carried different predictions from the registered ones")
    if data["fingerprints"] != prereg["trace_fingerprints_sha256"]:
        raise SystemExit("error: the run replayed different traffic from the registered traces")
    gpu = (data.get("environment", {}).get("gpus") or "").splitlines()
    gpu_name = gpu[0].split(",")[1].strip() if gpu and "," in gpu[0] else "unknown"

    out: dict[str, Any] = {"script": "scripts/c2_analyze.py"}
    decode, prefill = data.get("decode", []), data.get("prefill", [])
    dec_ctx, dec_flat = fit_decode(decode, design, True), fit_decode(decode, design, False)
    pre_ctx = fit_prefill(prefill, dec_ctx, True)
    pre_flat = fit_prefill(prefill, dec_flat, False)
    out["fit"] = {"decode": dec_ctx, "prefill": pre_ctx,
                  "context_free": {"decode": dec_flat, "prefill": pre_flat}}

    # C2a
    s128, s1536 = _median_step(decode, 128, 8), _median_step(decode, 1536, 8)
    ratio_a = s1536 / s128 if s128 and s1536 else None
    out["c2a"] = {"step_L128_b8_s": s128, "step_L1536_b8_s": s1536, "ratio": ratio_a,
                  "min_ratio": rules["c2a_min_ratio"],
                  "holds": ratio_a is not None and ratio_a >= rules["c2a_min_ratio"],
                  "decided": ratio_a is not None}

    # C2b
    cpu, cells = data.get("cpu", []), []
    for L, b in design["cpu"]["cells"]:
        alone = _median_step(cpu, L, b, "server_itl_s", "alone")
        busy = _median_step(cpu, L, b, "server_itl_s", "contended")
        cells.append({"L": L, "b": b, "alone_itl_s": alone, "contended_itl_s": busy,
                      "ratio": busy / alone if alone and busy else None,
                      "client_alone_s": _median_step(cpu, L, b, "step_s", "alone"),
                      "client_contended_s": _median_step(cpu, L, b, "step_s", "contended")})
    ratios = [c["ratio"] for c in cells if c["ratio"] is not None]
    decided_b = len(ratios) == len(cells) and bool(cells)
    ratio_b = statistics.median(ratios) if decided_b else None
    out["c2b"] = {"cells": cells, "median_ratio": ratio_b, "min_ratio": rules["c2b_min_ratio"],
                  "holds": decided_b and ratio_b >= rules["c2b_min_ratio"], "decided": decided_b}

    # C2c
    fitted = profile_from("t4-qwen3-4b-vllm0.30.0-ctx", dec_ctx, pre_ctx, e, gpu_name)
    flat = load_profile(REPO / cfg["profile"])
    workloads, holdout = prereg["workloads"], data.get("holdout", [])
    runs = []
    for m in holdout:
        name = m["workload"]
        valid = sorted(r["ttft_ms"] for r in m["requests"] if r["valid"] and r["ttft_ms"])
        meas = {"hit_rate": m["hit_rate"], "preemptions": m["preemptions"],
                "ttft_p50_ms": nearest_rank(valid, 0.50), "ttft_p95_ms": nearest_rank(valid, 0.95),
                "invalid": m["invalid"], "worst_send_lag_ms": m["worst_send_lag_ms"]}
        ctx, cf = replay(name, workloads[name], m, fitted), replay(name, workloads[name], m, flat)
        f = {q: factor(ctx[f"ttft_{q}_ms"], meas[f"ttft_{q}_ms"]) for q in ("p50", "p95")}
        runs.append({"workload": name, "measured": meas, "context_aware": ctx,
                     "context_free": cf, "factor": f,
                     "hit_rate_error": (abs(ctx["hit_rate"] - meas["hit_rate"]) / meas["hit_rate"]
                                        if meas["hit_rate"] else None)})
    decided_c = len(runs) == len(workloads) and all(
        v is not None for r in runs for v in r["factor"].values())
    out["c2c"] = {"runs": runs, "max_factor": rules["c2c_max_factor"], "decided": decided_c,
                  "holds": decided_c and all(v <= rules["c2c_max_factor"]
                                             for r in runs for v in r["factor"].values())}
    fitted = dataclasses.replace(fitted, calibrated=out["c2c"]["holds"])
    out["fitted_profile"] = dataclasses.asdict(fitted)
    out["predicted_outcomes"] = cfg["predictions"]
    return out


# ── the report ──────────────────────────────────────────────────────────────────────────────────


def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x * 1e3:.1f} ms"


def markdown(a: dict, data: dict, prereg: dict) -> str:
    cfg, fit = prereg["config"], a["fit"]
    verdict = {True: "HOLDS", False: "DOES NOT HOLD"}

    def v(rule: str) -> str:
        return verdict[a[rule]["holds"]] if a[rule]["decided"] else "UNDECIDED (data missing)"

    dc, pc = fit["decode"], fit["prefill"]
    fd, fp = fit["context_free"]["decode"], fit["context_free"]["prefill"]
    lines = [
        "# C2 — the timing model on a real T4",
        "",
        (f"Registered `configs/c2_timing.yaml` · predictions "
         f"`{prereg['predictions_sha256'][:16]}…` match the run · environment "
         f"{data.get('environment', {}).get('date_utc', '?')} · "
         "generated by `scripts/c2_analyze.py`, committed before the run."),
        "",
        "| rule | registered prediction | result |",
        "|---|---|---|",
        f"| **C2a** — context term is real | {cfg['predictions']['c2a']} | **{v('c2a')}** |",
        f"| **C2b** — CPU contention is real | {cfg['predictions']['c2b']} | **{v('c2b')}** |",
        (f"| **C2c** — held-out traffic predicted within {cfg['rules']['c2c_max_factor']}× | "
         f"{cfg['predictions']['c2c']} | **{v('c2c')}** |"),
        "",
        "## The fitted timing model",
        "",
        "| | context-aware (fitted) | context-free (same data) | the profile every result used |",
        "|---|---:|---:|---:|",
    ]
    flat = load_profile(REPO / cfg["profile"])
    lines += [
        (f"| decode base | {_ms(dc['base_s'])} | {_ms(fd['base_s'])} | "
         f"{_ms(flat.decode_base_s)} |"),
        (f"| decode per sequence | {_ms(dc['per_seq_s'])} | {_ms(fd['per_seq_s'])} | "
         f"{_ms(flat.decode_per_seq_s)} |"),
        (f"| decode per 1K context tokens (batch sum) | {_ms(dc['per_ctx_token_s'] * 1e3)} "
         "| — | — |"),
        f"| decode fit R² ({dc['points']} points) | {dc['r2']:.3f} | {fd['r2']:.3f} | |",
        (f"| prefill base | {_ms(pc['base_s'])} | {_ms(fp['base_s'])} | "
         f"{_ms(flat.prefill_base_s)} |"),
        (f"| prefill per new token | {pc['per_token_s'] * 1e3:.3f} ms | "
         f"{fp['per_token_s'] * 1e3:.3f} ms | {flat.prefill_per_token_s * 1e3:.3f} ms |"),
        (f"| prefill attention, per new × 1K attended tokens | {pc['attn_s'] * 1e6:.3f} ms "
         "| — | — |"),
        f"| prefill fit R² ({pc['points']} points) | {pc['r2']:.3f} | {fp['r2']:.3f} | |",
        "",
        "## Part A — decode step (median over reps; — = invalid)",
        "",
    ]
    grid = cfg["decode"]["grid"]
    Ls = sorted({L for L, _ in grid})
    bs = sorted({b for _, b in grid})
    lines += ["| L \\ b | " + " | ".join(str(b) for b in bs) + " |",
              "|---" * (len(bs) + 1) + "|"]
    for L in Ls:
        cells = []
        for b in bs:
            s = _median_step(data.get("decode", []), L, b) if [L, b] in grid else None
            cells.append("" if [L, b] not in grid else ("—" if s is None else f"{s * 1e3:.0f}"))
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    c2a = a["c2a"]
    ratio_a = "—" if c2a["ratio"] is None else f"{c2a['ratio']:.2f}"
    lines += [
        "",
        (f"**C2a:** step at L 1536, b 8 = {_ms(c2a['step_L1536_b8_s'])}; at L 128, b 8 = "
         f"{_ms(c2a['step_L128_b8_s'])}; ratio {ratio_a} (needed ≥ {c2a['min_ratio']})."),
        "",
        "## Part C — CPU contention (engine-side inter-token latency, medians over reps)",
        "",
        "| cell | alone | contended | ratio | client step alone · contended |",
        "|---|---:|---:|---:|---|",
    ]
    for c in a["c2b"]["cells"]:
        ratio = "—" if c["ratio"] is None else f"{c['ratio']:.2f}"
        lines.append(f"| L {c['L']}, b {c['b']} | {_ms(c['alone_itl_s'])} | "
                     f"{_ms(c['contended_itl_s'])} | {ratio} | {_ms(c['client_alone_s'])} · "
                     f"{_ms(c['client_contended_s'])} |")
    mr = a["c2b"]["median_ratio"]
    lines += ["", (f"**C2b:** median ratio {'—' if mr is None else f'{mr:.2f}'} (needed ≥ "
                   f"{a['c2b']['min_ratio']} for contention to count)."), "",
              "## Part D — held-out traffic on one engine",
              "",
              ("| workload | measured p50 · p95 | context-aware model | factor | "
               "context-free model | hit rate measured · model | preemptions measured · model |"),
              "|---|---|---|---|---|---|---|"]
    for r in a["c2c"]["runs"]:
        m, c, f, cf = r["measured"], r["context_aware"], r["factor"], r["context_free"]
        fac = " · ".join("—" if f[q] is None else f"{f[q]:.2f}×" for q in ("p50", "p95"))
        lines.append(
            f"| {r['workload']} | {m['ttft_p50_ms'] or 0:.0f} · {m['ttft_p95_ms'] or 0:.0f} ms | "
            f"{c['ttft_p50_ms'] or 0:.0f} · {c['ttft_p95_ms'] or 0:.0f} ms | {fac} | "
            f"{cf['ttft_p50_ms'] or 0:.0f} · {cf['ttft_p95_ms'] or 0:.0f} ms | "
            f"{m['hit_rate']:.4f} · {c['hit_rate']:.4f} | "
            f"{m['preemptions']} · {c['preemptions']} |")
    lines += ["", (f"**C2c:** holds only if every factor is ≤ {a['c2c']['max_factor']}. The "
                   "context-free column is the profile every earlier result used, replayed the "
                   "same way — reported, not tested."), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c2_kaggle_t4")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c2_prereg" / "predictions.json")
    args = ap.parse_args()
    data = json.loads((args.results / "c2.json").read_text())
    prereg = json.loads(args.prereg.read_text())
    a = analyze(data, prereg)
    (args.results / "analysis.json").write_text(json.dumps(a, indent=1, default=str) + "\n")
    prof = dict(a["fitted_profile"])
    prof = {"_what": ("Fitted by scripts/c2_analyze.py on C2's parts A and B (one T4, vLLM "
                      "0.30.0, Qwen3-4B). `calibrated` is true only if it predicted the held-out "
                      "part D within the registered factor (C2c)."),
            "_source": {"date_utc": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"), "fit": a["fit"]}, **prof}
    (args.results / "fitted_profile.json").write_text(json.dumps(prof, indent=1) + "\n")
    (args.results / "report.md").write_text(markdown(a, data, prereg))
    print((args.results / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
