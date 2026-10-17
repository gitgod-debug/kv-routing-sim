#!/usr/bin/env python3
"""Check on C2's data whether the cells `kvsim calibrate` measures are enough to fit its model.

    python scripts/c3_precheck.py     # artifacts/results/c2_kaggle_t4/c2.json
                                      #   -> artifacts/results/c3_prereg/precheck.{json,md}

Post hoc and CPU-only. C2 measured more cells than the command does (28 decode cells × 2, 15
prefill cells × 3). This takes the subset the command would measure, fits it with the command's
own code (`calib.calibrate.fit`), and replays C2's two held-out workloads with the result, using
C2c's replay. The `cold_only` rows show why the command also times prompts after a cached prefix.
It cannot show the command's own cells on a real engine (context 32 instead of 128, cold prompts
at 8 lengths instead of 6, no cache reset between cells); C3 measures those.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c2_analyze as c2

from kv_routing_sim.calib.calibrate import fit
from kv_routing_sim.traces.generator import AgenticWorkload

RESULTS = REPO / "artifacts" / "results" / "c2_kaggle_t4"
PREREG = REPO / "artifacts" / "results" / "c2_prereg" / "predictions.json"
OUT = REPO / "artifacts" / "results" / "c3_prereg"
# The command's decode cells are contexts 32, 512, 1024 × batches 1–16, each measured once. C2 has
# no 32-token cell; its shortest (128) stands in.
CONTEXTS, BATCHES = (128, 512, 1024), (1, 2, 4, 8, 16)


def _decode(data: dict, rep: int) -> list[dict]:
    return [{"batch": m["b"], "context": m["L"], "step_s": m["step_s"], "valid": m["valid"]}
            for m in data["decode"]
            if m["rep"] == rep and m["L"] in CONTEXTS and m["b"] in BATCHES]


def _prefill(data: dict, cached_cells: bool) -> list[dict]:
    return [{"prompt_tokens": m["prompt_tokens"],
             "uncached_tokens": m["prompt_tokens"] - m["cached_tokens"], "ttft_s": m["ttft_s"]}
            for m in data["prefill"] if m["valid"] and (cached_cells or m["c"] == 0)]


def _prefill_time(m: dict, spec: dict, base: float, per_token: float, attn: float) -> float:
    """Total prefill time a fit charges a held-out run's requests.

    Each request is charged at the cached tokens the engine reported for it.
    """
    trace, _ = AgenticWorkload(**spec).generate()
    length = {q.request_id: q.prompt_tokens for q in trace}
    total = 0.0
    for r in m["requests"]:
        c = r["cached_tokens"]
        n = length[r["request_id"]] - c
        total += base + per_token * n + attn * n * (c + n / 2.0)
    return total


def one(data: dict, prereg: dict, name: str, cached_cells: bool, rep: int) -> dict:
    """One row: the command's cells from C2's recording, fitted, and both held-out runs replayed."""
    cfg = prereg["config"]
    d = cfg["decode"]
    f = fit(_prefill(data, cached_cells), _decode(data, rep), output_tokens=d["output_tokens"],
            skip=d["skip_tokens"])
    profile = c2.profile_from(f"precheck-{name}-{rep}", f["decode_fit"], f["prefill_fit"],
                              cfg["engine"], "Tesla T4")
    runs = {}
    for m in data["holdout"]:
        valid = sorted(r["ttft_ms"] for r in m["requests"] if r["valid"] and r["ttft_ms"])
        meas = {q: c2.nearest_rank(valid, x) for q, x in (("p50", 0.50), ("p95", 0.95))}
        sim = c2.replay(m["workload"], prereg["workloads"][m["workload"]], m, profile)
        runs[m["workload"]] = {
            "measured_ms": meas,
            "predicted_ms": {q: sim[f"ttft_{q}_ms"] for q in meas},
            "factor": {q: c2.factor(sim[f"ttft_{q}_ms"], meas[q]) for q in meas},
        }
    return {"prefill_cells": name, "decode_rep": rep, "decode_points": f["decode_fit"]["points"],
            "prefill_points": f["prefill_fit"]["points"],
            "fit": {k: f[k] for k in ("decode_base_s", "decode_per_seq_s",
                                      "decode_per_ctx_token_s", "prefill_base_s",
                                      "prefill_per_token_s", "prefill_attn_s")},
            "holdout": runs}


def main() -> int:
    data = json.loads((RESULTS / "c2.json").read_text())
    prereg = json.loads(PREREG.read_text())
    full = json.loads((RESULTS / "analysis.json").read_text())["fit"]
    rows = [one(data, prereg, name, cached_cells, rep)
            for name, cached_cells in (("cold_only", False), ("with_cached_prefix", True))
            for rep in range(prereg["config"]["decode"]["reps"])]

    worst = {name: max(v for r in rows if r["prefill_cells"] == name
                       for w in r["holdout"].values() for v in w["factor"].values())
             for name in ("cold_only", "with_cached_prefix")}
    attn = {name: [r["fit"]["prefill_attn_s"] for r in rows if r["prefill_cells"] == name]
            for name in worst}
    # How much more prefill time the cold-only fit charges the traffic than C2's full fit does.
    cold, fp = rows[0]["fit"], full["prefill"]
    extra = {m["workload"]: _prefill_time(
        m, prereg["workloads"][m["workload"]], cold["prefill_base_s"],
        cold["prefill_per_token_s"], cold["prefill_attn_s"]) / _prefill_time(
        m, prereg["workloads"][m["workload"]], fp["base_s"], fp["per_token_s"], fp["attn_s"])
        for m in data["holdout"]}
    out = {"script": "scripts/c3_precheck.py", "post_hoc": True, "source": "c2_kaggle_t4/c2.json",
           "contexts": CONTEXTS, "batches": BATCHES, "c2_full_fit": full, "rows": rows,
           "worst_factor": worst, "cold_only_prefill_time_vs_c2_full_fit": extra,
           "attention_term_vs_c2_full_fit": {
               name: [a / full["prefill"]["attn_s"] for a in v] for name, v in attn.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precheck.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = [
        "# POST HOC — would `kvsim calibrate`'s own cells have been enough? (C2's data, no GPU)",
        "",
        ("⚠️ Post hoc. C2 measured more cells than the command does. Here the cells the command "
         "would have measured are fitted with the command's own code, and C2's two held-out "
         "workloads are replayed with the result — the replay C2c was decided on. Run before C3 "
         "to choose the command's design. Generated by `scripts/c3_precheck.py`."),
        "",
        (f"Decode cells: contexts {', '.join(map(str, CONTEXTS))} × batches "
         f"{', '.join(map(str, BATCHES))}, one rep (C2's 128 stands in for the command's 32). "
         "Prefill cells: **cold only** = cold prompts of 64–1,984 tokens × 3, what the command "
         "measured until 2026-10-01; **with cached prefix** = those plus 64, 256 and 704 new "
         "tokens after 256, 768 and 1,280 cached ones × 3."),
        "",
        ("| prefill cells | decode rep | decode: base + per seq + per 1K context | prefill: base "
         "+ per token + per token per 1K attended | K1_half p50 · p95 | K64_quarter p50 · p95 |"),
        "|---|---:|---|---|---|---|",
    ]
    for r in rows:
        f = r["fit"]
        cells = []
        for w in ("K1_half", "K64_quarter"):
            h = r["holdout"][w]
            cells.append(" · ".join(
                f"{h['predicted_ms'][q]:,.0f} ms ({h['factor'][q]:.2f}×)" for q in ("p50", "p95")))
        lines.append(
            f"| {r['prefill_cells'].replace('_', ' ')} | {r['decode_rep']} | "
            f"{f['decode_base_s'] * 1e3:.1f} ms + {f['decode_per_seq_s'] * 1e3:.2f} + "
            f"{f['decode_per_ctx_token_s'] * 1e6:.2f} | {f['prefill_base_s'] * 1e3:.1f} ms + "
            f"{f['prefill_per_token_s'] * 1e3:.3f} + {f['prefill_attn_s'] * 1e6:.3f} | "
            f"{cells[0]} | {cells[1]} |")
    fd, fp = full["decode"], full["prefill"]
    meas = rows[0]["holdout"]
    lines += [
        (f"| *C2's full fit (56 + 45 points)* | both | {fd['base_s'] * 1e3:.1f} ms + "
         f"{fd['per_seq_s'] * 1e3:.2f} + {fd['per_ctx_token_s'] * 1e6:.2f} | "
         f"{fp['base_s'] * 1e3:.1f} ms + {fp['per_token_s'] * 1e3:.3f} + "
         f"{fp['attn_s'] * 1e6:.3f} | *1.10× · 1.41×* | *1.31× · 1.11×* |"),
        "",
        ("Measured: " + "; ".join(
            f"{w} {m['measured_ms']['p50']:,.0f} · {m['measured_ms']['p95']:,.0f} ms"
            for w, m in meas.items()) + "."),
        "",
        (f"**Cold prompts alone are not enough.** Worst factor {worst['cold_only']:.2f}× — past "
         f"C2c's 2× limit — against {worst['with_cached_prefix']:.2f}× with the cached-prefix "
         "cells. On a cold prompt the tokens attended to are always half the prompt, so cold "
         "cells measure only n and n², and two different splits describe them equally well: the "
         "cold-only fit has a lower per-token cost, an attention term "
         f"{min(out['attention_term_vs_c2_full_fit']['cold_only']):.2f}× C2's, and a base that "
         "C2's fit does not have. Agentic traffic is the other case — a long cached prefix and "
         "few new tokens — and there the splits disagree: the cold-only fit charges "
         f"{(extra['K1_half'] - 1) * 100:.0f}% more prefill time on K1_half's requests "
         f"({(extra['K64_quarter'] - 1) * 100:.0f}% on K64_quarter's). K1_half sits at the knee "
         "of the load curve, where that much more service time becomes several times more "
         "queueing."),
        "",
        ("**What changed because of this:** since 2026-10-01 `kvsim calibrate` also times new "
         "tokens after a cached prefix (`--prefill-cached`, `--prefill-new`)."),
        "",
    ]
    (OUT / "precheck.md").write_text("\n".join(lines) + "\n")
    print((OUT / "precheck.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
