#!/usr/bin/env python3
"""POST HOC — why T1 failed where it did: was the timing model asked about batches it never saw?

    python scripts/t1_posthoc_range.py     # -> artifacts/results/t1_itl/posthoc_range.{json,md}

⚠️ Post hoc: written after T1 ran and did not hold. T1's five replays came out in two groups —
three within 1.07× of the engine's mean inter-token latency, two (the same workload, `K1_half`, in
both sessions) 1.5–1.7× too slow. This asks one question of the same five replays, and chooses
nothing in them: in each decode step, how large was the batch's SUMMED context (the S in
decode(b, S) = d0 + d1·b + d2·S) — and how does that compare with the largest S any calibration
cell measured?

Why it could matter. A calibration cell is b sequences with UNIQUE prompts, so b × context cannot
exceed the KV cache (24,592 tokens on this T4): the largest S measured is ~20,000. Traffic that
shares one system prompt is different — the shared blocks are stored once, so far more sequences
fit, while each still attends over its whole context. S can then be several times the cache, and
the model's decode cost there is a straight-line extrapolation.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import t1_itl_check as t1

from kv_routing_sim.sim.cost_model import CostProfile, load_profile

OUT = REPO / "artifacts" / "results" / "t1_itl"
STEPS: list[tuple[int, int]] = []  # (sequences, summed context) of every decode step of a replay


class Probe(CostProfile):
    """The same profile, noting what each decode step was asked to cost."""

    __slots__ = ()

    def decode_s(self, batch_size: int, context_tokens: int = 0) -> float:
        STEPS.append((batch_size, context_tokens))
        return CostProfile.decode_s(self, batch_size, context_tokens)


def fitted_range() -> dict[str, Any]:
    """The largest summed context among the decode cells each calibration measured."""
    c2 = json.loads((REPO / "artifacts/results/c2_kaggle_t4/c2.json").read_text())
    d = c2["design"]["decode"]
    mid = (d["skip_tokens"] + d["output_tokens"]) / 2.0
    a = max(m["b"] * (m["L"] + mid) for m in c2["decode"] if m["valid"])
    out = {"A": {"max_summed_context": a, "cells": sum(1 for m in c2["decode"] if m["valid"])}}
    for name, seed in (("B", 0), ("C", 1)):
        p = json.loads((REPO / f"artifacts/results/c3_kaggle_t4/calibrated_profile_seed{seed}.json")
                       .read_text())
        cells = [c for c in p["_source"]["decode_cells"] if c["valid"]]
        out[name] = {"max_summed_context": max(c["batch"] * (c["context"] + mid) for c in cells),
                     "cells": len(cells)}
    return out


def main() -> int:
    cfg = yaml.safe_load(t1.CONFIG.read_text())
    profile = load_profile(REPO / cfg["t1"]["profile"])
    probe = Probe(**dataclasses.asdict(profile))
    fitted = fitted_range()
    top = fitted["A"]["max_summed_context"]
    committed = {(r["session"], r["workload"]): r["A"]
                 for r in json.loads((OUT / "t1.json").read_text())["runs"]}
    rows = []
    for src in cfg["t1"]["runs"]:
        data = json.loads((REPO / src["results"]).read_text())
        specs = json.loads((REPO / src["prereg"]).read_text())["workloads"]
        by_name = {m["workload"]: m for m in data["holdout"]}
        for name in src["workloads"]:
            STEPS.clear()
            r = t1.replay(specs[name], by_name[name], probe)
            was = committed[(src["session"], name)]
            if abs(r.engine_itl_mean_ms() - was["predicted_itl_ms"]) > 1e-6:
                raise SystemExit(f"error: {name}: the probe changed the replay")
            tokens = sum(b for b, _ in STEPS)
            s_by_token = sorted(s for b, s in STEPS for _ in range(b))
            rows.append({
                "session": src["session"], "workload": name, "factor": was["factor"],
                "measured_itl_ms": was["measured_itl_ms"],
                "predicted_itl_ms": was["predicted_itl_ms"],
                "decode_steps": len(STEPS),
                "median_batch": statistics.median(b for b, _ in STEPS),
                "max_batch": max(b for b, _ in STEPS),
                # Per TOKEN, not per step: a step of 60 sequences is 60 tokens' worth of latency.
                "summed_context_median": statistics.median(s_by_token),
                "summed_context_p95": s_by_token[round(0.95 * (len(s_by_token) - 1))],
                "summed_context_max": s_by_token[-1],
                "share_of_tokens_beyond_fitted_range": sum(b for b, s in STEPS if s > top) / tokens,
            })
    out = {"script": "scripts/t1_posthoc_range.py", "fitted": fitted,
           "kv_cache_tokens": profile.kv_cache_blocks * profile.block_size, "runs": rows}
    (OUT / "posthoc_range.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        "# POST HOC — T1's failure, and the range the timing model was fitted on",
        "",
        ("⚠️ Post hoc: written after T1 ran and did not hold. Nothing here is a registered test. "
         "Generated by `scripts/t1_posthoc_range.py` from T1's own five replays (profile A)."),
        "",
        ("The decode model is `step = d0 + d1·b + d2·S`, with S the batch's summed context. The "
         f"largest S any calibration cell measured: **{top:,.0f} tokens** (C2, "
         f"{fitted['A']['cells']} cells); `kvsim calibrate`'s cells stop at "
         f"{fitted['B']['max_summed_context']:,.0f}. A cell's prompts are unique, so S cannot "
         f"exceed the KV cache ({out['kv_cache_tokens']:,} tokens). Traffic that shares a system "
         "prompt stores it once, fits far more sequences, and each still attends over all of it."),
        "",
        ("| session · workload | T1 factor | batch: median · max | summed context per token: "
         "median · p95 · max | tokens decoded beyond the fitted range |"),
        "|---|---:|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['session']} · {r['workload']} | {r['factor']:.2f}× | "
            f"{r['median_batch']:.0f} · {r['max_batch']} | {r['summed_context_median']:,.0f} · "
            f"{r['summed_context_p95']:,.0f} · {r['summed_context_max']:,.0f} | "
            f"{r['share_of_tokens_beyond_fitted_range']:.0%} |")
    (OUT / "posthoc_range.md").write_text("\n".join(lines) + "\n")
    print((OUT / "posthoc_range.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
