#!/usr/bin/env python3
"""C4, post hoc: a reported item the registration listed and the registered analysis left out.

    python scripts/c4_posthoc_fresh_cells.py
        # artifacts/results/c4_kaggle_t4/ -> posthoc_fresh_cells.{json,md}

configs/c4_per_token.yaml lists, under "Reported, not tested": "every cell's signed error against
profile A, and against each fresh fit". scripts/c4_analyze.py — committed before the run and not
changed since — compares the micro-benchmark cells with profile A only. This script adds the
other half: each run of `kvsim calibrate`'s own pooled fit, against the same measured cells.

⚠️ WRITTEN AFTER THE RESULTS WERE SEEN. It decides nothing and changes no verdict. The arithmetic
is the registration's own — `shared_step` and `stall_prediction` are imported from
scripts/c4_prepare.py, which made profile A's registered numbers with them — and the script stops
if it cannot reproduce those numbers from profile A first.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from c4_prepare import shared_step, stall_prediction  # the registration's own arithmetic

from kv_routing_sim.sim.cost_model import CostProfile, load_profile

RESULTS = REPO / "artifacts" / "results" / "c4_kaggle_t4"
PREREG = REPO / "artifacts" / "results" / "c4_prereg" / "predictions.json"


def compute(analysis: dict[str, Any], prereg: dict[str, Any],
            fits: dict[str, CostProfile]) -> dict[str, Any]:
    cfg = prereg["config"]
    s = cfg["decode_shared"]
    length = s["prefix_tokens"] + s["suffix_tokens"]
    decode, stall = [], []
    for c in analysis["decode"]["cells"]:
        row = {"b": c["b"], "summed_context": c["summed_context"],
               "measured_step_s": c["measured_step_s"], "fits": {}}
        for name, profile in fits.items():
            step = shared_step(profile, cfg, c["b"], length)["step_s"]
            row["fits"][name] = {"step_s": step, "over_measured": step / c["measured_step_s"]}
        decode.append(row)
    for c in analysis["d2"]["cells"]:
        row = {"b": c["b"], "new_tokens": c["new_tokens"],
               "measured_extra_s": c["measured_extra_s"], "fits": {}}
        for name, profile in fits.items():
            extra = stall_prediction(profile, cfg, c["b"], c["new_tokens"])["extra_gap_s"]
            row["fits"][name] = {"extra_gap_s": extra,
                                 "over_measured": extra / c["measured_extra_s"]}
        stall.append(row)
    # The guard: profile A through this script must give the registered numbers.
    for row, c in zip(decode, analysis["decode"]["cells"], strict=True):
        if abs(row["fits"]["A"]["step_s"] - c["predicted_step_s"]) > 1e-12:
            raise SystemExit(f"error: b={c['b']}: profile A does not reproduce its registration")
    for row, c in zip(stall, analysis["d2"]["cells"], strict=True):
        if abs(row["fits"]["A"]["extra_gap_s"] - c["predicted_extra_s"]) > 1e-12:
            raise SystemExit(f"error: stall b={c['b']}: profile A does not reproduce its "
                             "registration")
    return {"script": "scripts/c4_posthoc_fresh_cells.py", "post_hoc": True,
            "fits": list(fits), "decode_shared": decode, "stall": stall,
            "decode_per_1k_context_ms": {n: p.decode_per_ctx_token_s * 1e6
                                         for n, p in fits.items()}}


def _pct(x: float) -> str:
    pct = round((x - 1) * 100)
    return f"{pct:+d}%" if pct else "0%"


def markdown(out: dict[str, Any]) -> str:
    names = out["fits"]
    lines = [
        "# C4, post hoc — each fresh calibration against the micro-benchmark cells",
        "",
        ("⚠️ **Written after the results were seen.** The registration listed this as a reported "
         "item (*\"every cell's signed error against profile A, and against each fresh fit\"*) "
         "and the registered analysis computed only profile A's half. Same files, the "
         "registration's own arithmetic (`scripts/c4_prepare.py`), no verdict."),
        "",
        "Model against measured: above zero the model is too slow.",
        "",
        "## Decode steps of batches that share a cached prefix",
        "",
        "| b | summed context | measured | " + " | ".join(names) + " |",
        "|---:|---:|---:|" + "---:|" * len(names),
    ]
    for r in out["decode_shared"]:
        lines.append(f"| {r['b']} | {r['summed_context']:,.0f} | "
                     f"{r['measured_step_s'] * 1e3:.1f} ms | "
                     + " | ".join(_pct(r["fits"][n]["over_measured"]) for n in names) + " |")
    lines += ["", ("The fitted cost of context, in ms per 1,000 tokens of the batch's summed "
                   "context: " + " · ".join(
                       f"{n} {out['decode_per_1k_context_ms'][n]:.2f}" for n in names) + "."),
              "",
              "## What one arriving request costs the sequences already decoding",
              "",
              "| b | measured extra gap | " + " | ".join(names) + " |",
              "|---:|---:|" + "---:|" * len(names)]
    for r in out["stall"]:
        lines.append(f"| {r['b']} | {r['measured_extra_s'] * 1e3:.1f} ms | "
                     + " | ".join(_pct(r["fits"][n]["over_measured"]) for n in names) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    prereg = json.loads(PREREG.read_text())
    cfg = prereg["config"]
    fits = {"A": load_profile(REPO / cfg["profile"])}
    for seed in cfg["calibrate"]["seeds"]:
        fits[f"s{seed}"] = load_profile(RESULTS / f"calibrated_profile_seed{seed}.json")
    out = compute(analysis, prereg, fits)
    (RESULTS / "posthoc_fresh_cells.json").write_text(json.dumps(out, indent=1) + "\n")
    (RESULTS / "posthoc_fresh_cells.md").write_text(markdown(out))
    print((RESULTS / "posthoc_fresh_cells.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
