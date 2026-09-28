#!/usr/bin/env python3
"""C2's registration: the design checked, part D's traffic fingerprinted, the notebook built.

    python scripts/c2_prepare.py
        # configs/c2_timing.yaml -> artifacts/results/c2_prereg/predictions.json
        #                        -> notebooks/c2_timing_t4.ipynb

What is registered here: the design, part D's traffic (fingerprints, checked against the notebook's
port of the generator, as Phase 6's was), and what the CONTEXT-FREE profile predicts for part D —
reported beside the context-aware model, never tested. The context-aware model cannot predict
anything yet: it is fitted after the run, on parts A and B, by scripts/c2_analyze.py — and that
script, committed with this one, is the whole of what happens to the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _c2_notebook import build_notebook
from _spike_common import results_header
from phase6_prepare import _check_port, sha256

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.hashing import chain_hashes

MAX_BLOCKS = 1300  # of 1,536 usable: a decode cell above this could preempt, and time recomputation


def check_design(cfg: dict[str, Any]) -> None:
    """Every cell must fit the engine: no decode cell may need preemption, no prompt the max."""
    e, d = cfg["engine"], cfg["decode"]
    per = lambda L: math.ceil((L + d["output_tokens"]) / e["block_size"])
    for L, b in [*d["grid"], *cfg["cpu"]["cells"]]:
        if b * per(L) > MAX_BLOCKS:
            raise SystemExit(f"error: decode cell L={L}, b={b} needs {b * per(L)} blocks > "
                             f"{MAX_BLOCKS}: it could preempt")
        if L + d["output_tokens"] > e["max_model_len"]:
            raise SystemExit(f"error: L={L} + {d['output_tokens']} exceeds max_model_len")
    for c, n in cfg["prefill"]["grid"]:
        if c % e["block_size"] or c + n + 1 > e["max_model_len"]:
            raise SystemExit(f"error: prefill cell c={c}, n={n}: c must be whole blocks and "
                             "c + n + 1 must fit max_model_len")
    if not d["skip_tokens"] < d["output_tokens"] - 1:
        raise SystemExit("error: skip_tokens leaves no gaps to time")
    for name in ("L128", "L1536"):
        L = int(name[1:])
        if [L, 8] not in d["grid"]:
            raise SystemExit(f"error: C2a's rule needs the cell L={L}, b=8 in the decode grid")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=REPO / "configs" / "c2_timing.yaml")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c2_prereg" / "predictions.json")
    ap.add_argument("--notebook", type=pathlib.Path,
                    default=REPO / "notebooks" / "c2_timing_t4.ipynb")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    profile = load_profile(REPO / cfg["profile"])
    if profile.kv_cache_blocks != cfg["engine"]["num_gpu_blocks"]:
        raise SystemExit("error: the profile's capacity and the engine's must match")
    check_design(cfg)

    workloads = {name: dict(spec) for name, spec in cfg["holdout"]["workloads"].items()}
    traces, fingerprints = {}, {}
    for name, spec in workloads.items():
        pairs = list(AgenticWorkload(**spec).iter_with_tokens())
        traces[name] = pairs
        fingerprints[name] = sha256([[r.request_id, r.arrival_ts, r.session_id, ids]
                                     for r, ids in pairs])
        assert all(chain_hashes(ids) == r.block_hashes for r, ids in pairs)
    _check_port(workloads, fingerprints, traces)

    # The context-free profile, one replica, at the SCHEDULED times: reported, not tested.
    pred: dict[str, Any] = {"runs": {}}
    for name, pairs in traces.items():
        r = run([q for q, _ in pairs], policy="round-robin", n_replicas=1, seed=0,
                profile=profile, cache_model="vllm")
        pct = r.ttft_percentiles()
        pred["runs"][name] = {"context-free profile": {
            "hit_rate": round(r.hit_rate, 6), "prefix_cache_queries": r.cache_queries,
            "prefix_cache_hits": r.cache_hits, "preemptions": r.preemptions,
            "ttft_p50_ms": pct["p50_ms"], "ttft_p95_ms": pct["p95_ms"],
        }}
        print(f"  {name:12s} context-free profile: hit {r.hit_rate:.4f} · p50 "
              f"{pct['p50_ms']:.0f} ms · p95 {pct['p95_ms']:.0f} ms · preemptions {r.preemptions}")
    pred_sha = sha256(pred)

    payload = results_header("scripts/c2_prepare.py", seed=0, hardware_path="cpu-only",
                             extra={"config": cfg})
    payload.update({"profile": profile.provenance(), "workloads": workloads,
                    "trace_fingerprints_sha256": fingerprints, "predictions": pred,
                    "predictions_sha256": pred_sha})
    args.prereg.parent.mkdir(parents=True, exist_ok=True)
    args.prereg.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(build_notebook(cfg, workloads, fingerprints, pred, pred_sha))
    print(f"wrote {args.prereg}\nwrote {args.notebook}  (predictions sha256 {pred_sha[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
