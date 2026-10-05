#!/usr/bin/env python3
"""C4's registration: the wheel built from a clean tree, the predictions computed, the notebook
written.

    python scripts/c4_prepare.py
        # configs/c4_per_token.yaml -> artifacts/results/c4_prereg/predictions.json
        #                           -> notebooks/c4_per_token_t4.ipynb

What is registered here, before the run — every number from profile A, the shipped C2 fit:
  * the wheel the notebook carries (built from committed code only, so this refuses a dirty tree),
    and the command's default timing cells, read from that code;
  * the cache check `kvsim calibrate` will send, per seed, and the simulator's prediction for it;
  * part 3's traffic (fingerprints, checked against the notebook's port of the generator) and what
    profile A predicts for it at the scheduled times: first-token latency, time per output token
    (p50, p95, the share of requests within the limit) and the mean inter-token latency;
  * part 2's decode steps and part 2b's stalls, as profile A's model gives them.
What the fresh calibrations will say (rules E, F) cannot be registered: they do not exist until the
run. The procedure that will produce those numbers — scripts/c4_grids.py and scripts/c4_analyze.py
— is committed with this script instead.

    python scripts/c4_prepare.py --amend "WHY, as one sentence."

rebuilds the notebook from the current builder WITHOUT touching anything registered (see
scripts/c3_prepare.py, whose rules for an amendment these are).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _c4_notebook import build_notebook
from _spike_common import results_header
from c2_analyze import nearest_rank
from c3_prepare import build_wheel, command_defaults, predict_check
from phase6_prepare import _check_port, sha256

from kv_routing_sim.sim.cost_model import CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.hashing import chain_hashes

# Everything that ends up in the wheel or the notebook, or decides a verdict afterwards.
CODE = ("src", "pyproject.toml", "configs/c4_per_token.yaml", "configs/h2b_c2.yaml",
        "configs/phase5_crossover.yaml", "configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json",
        "scripts/_c1_notebook.py", "scripts/_c2_notebook.py", "scripts/_c3_notebook.py",
        "scripts/_c4_notebook.py", "scripts/_phase3_notebook.py", "scripts/_phase6_notebook.py",
        "scripts/c2_analyze.py", "scripts/c3_prepare.py", "scripts/c4_analyze.py",
        "scripts/c4_grids.py", "scripts/c4_prepare.py", "scripts/phase6_prepare.py")
MAX_BLOCK_SHARE = 0.85  # of the usable blocks: above this a cell could preempt


def shared_step(profile: CostProfile, cfg: dict[str, Any], b: int,
                length: int) -> dict[str, float]:
    """The model's decode step for b sequences of `length` context tokens — the quantity part 2
    times: S = b × (length + the middle of the tokens that are timed)."""
    s = cfg["decode_shared"]
    summed = b * (length + (s["skip_tokens"] + s["output_tokens"]) / 2.0)
    return {"summed_context": summed, "step_s": profile.decode_s(b, round(summed))}


def stall_prediction(profile: CostProfile, cfg: dict[str, Any], b: int,
                     new: int) -> dict[str, float]:
    """What the model charges the b decoding sequences when one more request arrives: the step
    that prefills it costs its prefill plus a decode step of b + 1 — against the usual step of b.
    Contexts are taken at the middle of the output, as for the decode cells."""
    s, t = cfg["decode_shared"], cfg["stall"]
    ctx = s["prefix_tokens"] + s["suffix_tokens"] + t["output_tokens"] / 2.0
    usual = profile.decode_s(b, round(b * ctx))
    late = s["prefix_tokens"] + new
    at = profile.prefill_s(new, s["prefix_tokens"]) + profile.decode_s(b + 1, round(b * ctx + late))
    return {"usual_gap_s": usual, "extra_gap_s": at - usual,
            "prefill_s": profile.prefill_s(new, s["prefix_tokens"])}


def predict_holdout(pairs: list, profile: CostProfile, limit_ms: float) -> dict[str, Any]:
    """Profile A, one replica, every request at its SCHEDULED time. Percentiles by the notebook's
    nearest-rank rule, over all requests."""
    r = run([q for q, _ in pairs], policy="round-robin", n_replicas=1, seed=0, profile=profile,
            cache_model="vllm", record_per_request=True)
    ttfts = sorted(v["ttft_ms"] for v in r.per_request.values())
    tpots = sorted(v["tpot_ms"] for v in r.per_request.values())
    p95 = nearest_rank(tpots, 0.95)
    return {"hit_rate": round(r.hit_rate, 6), "prefix_cache_queries": r.cache_queries,
            "prefix_cache_hits": r.cache_hits, "preemptions": r.preemptions,
            "ttft_p50_ms": round(nearest_rank(ttfts, 0.50), 3),
            "ttft_p95_ms": round(nearest_rank(ttfts, 0.95), 3),
            "tpot_p50_ms": round(nearest_rank(tpots, 0.50), 3), "tpot_p95_ms": round(p95, 3),
            "share_within_limit": round(sum(1 for t in tpots if t <= limit_ms) / len(tpots), 6),
            "itl_mean_ms": round(r.engine_itl_mean_ms(), 3),
            "meets_per_token_limit": p95 <= limit_ms,
            "margin": round(abs(p95 - limit_ms) / limit_ms, 6)}


def check_design(cfg: dict[str, Any], traces: dict[str, list]) -> None:
    """Every cell and every request must fit the engine, with room: nothing may preempt."""
    e, s, t = cfg["engine"], cfg["decode_shared"], cfg["stall"]
    bs, usable = e["block_size"], e["num_gpu_blocks"] - 1

    def blocks(tokens: int) -> int:
        return -(-tokens // bs)

    shared = s["prefix_tokens"] // bs
    if s["prefix_tokens"] % bs:
        raise SystemExit("error: the shared prefix must be whole blocks")
    for out, batches in ((s["output_tokens"], s["batches"]),
                         (t["output_tokens"], [b + 1 for b, _, _ in t["cells"]])):
        length = s["prefix_tokens"] + s["suffix_tokens"] + out
        if length + 1 > e["max_model_len"]:
            raise SystemExit(f"error: {length} tokens exceed max_model_len")
        for b in batches:
            need = shared + b * (blocks(length) - shared)
            if need > MAX_BLOCK_SHARE * usable:
                raise SystemExit(f"error: {b} sequences sharing the prefix need {need} blocks > "
                                 f"{MAX_BLOCK_SHARE:.0%} of {usable}: the cell could preempt")
    for length, b in s["controls"]:
        if b * blocks(length + s["output_tokens"]) > MAX_BLOCK_SHARE * usable:
            raise SystemExit(f"error: control L={length}, b={b} does not fit without preempting")
        if b not in s["batches"] or length != s["prefix_tokens"] + s["suffix_tokens"]:
            raise SystemExit("error: every control must have a shared cell of the same L and b")
    for _b, new, _delay in t["cells"]:
        if s["prefix_tokens"] + new + 1 > e["max_model_len"]:
            raise SystemExit("error: a late request exceeds max_model_len")
    if not s["skip_tokens"] < s["output_tokens"] - 1:
        raise SystemExit("error: skip_tokens leaves no gaps to time")
    for name, pairs in traces.items():
        longest = max(len(ids) for _, ids in pairs) + cfg["holdout"]["workloads"][name][
            "output_tokens"]
        if longest > e["max_model_len"]:
            raise SystemExit(f"error: {name}'s longest request needs {longest} tokens")
    if len(cfg["calibrate"]["seeds"]) < 2:
        raise SystemExit("error: rule E needs at least two runs of the command")


def predictions(cfg: dict[str, Any], traces: dict[str, list], profile: CostProfile) -> dict:
    s, limit = cfg["decode_shared"], cfg["slo"]["tpot_p95_ms"]
    length = s["prefix_tokens"] + s["suffix_tokens"]
    pred: dict[str, Any] = {"runs": {}, "check": {}}
    for name, pairs in traces.items():
        pred["runs"][name] = {"profile A": predict_holdout(pairs, profile, limit)}
    for seed in cfg["calibrate"]["seeds"]:
        pred["check"][str(seed)] = predict_check(cfg, seed)
    pred["decode_shared"] = {str(b): shared_step(profile, cfg, b, length) for b in s["batches"]}
    pred["decode_control"] = {f"{ln}x{b}": shared_step(profile, cfg, b, ln)
                              for ln, b in s["controls"]}
    pred["stall"] = {f"{b}+{new}": stall_prediction(profile, cfg, b, new)
                     for b, new, _ in cfg["stall"]["cells"]}
    return pred


def fitted_max_summed_context() -> float:
    """The largest summed context any of C2's decode cells measured — what profile A was fitted
    on. Rule D1b is about the cells beyond it."""
    c2 = json.loads((REPO / "artifacts/results/c2_kaggle_t4/c2.json").read_text())
    d = c2["design"]["decode"]
    mid = (d["skip_tokens"] + d["output_tokens"]) / 2.0
    return max(m["b"] * (m["L"] + mid) for m in c2["decode"] if m["valid"])


def rules_table(cfg: dict[str, Any], pred: dict[str, Any]) -> str:
    r, p = cfg["rules"], cfg["predictions"]
    limit = cfg["slo"]["tpot_p95_ms"]
    tested = [n for n, v in pred["runs"].items()
              if v["profile A"]["margin"] >= r["t3b_min_margin"]]
    rows = [
        ("C4a", ("`kvsim calibrate` exits 0, its cache check is exact, it reads the capacity, and "
                 f"its profile carries the fit of each of its {r['c4a_repeats']} repeats"),
         p["c4a"]),
        ("D1a", ("a batch that shares a prefix costs the same per step as one with unique prompts "
                 f"of the same length (within {r['d1a_max_factor']:g}×)"), p["d1a"]),
        ("D1b", (f"the model's decode step is within {r['d1b_max_factor']:g}× of the measured one "
                 "for every shared batch past the largest ever fitted"), p["d1b"]),
        ("D2", ("when one more request arrives, the extra gap every decoding sequence sees is "
                f"within {r['d2_max_factor']:g}× of what the model charges"), p["d2"]),
        ("T3a", (f"the model's p95 time per output token is within {r['t3a_max_factor']:g}× of "
                 "the measured one, on every workload"), p["t3a"]),
        ("T3b", (f"the model's verdict on the {limit:g} ms limit equals the engine's own, on "
                 f"{' and '.join(f'`{n}`' for n in tested)}"), p["t3b"]),
        ("E", ("the spread `kvsim replay` prints from one run's repeats covers another run's "
               f"count in ≥ {r['e_min_coverage']:.0%} of cases (decided afterwards, on a CPU)"),
         p["e"]),
        ("F1 · F2", ("the 64-prompt question on the fresh calibrations, under both targets "
                     "(decided afterwards, on a CPU)"), f"{p['f1']} · {p['f2']}"),
    ]
    return "\n".join(["| | Rule | Predicted |", "|---|---|---|",
                      *(f"| **{k}** | {text} | {v} |" for k, text, v in rows)])


def minutes(cfg: dict[str, Any], traces: dict[str, list], pred: dict[str, Any]) -> int:
    """A rough run time for the notebook's first page: start-up, the command, cells, traffic."""
    traffic = sum(max(q.arrival_ts for q, _ in pairs) + 40 for pairs in traces.values())
    s, t = cfg["decode_shared"], cfg["stall"]
    cells = s["reps"] * sum(v["step_s"] * s["output_tokens"] + 8
                            for v in pred["decode_shared"].values())
    stalls = t["reps"] * sum(v["usual_gap_s"] * t["output_tokens"] + 10
                             for v in pred["stall"].values())
    return round((12 * 60 + len(cfg["calibrate"]["seeds"]) * 360 + cells + stalls + traffic) / 60)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=REPO / "configs" / "c4_per_token.yaml")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c4_prereg" / "predictions.json")
    ap.add_argument("--notebook", type=pathlib.Path,
                    default=REPO / "notebooks" / "c4_per_token_t4.ipynb")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from uncommitted code (the notebook then says so); never for the "
                         "registered run")
    ap.add_argument("--no-wheel", action="store_true",
                    help="embed no wheel — for the dry-run, which uses the installed package")
    ap.add_argument("--amend", metavar="WHY",
                    help="rebuild the notebook WITHOUT touching anything registered, and record "
                         "WHY (one sentence) in the registration file")
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"error: commit first — the wheel must be the committed code:\n{dirty}")

    cfg = yaml.safe_load(args.config.read_text())
    profile = load_profile(REPO / cfg["profile"])
    if profile.kv_cache_blocks != cfg["engine"]["num_gpu_blocks"]:
        raise SystemExit("error: the profile's capacity and the engine's must match")
    header = results_header("scripts/c4_prepare.py", seed=0, hardware_path="cpu-only",
                            extra={"config": cfg})

    workloads = {name: dict(spec) for name, spec in cfg["holdout"]["workloads"].items()}
    traces, fingerprints = {}, {}
    for name, spec in workloads.items():
        pairs = list(AgenticWorkload(**spec).iter_with_tokens())  # arrival-sorted
        traces[name] = pairs
        fingerprints[name] = sha256([[r.request_id, r.arrival_ts, r.session_id, ids]
                                     for r, ids in pairs])
        assert all(chain_hashes(ids) == r.block_hashes for r, ids in pairs)
    check_design(cfg, traces)
    _check_port(workloads, fingerprints, traces)

    pred = predictions(cfg, traces, profile)
    for name, v in pred["runs"].items():
        a = v["profile A"]
        print(f"  {name:11s} profile A: hit {a['hit_rate']:.4f} · TTFT p95 {a['ttft_p95_ms']:.0f} "
              f"ms · ms per token p50 {a['tpot_p50_ms']:.1f}, p95 {a['tpot_p95_ms']:.1f} "
              f"({a['share_within_limit']:.1%} within the limit) · mean gap "
              f"{a['itl_mean_ms']:.1f} ms · preemptions {a['preemptions']}")
    top = fitted_max_summed_context()
    for b, v in pred["decode_shared"].items():
        print(f"  shared b {b:>3}: summed context {v['summed_context']:>9,.0f} "
              f"({v['summed_context'] / top:.1f}× the largest fitted) · step "
              f"{v['step_s'] * 1e3:.1f} ms")
    for k, v in pred["stall"].items():
        print(f"  stall {k}: usual gap {v['usual_gap_s'] * 1e3:.1f} ms, extra "
              f"{v['extra_gap_s'] * 1e3:.1f} ms")
    pred_sha = sha256(pred)
    kw = {"rules_table": rules_table(cfg, pred), "minutes": minutes(cfg, traces, pred),
          "s_max_x": max(v["summed_context"] for v in pred["decode_shared"].values()) / top}

    if args.amend:
        prereg = json.loads(args.prereg.read_text())
        old = args.notebook.read_text(encoding="utf-8")
        same = {"the config": cfg == prereg["config"],
                "the held-out workloads": workloads == prereg["workloads"],
                "the traces' fingerprints": fingerprints == prereg["trace_fingerprints_sha256"],
                "the predictions": pred == prereg["predictions"]
                and pred_sha == prereg["predictions_sha256"],
                "the command's default cells": command_defaults() == prereg["command_defaults"]}
        if not all(same.values()):
            raise SystemExit("error: not an amendment — these no longer come out as registered: "
                             f"{', '.join(k for k, ok in same.items() if not ok)}.")
        src = "".join("".join(c["source"]) for c in json.loads(old)["cells"]
                      if c["cell_type"] == "code")
        wheel = base64.b64decode(re.search(r'b64decode\("""([A-Za-z0-9+/=]*)"""\)', src).group(1))
        if hashlib.sha256(wheel).hexdigest() != prereg["wheel"]["sha256"]:
            raise SystemExit("error: the notebook no longer carries the registered wheel")
        moved = subprocess.run(["git", "diff", "--quiet", prereg["wheel"]["built_from"], "HEAD",
                                "--", "src", "pyproject.toml"], cwd=REPO, check=False).returncode
        if moved and not args.allow_dirty:
            raise SystemExit("error: src/ or pyproject.toml changed since the registered wheel "
                             "was built. That needs a new registration, not an amendment.")
        amendments = [*prereg.get("amendments", []),
                      {"date_utc": header["date_utc"],
                       "git_sha": header["git_sha"] + ("-dirty" if dirty else ""),
                       "why": args.amend, "notebook_sha256_before": _sha(old)}]
        new = build_notebook(cfg, workloads, fingerprints, pred, pred_sha,
                             wheel_b64=base64.b64encode(wheel).decode(),
                             wheel_sha=prereg["wheel"]["sha256"],
                             wheel_name=prereg["wheel"]["name"],
                             git_sha=prereg["wheel"]["built_from"], amendments=amendments, **kw)
        amendments[-1]["notebook_sha256_after"] = _sha(new)
        prereg["amendments"] = amendments
        args.notebook.write_text(new, encoding="utf-8")
        args.prereg.write_text(json.dumps(prereg, indent=1, default=str) + "\n")
        print(f"amended at {amendments[-1]['git_sha']}: {args.amend}")
        return 0

    data, name = (b"", "none.whl") if args.no_wheel else build_wheel()
    wheel_sha = hashlib.sha256(data).hexdigest()
    git = header["git_sha"] + ("-dirty" if dirty else "")
    header.update({"profile": profile.provenance(), "command_defaults": command_defaults(),
                   "wheel": {"name": name, "sha256": wheel_sha, "bytes": len(data),
                             "built_from": git},
                   "fitted_max_summed_context": top,
                   "workloads": workloads, "trace_fingerprints_sha256": fingerprints,
                   "predictions": pred, "predictions_sha256": pred_sha})
    args.prereg.parent.mkdir(parents=True, exist_ok=True)
    args.prereg.write_text(json.dumps(header, indent=1, default=str) + "\n")
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(
        build_notebook(cfg, workloads, fingerprints, pred, pred_sha,
                       wheel_b64=base64.b64encode(data).decode(), wheel_sha=wheel_sha,
                       wheel_name=name, git_sha=git, **kw), encoding="utf-8")
    print(f"wheel {name} · {len(data) / 1024:.0f} KiB · sha256 {wheel_sha[:16]}… · from {git}")
    print(f"wrote {args.prereg}\nwrote {args.notebook} "
          f"({args.notebook.stat().st_size / 1024:.0f} KiB) · predictions sha256 {pred_sha[:16]}… "
          f"· about {kw['minutes']} minutes on the GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
