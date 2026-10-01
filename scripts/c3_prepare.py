#!/usr/bin/env python3
"""C3's registration: the wheel built from a clean tree, the predictions computed, the notebook
written.

    python scripts/c3_prepare.py
        # configs/c3_calibrate.yaml -> artifacts/results/c3_prereg/predictions.json
        #                           -> notebooks/c3_calibrate_t4.ipynb

What is registered here, before the run:
  * the wheel the notebook carries — built from committed code only, which is why this refuses to
    run on a dirty tree — and the command's default timing cells, read from that code;
  * the cache check `kvsim calibrate` will send, per seed, and the simulator's prediction for it
    (C3a), as in C1;
  * part 2's traffic (fingerprints, checked against the notebook's port of the generator) and what
    C2's profile predicts for it at the scheduled times — hit rate, p50 and p95 TTFT (C3c).
What the command's own profile predicts (C3b) cannot be registered: it does not exist until the run.
The procedure that will produce it — scripts/c3_analyze.py — is committed with this script instead.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _c3_notebook import build_notebook
from _spike_common import results_header
from c2_analyze import nearest_rank
from phase6_prepare import _check_port, sha256

from kv_routing_sim.calib import calibrate as cal
from kv_routing_sim.cli import _ints, build_parser
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.hashing import chain_hashes

C2_PREREG = REPO / "artifacts" / "results" / "c2_prereg" / "predictions.json"
# Everything that ends up in the wheel or the notebook, or decides the verdict afterwards.
CODE = ("src", "pyproject.toml", "configs/c3_calibrate.yaml",
        "configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json", "scripts/_c1_notebook.py",
        "scripts/_c2_notebook.py", "scripts/_c3_notebook.py", "scripts/_phase3_notebook.py",
        "scripts/_phase6_notebook.py", "scripts/c2_analyze.py", "scripts/c3_analyze.py",
        "scripts/c3_prepare.py", "scripts/phase6_prepare.py")


def command_defaults() -> dict[str, Any]:
    """The timing cells `kvsim calibrate` measures when no option is passed — what C3 runs. Read
    from the command line's own parser: that, not the library's constants, is what the notebook
    invokes."""
    ns = build_parser().parse_args(["calibrate", "--out", "x.json"])
    cells = {k: list(_ints(getattr(ns, k), k)) for k in (
        "prefill_lengths", "prefill_cached", "prefill_new", "decode_batches", "decode_contexts")}
    return {**cells, "reps": ns.reps, "decode_tokens": ns.decode_tokens,
            "decode_skip_tokens": cal.DECODE_SKIP_TOKENS}


def check_design(cfg: dict[str, Any], traces: dict[str, list],
                 fingerprints: dict[str, str]) -> None:
    e = cfg["engine"]
    for name, pairs in traces.items():
        longest = max(len(ids) for _, ids in pairs) + cfg["holdout"]["workloads"][name][
            "output_tokens"]
        if longest > e["max_model_len"]:
            raise SystemExit(f"error: {name}'s longest request needs {longest} tokens > "
                             f"max_model_len {e['max_model_len']}")
    # "C2's two, unchanged" must be literally true: the same traces, by fingerprint.
    c2 = json.loads(C2_PREREG.read_text())["trace_fingerprints_sha256"]
    for name, fp in c2.items():
        if name in fingerprints and fingerprints[name] != fp:
            raise SystemExit(f"error: {name} is not the trace C2 measured (fingerprints differ)")
    if len(cfg["calibrate"]["seeds"]) != 2:
        raise SystemExit("error: the design is one tested run of the command and one reported")


def predict_check(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """The simulator's prediction for the cache check the command sends with this seed. The check
    fits in a fraction of any T4's cache, so nothing is evicted and timing does not enter it."""
    c = cfg["calibrate"]
    work = cal.check_workload(random.Random(f"check-{seed}"), c["check_sessions"],
                              c["check_turns"], tuple(c["token_range"]))
    sim = cal.predict_check(work, load_profile(REPO / cfg["profile"]))
    return {"requests": len(work), "queries": sim.cache_queries, "hits": sim.cache_hits,
            "hit_rate": round(sim.hit_rate, 6),
            "cached_tokens": [sim.per_request[f"c{i}"]["cached_tokens"]
                              for i in range(len(work))]}


def predict_holdout(pairs: list, profile) -> dict[str, Any]:
    """C2's profile, one replica, every request at its SCHEDULED time. Percentiles by the
    notebook's nearest-rank rule, over all requests."""
    r = run([q for q, _ in pairs], policy="round-robin", n_replicas=1, seed=0, profile=profile,
            cache_model="vllm", record_per_request=True)
    ttfts = sorted(v["ttft_ms"] for v in r.per_request.values())
    return {"hit_rate": round(r.hit_rate, 6), "prefix_cache_queries": r.cache_queries,
            "prefix_cache_hits": r.cache_hits, "preemptions": r.preemptions,
            "ttft_p50_ms": round(nearest_rank(ttfts, 0.50), 3),
            "ttft_p95_ms": round(nearest_rank(ttfts, 0.95), 3)}


def build_wheel() -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps", "-q",
                        "-w", tmp], check=True)
        whl = next(pathlib.Path(tmp).glob("*.whl"))
        return whl.read_bytes(), whl.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=pathlib.Path, default=REPO / "configs" / "c3_calibrate.yaml")
    ap.add_argument("--prereg", type=pathlib.Path,
                    default=REPO / "artifacts" / "results" / "c3_prereg" / "predictions.json")
    ap.add_argument("--notebook", type=pathlib.Path,
                    default=REPO / "notebooks" / "c3_calibrate_t4.ipynb")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from uncommitted code (the notebook then says so); never for the "
                         "registered run")
    ap.add_argument("--no-wheel", action="store_true",
                    help="embed no wheel — for the dry-run, which uses the installed package")
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"error: commit first — the wheel must be the committed code:\n{dirty}")

    cfg = yaml.safe_load(args.config.read_text())
    profile = load_profile(REPO / cfg["profile"])
    if profile.kv_cache_blocks != cfg["engine"]["num_gpu_blocks"]:
        raise SystemExit("error: the profile's capacity and the engine's must match")
    header = results_header("scripts/c3_prepare.py", seed=0, hardware_path="cpu-only",
                            extra={"config": cfg})

    workloads = {name: dict(spec) for name, spec in cfg["holdout"]["workloads"].items()}
    traces, fingerprints = {}, {}
    for name, spec in workloads.items():
        pairs = list(AgenticWorkload(**spec).iter_with_tokens())  # arrival-sorted
        traces[name] = pairs
        fingerprints[name] = sha256([[r.request_id, r.arrival_ts, r.session_id, ids]
                                     for r, ids in pairs])
        assert all(chain_hashes(ids) == r.block_hashes for r, ids in pairs)
    check_design(cfg, traces, fingerprints)
    _check_port(workloads, fingerprints, traces)

    pred: dict[str, Any] = {"runs": {}, "check": {}}
    for name, pairs in traces.items():
        p = predict_holdout(pairs, profile)
        pred["runs"][name] = {"C2 profile": p}
        print(f"  {name:12s} C2's profile: hit {p['hit_rate']:.4f} · p50 {p['ttft_p50_ms']:.0f} ms"
              f" · p95 {p['ttft_p95_ms']:.0f} ms · preemptions {p['preemptions']}")
    for seed in cfg["calibrate"]["seeds"]:
        c = predict_check(cfg, seed)
        pred["check"][str(seed)] = c
        print(f"  check, seed {seed}: hit rate {c['hit_rate']} ({c['hits']:,} of {c['queries']:,} "
              f"tokens, {c['requests']} requests)")
    pred_sha = sha256(pred)

    if args.no_wheel:
        data, name = b"", "none.whl"
    else:
        data, name = build_wheel()
    wheel_sha = hashlib.sha256(data).hexdigest()
    git = header["git_sha"] + ("-dirty" if dirty else "")

    header.update({"profile": profile.provenance(), "command_defaults": command_defaults(),
                   "wheel": {"name": name, "sha256": wheel_sha, "bytes": len(data),
                             "built_from": git},
                   "workloads": workloads, "trace_fingerprints_sha256": fingerprints,
                   "predictions": pred, "predictions_sha256": pred_sha})
    args.prereg.parent.mkdir(parents=True, exist_ok=True)
    args.prereg.write_text(json.dumps(header, indent=1, default=str) + "\n")
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(
        build_notebook(cfg, workloads, fingerprints, pred, pred_sha,
                       wheel_b64=base64.b64encode(data).decode(), wheel_sha=wheel_sha,
                       wheel_name=name, git_sha=git), encoding="utf-8")
    print(f"wheel {name} · {len(data) / 1024:.0f} KiB · sha256 {wheel_sha[:16]}… · from {git}")
    print(f"wrote {args.prereg}\nwrote {args.notebook} "
          f"({args.notebook.stat().st_size / 1024:.0f} KiB) · predictions sha256 {pred_sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
