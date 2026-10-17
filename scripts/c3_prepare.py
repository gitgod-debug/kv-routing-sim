#!/usr/bin/env python3
"""Prepare the C3 run: build the wheel, compute the predictions, and write the notebook.

    python scripts/c3_prepare.py
        # configs/c3_calibrate.yaml -> artifacts/results/c3_prereg/predictions.json
        #                           -> notebooks/c3_calibrate_t4.ipynb

The registration file records:
  * the wheel the notebook embeds, built from committed code only (so this refuses a dirty tree),
    and the command's default timing cells, read from that code;
  * per seed, the cache check `kvsim calibrate` will send and the simulator's prediction for it
    (C3a);
  * part 2's traffic (fingerprints, checked against the notebook's port of the generator) and the
    hit rate, p50 and p95 TTFT that C2's profile predicts for it at the scheduled times (C3c).
The command's own profile (C3b) exists only after the run; scripts/c3_analyze.py evaluates it.

    python scripts/c3_prepare.py --amend "WHY, as one sentence."

rebuilds the notebook from the current builder without changing anything registered, for plumbing
such as a repair of the GPU box's software. It refuses unless the config, the traffic, the
predictions and the command's default cells all come out as registered, re-embeds the registered
wheel byte for byte, and records the reason, the commit and the notebook's hash before and after
in the registration file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import random
import re
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
    """The timing cells `kvsim calibrate` measures when given no options, which is what C3 runs.

    Read from the CLI parser rather than the library's constants, since the notebook runs the CLI.
    """
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
    # Workloads that share a name with C2's must be the same traces, by fingerprint.
    c2 = json.loads(C2_PREREG.read_text())["trace_fingerprints_sha256"]
    for name, fp in c2.items():
        if name in fingerprints and fingerprints[name] != fp:
            raise SystemExit(f"error: {name} is not the trace C2 measured (fingerprints differ)")
    if len(cfg["calibrate"]["seeds"]) != 2:
        raise SystemExit("error: the design is one tested run of the command and one reported")


def predict_check(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """The simulator's prediction for the cache check the command sends with this seed.

    The check uses a fraction of any T4's cache, so nothing is evicted and timing plays no part.
    """
    c = cfg["calibrate"]
    work = cal.check_workload(random.Random(f"check-{seed}"), c["check_sessions"],
                              c["check_turns"], tuple(c["token_range"]))
    sim = cal.predict_check(work, load_profile(REPO / cfg["profile"]))
    return {"requests": len(work), "queries": sim.cache_queries, "hits": sim.cache_hits,
            "hit_rate": round(sim.hit_rate, 6),
            "cached_tokens": [sim.per_request[f"c{i}"]["cached_tokens"]
                              for i in range(len(work))]}


def predict_holdout(pairs: list, profile) -> dict[str, Any]:
    """Predict a workload with C2's profile on one replica, every request at its scheduled time.

    Percentiles use the notebook's nearest-rank rule over all requests.
    """
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def amend(args: argparse.Namespace, cfg: dict[str, Any], header: dict[str, Any], dirty: str,
          workloads: dict[str, Any], fingerprints: dict[str, str], pred: dict[str, Any],
          pred_sha: str) -> int:
    """Rebuild the notebook from the current builder, leaving everything registered as it is."""
    prereg = json.loads(args.prereg.read_text())
    old = args.notebook.read_text(encoding="utf-8")
    same = {
        "the config": cfg == prereg["config"],
        "the held-out workloads": workloads == prereg["workloads"],
        "the traces' fingerprints": fingerprints == prereg["trace_fingerprints_sha256"],
        "the predictions": pred == prereg["predictions"]
        and pred_sha == prereg["predictions_sha256"],
        "the command's default cells": command_defaults() == prereg["command_defaults"],
    }
    if not all(same.values()):
        raise SystemExit("error: not an amendment — these no longer come out as registered: "
                         f"{', '.join(k for k, ok in same.items() if not ok)}. Changing them "
                         "means a new registration (run without --amend, and say why in docs/09).")
    src = "".join("".join(c["source"]) for c in json.loads(old)["cells"]
                  if c["cell_type"] == "code")
    wheel = base64.b64decode(re.search(r'b64decode\("""([A-Za-z0-9+/=]*)"""\)', src).group(1))
    if hashlib.sha256(wheel).hexdigest() != prereg["wheel"]["sha256"]:
        raise SystemExit("error: the notebook no longer carries the registered wheel")
    # The registered wheel must still be the repo's code, or the notebook would run one thing
    # while the tests and the dry-run check another.
    moved = subprocess.run(["git", "diff", "--quiet", prereg["wheel"]["built_from"], "HEAD", "--",
                            "src", "pyproject.toml"], cwd=REPO, check=False).returncode
    if moved and not args.allow_dirty:
        raise SystemExit("error: src/ or pyproject.toml changed since "
                         f"{prereg['wheel']['built_from']}, the commit the registered wheel was "
                         "built from. That needs a new registration, not an amendment.")

    amendments = [*prereg.get("amendments", []),
                  {"date_utc": header["date_utc"],
                   "git_sha": header["git_sha"] + ("-dirty" if dirty else ""),
                   "why": args.amend, "notebook_sha256_before": _sha(old)}]
    new = build_notebook(cfg, workloads, fingerprints, pred, pred_sha,
                         wheel_b64=base64.b64encode(wheel).decode(),
                         wheel_sha=prereg["wheel"]["sha256"], wheel_name=prereg["wheel"]["name"],
                         git_sha=prereg["wheel"]["built_from"], amendments=amendments)
    amendments[-1]["notebook_sha256_after"] = _sha(new)
    prereg["amendments"] = amendments
    args.notebook.write_text(new, encoding="utf-8")
    args.prereg.write_text(json.dumps(prereg, indent=1, default=str) + "\n")
    print(f"amended at {amendments[-1]['git_sha']}: {args.amend}")
    print(f"registered and unchanged: wheel {prereg['wheel']['sha256'][:16]}… · predictions "
          f"{pred_sha[:16]}… · {len(fingerprints)} traces · the config")
    print(f"wrote {args.prereg}\nwrote {args.notebook} "
          f"({args.notebook.stat().st_size / 1024:.0f} KiB)")
    return 0


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
    ap.add_argument("--amend", metavar="WHY",
                    help="rebuild the notebook WITHOUT touching anything registered, and record "
                         "WHY (one sentence) in the registration file — see the top of this file")
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
    if args.amend:
        return amend(args, cfg, header, dirty, workloads, fingerprints, pred, pred_sha)

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
