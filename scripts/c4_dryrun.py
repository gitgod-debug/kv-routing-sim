#!/usr/bin/env python3
"""Dry-run the C4 notebook against a mock engine, then run the whole analysis on its output.

    python scripts/c4_dryrun.py [--keep]

Builds a small version of the notebook (fewer and smaller cells, short traffic, two runs of the
command; the registered notebook and predictions are left alone), runs every code cell as Run-All
would against Phase 3's mock engine, then runs scripts/c4_grids.py and scripts/c4_analyze.py on the
results.

The mock is the simulator's model with known context terms, and the small registration's
predictions use the mock's own profile, so every timing rule must hold with a factor near 1: the
shared-prefix cells, the stall cells, and the per-request token times on traffic. A failure is a
bug in the notebook or the analysis. Not covered: whether vLLM serves the two histograms as the
mock does, and any real number.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from c3_dryrun import MOCK_DECODE_PER_CTX_TOKEN_S, MOCK_PREFILL_ATTN_S, _cells, _fake_torch
from phase3_dryrun import _fake_vllm, _needles, _port_free, _shim

TIMING_RULES = ("c4a", "d1a", "d1b", "d2", "t3a")  # against the mock these must hold


def _smoke(work: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    cfg = yaml.safe_load((REPO / "configs" / "c4_per_token.yaml").read_text())
    e = cfg["engine"]
    # The mock's own profile (scripts/phase3_dryrun.py's MockEngine), with its two context terms.
    mock = {"name": "mock", "prefill_base_s": 1e-3, "prefill_per_token_s": 2e-5,
            "decode_base_s": 5e-4, "decode_per_seq_s": 2.5e-3,
            "kv_cache_blocks": e["num_gpu_blocks"], "block_size": e["block_size"],
            "max_num_seqs": e["max_num_seqs"], "gpu": "mock", "model": "mock",
            "engine_version": "mock", "calibrated": True, "calibration_error": None,
            "prefill_attn_s": MOCK_PREFILL_ATTN_S,
            "decode_per_ctx_token_s": MOCK_DECODE_PER_CTX_TOKEN_S}
    (work / "mock_profile.json").write_text(json.dumps(mock))
    cfg["profile"] = str(work / "mock_profile.json")
    cfg["calibrate"]["seeds"] = cfg["calibrate"]["seeds"][:2]
    cfg["decode_shared"].update(reps=1, batches=[8, 12, 16, 32])
    # A mock step of 8 sequences is ~67 ms: 1.5 s is ~22 tokens in, of 96.
    cfg["stall"].update(reps=1, cells=[[8, 128, 1.5], [16, 128, 3.0]])
    small = {}
    for name, spec in cfg["holdout"]["workloads"].items():
        small[f"{name}_small"] = dict(spec, n_sessions=16, think_time_s=2.0,
                                      arrival_rate=spec["arrival_rate"] * 6)
    cfg["holdout"] = {"workloads": small}
    cfg["grids"]["firmness"].update(prompts=[1], loads=[1], replicas_max=4)
    cfg["grids"]["prompts64"].update(loads=[1, 2], replicas_max=6)
    smoke = work / "c4_smoke.yaml"
    smoke.write_text(yaml.safe_dump(cfg))
    nb, prereg = work / "c4_smoke.ipynb", work / "predictions.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "c4_prepare.py"), "--config",
                    str(smoke), "--prereg", str(prereg), "--notebook", str(nb), "--allow-dirty",
                    "--no-wheel"], check=True, cwd=REPO)
    return nb, prereg, smoke


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    args = ap.parse_args()
    if not _port_free(8000):
        raise SystemExit("port 8000 is busy — stop whatever holds it first")
    work = pathlib.Path(tempfile.mkdtemp(prefix="c4_dryrun_"))
    nb, prereg, smoke = _smoke(work)
    cells = _cells(nb)
    _fake_vllm(work / "pkg", None, _needles(cells))
    _fake_torch(work / "pkg")
    _shim(work / "bin")
    smi = work / "bin" / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '0, Tesla T4, 15360 MiB, 7.5, 580.00'\n"
                   "echo '1, Tesla T4, 15360 MiB, 7.5, 580.00'\n")
    smi.chmod(0o755)

    saved_env, saved_cwd, saved_path = dict(os.environ), os.getcwd(), list(sys.path)
    os.environ["PATH"] = f"{work / 'bin'}{os.pathsep}{os.environ['PATH']}"
    # The pre-flight probes run in fresh interpreters: they find the fakes through PYTHONPATH.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        x for x in (str(work / "pkg"), os.environ.get("PYTHONPATH")) if x)
    os.environ["MOCK_DECODE_PER_CTX_TOKEN_S"] = str(MOCK_DECODE_PER_CTX_TOKEN_S)
    os.environ["MOCK_PREFILL_ATTN_S"] = str(MOCK_PREFILL_ATTN_S)
    sys.path.insert(0, str(work / "pkg"))
    sys.modules.pop("vllm", None)
    os.chdir(work)
    ns: dict = {"__name__": "__main__"}
    buf, err, t0 = io.StringIO(), None, time.time()
    try:
        with contextlib.redirect_stdout(buf):
            for i, src in enumerate(cells):
                exec(compile(src, f"cell{i}", "exec"), ns)  # noqa: S102 - the point of a dry-run
    except Exception as e:  # noqa: BLE001 - whatever a cell raises is the dry-run's finding
        err = f"{type(e).__name__} in code cell {i}: {e}"
    finally:
        with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
            ns["stop_all"]()
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path
        sys.modules.pop("vllm", None)
    run_s = time.time() - t0

    problems = [err] if err else []
    results = work / "c4_results"
    data: dict = {}
    if not (results / "c4.json").exists():
        problems.append("c4.json was not written")
    else:
        data = json.loads((results / "c4.json").read_text())
        want = {"calibrate": 2, "decode_shared": 4, "decode_control": 2, "stall": 2, "holdout": 3}
        for part, n in want.items():
            if len(data.get(part, [])) != n:
                problems.append(f"part {part}: {len(data.get(part, []))} records, expected {n}")
        for part in ("decode_shared", "decode_control", "stall"):
            bad = [m for m in data.get(part, []) if not m.get("valid")]
            if bad:
                problems.append(f"{len(bad)} invalid {part} cell(s), e.g. {bad[0]}")
        for c in data.get("calibrate", []):
            if not c["prediction_matches_registration"]:
                problems.append(f"seed {c['seed']}: the check's prediction is not the registered")
        for m in data.get("holdout", []):
            if m["invalid"] or m["without_tpot"]:
                problems.append(f"{m['workload']}: {m['invalid']} invalid, {m['without_tpot']} "
                                "without a time per token")
            if m.get("engine_share_within_limit") is None:
                problems.append(f"{m['workload']}: the engine's histogram share was not read")
        if not data.get("tpot_histogram_bounds"):
            problems.append("the per-request histogram's bounds were not read")
    if not (work / "c4_results.zip").exists():
        problems.append("c4_results.zip was not written")
    else:
        names = zipfile.ZipFile(work / "c4_results.zip").namelist()
        for wanted in ("c4_results/c4.json", "c4_results/calibrated_profile_seed0.json",
                       "c4_results/server_0.log", "c4_results/preflight.json"):
            if wanted not in names:
                problems.append(f"{wanted} is missing from the zip")

    analysis = None
    if not problems:
        for script, extra in (("c4_grids.py", ["--config", str(smoke), "--workers", "4"]),
                              ("c4_analyze.py", ["--prereg", str(prereg)])):
            out = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--results",
                                  str(results), *extra], capture_output=True, text=True,
                                 cwd=REPO, check=False)
            if out.returncode != 0:
                problems.append(f"{script} failed: {out.stderr[-1500:]}")
                break
        else:
            analysis = json.loads((results / "analysis.json").read_text())
            for rule in TIMING_RULES:
                if not analysis[rule]["holds"]:
                    problems.append(f"{rule.upper()} does not hold against the mock — where the "
                                    "model is the engine it must")
            for rule in ("t3b", "e", "f1", "f2"):
                if not analysis[rule]["decided"]:
                    problems.append(f"{rule.upper()} was left undecided")
            for s, c in analysis["commands"].items():
                if len(c["replicates"]) != 3:
                    problems.append(f"seed {s}: {len(c['replicates'])} repeat fits, expected 3")

    print(f"=== C4 dry-run, one mock engine   {run_s:6.1f}s   {'PASS' if not problems else 'FAIL'}")
    if analysis:
        for c in analysis["decode"]["cells"]:
            print(f"    | shared b {c['b']:>3}: measured {c['measured_step_s'] * 1e3:6.1f} ms · "
                  f"model {c['predicted_step_s'] * 1e3:6.1f} ms · factor {c['factor']:.3f}")
        for c in analysis["decode"]["controls"]:
            print(f"    | control b {c['b']:>2}: shared against unique {c['factor']:.3f}")
        for c in analysis["d2"]["cells"]:
            print(f"    | stall b {c['b']:>3}: extra gap measured "
                  f"{c['measured_extra_s'] * 1e3:6.1f} ms · model "
                  f"{c['predicted_extra_s'] * 1e3:6.1f} ms · factor {c['factor']:.3f}")
        for r in analysis["traffic"]["runs"]:
            m = r["measured"]
            print(f"    | {r['workload']:18s} p95 ms per token {m['tpot_p95_ms']:6.1f} · "
                  f"registered {r['registered']['tpot_p95_ms']:6.1f} · factor {r['factor']:.3f} · "
                  f"engine share {m['engine_share_within_limit']:.3f} / client "
                  f"{m['client_share_within_limit']:.3f} · verdict agrees {r['verdict_agrees']}")
        e = analysis["e"]
        print(f"    | E: {e['covered']} of {e['cases']} covered · F1 {analysis['f1']['holds']} · "
              f"F2 {analysis['f2']['holds']} (meaningless numbers here: the mock is no GPU)")
        print("    | " + " · ".join(f"{k.upper()} {analysis[k]['holds']}" for k in (
            "c4a", "d1a", "d1b", "d2", "t3a", "t3b")))
    for p in problems:
        print("    ⛔", p)
    if problems:
        print(buf.getvalue()[-5000:])
    if args.keep or problems:
        print(f"working directory kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
