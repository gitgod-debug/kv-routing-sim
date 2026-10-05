#!/usr/bin/env python3
"""Dry-run the C3 notebook on this machine against a mock engine — then the analysis, end to end.

    python scripts/c3_dryrun.py [--keep]

Builds a SMALL version of the notebook (short held-out traffic; the registered notebook and
predictions are never touched), executes every code cell the way Run-All does against Phase 3's
mock engine, and then runs scripts/c3_analyze.py — the registered analysis — on what came out. This
environment's installed package stands in for the embedded wheel: that one cell is skipped.

⭐ The mock runs with KNOWN context terms, and the "C2 profile" of the small config is the mock's
own profile. So here the model IS the engine, and all three rules must hold: the command's check
exact, its fit close enough to predict the traffic, and the registered predictions true. Anything
else is a bug in the notebook or the analysis, found before a GPU is spent. It says nothing about
vLLM.
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

from phase3_dryrun import _fake_vllm, _needles, _port_free, _shim

MOCK_DECODE_PER_CTX_TOKEN_S = 4e-6
MOCK_PREFILL_ATTN_S = 2e-8


def _smoke(work: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    cfg = yaml.safe_load((REPO / "configs" / "c3_calibrate.yaml").read_text())
    e = cfg["engine"]
    # The mock's own profile (scripts/phase3_dryrun.py's MockEngine), with the two terms set below.
    mock = {"name": "mock", "prefill_base_s": 1e-3, "prefill_per_token_s": 2e-5,
            "decode_base_s": 5e-4, "decode_per_seq_s": 2.5e-3,
            "kv_cache_blocks": e["num_gpu_blocks"], "block_size": e["block_size"],
            "max_num_seqs": e["max_num_seqs"], "gpu": "mock", "model": "mock",
            "engine_version": "mock", "calibrated": True, "calibration_error": None,
            "prefill_attn_s": MOCK_PREFILL_ATTN_S,
            "decode_per_ctx_token_s": MOCK_DECODE_PER_CTX_TOKEN_S}
    (work / "mock_profile.json").write_text(json.dumps(mock))
    cfg["profile"] = str(work / "mock_profile.json")
    small = {}
    for name, spec in cfg["holdout"]["workloads"].items():
        spec = dict(spec, n_sessions=16, think_time_s=2.0, arrival_rate=spec["arrival_rate"] * 4)
        small[f"{name}_small"] = spec  # renamed: these are not the traces C2 measured
    cfg["holdout"] = {"workloads": small}
    smoke = work / "c3_smoke.yaml"
    smoke.write_text(yaml.safe_dump(cfg))
    nb, prereg = work / "c3_smoke.ipynb", work / "predictions.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "c3_prepare.py"), "--config",
                    str(smoke), "--prereg", str(prereg), "--notebook", str(nb), "--allow-dirty",
                    "--no-wheel"], check=True, cwd=REPO)
    return nb, prereg


def _fake_torch(pkg: pathlib.Path) -> None:
    """What the pre-flight asks of torch, for the fresh interpreters it starts: a version, a CUDA
    build and a GPU. This environment has no torch, and needs none."""
    (pkg / "torch").mkdir(parents=True, exist_ok=True)
    (pkg / "torch" / "__init__.py").write_text(
        '__version__ = "0.0.0+dryrun"\n\n\nclass version:\n    cuda = "0.0"\n\n\n'
        "class cuda:\n    @staticmethod\n    def is_available():\n        return True\n\n"
        '    @staticmethod\n    def get_device_name(i):\n        return "Tesla T4 (dry-run)"\n')


def _cells(nb: pathlib.Path) -> list[str]:
    out = []
    for c in json.loads(nb.read_text())["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "@@KVSIM-WHEEL@@" in src:
            continue  # the installed package stands in for the embedded wheel
        src = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith("!"))
        out.append(src.replace("STARTUP_TIMEOUT_S = 1800", "STARTUP_TIMEOUT_S = 60"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    args = ap.parse_args()
    if not _port_free(8000):
        raise SystemExit("port 8000 is busy — stop whatever holds it first")
    work = pathlib.Path(tempfile.mkdtemp(prefix="c3_dryrun_"))
    nb, prereg = _smoke(work)
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
    except Exception as e:  # noqa: BLE001 - whatever a cell raises IS the dry-run's finding
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
    results = work / "c3_results"
    if not (results / "c3.json").exists():
        problems.append("c3.json was not written")
    else:
        data = json.loads((results / "c3.json").read_text())
        if len(data.get("calibrate", [])) != 2:
            problems.append(f"{len(data.get('calibrate', []))} runs of the command, expected 2")
        if len(data.get("holdout", [])) != 3:
            problems.append(f"{len(data.get('holdout', []))} held-out runs, expected 3")
        for c in data.get("calibrate", []):
            if not c["prediction_matches_registration"]:
                problems.append(f"seed {c['seed']}: the check's prediction is not the registered")
        pre = data.get("preflight") or {}
        if (pre.get("vllm_cli") or {}).get("returncode") != 0 or not pre.get("torch"):
            problems.append(f"the pre-flight record is missing or failed: {pre.get('vllm_cli')}")
        if pre.get("repairs"):
            problems.append(f"the pre-flight repaired something here: {pre['repairs']}")
    if not (work / "c3_results.zip").exists():
        problems.append("c3_results.zip was not written")
    else:
        names = zipfile.ZipFile(work / "c3_results.zip").namelist()
        for want in ("c3_results/c3.json", "c3_results/calibrated_profile_seed0.json",
                     "c3_results/server_0.log", "c3_results/preflight.json"):
            if want not in names:
                problems.append(f"{want} is missing from the zip")
    analysis = None
    if not problems:
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "c3_analyze.py"), "--results",
                              str(results), "--prereg", str(prereg)], capture_output=True,
                             text=True, cwd=REPO, check=False)
        if out.returncode != 0:
            problems.append(f"c3_analyze.py failed: {out.stderr[-1500:]}")
        else:
            analysis = json.loads((results / "analysis.json").read_text())
            for rule in ("c3a", "c3b", "c3c"):
                if not analysis[rule]["holds"]:
                    problems.append(f"{rule.upper()} does not hold against the mock — where the "
                                    "model is the engine, it must")
            got = analysis["commands"]["0"]["fit"]["decode_per_ctx_token_s"]
            if abs(got - MOCK_DECODE_PER_CTX_TOKEN_S) > 0.35 * MOCK_DECODE_PER_CTX_TOKEN_S:
                problems.append(f"the command did not recover the mock's decode context term: "
                                f"{got:.3g} against {MOCK_DECODE_PER_CTX_TOKEN_S:.3g}")

    print(f"=== C3 dry-run, one mock engine   {run_s:6.1f}s   {'PASS' if not problems else 'FAIL'}")
    if analysis:
        for seed, c in analysis["commands"].items():
            f = c["fit"]
            print(f"    | command, seed {seed}: exit {c['returncode']} in {c['wall_clock_s']:.0f} s"
                  f" · check {c['cached_tokens_equal']}/{c['cached_tokens_compared']} · decode per "
                  f"ctx token {f['decode_per_ctx_token_s']:.3g} s (mock "
                  f"{MOCK_DECODE_PER_CTX_TOKEN_S:.3g}) · prefill attention "
                  f"{f['prefill_attn_s']:.3g} s (mock {MOCK_PREFILL_ATTN_S:.3g})")
        for r in analysis["holdout"]:
            m = r["measured"]
            print(f"    | {r['workload']:18s} measured {m['ttft_p50_ms']:.0f}/"
                  f"{m['ttft_p95_ms']:.0f} ms · command's profile {r['command_factor']} · "
                  f"registered {r['c2_factor']}")
        print("    | " + " · ".join(f"{k.upper()} holds {analysis[k]['holds']}"
                                    for k in ("c3a", "c3b", "c3c")))
    for p in problems:
        print("    ⛔", p)
    if problems:
        print(buf.getvalue()[-4000:])
    if args.keep or problems:
        print(f"working directory kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
