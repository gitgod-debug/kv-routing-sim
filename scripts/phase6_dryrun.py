#!/usr/bin/env python3
"""Dry-run the Phase 6 notebook on this machine against two mock vLLM engines.

    python scripts/phase6_dryrun.py [--keep]

Builds a small version of the notebook in a temp directory (the same cells from the same builder,
with fewer sessions and faster arrivals; the committed notebook and predictions are not touched)
and executes every code cell in one namespace, the way Run-All does. Phase 3's mock engine stands
in for both `vllm serve` processes, one per port, through the same `vllm` shim on PATH.

This tests the plumbing: the regenerated traces and their fingerprints, two servers, resets on
both, the routers, streaming, the counters, the verdict and the zip. It cannot test whether vLLM
behaves like the simulator, because here the simulator is the engine (with a faster timing
profile, so routing and batching differ a little from the prediction).
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

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase3_dryrun import _fake_vllm, _needles, _port_free, _shim


def _smoke_notebook(work: pathlib.Path) -> pathlib.Path:
    """The Phase 6 config with a small, fast workload: about 20 s per run instead of 3 minutes."""
    cfg = yaml.safe_load((REPO / "configs" / "phase6_sim2real.yaml").read_text())
    for spec in cfg["workloads"].values():
        spec.update(n_sessions=16, arrival_rate=4.0, think_time_s=2.0)
    smoke = work / "phase6_smoke.yaml"
    smoke.write_text(yaml.safe_dump(cfg))
    nb = work / "phase6_smoke.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts" / "phase6_prepare.py"),
                    "--config", str(smoke), "--prereg", str(work / "predictions.json"),
                    "--notebook", str(nb)], check=True, cwd=REPO)
    return nb


def _cells(nb: pathlib.Path) -> list[str]:
    out = []
    for c in json.loads(nb.read_text())["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith("!"))
        out.append(src.replace("STARTUP_TIMEOUT_S = 1800", "STARTUP_TIMEOUT_S = 60"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    args = ap.parse_args()
    if not (_port_free(8000) and _port_free(8001)):
        raise SystemExit("ports 8000/8001 are busy — stop whatever holds them first")
    work = pathlib.Path(tempfile.mkdtemp(prefix="phase6_dryrun_"))
    cells = _cells(_smoke_notebook(work))
    _fake_vllm(work / "pkg", None, _needles(cells))
    _shim(work / "bin")
    # The environment cell must see Kaggle's two T4s; the check itself runs unmodified.
    smi = work / "bin" / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '0, Tesla T4, 15360 MiB, 7.5, 580.00'\n"
                   "echo '1, Tesla T4, 15360 MiB, 7.5, 580.00'\n")
    smi.chmod(0o755)

    saved_env, saved_cwd, saved_path = dict(os.environ), os.getcwd(), list(sys.path)
    os.environ["PATH"] = f"{work / 'bin'}{os.pathsep}{os.environ['PATH']}"
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

    runs = ns.get("RUNS", [])
    problems = [err] if err else []
    if len(runs) != 6:
        problems.append(f"expected 6 runs, got {len(runs)}")
    for r in runs:
        if r["invalid_requests"]:
            problems.append(f"{r['workload']} {r['policy']}: invalid {r['invalid_requests'][:3]}")
        if r["measured_hit_rate"] is None:
            problems.append(f"{r['workload']} {r['policy']}: no hit rate measured")
    if "VERDICT" not in ns:
        problems.append("no verdict computed")
    if not (work / "phase6_results.zip").exists() and err is None:
        problems.append("phase6_results.zip was not written")
    print(f"=== Phase 6 dry-run, two mock engines   {time.time() - t0:6.1f}s   "
          f"{'PASS' if not problems else 'FAIL'}")
    for line in buf.getvalue().splitlines():
        if any(k in line for k in ("hit ", "S2 ", "S1 ", "Phase 6:", "✓", "ready in")):
            print("    |", line.strip())
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
