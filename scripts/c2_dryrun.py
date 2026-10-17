#!/usr/bin/env python3
"""Dry-run the C2 notebook against two mock engines, then run the analysis on its output.

    python scripts/c2_dryrun.py [--keep]

Builds a small version of the notebook (one rep per part, short held-out traffic; the registered
notebook and predictions are left alone), runs every code cell as Run-All would against Phase 3's
mock engine on both ports, then runs scripts/c2_analyze.py on the results. The mocks have known
context terms (the MOCK_* constants), so the fit must recover them. This tests the notebook and
the analysis, not vLLM.
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

MOCK_DECODE_PER_CTX_TOKEN_S = 4e-6
MOCK_PREFILL_ATTN_S = 2e-8


def _smoke(work: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    cfg = yaml.safe_load((REPO / "configs" / "c2_timing.yaml").read_text())
    cfg["decode"]["reps"], cfg["prefill"]["reps"], cfg["cpu"]["reps"] = 1, 1, 1
    cfg["cpu"]["settle_s"] = 3
    cfg["cpu"]["background"]["concurrency"] = 8
    for spec in cfg["holdout"]["workloads"].values():
        spec.update(n_sessions=16, think_time_s=2.0)
        spec["arrival_rate"] = spec["arrival_rate"] * 4
    smoke = work / "c2_smoke.yaml"
    smoke.write_text(yaml.safe_dump(cfg))
    nb, prereg = work / "c2_smoke.ipynb", work / "predictions.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "c2_prepare.py"), "--config",
                    str(smoke), "--prereg", str(prereg), "--notebook", str(nb)],
                   check=True, cwd=REPO)
    return nb, prereg


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
    work = pathlib.Path(tempfile.mkdtemp(prefix="c2_dryrun_"))
    nb, prereg = _smoke(work)
    cells = _cells(nb)
    _fake_vllm(work / "pkg", None, _needles(cells))
    _shim(work / "bin")
    smi = work / "bin" / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '0, Tesla T4, 15360 MiB, 7.5, 580.00'\n"
                   "echo '1, Tesla T4, 15360 MiB, 7.5, 580.00'\n")
    smi.chmod(0o755)

    saved_env, saved_cwd, saved_path = dict(os.environ), os.getcwd(), list(sys.path)
    os.environ["PATH"] = f"{work / 'bin'}{os.pathsep}{os.environ['PATH']}"
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
    results = work / "c2_results"
    if not (results / "c2.json").exists():
        problems.append("c2.json was not written")
    else:
        data = json.loads((results / "c2.json").read_text())
        for part in ("decode", "prefill", "cpu", "holdout"):
            if not data.get(part):
                problems.append(f"part {part} is missing from c2.json")
        bad = [m for m in data.get("decode", []) if not m["valid"]]
        if bad:
            problems.append(f"{len(bad)} invalid decode cells, e.g. {bad[0]}")
    analysis = None
    if not problems:
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "c2_analyze.py"), "--results",
                              str(results), "--prereg", str(prereg)], capture_output=True,
                             text=True, cwd=REPO, check=False)
        if out.returncode != 0:
            problems.append(f"c2_analyze.py failed: {out.stderr[-1500:]}")
        else:
            analysis = json.loads((results / "analysis.json").read_text())
            got = analysis["fit"]["decode"]["per_ctx_token_s"]
            if abs(got - MOCK_DECODE_PER_CTX_TOKEN_S) > 0.35 * MOCK_DECODE_PER_CTX_TOKEN_S:
                problems.append(f"the decode fit did not recover the mock's context term: "
                                f"{got:.3g} against {MOCK_DECODE_PER_CTX_TOKEN_S:.3g}")
            if not analysis["c2a"]["holds"]:
                problems.append("C2a does not hold on a mock that has a context term")

    verdict = "PASS" if not problems else "FAIL"
    print(f"=== C2 dry-run, two mock engines   {run_s:6.1f}s   {verdict}")
    if analysis:
        f = analysis["fit"]
        print(f"    | decode per ctx token: fitted {f['decode']['per_ctx_token_s']:.3g} s · mock "
              f"{MOCK_DECODE_PER_CTX_TOKEN_S:.3g} s (R² {f['decode']['r2']:.3f})")
        print(f"    | prefill attention:    fitted {f['prefill']['attn_s']:.3g} s · mock "
              f"{MOCK_PREFILL_ATTN_S:.3g} s (R² {f['prefill']['r2']:.3f})")
        print(f"    | C2a ratio {analysis['c2a']['ratio']:.2f} · C2b median ratio "
              f"{analysis['c2b']['median_ratio']} · C2c holds {analysis['c2c']['holds']}")
        for r in analysis["c2c"]["runs"]:
            m, c = r["measured"], r["context_aware"]
            print(f"    | {r['workload']:12s} measured p50/p95 {m['ttft_p50_ms']:.0f}/"
                  f"{m['ttft_p95_ms']:.0f} ms · fitted model {c['ttft_p50_ms']:.0f}/"
                  f"{c['ttft_p95_ms']:.0f} ms · factors {r['factor']}")
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
