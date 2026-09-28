#!/usr/bin/env python3
"""Dry-run notebooks/c1_calibrate_t4.ipynb on this machine, against the mock engine.

    python scripts/c1_dryrun.py [--keep]

Every code cell runs in one namespace, the way Run-All does, with Phase 3's mock standing in for
`vllm serve` (through the same shim on PATH) and this environment's installed package standing in
for the embedded wheel — that one cell is skipped, so the editable install is not replaced.

The mock IS the simulator's model, so C1 must HOLD here: the check exact, the capacity read equal to
the mock's log line, and the check's prediction equal to the registered one. Anything else is a bug
in the notebook, found before a GPU is spent.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import phase3_dryrun as p3

NOTEBOOK = REPO / "notebooks" / "c1_calibrate_t4.ipynb"


def _cells() -> list[str]:
    out = []
    for c in json.loads(NOTEBOOK.read_text())["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "@@KVSIM-WHEEL@@" in src:
            continue  # the editable install stands in for the embedded wheel
        src = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith("!"))
        out.append(src.replace("STARTUP_TIMEOUT_S = 1800", "STARTUP_TIMEOUT_S = 60"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    args = ap.parse_args()
    if not p3._port_free(8000):
        raise SystemExit("port 8000 is busy — stop whatever holds it first")
    work = pathlib.Path(tempfile.mkdtemp(prefix="c1_dryrun_"))
    cells = _cells()
    p3._fake_vllm(work / "pkg", None, p3._needles(cells))
    p3._shim(work / "bin")
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
    except Exception as e:  # noqa: BLE001 - whatever a cell raises IS the dry-run's finding
        err = f"{type(e).__name__} in code cell {i}: {e}"
    finally:
        with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
            ns["stop_server"]()
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path
        sys.modules.pop("vllm", None)

    c1 = ns.get("C1") or {}
    problems = [err] if err else []
    for key in ("c1a", "c1b", "prediction_matches_registration"):
        if c1.get(key) is not True:
            problems.append(f"{key} is {c1.get(key)!r} — against the mock it must be True")
    if not (work / "c1_results.zip").exists():
        problems.append("c1_results.zip was not written")
    print(f"=== C1 dry-run against the mock   {time.time() - t0:>6.1f}s  "
          f"{'PASS' if not problems else 'FAIL'}")
    for line in buf.getvalue().splitlines():
        if line.startswith(("C1a", "C1b", "prediction", "==>", "check", "decode", "prefill",
                            "engine log")):
            print("    |", line.strip())
    for p in problems:
        print("    ⛔", p)
    if problems:
        print(buf.getvalue()[-4000:])
    if args.keep or problems:
        print(f"    working directory kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
