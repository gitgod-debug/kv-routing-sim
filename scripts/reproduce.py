#!/usr/bin/env python3
"""Re-derive the stored results on this machine and compare them with the committed files.

    python scripts/reproduce.py              # every step; no GPU is used
    python scripts/reproduce.py --quick      # leave out the steps that take minutes
    python scripts/reproduce.py --list       # the steps, and what each one re-derives
    python scripts/reproduce.py --only c3    # the steps whose script name contains "c3"

It needs a clone of the repository and numpy: pip install -e ".[analysis]"

The repository's tracked files are copied to a temporary folder and the scripts run there, so
nothing in the working tree is touched. Every file a script writes is then compared with the
committed one. Text, whole numbers and the shape of the data must be equal. Other numbers must
agree to twelve digits: three steps fit a line with numpy, and the last digits of a fit depend on
the CPU and on numpy's version. Where a number differs after the twelfth digit, the step says so.
Fields that describe the run and not the result (date, commit, machine, package versions, run
time) are left out of the comparison.

GPU measurements cannot be made again on a laptop. For those, the analysis is run again on the
raw files that the GPU run left in the repository. Not re-made here either: the stored grids that
some analyses read (T2's three, C4's, and that of calibration C), the first two sweeps,
H6, the scale test and the real-trace runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[1]
# Fields that say when, where and how long, at any depth of a results file.
RUN_FIELDS = frozenset({
    "date_utc", "created_utc", "git_sha", "git_dirty", "git_dirty_paths", "hardware", "packages",
    "runtime_s", "build_wall_clock_s",
})
# A report's own words about when, at which commit and in how many seconds it was made.
GENERATED = re.compile(
    r"generated `[^`]*` at `[^`]*`(?: \(dirty tree\))?|(?<=runs in )\d+(?= s\b)")
# The folder the repository sat in, where a result stores an absolute path into it.
ROOT = re.compile(r"/[^\s\"'`:,]*?/(?=(?:artifacts|configs|scripts|src|notebooks|docs)/)")
TEXT = (".md", ".yaml", ".yml", ".txt")
# How close two computed numbers must be to count as the same result. Three steps fit a line with
# numpy. On the same inputs, another CPU or another numpy version gave fits up to 3e-15 apart
# (relative), so equal digits cannot be asked for. Twelve digits leave a wide margin.
REL_TOL = 1e-12
ABS_TOL = 1e-15


@dataclass(frozen=True)
class Step:
    what: str
    argv: tuple[str, ...]
    slow: bool = False  # takes minutes, not seconds
    # Fields today's code writes that the committed file, written by older code, does not have.
    added_since: tuple[str, ...] = ()
    note: str = ""
    writes: bool = True  # a step that should write files and writes none has failed
    needs: tuple[str, ...] = ()  # modules the step's script imports, beyond the package's own


STEPS = (
    Step("the calibration report, from the GPU run's raw files (Phase 3)",
         ("scripts/phase3_analyze.py",), needs=("numpy",)),
    Step("the same calibration report for vLLM 0.34.0, from that run's raw files (V1)",
         ("scripts/phase3_analyze.py", "artifacts/results/v034_kaggle_t4", "--prereg",
          "artifacts/results/v034_prereg/predictions.json"), needs=("numpy",)),
    Step("hit rates on two live replicas, from the raw files (Phase 6)",
         ("scripts/phase6_analyze.py",)),
    Step("the calibration command on a real GPU (C1)", ("scripts/c1_analyze.py",)),
    Step("the context-aware timing model and its three rules (C2)", ("scripts/c2_analyze.py",)),
    Step("a profile fitted on a real GPU against unseen traffic (C3), with its grid",
         ("scripts/c3_analyze.py", "--grid"), slow=True,
         note="the stored grid's config holds the folder the repository sat in when it ran; "
              "paths into the repository are compared from its root."),
    Step("per-token timing on a real GPU (C4)", ("scripts/c4_analyze.py",)),
    Step("per-token latency on five held-out runs (T1)", ("scripts/t1_itl_check.py",)),
    Step("the two-part target, from its stored grids (T2)", ("scripts/t2_analyze.py",)),
    Step("GPU counts across three calibrations, from their stored grids",
         ("scripts/c3_posthoc_grid.py",)),
    Step("the grid behind the finding and the demo page (H2b on the checked timing model)",
         ("scripts/build_grid.py", "--source", "h2b_c2"), slow=True),
    Step("what one agent session costs (S1)", ("scripts/session_cost.py",), slow=True),
    # Analyses made after their results were seen. Each is quick, and none was covered until one
    # of them was found stale: a stored file that no check runs again can drift unseen.
    Step("Phase 3's two limits, measured (post hoc)", ("scripts/phase3_posthoc.py",),
         needs=("numpy",)),
    Step("the checked timing model against Phase 6's two replicas (post hoc)",
         ("scripts/c2_posthoc_phase6.py",)),
    Step("whether the cells the calibration command measures are enough (C3's pre-check)",
         ("scripts/c3_precheck.py",)),
    Step("each fresh C4 calibration on the micro-benchmark cells (post hoc)",
         ("scripts/c4_posthoc_fresh_cells.py",)),
    Step("the edge check on T1's five runs (post hoc)", ("scripts/edge_posthoc_t1.py",)),
    Step("T1's replays against the range the model was fitted on (post hoc)",
         ("scripts/t1_posthoc_range.py",)),
    Step("two published policies on a production trace, from the stored grid (R3)",
         ("scripts/r3_analyze.py",)),
    Step("every registered prediction, computed again with both engines (post hoc)",
         ("scripts/registered_predictions_again.py",)),
    Step("the README's evidence ledger against the files it names",
         ("scripts/build_ledger.py", "--check"), writes=False),
)


def tracked_files() -> list[str]:
    """The repository's files as git lists them, or every file outside the usual clutter."""
    out = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=False)
    if out.returncode == 0 and out.stdout.strip():
        return [f for f in out.stdout.splitlines() if (REPO / f).is_file()]
    skip = {".git", ".venv", ".venv_kv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return [str(p.relative_to(REPO)) for p in REPO.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(REPO).parts)
            and p.relative_to(REPO).parts[:2] not in (("artifacts", "traces"),
                                                      ("artifacts", "logs"))]


def without(node, fields: frozenset[str]):
    """The same JSON value with the named fields removed at every depth."""
    if isinstance(node, dict):
        return {k: without(v, fields) for k, v in node.items() if k not in fields}
    if isinstance(node, list):
        return [without(v, fields) for v in node]
    return node


def later_digits(new, old) -> int | None:
    """How many numbers of two JSON values differ only after the twelfth digit.

    None if the values differ in any other way. Whole numbers, text, true and false, and the
    shape of the data must be equal. Only a number with a fraction may differ, and only within
    REL_TOL.
    """
    if isinstance(new, dict) and isinstance(old, dict):
        if new.keys() != old.keys():
            return None
        counts = [later_digits(new[k], old[k]) for k in new]
    elif isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            return None
        counts = [later_digits(a, b) for a, b in zip(new, old, strict=True)]
    elif isinstance(new, bool) or isinstance(old, bool):
        return 0 if new is old else None
    elif isinstance(new, float) and isinstance(old, float):
        if new == old or (math.isnan(new) and math.isnan(old)):
            return 0
        return 1 if math.isclose(new, old, rel_tol=REL_TOL, abs_tol=ABS_TOL) else None
    else:
        return 0 if new == old else None
    return None if None in counts else sum(counts)


def compare(new: pathlib.Path, old: pathlib.Path, added_since: tuple[str, ...] = ()) -> int | None:
    """Compare the recomputed file `new` with the committed file `old`.

    0 if they hold the same result, n if n numbers differ only after the twelfth digit, None if
    the results differ.

    The same result means equal bytes; or, for JSON, equal values once the run fields are left
    out; or, for a report or a config, equal text once its "generated ... at ..." words and its
    "runs in N s" are masked. In both cases an absolute path into the repository is compared from
    the repository's root, because a result made in another folder stores another path.
    `added_since` names fields that only the recomputed file may have; the committed file must
    not have them at all.
    """
    a, b = new.read_bytes(), old.read_bytes()
    if a == b:
        return 0
    if new.suffix not in (".json", *TEXT):
        return None
    try:
        recomputed, committed = (ROOT.sub("<repository>/", raw.decode()) for raw in (a, b))
    except UnicodeDecodeError:
        return None
    if new.suffix in TEXT:
        return 0 if GENERATED.sub("", recomputed) == GENERATED.sub("", committed) else None
    try:
        recomputed, committed = json.loads(recomputed), json.loads(committed)
    except ValueError:
        return None
    if without(committed, frozenset(added_since)) != committed:
        return None
    return later_digits(without(recomputed, RUN_FIELDS | frozenset(added_since)),
                        without(committed, RUN_FIELDS))


def same_result(new: pathlib.Path, old: pathlib.Path, added_since: tuple[str, ...] = ()) -> bool:
    """True if `new` holds the same result as `old`, to twelve digits (see `compare`)."""
    return compare(new, old, added_since) is not None


def snapshot(root: pathlib.Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def run_step(step: Step, copy: pathlib.Path) -> tuple[bool, list[str], str, int]:
    """Run one step in the copy.

    Returns (ok, one line per written file, the script's last lines, how many numbers differ
    only after the twelfth digit).
    """
    before = snapshot(copy)
    env = dict(os.environ, PYTHONPATH=str(copy / "src"))
    done = subprocess.run([sys.executable, *step.argv], cwd=copy, env=env, capture_output=True,
                          text=True, check=False)
    tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-6:])
    if done.returncode != 0:
        return False, [], tail, 0
    lines, ok, later = [], True, 0
    for name, stamp in sorted(snapshot(copy).items()):
        if before.get(name) == stamp:
            continue
        original = REPO / name
        if not original.exists():
            lines.append(f"    new file, nothing to compare with   {name}")
            ok = False
            continue
        n = compare(copy / name, original, step.added_since)
        if n is None:
            lines.append(f"    DIFFERS from the committed file     {name}")
            ok = False
        elif n == 0:
            lines.append(f"    stored = recomputed                 {name}")
        else:
            lines.append(f"    stored = recomputed to 12 digits    {name}  "
                         f"({n} number{'s' if n > 1 else ''} differ{'' if n > 1 else 's'} "
                         "after that)")
            later += n
    if step.writes and not lines:
        return False, ["    wrote no file, though it should have"], tail, 0
    return ok, lines, tail, later


def missing_modules(steps: tuple[Step, ...] | list[Step]) -> list[str]:
    """The modules these steps need that cannot be imported here."""
    return sorted({m for s in steps for m in s.needs if importlib.util.find_spec(m) is None})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="leave out the steps that take minutes")
    ap.add_argument("--list", action="store_true", help="print the steps and stop")
    ap.add_argument("--only", metavar="TEXT", help="run only the steps whose script name has TEXT")
    ap.add_argument("--keep", action="store_true", help="keep the temporary copy")
    args = ap.parse_args()
    steps = [s for s in STEPS if not (args.quick and s.slow)
             and (not args.only or args.only in s.argv[0])]
    if args.list:
        for s in steps:
            print(f"{'minutes' if s.slow else 'seconds':8s} {' '.join(s.argv):44s} {s.what}")
        return 0
    missing = missing_modules(steps)
    if missing:
        blocked = sum(1 for s in steps if set(s.needs) & set(missing))
        print(f"{' and '.join(missing)} is not installed here, and {blocked} of these "
              f"{len(steps)} steps need it. Install it and run this again:\n"
              '    pip install -e ".[analysis]"', file=sys.stderr)
        return 2

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kvsim-reproduce-"))
    copy = tmp / "repo"
    for name in tracked_files():
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, copy / name)
    print(f"copied {len(tracked_files())} files to {copy}")
    failed = later = 0
    for s in steps:
        t0 = time.monotonic()
        ok, lines, tail, n = run_step(s, copy)
        failed += not ok
        later += n
        print(f"\n{'ok  ' if ok else 'FAIL'} {s.what}  ({time.monotonic() - t0:.0f} s)")
        print("\n".join(lines) if lines else "    wrote no file")
        if s.note:
            print(f"    note: {s.note}")
        if not ok and tail:
            print("    the script's last lines:\n      " + tail.replace("\n", "\n      "))
    if args.keep:
        print(f"\nkept {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(steps)
    print(f"\n{total - failed} of {total} steps re-derived what is committed"
          + ("" if not failed else f"; {failed} did not"))
    if later:
        print(f"{later} number{'s' if later > 1 else ''} agree{'' if later > 1 else 's'} to "
              "twelve digits and differ after that. Three steps fit a line with numpy, and the "
              "last digits of a fit depend on the CPU and on numpy's version.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
