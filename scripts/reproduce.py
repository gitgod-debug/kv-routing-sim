#!/usr/bin/env python3
"""Re-derive the stored results on this machine and compare them with the committed files.

    python scripts/reproduce.py              # every step; no GPU is used
    python scripts/reproduce.py --quick      # leave out the steps that take minutes
    python scripts/reproduce.py --list       # the steps, and what each one re-derives
    python scripts/reproduce.py --only c3    # the steps whose script name contains "c3"

The repository's tracked files are copied to a temporary folder and the scripts run there, so
nothing in the working tree is touched. Every file a script writes is then compared with the
committed one. Numbers and text must be equal. Fields that describe the run and not the result
(date, commit, machine, package versions, run time) are left out of the comparison.

GPU measurements cannot be made again on a laptop. For those, the analysis is run again on the
raw files that the GPU run left in the repository. Not re-made here either: the stored grids that
some analyses read (T2's three, C4's, and those of calibrations B and C), the first two sweeps,
H6, the scale test and the real-trace runs.
"""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class Step:
    what: str
    argv: tuple[str, ...]
    slow: bool = False  # takes minutes, not seconds
    # Fields today's code writes that the committed file, written by older code, does not have.
    added_since: tuple[str, ...] = ()
    note: str = ""
    writes: bool = True  # a step that should write files and writes none has failed


STEPS = (
    Step("the calibration report, from the GPU run's raw files (Phase 3)",
         ("scripts/phase3_analyze.py",),
         added_since=("prefill_attn_s", "decode_per_ctx_token_s"),
         note="a profile gained two timing fields on 2026-09-30 (C2); the committed Phase 3 "
              "files are older and do not have them. Every other value is compared."),
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


def same_result(new: pathlib.Path, old: pathlib.Path, added_since: tuple[str, ...] = ()) -> bool:
    """True if the recomputed file `new` holds the same result as the committed file `old`.

    Equal bytes; or, for JSON, equal values once the run fields are left out; or, for a report
    or a config, equal text once its "generated ... at ..." words and its "runs in N s" are
    masked. In both cases an
    absolute path into the repository is compared from the repository's root, because a result
    made in another folder stores another path. `added_since` names fields that only the
    recomputed file may have; the committed file must not have them at all.
    """
    a, b = new.read_bytes(), old.read_bytes()
    if a == b:
        return True
    if new.suffix not in (".json", *TEXT):
        return False
    try:
        recomputed, committed = (ROOT.sub("<repository>/", raw.decode()) for raw in (a, b))
    except UnicodeDecodeError:
        return False
    if new.suffix in TEXT:
        return GENERATED.sub("", recomputed) == GENERATED.sub("", committed)
    try:
        recomputed, committed = json.loads(recomputed), json.loads(committed)
    except ValueError:
        return False
    if without(committed, frozenset(added_since)) != committed:
        return False
    return (without(recomputed, RUN_FIELDS | frozenset(added_since))
            == without(committed, RUN_FIELDS))


def snapshot(root: pathlib.Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def run_step(step: Step, copy: pathlib.Path) -> tuple[bool, list[str], str]:
    """Run one step in the copy; return (ok, one line per written file, the script's last lines)."""
    before = snapshot(copy)
    env = dict(os.environ, PYTHONPATH=str(copy / "src"))
    done = subprocess.run([sys.executable, *step.argv], cwd=copy, env=env, capture_output=True,
                          text=True, check=False)
    tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-6:])
    if done.returncode != 0:
        return False, [], tail
    lines, ok = [], True
    for name, stamp in sorted(snapshot(copy).items()):
        if before.get(name) == stamp:
            continue
        original = REPO / name
        if not original.exists():
            lines.append(f"    new file, nothing to compare with   {name}")
            ok = False
        elif same_result(copy / name, original, step.added_since):
            lines.append(f"    stored = recomputed                 {name}")
        else:
            lines.append(f"    DIFFERS from the committed file     {name}")
            ok = False
    if step.writes and not lines:
        return False, ["    wrote no file, though it should have"], tail
    return ok, lines, tail


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

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kvsim-reproduce-"))
    copy = tmp / "repo"
    for name in tracked_files():
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, copy / name)
    print(f"copied {len(tracked_files())} files to {copy}")
    failed = 0
    for s in steps:
        t0 = time.monotonic()
        ok, lines, tail = run_step(s, copy)
        failed += not ok
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
