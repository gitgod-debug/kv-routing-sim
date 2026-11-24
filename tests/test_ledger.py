"""The README's evidence ledger against the files it points to (scripts/build_ledger.py)."""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ledger = _load("build_ledger")


def _cells(table: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip().strip("|").split(" | ")]
            for line in table.splitlines()[2:]]


def _full_history() -> bool:
    """True in a full clone. A shallow one, as CI makes, cannot say which commit changed a file."""
    out = subprocess.run(
        ["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "rev-parse",
         "--is-shallow-repository"], capture_output=True, text=True, check=False)
    return out.returncode == 0 and out.stdout.strip() == "false"


def test_every_row_reads_its_number_from_its_file():
    """`table` stops if a file is missing, was made by another script or holds another number."""
    built = _cells(ledger.table(commit_of=lambda path: "-"))
    assert len(built) == len(ledger.ROWS) >= 20


def test_the_readme_holds_the_ledger_the_script_builds():
    shown = _cells(ledger.in_readme())
    built = _cells(ledger.table(commit_of=lambda path: "-"))
    assert [row[:4] for row in shown] == [row[:4] for row in built]
    if _full_history():
        assert shown == _cells(ledger.table())


def test_a_row_whose_file_holds_another_number_is_refused():
    wrong = dataclasses.replace(ledger.ROWS[0], shown="2 of 3 runs, on both")
    with pytest.raises(SystemExit, match="holds"):
        ledger.verify(wrong)


def test_a_row_that_names_the_wrong_script_is_refused():
    by_script = next(r for r in ledger.ROWS if r.made_by == "scripts/session_cost.py")
    with pytest.raises(SystemExit, match="made by"):
        ledger.verify(dataclasses.replace(by_script, made_by="scripts/build_grid.py"))
