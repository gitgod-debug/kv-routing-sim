"""Tests for how scripts/reproduce.py compares a recomputed file with the committed one.

The reproduction itself takes minutes and is not run here.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rep = _load("reproduce")

COMMITTED = {
    "script": "scripts/x.py", "date_utc": "2026-10-05T11:12:53+00:00", "git_sha": "0be7bec",
    "git_dirty": False, "hardware": {"usable_cpus": 12}, "packages": {"numpy": "2.5.3"},
    "runtime_s": 791.6,
    "cells": [{"hit_rate": 0.9197, "replicas": 5, "profile": {"runtime_s": 3.0, "blocks": 1537}}],
}


def _pair(tmp_path, new, old, suffix=".json"):
    a, b = tmp_path / f"new{suffix}", tmp_path / f"old{suffix}"
    for path, content in ((a, new), (b, old)):
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    return a, b


def test_a_file_made_on_another_day_and_machine_holds_the_same_result(tmp_path):
    again = json.loads(json.dumps(COMMITTED))
    again.update(date_utc="2026-11-25T00:00:00+00:00", git_sha="unknown", git_dirty=True,
                 hardware={"usable_cpus": 4}, packages={"numpy": "2.6.0"}, runtime_s=12.0)
    again["cells"][0]["profile"]["runtime_s"] = 9.0  # a run field, at any depth
    assert rep.same_result(*_pair(tmp_path, again, COMMITTED))


def test_another_number_is_another_result(tmp_path):
    again = json.loads(json.dumps(COMMITTED))
    again["cells"][0]["hit_rate"] = 0.9198
    assert not rep.same_result(*_pair(tmp_path, again, COMMITTED))
    again = json.loads(json.dumps(COMMITTED))
    again["script"] = "scripts/y.py"  # what made the file is part of the result
    assert not rep.same_result(*_pair(tmp_path, again, COMMITTED))


def test_a_field_added_since_is_allowed_only_where_the_step_names_it(tmp_path):
    again = json.loads(json.dumps(COMMITTED))
    again["cells"][0]["prefill_attn_s"] = 0.0
    new, old = _pair(tmp_path, again, COMMITTED)
    assert not rep.same_result(new, old)
    assert rep.same_result(new, old, added_since=("prefill_attn_s",))
    # A committed file that has the field is not "older code": the exemption does not apply.
    assert not rep.same_result(old, new, added_since=("prefill_attn_s",))


def test_a_report_is_compared_apart_from_its_own_generated_line(tmp_path):
    old = "T1 · generated `2026-10-05T11:12:53+00:00` at `0be7bec` by `scripts/t1.py`\n\n1.54×\n"
    new = old.replace("2026-10-05T11:12:53", "2026-11-25T00:00:00").replace("0be7bec", "unknown")
    assert rep.same_result(*_pair(tmp_path, new, old, ".md"))
    assert not rep.same_result(*_pair(tmp_path, new.replace("1.54", "1.55"), old, ".md"))
    dirty = old.replace("`0be7bec`", "`0be7bec` (dirty tree)")  # the stored file says so itself
    assert rep.same_result(*_pair(tmp_path, new, dirty, ".md"))
    grid = "3 seeds · 6,912 runs in 306 s · five T4s\n"
    assert rep.same_result(*_pair(tmp_path, grid.replace("306 s", "436 s"), grid, ".md"))
    assert not rep.same_result(*_pair(tmp_path, grid.replace("6,912", "6,913"), grid, ".md"))


def test_every_step_names_a_script_that_exists_and_the_copy_lists_real_files():
    for step in rep.STEPS:
        assert (REPO / step.argv[0]).is_file(), step.argv[0]
        assert step.note or not step.added_since  # an exemption always says why
    files = rep.tracked_files()
    assert "README.md" in files and "scripts/build_grid.py" in files
    assert all((REPO / f).is_file() for f in files)
    assert not any(f.startswith("artifacts/traces/") for f in files)


def test_a_path_into_the_repository_is_compared_from_the_repositorys_root(tmp_path):
    old = "profile: /home/someone/work/sim/artifacts/results/c3/profile.json\nalpha: 16\n"
    new = old.replace("/home/someone/work/sim", "/tmp/kvsim-reproduce-abc/repo")
    assert rep.same_result(*_pair(tmp_path, new, old, ".yaml"))
    assert not rep.same_result(*_pair(tmp_path, new.replace("16", "17"), old, ".yaml"))
    other_file = new.replace("profile.json", "another.json")
    assert not rep.same_result(*_pair(tmp_path, other_file, old, ".yaml"))
    stored = {"config": {"profile": "/a/b/artifacts/results/c3/profile.json"}, "replicas": 5}
    again = {"config": {"profile": "/x/artifacts/results/c3/profile.json"}, "replicas": 5}
    assert rep.same_result(*_pair(tmp_path, again, stored))
    assert not rep.same_result(*_pair(tmp_path, {**again, "replicas": 6}, stored))
