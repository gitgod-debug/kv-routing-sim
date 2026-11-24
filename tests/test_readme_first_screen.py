"""The README's first screen against the files it summarises.

Three tables sit at the top of the README: where the numbers hold, what one agent session costs,
and every registered test with its outcome. These tests fail if a table drifts from the log, the
stored result or the profile it was copied from.
"""

from __future__ import annotations

import json
import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results"
README = (REPO / "README.md").read_text(encoding="utf-8")
LOG = (REPO / "docs" / "09-hypotheses.md").read_text(encoding="utf-8")

HELD, FAILED, PART = "✅ held", "⛔ failed", "✅⛔"
# One entry per row of the log's status table. A test added to the log must be added here and to
# the README, or the first test below fails.
OUTCOMES = {
    "H1": FAILED, "H2": FAILED, "H2b": PART, "H3": HELD, "H4": HELD, "H5": PART, "H6": PART,
    "R1": FAILED, "R2": HELD, "R1, R2 again": PART, "R3": PART, "Phase 6": HELD,
    "C1": HELD, "C2": HELD, "C3": HELD, "C4": PART, "T1": FAILED, "T2": HELD, "S1": HELD,
}


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


def _named_rows(table: str) -> dict[str, list[str]]:
    """The cells of every table row whose first cell is a bold name, keyed by that name."""
    rows = [[c.strip() for c in line.strip().strip("|").split(" | ")]
            for line in table.splitlines() if line.startswith("| **")]
    return {row[0].strip("*"): row for row in rows}


def _log_status() -> dict[str, str]:
    table = LOG[LOG.index("\n## Status"):LOG.index("**Amendments:**")]
    return {name: row[-1] for name, row in _named_rows(table).items()}


def test_every_registered_test_is_on_the_first_screen_with_its_outcome():
    assert set(_log_status()) == set(OUTCOMES), "the log's status table gained or lost a test"
    shown = _named_rows(_section(README, "## Every registered test"))
    assert set(shown) == set(OUTCOMES)
    for name, mark in OUTCOMES.items():
        assert shown[name][2].startswith(mark), name


def test_the_first_screen_shows_no_success_the_log_does_not_record():
    """A row shown as held is not one the log calls falsified, and the reverse."""
    log = _log_status()
    for name, mark in OUTCOMES.items():
        missed = any(word in log[name] for word in ("FALSIFIED", "DOES NOT HOLD"))
        if mark == HELD:
            assert "✅" in log[name] and not missed, name
        elif mark == FAILED:
            assert missed and "✅" not in log[name], name


def test_the_count_above_the_table_matches_its_rows():
    marks = list(OUTCOMES.values())
    said = re.search(r"Of (\d+) rows, (\d+) held .*?, (\d+) failed, and (\d+) held in part",
                     _section(README, "## Every registered test"), re.DOTALL)
    assert [int(n) for n in said.groups()] == [
        len(marks), marks.count(HELD), marks.count(FAILED), marks.count(PART)]


def test_the_session_cost_table_is_the_stored_result():
    stored = json.loads((RESULTS / "session_cost" / "session_cost.json").read_text())
    cell = stored["headline_cell"]
    names = {"radix-affinity-balanced@16": "balanced prefix-affinity"}
    want = []
    for target in ("first_token", "two_part"):
        where = (target, cell["prompts"], cell["load"])
        rows = [r for r in stored["rows"] if (r["target"], r["prompts"], r["load"]) == where]
        for r in sorted(rows, key=lambda r: r["inr_per_session"]):
            ci = r["re_prefilled_ci"]
            want.append([
                names.get(r["policy"], r["policy"]), str(r["replicas"]), f"{r['hit_rate']:.1%}",
                (f"{r['re_prefilled_tokens_per_session']:,.0f} "
                 f"({ci['ci_lo']:,.0f} to {ci['ci_hi']:,.0f})"),
                f"{r['inr_per_session']:.3f}",
            ])
    section = _section(README, "## What one agent session costs")
    table = [line for line in section.splitlines() if line.startswith("|")][2:]
    shown = [[c.strip() for c in line.split("|")][2:7] for line in table]
    assert shown == want
    first = next(r for r in stored["rows"] if (r["prompts"], r["load"]) == (1, 2))
    assert f"{first['prompt_tokens_per_session']:,.0f} prompt tokens" in section
    assert f"{first['new_tokens_per_session']:,.0f} of them are new" in section
    assert f"{stored['requests_per_session']:g} requests" in section


def test_the_validity_table_states_what_the_profile_and_the_grids_record():
    section = _section(README, "## Validity")
    profile = json.loads(
        (REPO / "configs" / "profiles" / "t4-qwen3-4b-vllm0.30.0-ctx.json").read_text())
    assert f"vLLM **{profile['engine_version']}**" in section
    assert profile["gpu"] in section
    assert f"`{profile['model']}`" in section
    assert f"blocks of {profile['block_size']} tokens" in section
    assert f"{profile['kv_cache_blocks']:,} blocks on a T4" in section
    first = json.loads((RESULTS / "demo_grid_ctx.json").read_text())["axes"]["replicas"]
    both = json.loads((RESULTS / "t2_two_part_slo" / "grid_A.json").read_text())["axes"]["replicas"]
    assert f"1 to {max(first)}, and to {max(both)} under the two-part target" in section
    exact = yaml.safe_load((REPO / "configs" / "phase3_workloads.yaml").read_text())["engine"]
    assert f"{exact['num_gpu_blocks']} in the exactness test" in section
