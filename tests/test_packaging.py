"""Tests for the profile and cost model shipped inside the package for users without a clone.

They are copies of the files in `configs/` and must stay byte-identical to them, so one name never
refers to two different measured profiles.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from kv_routing_sim import data
from kv_routing_sim.cli import main
from kv_routing_sim.eval.cost import CostModel
from kv_routing_sim.sim.cost_model import load_profile, resolve_profile
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
T4 = "t4-qwen3-4b-vllm0.30.0"


def test_the_shipped_copies_are_byte_identical_to_configs():
    assert data.cost_model().read_bytes() == (REPO / "configs" / "cost_model.yaml").read_bytes()
    shipped = data.profiles()
    assert set(shipped) == {p.stem for p in (REPO / "configs" / "profiles").glob("*.json")}
    for name, ref in shipped.items():
        assert ref.read_bytes() == (REPO / "configs" / "profiles" / f"{name}.json").read_bytes()


def test_a_profile_resolves_by_path_or_by_shipped_name():
    by_name = resolve_profile(T4)
    assert by_name == load_profile(REPO / "configs" / "profiles" / f"{T4}.json")
    assert by_name.calibrated and by_name.kv_cache_blocks == 1537
    assert resolve_profile(f"{T4}.json") == by_name


def test_an_unknown_profile_fails_and_names_the_shipped_ones():
    with pytest.raises(FileNotFoundError, match=T4):
        resolve_profile("configs/profiles/no-such-gpu.json")


def test_the_default_cost_model_is_the_cited_one():
    cited = CostModel.from_yaml(REPO / "configs" / "cost_model.yaml")
    assert CostModel.from_yaml("default") == cited


def test_replay_works_outside_a_clone(tmp_path, monkeypatch):
    """The README's command, run from a directory with no configs/ in it."""
    trace = tmp_path / "trace.jsonl"
    AgenticWorkload(n_sessions=12, turns_per_session=2, arrival_rate=1.0, seed=0).write_jsonl(trace)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "rec.json"
    assert main(["replay", str(trace), "--policies", "round-robin,session-sticky",
                 "--replicas", "1-2", "--seeds", "1", "--profile", T4, "--cost-model", "default",
                 "--format", "json", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["profile"]["calibrated"] is True
    assert payload["recommendation"]["recommended"]["inr_per_1k_requests"] is not None
