"""Tests for the measured profiles' index and the distance of a setup from them."""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from kv_routing_sim.cli import main
from kv_routing_sim.profiles import (
    MeasuredProfile,
    Setup,
    differences,
    longest_prompt_measured,
    measured_profiles,
    nearest,
)
from kv_routing_sim.sim.cost_model import resolve_profile
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
T4 = Setup(gpu="Tesla T4", model="Qwen/Qwen3-4B", engine_version="0.30.0", block_size=16,
           max_num_seqs=256)


def _profile(name: str, **key) -> MeasuredProfile:
    base = {"gpu": "Tesla T4", "model": "m", "engine_version": "0.30.0", "block_size": 16,
            "max_num_seqs": 256}
    return MeasuredProfile(name, Setup(**{**base, **key}), 100, 2000, (), "somewhere")


def test_the_index_agrees_with_every_shipped_profile_and_its_source():
    index = yaml.safe_load((REPO / "configs" / "profiles" / "index.yaml").read_text())
    shipped = REPO / "src" / "kv_routing_sim" / "data" / "profiles"
    assert (shipped / "index.yaml").read_bytes() == \
        (REPO / "configs" / "profiles" / "index.yaml").read_bytes()
    names = {p["name"] for p in index["profiles"]}
    assert names == {p.stem for p in (REPO / "configs" / "profiles").glob("*.json")}
    for entry in index["profiles"]:
        profile = resolve_profile(entry["name"])
        assert entry["key"] == {f: getattr(profile, f) for f in entry["key"]}
        assert entry["kv_cache_blocks"] == profile.kv_cache_blocks
        assert (REPO / entry["evidence"]).is_file()


def test_the_longest_prompt_is_the_longest_one_the_fit_measured():
    c2 = json.loads((REPO / "artifacts" / "results" / "c2_kaggle_t4" / "c2.json").read_text())
    design = c2["design"]
    fitted = max([length for length, _ in design["decode"]["grid"]]
                 + [c + n for c, n in design["prefill"]["grid"]])
    assert longest_prompt_measured("t4-qwen3-4b-vllm0.30.0-ctx") == fitted == 1984
    own = REPO / "artifacts" / "results" / "c4_kaggle_t4" / "calibrated_profile_seed0.json"
    source = json.loads(own.read_text())["_source"]
    expected = max([c["context"] for c in source["decode_cells"]]
                   + [p["prompt_tokens"] for p in source["prefill_points"]])
    assert longest_prompt_measured(own) == expected


def test_a_profile_that_does_not_say_its_range_has_none(tmp_path):
    path = tmp_path / "guess.json"
    path.write_text(json.dumps({"name": "guess"}))
    assert longest_prompt_measured(path) is None
    assert longest_prompt_measured("no-such-profile") is None


def test_the_same_setup_has_no_differences_whatever_its_spelling():
    assert differences(T4, T4) == []
    assert differences(Setup("tesla t4", "Qwen/Qwen3-4B", "v0.30.0", 16, 256), T4) == []


def test_each_differing_or_unknown_field_is_named_with_what_it_means():
    found = differences(Setup(gpu="Tesla T4", model="Qwen/Qwen3-4B", engine_version="0.34.0",
                              block_size=16), T4)
    assert [(d.field, d.yours, d.measured) for d in found] == [
        ("engine_version", "0.34.0", "0.30.0"), ("max_num_seqs", None, 256)]
    assert all(d.meaning for d in found)


def test_the_nearest_has_the_fewest_differences_and_the_gpu_breaks_ties():
    a100 = _profile("a100", gpu="A100")
    other_model = _profile("other-model", model="x")
    yours = Setup(gpu="A100", model="x", engine_version="0.30.0", block_size=16, max_num_seqs=256)
    best, found = nearest(yours, [other_model, a100])  # one difference each
    assert (best.name, [d.field for d in found]) == ("a100", ["model"])
    exact = _profile("exact", gpu="A100", model="x")
    assert nearest(yours, [a100, other_model, exact])[0].name == "exact"
    with pytest.raises(ValueError, match="no measured profile"):
        nearest(yours, [])


def test_the_command_lists_the_profiles_and_gives_a_distance(capsys):
    assert main(["profiles"]) == 0
    listed = capsys.readouterr().out
    assert all(p.name in listed for p in measured_profiles())
    assert main(["profiles", "--gpu", "A100", "--model", "Qwen/Qwen3-4B", "--engine-version",
                 "0.34.0", "--block-size", "16", "--max-num-seqs", "256"]) == 0
    out = capsys.readouterr().out
    assert "A100  differs: measured on Tesla T4" in out
    assert "2 of 5 differ or are unknown" in out
    assert "Nothing is interpolated between profiles" in out


@pytest.mark.parametrize(("turns", "outside"), [(4, False), (6, True)])
def test_a_replay_says_when_prompts_are_longer_than_the_profile_was_timed_on(tmp_path, turns,
                                                                             outside):
    """Six turns of the generator's default session pass 1,984 tokens; four do not."""
    trace = tmp_path / "trace.jsonl"
    AgenticWorkload(n_sessions=10, turns_per_session=turns, arrival_rate=0.5,
                    seed=0).write_jsonl(trace)
    out = tmp_path / "rec.json"
    assert main(["replay", str(trace), "--policies", "session-sticky", "--replicas", "1-2",
                 "--seeds", "1", "--profile", "t4-qwen3-4b-vllm0.30.0-ctx", "--cost-model",
                 "default", "--no-edge-check", "--format", "json", "--out", str(out)]) == 0
    prompts = json.loads(out.read_text())["prompt_range"]
    assert prompts["longest_measured"] == 1984
    assert (prompts["requests_longer"] > 0) is outside
    md = tmp_path / "rec.md"
    main(["replay", str(trace), "--policies", "session-sticky", "--replicas", "1-2", "--seeds", "1",
          "--profile", "t4-qwen3-4b-vllm0.30.0-ctx", "--cost-model", "default", "--no-edge-check",
          "--out", str(md)])
    assert ("Outside the measured range" in md.read_text()) is outside


def test_a_source_that_is_not_an_object_means_the_spread_and_range_are_unknown(tmp_path):
    from kv_routing_sim.sim.cost_model import replicate_profiles

    raw = json.loads((REPO / "configs" / "profiles" / "t4-qwen3-4b-vllm0.30.0.json").read_text())
    reps = [{"rep": r, "prefill_base_s": 0.01, "prefill_per_token_s": 1e-4, "prefill_attn_s": 0.0,
             "decode_base_s": 0.02, "decode_per_seq_s": 1e-3, "decode_per_ctx_token_s": 0.0}
            for r in range(2)]
    path = tmp_path / "p.json"
    for source in ([1, 2], "text", {"replicates": "many", "decode_cells": [3, None]},
                   {"replicates": [reps[0], {"rep": 1}]},  # a repeat without its fit
                   {"replicates": [reps[0], {**reps[1], "decode_base_s": -1.0}]},
                   {"replicates": [reps[0], ["not", "an", "object"]]}):
        path.write_text(json.dumps({**raw, "_source": source}))
        assert replicate_profiles(path) == []
        assert longest_prompt_measured(path) is None
    path.write_text(json.dumps({**raw, "_source": {"replicates": reps}}))
    assert [p.name for p in replicate_profiles(path)] == [f"{raw['name']}#repeat{r}"
                                                         for r in range(2)]
