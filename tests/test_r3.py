"""R3's plumbing: the grid runner on a made-up trace in the Qwen-Bailian format. Nothing here is
R3's registered design or says anything about its result."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import random
import subprocess
import sys

import pytest
import yaml

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run

REPO = pathlib.Path(__file__).resolve().parents[1]

TOY = CostProfile(
    name="toy", prefill_base_s=0.0, prefill_per_token_s=0.0005, decode_base_s=0.02,
    decode_per_seq_s=0.0005, kv_cache_blocks=2_000, block_size=16, max_num_seqs=32,
    calibrated=True, prefill_attn_s=1e-7, decode_per_ctx_token_s=1e-6,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r3 = _load("run_r3")


def _made_up_trace(path: pathlib.Path, n_sessions: int = 30, seed: int = 0) -> int:
    """Sessions of 1 to 6 turns behind two shared prompts, in the Bailian format."""
    rng = random.Random(seed)
    rows, chat, next_id = [], 0, 1000
    for s in range(n_sessions):
        t = s * 1.5 + rng.random()
        ids = [(s % 2) * 100 + j for j in range(8)]  # one of two 8-block "system prompts"
        parent = -1
        for turn in range(1, rng.randint(1, 6) + 1):
            ids = [*ids, *range(next_id, next_id + 4)]
            next_id += 4
            rows.append({"chat_id": chat, "parent_chat_id": parent, "timestamp": round(t, 3),
                         "input_length": len(ids) * 16 - 3, "output_length": 12, "type": "coder",
                         "turn": turn, "hash_ids": list(ids)})
            parent, chat = chat, chat + 1
            t += 2.0 + rng.random()
    rows.sort(key=lambda r: r["timestamp"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


@pytest.fixture
def design(tmp_path):
    n = _made_up_trace(tmp_path / "trace.jsonl")
    (tmp_path / "toy.json").write_text(json.dumps(dataclasses.asdict(TOY)))
    cfg = {
        "results_dir": str(tmp_path / "out"), "trace": str(tmp_path / "trace.jsonl"),
        "adapter_name": "bailian", "adapter": {"token_scale": 1.0, "max_total_tokens": 2048},
        "window_s": 10_000, "warmup_s": 5, "cache_model": "vllm", "replicas_max": 3,
        "seeds": [0, 1], "probe_speeds": [1, 4], "speeds": [1, 2],
        "slo": {"ttft_p95_ms": 2000, "tpot_p95_ms": 100},
        "calibrations": {"X": str(tmp_path / "toy.json")},
        "policies": [
            {"label": "round-robin", "policy": "round-robin"},
            {"label": "session-sticky", "policy": "session-sticky"},
            {"label": "radix-affinity-balanced@16", "policy": "radix-affinity-balanced",
             "kwargs": {"load_penalty": 16}},
            {"label": "lmetric", "policy": "lmetric"},
            {"label": "smetric-local", "policy": "smetric-local", "kwargs": {"ttft_slo_s": 2.0}},
        ],
    }
    (tmp_path / "r3.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path, cfg, n


def _run(tmp_path, *extra):
    # As it is really run — its own process — so its worker pool is not forked from pytest's.
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_r3.py"), "--config",
                           str(tmp_path / "r3.yaml"), "--workers", "2", *extra],
                          check=True, capture_output=True, text=True, cwd=REPO)


def test_the_probe_runs_round_robin_alone(design):
    tmp_path, _, _ = design
    out = _run(tmp_path, "--probe").stdout
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["probe"] is True
    assert probe["axes"]["policies"] == ["round-robin"] and probe["axes"]["seeds"] == [0]
    assert probe["axes"]["speeds"] == ["1", "4"] and probe["n_runs"] == 2 * 3
    assert set(probe["cells"]["X"]["4"]) == {"round-robin"}
    assert "round-robin at 1×" in out and "round-robin at 4×" in out
    assert not (tmp_path / "out" / "grid.json").exists()


def test_the_grid_runner_writes_every_run_and_decides_nothing(design):
    tmp_path, cfg, n_requests = design
    _run(tmp_path)
    grid = json.loads((tmp_path / "out" / "grid.json").read_text())
    labels = [p["label"] for p in cfg["policies"]]
    assert grid["axes"] == {"calibrations": ["X"], "speeds": ["1", "2"], "policies": labels,
                            "replicas": [1, 2, 3], "seeds": [0, 1]}
    assert grid["n_runs"] == 2 * 5 * 3 * 2 and grid["probe"] is False
    assert grid["adapter_stats"]["requests"] == n_requests
    assert len(grid["trace_sha256"]) == 64
    assert "verdict" not in json.dumps(grid) and "holds" not in json.dumps(grid)

    s = grid["structure"]
    assert s["requests"] == n_requests and s["sessions"] == 30
    assert s["requests_per_session"]["max"] <= 6
    # Two shared prompts and sessions that re-send their history: reuse of both kinds.
    assert 0 < s["reusable_from_the_same_session"] < s["reusable_from_any_earlier_request"] < 1

    for label in labels:
        cell = grid["cells"]["X"]["2"][label]
        assert [len(cell[m]) for m in ("ttft_p95_ms", "tpot_p95_ms", "hit_rate",
                                       "busiest_over_mean", "preemptions")] == [3] * 5
        assert all(len(row) == 2 for row in cell["ttft_p95_ms"]) and cell["all_completed"]
    # Only a policy that reports on itself has a `router` entry.
    assert set(grid["cells"]["X"]["1"]["smetric-local"]["router"]) == {"stuck", "migrated",
                                                                      "balanced"}
    assert "router" not in grid["cells"]["X"]["1"]["session-sticky"]

    # The stored numbers are the run's own: one cell, recomputed here.
    trace, _ = r3.read_trace(cfg)
    warm = sum(1 for q in trace if q.arrival_ts < cfg["warmup_s"])
    at2 = tuple(dataclasses.replace(q, arrival_ts=q.arrival_ts / 2.0) for q in trace)
    r = run(at2, policy="lmetric", n_replicas=2, seed=1, profile=TOY, cache_model="vllm",
            warmup_requests=warm)
    cell = grid["cells"]["X"]["2"]["lmetric"]
    assert cell["ttft_p95_ms"][1][1] == r.ttft_percentiles()["p95_ms"]
    assert cell["tpot_p95_ms"][1][1] == r.tpot_percentiles()["p95_ms"]


def test_the_runner_refuses_a_trace_that_is_not_the_registered_file(design):
    tmp_path, cfg, _ = design
    cfg["trace_sha256"] = "0" * 64
    (tmp_path / "r3.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(subprocess.CalledProcessError) as err:
        _run(tmp_path, "--probe")
    assert "not the registered file" in err.value.stderr


def test_structure_separates_a_sessions_own_reuse_from_reuse_across_sessions():
    from kv_routing_sim.traces.schema import NormalizedRequest as Req

    shared = (1, 2, 3, 4)
    trace = (
        Req("a1", 0.0, (*shared, 10, 11), 100, 4, "A"),
        Req("b1", 1.0, (*shared, 20, 21), 100, 4, "B"),          # 4 blocks, from A's session
        Req("a2", 2.0, (*shared, 10, 11, 12, 13), 132, 4, "A"),  # 6 blocks, all its own
    )
    s = r3.structure(trace, 16)
    assert (s["blocks"], s["distinct_blocks"], s["sessions"]) == (20, 10, 2)
    assert s["reusable_from_any_earlier_request"] == round(10 / 20, 4)
    assert s["reusable_from_the_same_session"] == round(6 / 20, 4)
