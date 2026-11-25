"""Tests for L1's own code (scripts/run_l1.py) on a hand-made trace in the AgentX format.

Nothing here says anything about L1's result: the sessions, the counts and the rows are made up.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
B = 64


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _request(t: float, ids: list[int], tail: int = 5, out: int = 70, model: str = "m") -> dict:
    return {"t": t, "type": "n", "model": model, "in": len(ids) * B + tail, "out": out,
            "hash_ids": ids, "api_time": 2.0}


def _sessions() -> list[dict]:
    """One session that grows, is cut back and grows again, with a sub-agent; one that repeats."""
    return [
        {"id": "a", "models": ["m"], "requests": [
            _request(0.0, [1, 2, 3]),
            _request(10.0, [1, 2, 3, 4]),
            _request(20.0, [1, 2, 9]),                     # an edited context: only two still match
            _request(25.0, [1, 2, 3, 4, 5], model="x"),    # another model: not this fleet's
            _request(30.0, [1, 2, 9, 4]),                  # block 4 again, after another block 9
            {"t": 40.0, "type": "subagent", "agent_id": "s", "requests": [
                _request(0.0, [1, 2]), _request(5.0, [1, 2, 7])]},
        ]},
        {"id": "b", "models": ["m"], "requests": [
            _request(0.0, [1, 2], tail=0), _request(5.0, [1, 2], tail=0)]},
    ]


@pytest.fixture
def l1(tmp_path):
    """The script's module, loaded with the made-up trace as its registered one."""
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in _sessions()))
    mod = _load("run_l1")
    cfg = yaml.safe_load((REPO / "configs" / "l1_agentx_full.yaml").read_text())
    cfg.update(trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               never_forgets_blocks=1000, capacity_tokens=[64 * B, 128 * B], replicas=2,
               policies={"round-robin": [0], "session-sticky": [0, 1], "radix-affinity": [0, 1]},
               bootstrap_resamples=200)
    cfg["adapter"] = {**cfg["adapter"], "concurrency": 2, "stagger_s": 0.0}
    mod.facts = mod.load(cfg)
    mod.cfg = cfg
    return mod


def test_the_raw_count_follows_the_chain_and_stops_before_the_last_token(l1):
    rows = l1.raw_count(pathlib.Path(l1.cfg["trace"]), B)["servable"]
    assert [c for _, c in rows["a"]] == [0, 3 * B, 2 * B, 3 * B]  # 4 after another 9 is new
    assert [c for _, c in rows["a/s"]] == [0, 2 * B]  # a sub-agent has a cache scope of its own
    assert rows["b"] == [(2 * B, 0), (2 * B, B)]  # all of it was sent before; one token must run


def test_the_trace_and_the_raw_count_describe_the_same_requests(l1):
    assert l1.facts["requests"] == 8 and l1.facts["sessions"] == 2
    assert l1.facts["sessions_with_sub_agents"] == 3
    assert l1.facts["other_model_requests_left_out"] == {"other_model": 1}
    assert l1.facts["servable_tokens"] == sum(l1.SERVABLE) == (3 + 2 + 3 + 2 + 1) * B


def test_another_file_than_the_registered_one_is_refused(l1, tmp_path):
    pathlib.Path(l1.cfg["trace"]).write_text("{}\n")
    with pytest.raises(SystemExit, match="not the registered file"):
        l1.load(l1.cfg)


def test_a_replica_that_never_evicts_serves_exactly_the_raw_count(l1):
    e = l1.never_forgets()
    assert (e["requests_equal"], e["requests"], e["evictions"]) == (8, 8, 0)
    assert e["hit_tokens"] == e["servable_tokens"]


def test_round_robin_computes_again_what_the_other_replica_holds(l1):
    rr = l1.grid_run(("round-robin", 128 * B, 0))
    sticky = l1.grid_run(("session-sticky", 128 * B, 0))
    assert sticky["re_prefilled_tokens"] == 0 and sticky["hit_tokens"] == sum(l1.SERVABLE)
    assert rr["re_prefilled_tokens"] == sum(l1.SERVABLE) - rr["hit_tokens"] > 0
    assert sum(rr["per_session"].values()) == rr["re_prefilled_tokens"]


def _rows(sticky: float, rr: float, affinity: float, seconds: float = 60.0) -> list[dict]:
    caps = [1, 2]
    return [{"policy": p, "capacity_tokens": c, "re_prefilled_share": v, "seconds_max": seconds}
            for c in caps for p, v in (("round-robin", rr), ("session-sticky", sticky),
                                       ("radix-affinity", affinity))]


def test_the_four_predictions_are_scored_by_their_registered_rules(l1):
    cfg = {**l1.cfg, "capacity_tokens": [1, 2],
           "rules": {"l1b_max_minutes": 10, "l1c_min_ratio": 4, "l1d_max_points": 2}}
    good = {"requests": 8, "requests_equal": 8, "hit_tokens": 100, "servable_tokens": 100}
    scored = l1.verdicts(cfg, good, _rows(0.05, 0.20, 0.06))
    assert [scored[k]["holds"] for k in ("L1a", "L1b", "L1c", "L1d")] == [True] * 4
    bad = {**good, "requests_equal": 7}
    scored = l1.verdicts(cfg, bad, _rows(0.05, 0.19, 0.0701, seconds=601.0))
    assert [scored[k]["holds"] for k in ("L1a", "L1b", "L1c", "L1d")] == [False] * 4


def test_the_summary_has_an_interval_and_a_paired_one_against_round_robin(l1):
    keys = [(p, c, s) for c in l1.cfg["capacity_tokens"] for p, seeds in l1.cfg["policies"].items()
            for s in seeds]
    sent = {"a": 1000, "b": 500}
    rows = l1.summarise(l1.cfg, [l1.grid_run(k) for k in keys], sent)
    assert [(r["policy"], r["seeds"]) for r in rows[:3]] == [
        ("round-robin", 1), ("session-sticky", 2), ("radix-affinity", 2)]
    assert rows[0]["minus_round_robin_ci"]["point"] == 0.0
    assert rows[1]["minus_round_robin_ci"]["point"] < 0.0
    assert rows[1]["re_prefilled_ci"]["n_resamples"] == 200
