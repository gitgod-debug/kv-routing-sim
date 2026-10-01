"""C3's analysis (scripts/c3_analyze.py) — committed before the run, so it is tested before the run.

The end-to-end cases build a tiny "GPU run" in which the engine IS a known profile: the simulator
makes the measurements. A profile equal to that engine must then predict them exactly, a profile
three times too fast must fail, and a missing one must leave its rule undecided — never passed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib

import pytest

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c3_analyze", REPO / "scripts" / "c3_analyze.py")
c3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c3)

SPEC = {"n_sessions": 12, "turns_per_session": 3, "session_context_tokens": 150,
        "user_turn_tokens": 40, "assistant_turn_tokens": 60, "output_tokens": 16,
        "think_time_s": 1.0, "seed": 0, "prefix_sharing_ratio": 0.8, "arrival_rate": 2.0}
TRUTH = CostProfile(name="truth", prefill_base_s=0.0, prefill_per_token_s=3e-4, decode_base_s=0.04,
                    decode_per_seq_s=3e-4, kv_cache_blocks=1537, block_size=16, max_num_seqs=256,
                    prefill_attn_s=7.5e-7, decode_per_ctx_token_s=3.5e-6, calibrated=True)
CHECK = {"requests": 48, "queries": 13776, "hits": 9840}


def _profile_file(profile: CostProfile, *, equal: int = 48) -> dict:
    """What `kvsim calibrate` writes, as far as the analysis reads it."""
    fit = {"points": 15, "r2": 0.99}
    return {"_source": {"check": {"measured": {"hits": 9840.0, "queries": 13776.0},
                                  "cached_tokens_equal": equal, "cached_tokens_compared": 48},
                        "decode_fit": fit, "prefill_fit": fit, "notes": []},
            **dataclasses.asdict(profile)}


def _run(tmp_path: pathlib.Path, seed0: dict | None, *, holdout: bool = True):
    """A results directory and its registration, with TRUTH as the engine."""
    trace, _ = AgenticWorkload(**SPEC).generate()
    r = run(trace, policy="round-robin", n_replicas=1, seed=0, profile=TRUTH, cache_model="vllm",
            record_per_request=True)
    ttfts = sorted(v["ttft_ms"] for v in r.per_request.values())
    requests = [{"request_id": q.request_id, "valid": True, "send_offset_s": q.arrival_ts,
                 "ttft_ms": r.per_request[q.request_id]["ttft_ms"]} for q in trace]
    (tmp_path / "truth.json").write_text(json.dumps(dataclasses.asdict(TRUTH)))
    prereg = {
        "config": {"rules": {"c3b_max_factor": 2.0, "c3c_max_factor": 2.0,
                             "hit_rate_max_relative_error": 0.10},
                   "calibrate": {"seeds": [0, 1]}, "engine": {"block_size": 16},
                   "profile": str(tmp_path / "truth.json"),
                   "predictions": {"c3a": "holds", "c3b": "holds", "c3c": "holds"}},
        "predictions_sha256": "p" * 64, "trace_fingerprints_sha256": {"W": "f"},
        "wheel": {"sha256": "w" * 64, "built_from": "test"}, "workloads": {"W": SPEC},
        "predictions": {"check": {"0": CHECK, "1": CHECK},
                        "runs": {"W": {"C2 profile": {
                            "hit_rate": r.hit_rate, "preemptions": r.preemptions,
                            "ttft_p50_ms": c3.c2.nearest_rank(ttfts, 0.50),
                            "ttft_p95_ms": c3.c2.nearest_rank(ttfts, 0.95)}}}},
    }
    data = {"predictions_sha256": "p" * 64, "fingerprints": {"W": "f"}, "wheel_sha256": "w" * 64,
            "environment": {"date_utc": "2026-10-01T00:00:00+00:00"},
            "calibrate": [{"seed": 0, "returncode": 0, "wall_clock_s": 200.0}],
            "holdout": [{"workload": "W", "hit_rate": r.hit_rate, "preemptions": r.preemptions,
                         "invalid": 0, "worst_send_lag_ms": 0.0, "requests": requests}]
            if holdout else []}
    (tmp_path / "server_0.log").write_text("INFO GPU KV cache size: 24,592 tokens, …\n")
    if seed0 is not None:
        (tmp_path / "calibrated_profile_seed0.json").write_text(json.dumps(seed0))
    return data, prereg


def test_a_profile_that_is_the_engine_predicts_it_exactly(tmp_path):
    data, prereg = _run(tmp_path, _profile_file(TRUTH))
    a = c3.analyze(data, prereg, tmp_path)
    assert a["c3a"] == {"seed": 0, "decided": True, "holds": True}
    assert a["c3b"]["holds"] and a["c3c"]["holds"]
    row = a["holdout"][0]
    assert row["command_factor"] == pytest.approx({"p50": 1.0, "p95": 1.0})
    assert row["c2_factor"] == pytest.approx({"p50": 1.0, "p95": 1.0})
    assert a["commands"]["1"]["decided"] is False  # the second run of the command never happened
    report = c3.markdown(a, data, prereg)
    assert report.count("**HOLDS**") == 3 and "not run" in report


def test_a_profile_three_times_too_fast_fails_c3b_and_only_c3b(tmp_path):
    fast = dataclasses.replace(
        TRUTH, **{k: getattr(TRUTH, k) / 3 for k in (
            "prefill_per_token_s", "prefill_attn_s", "decode_base_s", "decode_per_seq_s",
            "decode_per_ctx_token_s")})
    data, prereg = _run(tmp_path, _profile_file(fast))
    a = c3.analyze(data, prereg, tmp_path)
    assert a["c3b"]["decided"] and not a["c3b"]["holds"]
    assert a["c3a"]["holds"] and a["c3c"]["holds"]
    assert min(a["holdout"][0]["command_factor"].values()) > 2.0


def test_one_check_request_off_fails_c3a_whatever_the_totals_say(tmp_path):
    data, prereg = _run(tmp_path, _profile_file(TRUTH, equal=47))
    a = c3.analyze(data, prereg, tmp_path)
    cond = a["commands"]["0"]["conditions"]
    assert cond["check counters equal the registered prediction"]
    assert not cond["every check request's cached tokens equal the model's"]
    assert a["c3a"]["decided"] and not a["c3a"]["holds"]


def test_missing_data_leaves_a_rule_undecided_never_passed(tmp_path):
    data, prereg = _run(tmp_path, None)  # the command wrote no profile
    a = c3.analyze(data, prereg, tmp_path)
    assert a["c3a"]["decided"] and not a["c3a"]["holds"]
    assert not a["c3b"]["decided"] and not a["c3b"]["holds"]
    assert a["c3c"]["holds"]  # C2's profile needs nothing from the command
    assert "UNDECIDED" in c3.markdown(a, data, prereg)

    data, prereg = _run(tmp_path, _profile_file(TRUTH), holdout=False)  # stopped before part 2
    a = c3.analyze(data, prereg, tmp_path)
    assert a["c3a"]["holds"]
    assert not a["c3b"]["decided"] and not a["c3c"]["decided"]
    assert not a["c3b"]["holds"] and not a["c3c"]["holds"]


def test_a_run_that_is_not_the_registered_one_is_refused(tmp_path):
    data, prereg = _run(tmp_path, _profile_file(TRUTH))
    for key in ("predictions_sha256", "wheel_sha256"):
        with pytest.raises(SystemExit, match="different"):
            c3.analyze({**data, key: "x"}, prereg, tmp_path)
    with pytest.raises(SystemExit, match="different traffic"):
        c3.analyze({**data, "fingerprints": {"W": "other"}}, prereg, tmp_path)


def test_capacity_is_read_from_the_engine_log(tmp_path):
    assert c3.log_blocks(tmp_path, 16) is None  # no log: C3a's capacity condition cannot hold
    (tmp_path / "server_0.log").write_text("x\nGPU KV cache size: 24,592 tokens\n")
    assert c3.log_blocks(tmp_path, 16) == 1537
    facts = c3.command_facts({"returncode": 0, "wall_clock_s": 1.0},
                             {**_profile_file(TRUTH), "kv_cache_blocks": 1500}, CHECK, 1537)
    assert not facts["conditions"]["capacity read equals the engine log"] and not facts["holds"]


def test_the_grid_helper_counts_what_the_readme_claims():
    """`--grid` compares a second calibration's replica table with the committed one. Its count of
    points where session-sticky is best or tied must be the README's 22 of 24 on that table."""
    committed = json.loads((REPO / "artifacts/results/h2b_c2/h2b_c2.json").read_text())
    req = committed["required_replicas"]
    assert sum(len(v) for v in req.values()) == 24
    assert c3._sticky_best(req, committed["not_met"]) == 22

