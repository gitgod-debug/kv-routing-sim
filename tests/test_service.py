"""The replay service shared by the command line and the browser page."""

from __future__ import annotations

import json

import pytest

from kv_routing_sim.cli import main
from kv_routing_sim.service import ConfigError, ReplayOptions, replay
from kv_routing_sim.traces.generator import AgenticWorkload

PROFILE = "t4-qwen3-4b-vllm0.30.0-ctx"
T4_PRICE_USD = 0.526


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    path = tmp_path_factory.mktemp("service") / "t.jsonl"
    AgenticWorkload(n_sessions=20, turns_per_session=3, arrival_rate=0.5,
                    seed=1).write_jsonl(str(path))
    return path


def _options(**overrides) -> ReplayOptions:
    base = {"policies": "round-robin,session-sticky", "replicas": "1-3", "seeds": 2,
            "profile": PROFILE, "cost_model": "default"}
    return ReplayOptions(**{**base, **overrides})


def test_the_service_and_the_command_line_produce_the_same_report(trace, tmp_path):
    report = replay(trace, _options())
    out = tmp_path / "r.json"
    assert main(["replay", str(trace), "--policies", "round-robin,session-sticky",
                 "--replicas", "1-3", "--seeds", "2", "--profile", PROFILE,
                 "--cost-model", "default", "--format", "json", "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(json.dumps(report.as_dict()))


def test_progress_is_reported_after_every_run_of_every_stage(trace):
    calls = []
    report = replay(trace, _options(), progress=lambda *c: calls.append(c))
    sweep_calls = [c for c in calls if c[0] == "sweep"]
    assert sweep_calls == [("sweep", done, 12) for done in range(13)]
    if report.edge is not None:
        assert calls[-1] == ("edge", 2, 2)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"replicas": "two"}, "could not parse replicas"),
    ({"replicas": "0-2"}, "positive integers"),
    ({"policies": "nope"}, "unknown policies"),
    ({"seeds": 0}, "seeds must be at least 1"),
    ({"block_size": 32}, "does not match the profile"),
    ({"trace_format": "csv"}, "unknown trace format"),
    ({"profile": "no-such-profile"}, "could not read profile"),
    ({"ttft_slo_ms": 0}, "first-token limit"),
    ({"gpu_price_per_hour_usd": -1}, "GPU price"),
])
def test_invalid_options_raise_a_config_error(trace, overrides, message):
    with pytest.raises(ConfigError, match=message):
        replay(trace, _options(**overrides))


def test_a_missing_trace_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="could not read trace"):
        replay(tmp_path / "missing.jsonl", _options())


def test_the_gpu_price_scales_the_cost_and_nothing_else(trace):
    base = replay(trace, _options(edge_check=False))
    doubled = replay(trace, _options(edge_check=False, gpu_price_per_hour_usd=2 * T4_PRICE_USD))

    def without_cost(point):
        return {k: v for k, v in point.items() if k != "inr_per_1k_requests"}

    for a, b in zip(base.points, doubled.points, strict=True):
        assert without_cost(a) == without_cost(b)
        if a["inr_per_1k_requests"] is not None:
            assert b["inr_per_1k_requests"] == pytest.approx(2 * a["inr_per_1k_requests"],
                                                             rel=1e-3)


def test_a_first_token_limit_replaces_the_cost_models(trace):
    tight = replay(trace, _options(edge_check=False, tpot_slo_ms=0, ttft_slo_ms=1))
    loose = replay(trace, _options(edge_check=False, tpot_slo_ms=0, ttft_slo_ms=600_000))
    assert not any(p["meets_slo"] for p in tight.points)
    assert all(p["meets_slo"] for p in loose.points)
    assert "set with `--ttft-slo-ms`" in tight.slo
