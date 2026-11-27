"""The replay service shared by the command line and the browser page."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys

import pytest

from kv_routing_sim import browser
from kv_routing_sim.cli import main
from kv_routing_sim.report import recommendation_notes
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
    ({"policies": ""}, "choose at least one routing policy"),
    ({"policies": " , "}, "choose at least one routing policy"),
    ({"replicas": "1-99999999999999"}, "99,999,999,999,999 replica counts. Ask for at most 1,024"),
    ({"replicas": ",".join(["1"] * 1025)}, "Ask for at most 1,024"),
    ({"ttft_slo_ms": float("nan")}, "first-token limit"),
    ({"ttft_slo_ms": float("inf")}, "first-token limit"),
    ({"gpu_price_per_hour_usd": float("nan")}, "GPU price"),
    ({"tpot_slo_ms": -1}, "per-token limit"),
    ({"tpot_slo_ms": float("nan")}, "per-token limit"),
    # The limits are checked without a cost model too, where they would have no effect.
    ({"cost_model": None, "ttft_slo_ms": float("nan")}, "first-token limit"),
    ({"cost_model": None, "gpu_price_per_hour_usd": 0}, "GPU price"),
    ({"cost_model": None, "tpot_slo_ms": float("inf")}, "per-token limit"),
    ({"token_scale": 0.5}, "works only with the agentx, bailian and mooncake formats"),
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


def _long_trace(tmp_path, prompt=30_000, output=100):
    """Three requests. The last needs 30,100 tokens, and the shipped T4 profile holds 24,576."""
    rows = [{"request_id": f"r{i}", "timestamp": i, "prompt_token_ids": list(range(n)),
             "num_generated_tokens": out} for i, (n, out) in
            enumerate([(64, 8), (128, 8), (prompt, output)])]
    path = tmp_path / "long.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_a_request_too_long_for_the_profile_is_refused_before_the_sweep(tmp_path):
    with pytest.raises(ConfigError) as caught:
        replay(_long_trace(tmp_path), _options())
    assert str(caught.value) == (
        "1 of 3 requests needs more KV cache than one replica of this profile holds. The longest "
        "needs 30,100 tokens, and a replica holds 24,576. vLLM would not start with a context "
        "this long on this GPU. Use a profile measured with more KV cache.")
    # The same request from a reader that can scale tokens points to --token-scale as well.
    rows = [{"timestamp": 0, "input_length": 30_000, "output_length": 100,
             "hash_ids": list(range(59))}]
    moon = tmp_path / "m.jsonl"
    moon.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ConfigError, match=r"or read the trace with a smaller --token-scale\.$"):
        replay(moon, _options(trace_format="mooncake"))
    # At 1/16 it fits: 1,875 + 6 tokens.
    report = replay(moon, _options(trace_format="mooncake", token_scale=1 / 16, replicas="1",
                                   seeds=1, edge_check=False))
    assert report.trace["scaling"] == {"token_scale": 0.0625}


def test_a_too_long_request_is_an_error_line_on_the_page_and_the_command_line(tmp_path):
    long = _long_trace(tmp_path)
    page = browser.run({"traffic": {"source": "file", "path": str(long), "format": "vllm-jsonl"},
                        "profile": {"source": "shipped"}, "policies": "round-robin",
                        "replicas": "1", "seeds": 1}, work_dir=tmp_path)
    assert page["error"].startswith("1 of 3 requests needs more KV cache")
    done = subprocess.run([sys.executable, "-m", "kv_routing_sim.cli", "replay", str(long),
                           "--profile", PROFILE, "--replicas", "1", "--seeds", "1"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 1
    assert done.stderr.startswith("error: 1 of 3 requests needs more KV cache")
    assert "Traceback" not in done.stderr


def test_a_compressed_trace_that_ends_early_is_a_config_error(trace, tmp_path):
    whole = gzip.compress(trace.read_bytes())
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(ConfigError, match=r"could not read trace .*cut\.jsonl\.gz'?: the file ends "
                                          "early"):
        replay(cut, _options())


def test_requests_with_one_id_are_renamed_and_the_replay_completes(tmp_path):
    rows = [{"request_id": "same", "timestamp": i * 0.5, "prompt_token_ids": list(range(64)),
             "num_generated_tokens": 8} for i in range(6)]
    path = tmp_path / "dup.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = replay(path, _options(replicas="1-2", seeds=1, edge_check=False))
    assert report.trace["renamed_duplicate_ids"] == 5
    assert report.trace["requests"] == 6 and len(report.points) == 4


def test_an_agentx_replay_says_its_tokens_were_scaled(tmp_path):
    session = {"id": "s", "models": ["m"], "requests": [
        {"t": float(i), "type": "n", "model": "m", "in": 1024 + 64 * i, "out": 64,
         "hash_ids": list(range(16 + i)), "api_time": 1.0} for i in range(4)]}
    path = tmp_path / "agentx.jsonl"
    path.write_text(json.dumps(session) + "\n")
    report = replay(path, _options(trace_format="agentx", replicas="1", seeds=1,
                                   edge_check=False))
    assert report.trace["scaling"] == {"token_scale": 0.03125, "concurrency": 32,
                                       "stagger_s": 600.0}
    (note,) = [n for n in recommendation_notes(report.as_dict()) if "scaled" in n.text]
    assert note.warning
    assert note.text == (
        "Token counts were scaled by 1/32 when the trace was read: the AgentX reader's default, "
        "so that it fits a T4. With 16-token blocks the largest scale is 1/4 (--token-scale "
        "0.25). The trace at its recorded size needs a profile with 64-token blocks.")
    larger = replay(path, _options(trace_format="agentx", replicas="1", seeds=1,
                                   edge_check=False, token_scale=0.25))
    assert larger.trace["scaling"]["token_scale"] == 0.25
    with pytest.raises(ConfigError, match="does not divide a 16-token block"):
        replay(path, _options(trace_format="agentx", token_scale=1.0))
