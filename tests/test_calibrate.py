"""`kvsim calibrate`, against the repo's mock vLLM — the one engine whose true timings are known.

The mock (scripts/phase3_dryrun.py) is a real HTTP server that schedules requests in real time with
the simulator's own vLLM model and a known profile: prefill 1 ms + 0.02 ms per token, decode
0.5 ms + 2.5 ms per sequence in the batch. So a correct calibration must recover those numbers, and
its cache check must come out exact — here the model IS the engine, as in the Phase 3 dry-run.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import threading
from http.server import ThreadingHTTPServer

import pytest

from kv_routing_sim.calib.calibrate import CalibrationError, Engine, calibrate, ols, probe
from kv_routing_sim.sim.cost_model import load_profile

REPO = pathlib.Path(__file__).resolve().parents[1]
QUIET = {"log": lambda *_: None}
SMALL = {"prefill_lengths": (64, 256, 1024, 2048), "reps": 2, "decode_batches": (1, 2, 4, 8),
         "decode_tokens": 32, "check_sessions": 8, "check_turns": 3}


@pytest.fixture(scope="module")
def mock_url():
    spec = importlib.util.spec_from_file_location("phase3_dryrun",
                                                  REPO / "scripts" / "phase3_dryrun.py")
    dryrun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dryrun)
    engine = dryrun.MockEngine(num_blocks=4000, block_size=16, max_num_seqs=64)
    server = ThreadingHTTPServer(("127.0.0.1", 0), dryrun._handler(engine, False, 4000))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_calibration_recovers_the_mock_engines_own_timings(mock_url, tmp_path):
    profile, ok = calibrate(Engine(mock_url, poll_s=0.05), model="mock", gpu="mock GPU",
                            **SMALL, **QUIET)
    assert ok and profile["calibrated"] is True
    assert profile["calibration_error"] == 0.0  # the model is the engine: exact, as in Phase 3
    check = profile["_source"]["check"]
    assert check["cached_tokens_equal"] == check["cached_tokens_compared"] == check["requests"]
    assert profile["kv_cache_blocks"] == 4000  # read from the engine, never estimated
    # The mock's true coefficients, within what HTTP and JSON overhead can add on a busy machine.
    assert profile["decode_per_seq_s"] == pytest.approx(2.5e-3, rel=0.3)
    assert profile["prefill_per_token_s"] == pytest.approx(2e-5, rel=0.5)
    out = tmp_path / "mock.json"
    out.write_text(json.dumps(profile))
    loaded = load_profile(out)  # the file is a profile the simulator accepts as it is
    assert loaded.calibrated and loaded.kv_cache_blocks == 4000


def test_a_wrong_block_size_is_caught_by_the_check(mock_url):
    """With 32-token blocks the model mispredicts almost every request's cached tokens, while the
    total hit rate moves by only a few percent — which is why every request is compared."""
    profile, ok = calibrate(Engine(mock_url, poll_s=0.05), model="mock", block_size=32,
                            **SMALL, **QUIET)
    assert not ok and profile["calibrated"] is False
    check = profile["_source"]["check"]
    assert check["cached_tokens_equal"] < check["cached_tokens_compared"]
    assert check["requests_that_differ"]


class _Canned(Engine):
    """An engine that answers from a dict of (method, path) -> (status, body). No network."""

    def __init__(self, answers):
        super().__init__("http://canned", poll_s=0.0)
        self.answers = answers

    def request(self, method, path, body=None, timeout=30.0):
        return self.answers.get((method, path.split("?")[0]), (404, '{"detail":"Not Found"}'))


def _answers(metrics: str) -> dict:
    return {
        ("GET", "/health"): (200, ""),
        ("GET", "/metrics"): (200, metrics),
        ("GET", "/v1/models"): (200, json.dumps({"data": [{"id": "Qwen/Qwen3-4B",
                                                           "max_model_len": 4096}]})),
        ("GET", "/version"): (200, json.dumps({"version": "0.30.0"})),
    }


PREFIX = ('vllm:prefix_cache_queries_total{engine="0"} 0.0\n'
          'vllm:prefix_cache_hits_total{engine="0"} 0.0\n')


def test_probe_reads_the_capacity_the_engine_reports():
    info_line = ('vllm:cache_config_info{block_size="16",cache_dtype="auto",'
                 'num_gpu_blocks="1537",enable_prefix_caching="True"} 1.0\n')
    info = probe(_Canned(_answers(PREFIX + info_line)))
    assert (info.model, info.version, info.kv_cache_blocks, info.block_size,
            info.max_model_len) == ("Qwen/Qwen3-4B", "0.30.0", 1537, 16, 4096)
    assert any("from vllm:cache_config_info" in n for n in info.notes)


def test_probe_never_guesses_the_capacity():
    with pytest.raises(CalibrationError, match="--kv-blocks"):
        probe(_Canned(_answers(PREFIX)))
    assert probe(_Canned(_answers(PREFIX)), kv_blocks=900).kv_cache_blocks == 900


def test_probe_refuses_an_engine_without_prefix_caching():
    with pytest.raises(CalibrationError, match="enable-prefix-caching"):
        probe(_Canned(_answers('vllm:num_requests_running{engine="0"} 0.0\n')))


def test_a_line_needs_two_distinct_points():
    assert ols([1.0, 2.0, 4.0], [3.0, 5.0, 9.0])["slope"] == pytest.approx(2.0)
    with pytest.raises(CalibrationError, match="two distinct points"):
        ols([4.0, 4.0], [1.0, 2.0])


PHASE3 = REPO / "artifacts" / "results" / "phase3_kaggle_t4"


def _real(metrics: str) -> dict:
    return {**_answers(metrics),
            ("GET", "/server_info"): (200, (PHASE3 / "server_info.json").read_text())}


def test_probe_reads_what_a_real_vllm_0_30_served():
    """Parsed from what vLLM 0.30.0 actually served on the T4 in Phase 3 — not from the mock."""
    metrics = (PHASE3 / "metrics_W2_short_r1_before.txt").read_text()
    info = probe(_Canned(_real(metrics)))
    assert (info.kv_cache_blocks, info.block_size, info.max_num_seqs) == (512, 16, 256)
    assert any("from vllm:cache_config_info" in n for n in info.notes)
    # Without that gauge, the same number comes from /server_info.
    stripped = "\n".join(ln for ln in metrics.splitlines() if "cache_config_info" not in ln)
    info = probe(_Canned(_real(stripped)))
    assert info.kv_cache_blocks == 512
    assert any("from GET /server_info" in n for n in info.notes)
