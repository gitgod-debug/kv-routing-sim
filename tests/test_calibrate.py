"""`kvsim calibrate`, against the repo's mock vLLM — the one engine whose true timings are known.

The mock (scripts/phase3_dryrun.py) is a real HTTP server that schedules requests in real time with
the simulator's own vLLM model and a known profile: prefill 1 ms + 0.02 ms per token, decode
0.5 ms + 2.5 ms per sequence in the batch. So a correct calibration must recover those numbers, and
its cache check must come out exact — here the model IS the engine, as in the Phase 3 dry-run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import random
import threading
from http.server import ThreadingHTTPServer

import pytest

from kv_routing_sim.calib.calibrate import (
    COEFFICIENTS,
    CalibrationError,
    Engine,
    EngineInfo,
    calibrate,
    fit,
    fit_replicates,
    measure_prefill,
    ols,
    probe,
)
from kv_routing_sim.sim.cost_model import load_profile, replicate_profiles

REPO = pathlib.Path(__file__).resolve().parents[1]
QUIET = {"log": lambda *_: None}
SMALL = {"prefill_lengths": (64, 256, 1024, 2048), "reps": 2, "prefill_cached": (256, 1024),
         "prefill_new": (64, 512), "decode_batches": (1, 2, 4, 8), "decode_contexts": (32, 1024),
         "decode_tokens": 32, "check_sessions": 8, "check_turns": 3}
# The context terms a second mock is given — C2's T4 fit is 3.5e-6 and 7.6e-7; these are close.
CTX = {"MOCK_DECODE_PER_CTX_TOKEN_S": "4e-6", "MOCK_PREFILL_ATTN_S": "2e-8"}


def _serve(env: dict[str, str]):
    """A mock engine on a free port. `env` sets the mock's context terms while it is built."""
    spec = importlib.util.spec_from_file_location("phase3_dryrun",
                                                  REPO / "scripts" / "phase3_dryrun.py")
    dryrun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dryrun)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        engine = dryrun.MockEngine(num_blocks=4000, block_size=16, max_num_seqs=64)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    server = ThreadingHTTPServer(("127.0.0.1", 0), dryrun._handler(engine, False, 4000))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture(scope="module")
def mock_url():
    server = _serve({})
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def ctx_mock_url():
    server = _serve(CTX)
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
    # This mock has no context terms: the fit must not invent a large one.
    assert profile["decode_per_ctx_token_s"] < 0.1 * float(CTX["MOCK_DECODE_PER_CTX_TOKEN_S"])
    out = tmp_path / "mock.json"
    out.write_text(json.dumps(profile))
    loaded = load_profile(out)  # the file is a profile the simulator accepts as it is
    assert loaded.calibrated and loaded.kv_cache_blocks == 4000
    # Every timing cell was measured `reps` times, and the profile keeps the fit of each repeat —
    # what `kvsim replay` sizes on to show how firm a replica count is.
    src = profile["_source"]
    cells = {(c["batch"], c["context"]) for c in src["decode_cells"]}
    assert len(src["decode_cells"]) == SMALL["reps"] * len(cells)
    assert sorted({c["rep"] for c in src["decode_cells"]}) == list(range(SMALL["reps"]))
    assert [r["rep"] for r in src["replicates"]] == list(range(SMALL["reps"]))
    for rep in src["replicates"]:
        assert set(COEFFICIENTS) <= set(rep)
        # The mock is the same engine on every repeat: each one finds its per-sequence cost.
        assert rep["decode_per_seq_s"] == pytest.approx(2.5e-3, rel=0.3)
    reps = replicate_profiles(out)
    assert len(reps) == SMALL["reps"] and all(r.kv_cache_blocks == 4000 for r in reps)


def test_calibration_recovers_known_context_terms(ctx_mock_url):
    """C2's point: decode and prefill cost grow with context. A mock given known context terms must
    have them recovered — and the profile must carry them, or replicas would be sized optimistic."""
    profile, ok = calibrate(Engine(ctx_mock_url, poll_s=0.05), model="mock", **SMALL, **QUIET)
    assert ok and profile["calibration_error"] == 0.0
    assert profile["decode_per_ctx_token_s"] == pytest.approx(
        float(CTX["MOCK_DECODE_PER_CTX_TOKEN_S"]), rel=0.35)
    assert profile["prefill_attn_s"] == pytest.approx(float(CTX["MOCK_PREFILL_ATTN_S"]), rel=0.5)
    assert profile["decode_per_seq_s"] == pytest.approx(2.5e-3, rel=0.3)


def test_prefill_is_also_timed_after_a_cached_prefix(ctx_mock_url):
    """Agentic traffic is a long cached prefix and few new tokens — the case cold prompts never
    measure (scripts/c3_precheck.py). Each such cell must reach the fit with the cached tokens the
    ENGINE reported, not the ones the design hoped for."""
    profile, _ = calibrate(Engine(ctx_mock_url, poll_s=0.05), model="mock", **SMALL, **QUIET)
    points = profile["_source"]["prefill_points"]
    after = [(p["prompt_tokens"] - p["uncached_tokens"], p["uncached_tokens"]) for p in points
             if p["uncached_tokens"] < p["prompt_tokens"]]
    cells = {(c, n) for c in SMALL["prefill_cached"] for n in SMALL["prefill_new"]}
    assert set(after) == cells and len(after) == SMALL["reps"] * len(cells)
    cold = [p for p in points if p["uncached_tokens"] == p["prompt_tokens"]]
    assert len(cold) == SMALL["reps"] * len(SMALL["prefill_lengths"])
    assert profile["_source"]["notes"] == []  # this engine reports cached tokens: nothing assumed


def _synthetic(scale_by_rep: dict[int, float]) -> tuple[list[dict], list[dict]]:
    """Timing points an engine with known coefficients would give — each repeat's engine the true
    one scaled, so the fit of that repeat alone must come out scaled the same."""
    d0, d1, d2 = 0.04, 3e-4, 3.5e-6
    p0, p1, p2 = 0.0, 3e-4, 7.5e-7
    mid = (8 + 64) / 2
    prefill, decode = [], []
    for rep, x in scale_by_rep.items():
        for L in (128, 512, 1024):
            for b in (1, 2, 4, 8):
                decode.append({"rep": rep, "batch": b, "context": L, "valid": True,
                               "step_s": x * (d0 + d1 * b + d2 * b * (L + mid))})
        for c, n in ((0, 64), (0, 512), (0, 1024), (256, 64), (768, 256), (1280, 704)):
            ttft = x * (p0 + p1 * n + p2 * n * (c + n / 2) + d0 + d1 + d2 * (c + n))
            prefill.append({"rep": rep, "prompt_tokens": c + n, "uncached_tokens": n,
                            "ttft_s": ttft})
    return prefill, decode


def test_the_fit_of_each_repeat_alone_shows_how_much_the_engine_moved():
    """Three repeats; the engine is 10% slower in the second and 5% faster in the third. The pooled
    fit lands in between — and hides that. The per-repeat fits show it."""
    prefill, decode = _synthetic({0: 1.0, 1: 1.1, 2: 0.95})
    pooled = fit(prefill, decode, output_tokens=64)
    reps = fit_replicates(prefill, decode, output_tokens=64)
    assert [r["rep"] for r in reps] == [0, 1, 2]
    for r, x in zip(reps, (1.0, 1.1, 0.95), strict=True):
        assert r["decode_per_ctx_token_s"] == pytest.approx(3.5e-6 * x)
        assert r["decode_base_s"] == pytest.approx(0.04 * x)
        assert r["prefill_per_token_s"] == pytest.approx(3e-4 * x)
        assert r["prefill_attn_s"] == pytest.approx(7.5e-7 * x)
        assert (r["decode_cells"], r["prefill_points"]) == (12, 6)
    lo, hi = (f(r["decode_per_ctx_token_s"] for r in reps) for f in (min, max))
    assert lo < pooled["decode_per_ctx_token_s"] < hi


def test_one_repeat_has_nothing_to_compare_and_a_broken_one_is_left_out():
    prefill, decode = _synthetic({0: 1.0})
    assert fit_replicates(prefill, decode, output_tokens=64) == []  # one repeat: no spread
    # Points from before repeats were tagged (no "rep" key) are one repeat, not an error.
    untagged = ([{k: v for k, v in p.items() if k != "rep"} for p in prefill],
                [{k: v for k, v in c.items() if k != "rep"} for c in decode])
    assert fit_replicates(*untagged, output_tokens=64) == []
    # A repeat whose decode cells all preempted has no fit of its own: it is dropped, not guessed.
    prefill, decode = _synthetic({0: 1.0, 1: 1.1, 2: 0.95})
    for c in decode:
        if c["rep"] == 1:
            c["valid"] = False
    assert [r["rep"] for r in fit_replicates(prefill, decode, output_tokens=64)] == [0, 2]
    # …and if that leaves a single one, there is again nothing to compare.
    for c in decode:
        if c["rep"] == 2:
            c["valid"] = False
    assert fit_replicates(prefill, decode, output_tokens=64) == []


class _Silent(Engine):
    """Answers every completion at once and reports no per-request cached tokens. No network."""

    def __init__(self):
        super().__init__("http://silent", poll_s=0.0)
        self.sent = []

    def stream(self, model, ids, max_tokens):
        self.sent.append(len(ids))
        return {"t_send": 0.0, "t_first": 0.1, "t_last": 0.1, "t_chunks": [0.1], "error": None,
                "usage": {"prompt_tokens": len(ids), "completion_tokens": max_tokens}}


def test_cached_prefix_cells_that_do_not_fit_the_engine_are_skipped():
    engine, notes = _Silent(), []
    info = EngineInfo("http://silent", "m", "0", kv_cache_blocks=4000, block_size=16,
                      max_num_seqs=8, max_model_len=1024)
    points = measure_prefill(engine, info, random.Random(0), lengths=(64, 4096), reps=1,
                             token_range=(1000, 2000), cached=(250, 768, 1280), new=(64, 704),
                             notes=notes)
    # 4096 is cut to what the engine accepts; 250 becomes 240, whole blocks; 768 + 704 and both
    # 1280 cells exceed 1,024 tokens and are not sent.
    assert [(p["prompt_tokens"], p["uncached_tokens"]) for p in points] == [
        (64, 64), (1016, 1016), (304, 64), (944, 704), (832, 64)]
    assert max(engine.sent) <= 1016
    # No cached-token report from this engine: the designed prefix is assumed, and said so once.
    assert len(notes) == 1 and "assumed" in notes[0]


def test_the_command_lines_default_cells_are_the_librarys():
    """The default cells are named twice — the parser's strings and the library's constants. If
    they drift apart, the documented procedure and the one that runs are different procedures."""
    from kv_routing_sim.calib import calibrate as lib
    from kv_routing_sim.cli import _ints, build_parser

    ns = build_parser().parse_args(["calibrate", "--out", "x.json"])
    assert _ints(ns.prefill_lengths, "") == lib.DEFAULT_PREFILL_LENGTHS
    assert _ints(ns.prefill_cached, "") == lib.DEFAULT_PREFILL_CACHED
    assert _ints(ns.prefill_new, "") == lib.DEFAULT_PREFILL_NEW
    assert _ints(ns.decode_batches, "") == lib.DEFAULT_DECODE_BATCHES
    assert _ints(ns.decode_contexts, "") == lib.DEFAULT_DECODE_CONTEXTS


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
