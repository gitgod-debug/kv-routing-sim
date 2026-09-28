"""Trace ingestion: the schema contract, dirty input, and the generator's honesty about its knob."""

from __future__ import annotations

import itertools
import json

import pytest

from kv_routing_sim.traces import adapters
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.schema import SCHEMA_VERSION, NormalizedRequest, TraceStats


def test_schema_rejects_impossible_records():
    with pytest.raises(ValueError):
        NormalizedRequest("r", -1.0, (), 10, 10)
    with pytest.raises(ValueError):
        NormalizedRequest("r", 0.0, (), -1, 10)


def test_normalized_request_carries_no_prompt_text():
    """ADR-002: hashes only. A user's prompts must never enter the simulator or an artifact."""
    fields = set(NormalizedRequest.__slots__)
    assert not fields & {"prompt", "text", "token_ids", "prompt_token_ids"}
    assert "block_hashes" in fields


def test_malformed_lines_are_skipped_and_counted(tmp_path):
    """A run that dropped 12% of its input is a different run, and must say so."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {"request_id": "a", "timestamp": 1.0, "prompt_token_ids": list(range(32))}
                ),
                "{ broken json",
                json.dumps([1, 2, 3]),
                json.dumps({"request_id": "no-timestamp", "prompt_token_ids": [1]}),
                "",
                json.dumps(
                    {"request_id": "b", "timestamp": 2.0, "prompt_token_ids": list(range(32))}
                ),
            ]
        )
    )
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(p, stats=stats))
    assert len(out) == 2
    assert stats.requests == 2
    assert stats.skipped_lines == 3
    assert set(stats.skip_reasons) == {
        "json_decode_error",
        "not_an_object",
        "missing_or_bad_fields",
    }
    assert stats.as_dict()["schema_version"] == SCHEMA_VERSION


def test_timestamps_are_normalized_to_trace_start(tmp_path):
    """Absolute epochs differ between logs and are not comparable; elapsed time is."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"r{i}",
                    "timestamp": 1_700_000_000 + i,
                    "prompt_token_ids": [1] * 32,
                }
            )
            for i in range(3)
        )
    )
    out = list(adapters.get("vllm-jsonl").read(p))
    assert [r.arrival_ts for r in out] == [0.0, 1.0, 2.0]


def test_openai_adapter_preserves_message_order(tmp_path):
    """Reordering messages would destroy the shared prefix the whole simulation is about."""
    p = tmp_path / "t.jsonl"
    common = [{"role": "system", "content": "you are a tool-using agent " * 20}]
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"r{i}",
                    "created": i,
                    "messages": common + [{"role": "user", "content": f"question {i}"}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 10},
                }
            )
            for i in range(2)
        )
    )
    a, b = adapters.get("openai-log").read(p)
    shared = sum(1 for x, y in zip(a.block_hashes, b.block_hashes, strict=False) if x == y)
    assert shared > 0  # the identical system message produces identical leading blocks


def test_unknown_format_fails_loudly():
    with pytest.raises(ValueError, match="unknown trace format"):
        adapters.get("parquet")


def test_generator_reports_what_it_achieved_not_what_was_asked():
    """The knob is a target. Reporting it as the outcome would be a quiet substitution."""
    for requested in (0.0, 0.3, 0.6, 0.85):
        wl = AgenticWorkload(n_sessions=60, prefix_sharing_ratio=requested, seed=0)
        _, report = wl.generate()
        assert report["requested_prefix_sharing_ratio"] == requested
        assert abs(report["achieved_cross_session_sharing"] - requested) < 0.03


def test_zero_cross_session_sharing_still_has_within_session_reuse():
    """The two numbers answer different questions and must not be collapsed into one."""
    _, report = AgenticWorkload(n_sessions=40, prefix_sharing_ratio=0.0, seed=0).generate()
    assert report["achieved_cross_session_sharing"] == 0.0
    assert report["achieved_total_block_reuse"] > 0.5


def test_turn_n_is_a_prefix_of_turn_n_plus_one():
    """The defining structure of agentic traffic."""
    trace, _ = AgenticWorkload(n_sessions=3, turns_per_session=4, seed=0).generate()
    s0 = [r for r in trace if r.session_id == "s0"]
    for a, b in itertools.pairwise(s0):
        assert b.block_hashes[: len(a.block_hashes)] == a.block_hashes


def test_generator_rejects_impossible_settings():
    with pytest.raises(ValueError, match="prefix_sharing_ratio"):
        AgenticWorkload(prefix_sharing_ratio=1.0)
    with pytest.raises(ValueError, match="arrival_rate"):
        AgenticWorkload(arrival_rate=0.0)


def test_written_trace_round_trips_with_real_token_ids(tmp_path):
    """Placeholder ids would make every prompt share a prefix and report a fake ~99% hit rate."""
    out = tmp_path / "t.jsonl"
    wl = AgenticWorkload(n_sessions=20, prefix_sharing_ratio=0.6, seed=0)
    report = wl.write_jsonl(str(out))
    reread = list(adapters.get("vllm-jsonl").read(out))
    in_memory, _ = wl.generate()
    assert len(reread) == len(in_memory) == report["requests"]
    assert [r.block_hashes for r in reread] == [r.block_hashes for r in in_memory]


def test_one_shared_prefix_is_the_trace_every_earlier_run_used():
    """K = 1 must draw nothing new: Phase 3's fingerprint and Phase 4's tables depend on it."""
    kw = {"n_sessions": 30, "turns_per_session": 3, "prefix_sharing_ratio": 0.6, "seed": 4}
    before, report = AgenticWorkload(**kw).generate()
    same, report_k1 = AgenticWorkload(**kw, n_shared_prefixes=1).generate()
    assert [(r.request_id, r.arrival_ts, r.block_hashes) for r in before] == [
        (r.request_id, r.arrival_ts, r.block_hashes) for r in same
    ]
    assert report == report_k1  # no new keys for K = 1, so old reports are unchanged too


def test_sessions_share_a_prefix_only_with_sessions_that_drew_the_same_one():
    """K prompts, one per tenant: prefix k's blocks are shared by k's sessions and nobody else."""
    wl = AgenticWorkload(n_sessions=40, turns_per_session=1, prefix_sharing_ratio=0.6,
                         n_shared_prefixes=4, seed=0)
    trace, report = wl.generate()
    drawn = wl.prefix_assignment()
    prefix_blocks = report["shared_prefix_blocks"]
    first = {r.session_id: r.block_hashes[:prefix_blocks] for r in trace}
    for a, b in itertools.combinations(range(wl.n_sessions), 2):
        same_prefix = first[f"s{a}"] == first[f"s{b}"]
        assert same_prefix == (drawn[a] == drawn[b])
    assert report["distinct_prefixes_used"] == len(set(drawn)) > 1
    assert report["shared_working_set_blocks"] == len(set(drawn)) * prefix_blocks


def test_generator_rejects_impossible_prefix_counts():
    with pytest.raises(ValueError, match="n_shared_prefixes"):
        AgenticWorkload(n_shared_prefixes=0)
    with pytest.raises(ValueError, match="n_shared_prefixes"):
        AgenticWorkload(n_shared_prefixes=101)
    with pytest.raises(ValueError, match="prefix_zipf_s"):
        AgenticWorkload(n_shared_prefixes=4, prefix_zipf_s=-1.0)


def _mooncake(tmp_path, rows):
    p = tmp_path / "m.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_mooncake_blocks_share_exactly_at_the_traces_granularity(tmp_path):
    """A 512-token trace block becomes whole 16-token blocks; equal trace ids hash equal, and a
    partial trace block contributes only its full 16-token blocks."""
    p = _mooncake(tmp_path, [
        {"timestamp": 0, "input_length": 1024, "output_length": 8, "hash_ids": [7, 8]},
        {"timestamp": 500, "input_length": 1100, "output_length": 8, "hash_ids": [7, 9, 10]},
    ])
    a, b = adapters.get("mooncake").read(p, infer_sessions=False)
    assert len(a.block_hashes) == 64  # 2 trace blocks x 32
    assert len(b.block_hashes) == 64 + 76 // 16  # 2 full blocks + the 76-token tail's 4 blocks
    assert a.block_hashes[:32] == b.block_hashes[:32]  # trace block 7, shared
    assert a.block_hashes[32:64] != b.block_hashes[32:64]  # 8 vs 9: not shared
    assert b.arrival_ts == 0.5 and b.session_id is None


def test_mooncake_token_scale_keeps_structure_and_shrinks_sizes(tmp_path):
    p = _mooncake(tmp_path, [
        {"timestamp": 0, "input_length": 1024, "output_length": 160, "hash_ids": [1, 2]},
    ])
    (r,) = adapters.get("mooncake").read(p, token_scale=1 / 16, infer_sessions=False)
    assert r.prompt_tokens == 64 and r.output_tokens == 10
    assert len(r.block_hashes) == 4  # 2 trace blocks x (512 / 16 / 16)
    with pytest.raises(ValueError, match="whole number"):
        list(adapters.get("mooncake").read(p, token_scale=1 / 3))


def test_mooncake_sessions_follow_histories_but_never_a_shared_prompt(tmp_path):
    """A turn that re-sends a request's full blocks joins its session. Requests that only share a
    hot system prompt — one many different blocks follow — stay separate."""
    # five users behind shared prompt 0: its fan-out is 5
    rows = [{"timestamp": i, "input_length": 1536, "output_length": 1,
             "hash_ids": [0, 100 + i, 200 + i]} for i in range(5)]
    rows.append({"timestamp": 9, "input_length": 2048, "output_length": 1,
                 "hash_ids": [0, 100, 200, 300]})  # user 0's next turn
    st = TraceStats()
    out = list(adapters.get("mooncake").read(_mooncake(tmp_path, rows), hot_fanout=3, stats=st))
    assert len({r.session_id for r in out[:5]}) == 5  # the shared prompt merged no one
    assert out[5].session_id == out[0].session_id  # the continuation joined user 0
    assert st.requests == 6


def test_mooncake_bad_lines_are_skipped_and_counted(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"timestamp": 0, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]}\n'
        "not json\n"
        '{"timestamp": 1, "input_length": 1024, "output_length": 1, "hash_ids": [1]}\n'
        '{"timestamp": 2, "input_length": 1024, "output_length": 1}\n'
    )
    st = TraceStats()
    out = list(adapters.get("mooncake").read(p, stats=st))
    assert len(out) == 1 and st.skipped_lines == 3
    assert st.skip_reasons == {"json_decode_error": 1, "block_count_mismatch": 1,
                               "missing_or_bad_fields": 1}


def _agentx(tmp_path, bad_count=False):
    """Two tiny AgentX sessions; the second reuses the first's LOCAL ids on purpose."""
    s1 = {"id": "t1", "models": ["main-model"], "block_size": 64, "hash_id_scope": "local",
          "requests": [
              {"t": 0.0, "type": "n", "model": "main-model", "in": 1024, "out": 64,
               "hash_ids": list(range(1, 17)), "api_time": 5.0},
              {"t": 10.0, "type": "n", "model": "main-model", "in": 2100, "out": 64,
               "hash_ids": list(range(1, 33 - int(bad_count))), "api_time": 5.0},
              {"t": 12.0, "type": "s", "model": "side-model", "in": 640, "out": 8,
               "hash_ids": list(range(100, 110)), "api_time": 1.0},
              {"t": 20.0, "type": "subagent", "agent_id": "a1", "requests": [
                  {"t": 1.0, "type": "n", "model": "main-model", "in": 1024, "out": 8,
                   "hash_ids": list(range(500, 516)), "api_time": 2.0}]},
          ]}
    s2 = {"id": "t2", "models": ["main-model"], "block_size": 64, "hash_id_scope": "local",
          "requests": [{"t": 0.0, "type": "n", "model": "main-model", "in": 1024, "out": 64,
                        "hash_ids": list(range(1, 17)), "api_time": 3.0}]}
    p = tmp_path / "agentx.jsonl"
    p.write_text(json.dumps(s1) + "\n" + json.dumps(s2) + "\n")
    return p


def test_agentx_hash_scope_is_local_and_blocks_group_at_the_scale(tmp_path):
    """At 1/64, sixteen 64-token trace blocks make one 16-token block. Equal ids share within a
    session; the same ids in ANOTHER session must not — the dataset's ids are session-local."""
    reqs = list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 64,
                                            concurrency=2, stagger_s=0.0))
    t1 = [r for r in reqs if r.session_id == "t1"]
    (t2,) = [r for r in reqs if r.session_id == "t2"]
    assert [len(r.block_hashes) for r in t1] == [1, 2]
    assert t1[0].block_hashes[0] == t1[1].block_hashes[0]  # shared within the session
    assert t2.block_hashes[0] != t1[0].block_hashes[0]  # the same local ids: NOT shared
    assert (t1[0].prompt_tokens, t1[0].output_tokens) == (16, 1)


def test_agentx_keeps_one_model_and_subagents_as_their_own_sessions(tmp_path):
    stats = TraceStats()
    reqs = list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 64,
                                            concurrency=2, stagger_s=0.0, stats=stats))
    assert stats.skip_reasons == {"other_model": 1}  # the side-model request
    (sub,) = [r for r in reqs if r.session_id == "t1/a1"]
    assert sub.arrival_ts == 21.0  # the sub-agent's start (20) + its request's own t (1)
    assert [r.arrival_ts for r in reqs] == sorted(r.arrival_ts for r in reqs)
    kept = list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 64,
                                            concurrency=2, stagger_s=0.0, models="all"))
    assert len(kept) == len(reqs) + 1


def test_agentx_one_slot_plays_sessions_back_to_back(tmp_path):
    """With one slot, the second session starts when the first would have finished (its last
    request's t + api_time; session t1's is its sub-agent's, 20 + 1 + 2 = 23)."""
    reqs = list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 64,
                                            concurrency=1, stagger_s=0.0))
    first = reqs[0].session_id.split("/")[0]
    later = [r for r in reqs if not r.session_id.startswith(first)]
    assert min(r.arrival_ts for r in later) == {"t1": 23.0, "t2": 3.0}[first]


def test_agentx_window_and_malformed_requests(tmp_path):
    stats = TraceStats()
    reqs = list(adapters.get("agentx").read(_agentx(tmp_path, bad_count=True), token_scale=1 / 64,
                                            concurrency=2, stagger_s=0.0, window_s=11.0,
                                            stats=stats))
    assert all(r.arrival_ts < 11.0 for r in reqs)  # the sub-agent, at 21, is outside the window
    assert stats.skip_reasons == {"block_count_mismatch": 1}  # 2100 tokens need 32 ids, not 31
    with pytest.raises(ValueError, match="does not divide"):
        list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 3))
