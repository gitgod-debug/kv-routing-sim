"""Tests for trace ingestion (schema, dirty input, format adapters) and the workload generator."""

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
                json.dumps({"request_id": "a", "timestamp": 1.0,
                            "prompt_token_ids": list(range(32)), "num_generated_tokens": 8}),
                "{ broken json",
                json.dumps([1, 2, 3]),
                json.dumps({"request_id": "no-timestamp", "prompt_token_ids": [1]}),
                "",
                json.dumps({"request_id": "b", "timestamp": 2.0,
                            "prompt_token_ids": list(range(32)), "num_generated_tokens": 8}),
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
                    "num_generated_tokens": 8,
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
    """The sharing ratio is a target; the report gives the achieved value alongside it."""
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
    hot system prompt (one that many different blocks follow) stay separate."""
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
    """Two tiny AgentX sessions; the second reuses the first's local ids on purpose."""
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
    """At 1/64, sixteen 64-token trace blocks make one 16-token block. Equal ids are shared within a
    session but not across sessions, because the dataset's ids are session-local."""
    reqs = list(adapters.get("agentx").read(_agentx(tmp_path), token_scale=1 / 64,
                                            concurrency=2, stagger_s=0.0))
    t1 = [r for r in reqs if r.session_id == "t1"]
    (t2,) = [r for r in reqs if r.session_id == "t2"]
    assert [len(r.block_hashes) for r in t1] == [1, 2]
    assert t1[0].block_hashes[0] == t1[1].block_hashes[0]  # shared within the session
    assert t2.block_hashes[0] != t1[0].block_hashes[0]  # the same local ids: not shared
    assert (t1[0].prompt_tokens, t1[0].output_tokens) == (16, 1)


def test_agentx_hashes_each_distinct_block_once_and_shares_the_value(tmp_path):
    """The chained hash is what it was, and a repeated prefix is the same objects, not copies."""
    from kv_routing_sim.traces.adapters import agentx

    reqs = list(adapters.get("agentx").read(_agentx(tmp_path), block_size=64, token_scale=1.0,
                                            concurrency=2, stagger_s=0.0))
    first, second = (r for r in reqs if r.session_id == "t1")
    expected, prev = [], 0
    for block_id in range(1, 33):
        prev = agentx._hash("t1", prev, [block_id])
        expected.append(prev)
    assert (list(first.block_hashes), list(second.block_hashes)) == (expected[:16], expected)
    assert all(a is b for a, b in zip(first.block_hashes, second.block_hashes, strict=False))
    (other,) = [r for r in reqs if r.session_id == "t2"]  # the same local ids, another session
    assert not set(other.block_hashes) & set(second.block_hashes)


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


def _bailian(tmp_path, rows):
    p = tmp_path / "b.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _brow(chat, parent, ts, n_in, n_out, ids, turn=1):
    return {"chat_id": chat, "parent_chat_id": parent, "timestamp": ts, "input_length": n_in,
            "output_length": n_out, "type": "coder", "turn": turn, "hash_ids": ids}


def test_bailian_sessions_are_the_traces_own_and_prefixes_are_chained(tmp_path):
    """A request joins the session at the root of its parent chain. Blocks are shared only as a
    run from the start of the prompt: the trace hashes each block alone, the replay chains them."""
    p = _bailian(tmp_path, [
        _brow(0, -1, 0.0, 64, 8, [5, 6, 7, 8]),
        _brow(1, -1, 1.5, 64, 8, [5, 6, 9, 8]),           # same first two blocks, then differs
        _brow(2, 0, 4.0, 100, 8, [5, 6, 7, 8, 3, 4, 2], turn=2),   # follows chat 0
        _brow(3, 2, 9.0, 120, 8, [5, 6, 7, 8, 3, 4, 1, 1], turn=3),  # follows chat 2
        _brow(4, 77, 9.5, 32, 8, [5, 6]),                 # its parent is not in the file
    ])
    a, b, c, d, e = adapters.get("bailian").read(p)
    assert [r.session_id for r in (a, b, c, d, e)] == ["b0", "b1", "b0", "b0", "b4"]
    assert a.block_hashes[:2] == b.block_hashes[:2]
    assert a.block_hashes[2] != b.block_hashes[2]
    # Block 8 is at position 3 in both, after different prefixes: it is not shared.
    assert a.block_hashes[3] != b.block_hashes[3]
    assert c.block_hashes[:4] == a.block_hashes and len(c.block_hashes) == 6  # 100 // 16
    assert d.block_hashes[:6] == c.block_hashes and len(d.block_hashes) == 7  # 120 // 16
    assert (b.arrival_ts, c.prompt_tokens, c.output_tokens) == (1.5, 100, 8)
    assert a.request_id == "b0" and e.block_hashes == a.block_hashes[:2]


def test_bailian_token_scale_groups_blocks_and_shrinks_sizes(tmp_path):
    """At 1/4, one block of the replay stands for four trace blocks in a row, and is shared only
    if all four are."""
    p = _bailian(tmp_path, [
        _brow(0, -1, 0.0, 16 * 9, 40, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        _brow(1, -1, 1.0, 16 * 9, 40, [1, 2, 3, 4, 5, 6, 7, 0, 9]),  # differs inside group two
    ])
    a, b = adapters.get("bailian").read(p, token_scale=1 / 4)
    assert (a.prompt_tokens, a.output_tokens, len(a.block_hashes)) == (36, 10, 2)
    assert a.block_hashes[0] == b.block_hashes[0] and a.block_hashes[1] != b.block_hashes[1]
    (c, _) = adapters.get("bailian").read(p, token_scale=1 / 3)  # any whole k groups cleanly
    assert (c.prompt_tokens, len(c.block_hashes)) == (48, 3)
    for bad in (0.3, 2.0, 0.0):
        with pytest.raises(ValueError, match="1/k"):
            list(adapters.get("bailian").read(p, token_scale=bad))
    with pytest.raises(ValueError, match="16-token blocks"):
        list(adapters.get("bailian").read(p, block_size=32))


def test_bailian_applies_the_engines_length_limit_and_counts_it(tmp_path):
    from kv_routing_sim.traces.adapters.bailian import outputs_capped

    p = _bailian(tmp_path, [
        _brow(0, -1, 0.0, 160, 30, list(range(10))),
        _brow(1, -1, 1.0, 160, 500, list(range(10))),     # output cut to fit
        _brow(2, -1, 2.0, 320, 5, list(range(20))),       # the prompt alone reaches the limit
        _brow(3, -1, 3.0, 160, 5, list(range(9))),        # 160 tokens need 10 ids, not 9
    ])
    st = TraceStats()
    out = list(adapters.get("bailian").read(p, max_total_tokens=320, stats=st))
    assert [(r.prompt_tokens, r.output_tokens) for r in out] == [(160, 30), (160, 160)]
    assert outputs_capped(st) == 1 and st.requests == 2
    assert st.skipped_lines == 2  # a cut output is counted, but it is not a skipped line
    assert {k: v for k, v in st.skip_reasons.items() if k != "output_capped"} == {
        "prompt_too_long": 1, "block_count_mismatch": 1}


def _jsonl(tmp_path, rows, name="t.jsonl"):
    """Write rows as JSONL. A str row is written as it is, for text json.dumps cannot produce."""
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return p


def _vllm(rid, ts, n=32, out=8, **extra):
    return {"request_id": rid, "timestamp": ts, "prompt_token_ids": list(range(n)),
            "num_generated_tokens": out, **extra}


def test_a_file_in_newest_first_order_is_read_in_time_order(tmp_path):
    """Time starts at the earliest request, not the first line, and nothing is clamped to 0."""
    p = _jsonl(tmp_path, [_vllm("c", 130.0), _vllm("b", 110.0), _vllm("a", 100.0),
                          _vllm("a2", 100.0)])
    out = list(adapters.get("vllm-jsonl").read(p))
    assert [(r.request_id, r.arrival_ts) for r in out] == [
        ("a", 0.0), ("a2", 0.0), ("b", 10.0), ("c", 30.0)]  # equal times keep the file's order


def test_a_bad_timestamp_is_skipped_and_never_sets_the_clock(tmp_path):
    rows = [_vllm("bad", "soon"), _vllm("nan", "NaN"), _vllm("bool", True), _vllm("inf", "1e999"),
            _vllm("x", 50.0), _vllm("y", 52.5)]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [(r.request_id, r.arrival_ts) for r in out] == [("x", 0.0), ("y", 2.5)]
    assert stats.skip_reasons == {"bad_timestamp": 4}


def test_timestamps_may_be_numeric_or_iso_8601_strings(tmp_path):
    rows = [_vllm("a", "2026-11-27T10:00:00Z"), _vllm("b", "2026-11-27T10:00:01.5"),
            _vllm("c", "2026-11-27T11:00:02+01:00"), _vllm("d", str(1_795_773_603))]
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows)))
    # A time without an offset is UTC. 11:00:02+01:00 is 10:00:02Z, and the number is 10:00:03Z.
    assert [(r.request_id, r.arrival_ts) for r in out] == [
        ("a", 0.0), ("b", 1.5), ("c", 2.0), ("d", 3.0)]
    # Mooncake counts in milliseconds: a numeric string is milliseconds, a date is converted.
    for stamps, expected in ((["2026-11-27T10:00:00Z", "2026-11-27T10:00:02.5Z"], [0.0, 2.5]),
                             (["4000", 7500], [0.0, 3.5])):
        rows = [{"timestamp": t, "input_length": 512, "output_length": 1, "hash_ids": [i]}
                for i, t in enumerate(stamps)]
        moon = list(adapters.get("mooncake").read(_mooncake(tmp_path, rows)))
        assert [r.arrival_ts for r in moon] == expected


def test_mooncake_and_bailian_read_newest_first_files_in_time_order(tmp_path):
    """Their sessions follow the order of arrival too: a follow-up joins the request before it."""
    rows = [{"timestamp": 9000, "input_length": 1536, "output_length": 1, "hash_ids": [1, 2, 3]},
            {"timestamp": 3000, "input_length": 1024, "output_length": 1, "hash_ids": [1, 2]}]
    first, second = adapters.get("mooncake").read(_mooncake(tmp_path, rows))
    assert (first.request_id, first.arrival_ts, second.arrival_ts) == ("m1", 0.0, 6.0)
    assert second.session_id == first.session_id  # extends the earlier request's full blocks

    p = _bailian(tmp_path, [_brow(2, 1, 9.0, 32, 8, [5, 6], turn=3),
                            _brow(1, 0, 4.0, 32, 8, [5, 6], turn=2),
                            _brow(0, -1, 2.5, 32, 8, [5, 6])])
    out = list(adapters.get("bailian").read(p))
    assert [(r.request_id, r.arrival_ts, r.session_id) for r in out] == [
        ("b0", 0.0, "b0"), ("b1", 1.5, "b0"), ("b2", 6.5, "b0")]


def test_a_missing_output_length_is_skipped_and_an_explicit_zero_is_kept(tmp_path):
    rows = [{"request_id": "none", "timestamp": 1, "prompt_token_ids": list(range(32))},
            _vllm("zero", 2, out=0), _vllm("five", 3, out=5)]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [(r.request_id, r.output_tokens) for r in out] == [("zero", 0), ("five", 5)]
    assert stats.skip_reasons == {"no_output_length": 1}


def test_openai_output_length_is_the_usage_else_the_requests_limit(tmp_path):
    msgs = [{"role": "user", "content": "hello there " * 20}]
    rows = [{"id": "usage", "created": 1, "messages": msgs, "max_tokens": 99,
             "usage": {"prompt_tokens": 40, "completion_tokens": 7}},
            {"id": "top", "created": 2, "messages": msgs, "max_tokens": 11},
            {"id": "body", "created": 3, "body": {"messages": msgs, "max_tokens": 13}},
            {"id": "none", "created": 4, "messages": msgs}]
    stats = TraceStats()
    out = list(adapters.get("openai-log").read(_jsonl(tmp_path, rows), stats=stats))
    assert [(r.request_id, r.output_tokens) for r in out] == [("usage", 7), ("top", 11),
                                                              ("body", 13)]
    assert stats.skip_reasons == {"no_output_length": 1}


def test_a_prompt_that_is_a_list_of_integers_is_token_ids(tmp_path):
    ids = list(range(1000, 1040))
    rows = [{"request_id": "ids", "timestamp": 0, "prompt_token_ids": ids, "max_tokens": 4},
            {"request_id": "prompt", "timestamp": 1, "prompt": ids, "max_tokens": 4}]
    a, b = adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows))
    assert (b.block_hashes, b.prompt_tokens) == (a.block_hashes, 40)


def test_an_empty_prompt_is_skipped_and_counted(tmp_path):
    rows = [{"request_id": "ids", "timestamp": 0, "prompt_token_ids": [], "max_tokens": 4},
            {"request_id": "text", "timestamp": 1, "prompt": "   ", "max_tokens": 4},
            {"request_id": "count", "timestamp": 2, "prompt_token_ids": [1, 2],
             "num_prompt_tokens": 0, "max_tokens": 4},
            _vllm("ok", 3)]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [r.request_id for r in out] == ["ok"]
    assert stats.skip_reasons == {"empty_prompt": 3}
    chat = [{"id": "empty", "created": 0, "messages": [], "max_tokens": 4},
            {"id": "ok", "created": 1, "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 4}]
    stats = TraceStats()
    out = list(adapters.get("openai-log").read(_jsonl(tmp_path, chat, "c.jsonl"), stats=stats))
    assert [r.request_id for r in out] == ["ok"] and stats.skip_reasons == {"empty_prompt": 1}


def test_a_byte_order_mark_does_not_cost_the_first_record(tmp_path):
    p = tmp_path / "bom.jsonl"
    p.write_text("﻿" + json.dumps(_vllm("a", 0)) + "\n" + json.dumps(_vllm("b", 1)) + "\n",
                 encoding="utf-8")
    stats = TraceStats()
    assert [r.request_id for r in adapters.get("vllm-jsonl").read(p, stats=stats)] == ["a", "b"]
    assert stats.skipped_lines == 0
    m = tmp_path / "bom_m.jsonl"
    m.write_text("﻿" + json.dumps({"timestamp": 0, "input_length": 512, "output_length": 1,
                                        "hash_ids": [1]}) + "\n", encoding="utf-8")
    assert len(list(adapters.get("mooncake").read(m))) == 1


def test_one_bad_record_is_skipped_never_a_traceback(tmp_path):
    """Values that convert badly, in each adapter: each skips its own record and is counted."""
    base = [
        _vllm("text-count", 1, num_prompt_tokens="n/a"),
        {"request_id": "none-in-ids", "timestamp": 2, "prompt_token_ids": [1] * 20 + [None],
         "max_tokens": 4},  # in the partial last block, which is never hashed
        '{"request_id": "huge", "timestamp": 3, "prompt_token_ids": [1, 2], "max_tokens": 1e999}',
        '{"request_id": "nan", "timestamp": 4, "prompt_token_ids": [1, 2], "max_tokens": NaN}',
        _vllm("big-id", 5, n=0, prompt_token_ids=[2**64] * 16),
        _vllm("ok", 6),
    ]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, base), stats=stats))
    assert [r.request_id for r in out] == ["ok"] and stats.skip_reasons == {"invalid_record": 5}

    def moon(**row):
        return {"timestamp": 0, "input_length": 512, "output_length": 1, "hash_ids": [1], **row}

    rows = [moon(hash_ids=[2**64]), moon(hash_ids=[-1]), moon(output_length=10**400),
            '{"timestamp": 0, "input_length": 1e999, "output_length": 1, "hash_ids": [1]}',
            moon(timestamp=7)]
    stats = TraceStats()
    out = list(adapters.get("mooncake").read(_jsonl(tmp_path, rows, "m.jsonl"), stats=stats))
    assert len(out) == 1 and stats.skip_reasons == {"invalid_record": 4}

    huge_chat = json.dumps(_brow(2, -1, 2.0, 32, 8, [1, 2])).replace('"chat_id": 2',
                                                                     '"chat_id": 1e999')
    rows = [_brow(0, -1, 0.0, 32, 8, [2**64, 1]), _brow(1, -1, 1.0, 32, 10**400, [1, 2]),
            huge_chat, _brow(3, -1, 3.0, 32, 8, [1, 2])]
    stats = TraceStats()
    out = list(adapters.get("bailian").read(_jsonl(tmp_path, rows, "b.jsonl"), stats=stats))
    assert [r.request_id for r in out] == ["b3"] and stats.skip_reasons == {"invalid_record": 3}


def test_agentx_skips_a_session_or_request_it_cannot_read(tmp_path):
    good = {"id": "ok", "models": ["m"], "requests": [
        {"t": 0.0, "type": "n", "model": "m", "in": 1024, "out": 8, "hash_ids": list(range(16)),
         "api_time": 1.0}]}

    def session(sid, **changes):
        return {**good, "id": sid, **changes}

    def request(**changes):
        return [{**good["requests"][0], **changes}]

    rows = [[1, 2, 3], '"a string"',
            session("models", models="m"),
            session("entry", requests=["not an object"]),
            session("negative", requests=request(t=-1.0)),
            '{"id": "nan", "models": ["m"], "requests": [{"t": NaN, "in": 1024, "out": 8}]}',
            session("big-id", requests=request(hash_ids=[2**64] * 16)),
            session("huge-out", requests=request(out=10**400)),
            good]
    stats = TraceStats()
    out = list(adapters.get("agentx").read(_jsonl(tmp_path, rows, "x.jsonl"), stats=stats))
    assert [r.session_id for r in out] == ["ok"]
    assert stats.skip_reasons == {"invalid_record": 8}


def test_repeated_request_ids_are_made_unique_and_counted(tmp_path):
    rows = [_vllm("x", 0), _vllm("x", 1), _vllm("x#2", 2), _vllm("y", 3)]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [r.request_id for r in out] == ["x", "x#2", "x#2#2", "y"]
    assert stats.renamed_duplicate_ids == 2 and stats.as_dict()["renamed_duplicate_ids"] == 2
    p = _bailian(tmp_path, [_brow(5, -1, 0.0, 32, 8, [1, 2]), _brow(5, -1, 1.0, 32, 8, [1, 2])])
    stats = TraceStats()
    assert [r.request_id for r in adapters.get("bailian").read(p, stats=stats)] == ["b5", "b5#2"]
    assert stats.renamed_duplicate_ids == 1


def test_agentx_reads_a_compressed_file_as_it_reads_the_plain_one(tmp_path):
    import gzip

    plain = _agentx(tmp_path)
    packed = tmp_path / "agentx.jsonl.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))
    kw = {"token_scale": 1 / 64, "concurrency": 1, "stagger_s": 0.0}
    assert list(adapters.get("agentx").read(packed, **kw)) == \
        list(adapters.get("agentx").read(plain, **kw))


def test_the_trace_statistics_state_how_the_reader_scaled_it(tmp_path):
    stats = TraceStats()
    list(adapters.get("agentx").read(_agentx(tmp_path), stats=stats))
    assert stats.as_dict()["scaling"] == {"token_scale": 0.03125, "concurrency": 32,
                                          "stagger_s": 600.0}
    rows = [{"timestamp": 0, "input_length": 1024, "output_length": 160, "hash_ids": [1, 2]}]
    for scale, scaling in ((1 / 16, {"token_scale": 0.0625}), (1.0, None)):
        stats = TraceStats()
        list(adapters.get("mooncake").read(_mooncake(tmp_path, rows), token_scale=scale,
                                           stats=stats))
        assert stats.as_dict().get("scaling") == scaling
    p = _bailian(tmp_path, [_brow(0, -1, 0.0, 64, 8, [5, 6, 7, 8])])
    stats = TraceStats()
    list(adapters.get("bailian").read(p, token_scale=1 / 2, stats=stats))
    assert stats.as_dict()["scaling"] == {"token_scale": 0.5}


def test_the_generator_refuses_no_sessions_and_a_rate_that_is_not_a_number():
    with pytest.raises(ValueError, match="n_sessions"):
        AgenticWorkload(n_sessions=0)
    for rate in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="arrival_rate"):
            AgenticWorkload(arrival_rate=rate)


# A second review before release 0.2.0: what a reader assumes or refuses must be said.


def test_a_log_in_epoch_milliseconds_is_refused(tmp_path):
    """Read as seconds, milliseconds stretch the log a thousand times and the load looks tiny."""
    rows = [_vllm(f"r{i}", 1_795_000_000_000 + 100 * i) for i in range(4)]
    with pytest.raises(ValueError, match="too large to be seconds"):
        list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows)))
    # Seconds since the epoch, and an ISO time, are fine.
    rows = [_vllm("a", 1_795_000_000.0), _vllm("b", "2026-11-28T10:00:00Z")]
    assert len(list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows)))) == 2


def test_a_limit_that_stands_in_for_an_answer_length_is_counted(tmp_path):
    """`max_tokens` is the longest an answer could be, not its length. The reader says how often
    it had nothing better."""
    base = {"timestamp": 1, "prompt_token_ids": list(range(32))}
    rows = [{**base, "request_id": "real", "num_generated_tokens": 7, "max_tokens": 500},
            {**base, "request_id": "limit", "max_tokens": 500},
            {**base, "request_id": "none"}]
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [(r.request_id, r.output_tokens) for r in out] == [("real", 7), ("limit", 500)]
    assert stats.assumed == {"max_tokens_as_output_length": 1}
    assert stats.as_dict()["assumed"] == {"max_tokens_as_output_length": 1}
    assert stats.skip_reasons == {"no_output_length": 1}
    # A reader that assumed nothing writes no such key, so earlier results read the same.
    clean = TraceStats()
    list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, [_vllm("a", 1)], "c.jsonl"), stats=clean))
    assert "assumed" not in clean.as_dict()


def test_openai_limits_are_counted_too(tmp_path):
    msgs = [{"role": "user", "content": "hello there " * 20}]
    rows = [{"id": "usage", "created": 1, "messages": msgs, "max_tokens": 99,
             "usage": {"prompt_tokens": 40, "completion_tokens": 7}},
            {"id": "top", "created": 2, "messages": msgs, "max_tokens": 11},
            {"id": "body", "created": 3, "body": {"messages": msgs, "max_tokens": 13}}]
    stats = TraceStats()
    list(adapters.get("openai-log").read(_jsonl(tmp_path, rows), stats=stats))
    assert stats.assumed == {"max_tokens_as_output_length": 2}


@pytest.mark.parametrize(("change", "reason"), [
    ({"num_generated_tokens": True}, "invalid_record"),  # int(True) is 1
    ({"num_generated_tokens": 20.7}, "invalid_record"),  # int(20.7) is 20
    ({"num_generated_tokens": "2_0"}, "invalid_record"),  # int("2_0") is 20
    ({"prompt_token_ids": "123456789"}, "invalid_record"),  # a string is not a list of ids
    ({"prompt_token_ids": [1, True, 3] * 11}, "invalid_record"),
    ({"timestamp": "1_0"}, "bad_timestamp"),  # float("1_0") is 10
])
def test_a_value_of_the_wrong_type_skips_the_record(tmp_path, change, reason):
    """Python's int() and float() accept more than a log means. Such a record is skipped."""
    stats = TraceStats()
    rows = [_vllm("good", 1), {**_vllm("odd", 2), **change}]
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [r.request_id for r in out] == ["good"]
    assert stats.skip_reasons == {reason: 1}


def test_a_count_may_be_a_whole_float_or_a_string_of_digits(tmp_path):
    rows = [{**_vllm("a", 1), "num_generated_tokens": 8.0}, {**_vllm("b", 2),
                                                             "num_generated_tokens": " 9 "}]
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows)))
    assert [r.output_tokens for r in out] == [8, 9]


@pytest.mark.parametrize("fmt", ["vllm-jsonl", "openai-log", "mooncake", "bailian", "agentx"])
def test_a_line_the_decoder_gives_up_on_is_skipped(tmp_path, fmt):
    """Nesting a hundred thousand deep, or an integer of thousands of digits, is one bad line."""
    rows = ["[" * 100_000 + "]" * 100_000, '{"timestamp": ' + "9" * 5000 + "}"]
    stats = TraceStats()
    assert list(adapters.get(fmt).read(_jsonl(tmp_path, rows), stats=stats)) == []
    assert stats.skipped_lines == 2


def test_text_that_cannot_be_encoded_is_one_bad_record(tmp_path):
    rows = [_vllm("good", 1),
            '{"request_id": "odd", "timestamp": 2, "num_generated_tokens": 4, "prompt": "\\ud800"}']
    stats = TraceStats()
    out = list(adapters.get("vllm-jsonl").read(_jsonl(tmp_path, rows), stats=stats))
    assert [r.request_id for r in out] == ["good"] and stats.skip_reasons == {"invalid_record": 1}


def test_agentx_sessions_with_one_id_are_kept_apart(tmp_path):
    """Hash ids are local to a session, and its id keeps them apart. Two sessions with one id,
    as in two files joined, must not look like one session reading its own blocks again."""
    def session(sid):
        return {"id": sid, "models": ["m"], "requests": [
            {"t": 0.0, "type": "n", "model": "m", "in": 1024, "out": 8,
             "hash_ids": list(range(16)), "api_time": 1.0}]}

    def read(rows):
        stats = TraceStats()
        out = list(adapters.get("agentx").read(_jsonl(tmp_path, rows, "x.jsonl"), stats=stats))
        return out, stats

    apart, stats = read([session("a"), session("b")])
    assert set(apart[0].block_hashes).isdisjoint(apart[1].block_hashes)
    assert "assumed" not in stats.as_dict()
    same, stats = read([session("a"), session("a"), session("a#2")])
    hashes = [set(r.block_hashes) for r in same]
    assert all(x.isdisjoint(y) for x, y in itertools.combinations(hashes, 2))
    assert len({r.session_id for r in same}) == 3
    # The second "a" becomes "a#2", so the session that was named "a#2" is renamed as well.
    assert stats.assumed == {"repeated_session_id_renamed": 2}


def test_agentx_needs_an_id_and_a_list_of_requests(tmp_path):
    """Without an id a session's blocks cannot be kept apart, and a line without `requests` is a
    record of another format. Both are counted, so a wrong format is not a silent empty trace."""
    good = {"id": "ok", "models": ["m"], "requests": [
        {"t": 0.0, "type": "n", "model": "m", "in": 1024, "out": 8, "hash_ids": list(range(16)),
         "api_time": 1.0}]}
    rows = [{**good, "id": None},
            {"request_id": "q", "timestamp": 1, "prompt_token_ids": [1]}, good]
    stats = TraceStats()
    out = list(adapters.get("agentx").read(_jsonl(tmp_path, rows, "x.jsonl"), stats=stats))
    assert [r.session_id for r in out] == ["ok"] and stats.skip_reasons == {"bad_session": 2}


def test_the_generator_refuses_a_rate_too_small_for_finite_times():
    with pytest.raises(ValueError, match="at least 1e-09"):
        AgenticWorkload(n_sessions=3, arrival_rate=1e-320)
