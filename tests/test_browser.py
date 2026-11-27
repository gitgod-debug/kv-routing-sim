"""The entry point that the in-browser page calls under Pyodide."""

from __future__ import annotations

import json
import shlex

from kv_routing_sim import browser, data
from kv_routing_sim.cli import main

SETTINGS = {
    "traffic": {"source": "generate", "sessions": 20, "turns": 3, "sharing": 0.6, "rate": 0.5,
                "seed": 1},
    "profile": {"source": "shipped"},
    "policies": "round-robin,session-sticky",
    "replicas": "1-3",
    "seeds": 2,
    "tpot_slo_ms": 100,
}


def test_the_command_the_page_shows_reproduces_its_report(tmp_path, monkeypatch, capsys):
    result = browser.run(SETTINGS, work_dir=tmp_path / "page")
    assert "error" not in result
    assert result["generated"] is not None and "## Recommendation" in result["markdown"]

    monkeypatch.chdir(tmp_path)
    generate, replay = result["command"].splitlines()
    assert main(shlex.split(generate)[1:]) == 0
    assert main([*shlex.split(replay)[1:], "--format", "json", "--out", "cli.json"]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "cli.json").read_text()) == result["report"]


def test_invalid_settings_come_back_as_an_error_message(tmp_path):
    def error(**changes):
        return browser.run({**SETTINGS, **changes}, work_dir=tmp_path)["error"]

    assert "could not parse replicas" in error(replicas="zero")
    assert "sessions must be a whole number" in error(traffic={**SETTINGS["traffic"],
                                                               "sessions": 0})
    assert "prefix_sharing_ratio" in error(traffic={**SETTINGS["traffic"], "sharing": 1.5})
    assert "choose a trace file" in error(traffic={"source": "file"})
    assert "choose a profile file" in error(profile={"source": "file"})


def test_a_trace_file_and_a_profile_file_are_read_from_their_paths(tmp_path, capsys):
    trace = tmp_path / "log.jsonl"
    main(["generate", "--out", str(trace), "--sessions", "10", "--turns", "2", "--rate", "0.5"])
    capsys.readouterr()
    profile = tmp_path / "mine.json"
    profile.write_text(data.profiles()[browser.SHIPPED_PROFILE].read_text())
    result = browser.run({
        **SETTINGS,
        "traffic": {"source": "file", "path": str(trace), "name": "log.jsonl",
                    "format": "vllm-jsonl"},
        "profile": {"source": "file", "path": str(profile), "name": "mine.json"},
    }, work_dir=tmp_path)
    assert "error" not in result and result["generated"] is None
    assert ("kvsim replay log.jsonl --format-in vllm-jsonl --profile mine.json"
            in result["command"])


def test_run_json_keeps_null_values(tmp_path):
    result = json.loads(browser.run_json(json.dumps(SETTINGS), None, str(tmp_path)))
    assert result["report"]["replica_spread"] is None


def test_numeric_settings_get_messages_of_their_own(tmp_path):
    def error(traffic=None, **changes):
        settings = {**SETTINGS, **changes}
        if traffic is not None:
            settings["traffic"] = {**SETTINGS["traffic"], **traffic}
        return browser.run(settings, work_dir=tmp_path)["error"]

    # 3.7 would have run as seed 3, while the command shown said 3.7.
    assert error(traffic={"seed": 3.7}) == "seed must be a whole number of at least 0"
    assert error(traffic={"seed": -1}) == "seed must be a whole number of at least 0"
    assert error(traffic={"rate": None}) == "enter a number for arrival rate"
    assert error(traffic={"sharing": ""}) == "enter a number for sharing"
    assert error(traffic={"sessions": "  "}) == "enter a number for sessions"
    assert error(traffic={"rate": "fast"}) == "arrival rate must be a number"
    assert error(traffic={"rate": float("nan")}) == "arrival rate must be a finite number"
    assert error(seeds=None) == "enter a number for seeds"
    assert error(seeds=2.5) == "seeds must be a whole number of at least 1"
    assert error(tpot_slo_ms=None) == "enter a number for the per-token limit"
    assert error(ttft_slo_ms="soon") == "the first-token limit must be a number"
    assert error(gpu_price_per_hour_usd=[1]) == "the GPU price must be a number"


def test_a_trace_files_token_scale_reaches_the_reader_and_the_command(tmp_path):
    rows = [{"timestamp": 1000 * i, "input_length": 1024, "output_length": 160,
             "hash_ids": [i, i + 100]} for i in range(4)]
    trace = tmp_path / "m.jsonl"
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = browser.run({
        **SETTINGS,
        "traffic": {"source": "file", "path": str(trace), "name": "m.jsonl", "format": "mooncake",
                    "token_scale": 0.0625},
    }, work_dir=tmp_path)
    assert result["report"]["trace"]["scaling"] == {"token_scale": 0.0625}
    assert "--format-in mooncake --token-scale 0.0625" in result["command"]
    assert any("scaled by 1/16" in n["text"] and n["warning"] for n in result["notes"])


def test_run_json_writes_strict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "run", lambda *a, **k: {"report": {"x": float("nan")},
                                                         "y": [float("inf")]})

    def refuse(token):
        raise ValueError(token)

    out = json.loads(browser.run_json("{}", None, str(tmp_path)), parse_constant=refuse)
    assert out == {"report": {"x": None}, "y": [None]}
