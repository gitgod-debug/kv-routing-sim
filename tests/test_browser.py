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
