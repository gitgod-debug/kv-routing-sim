"""Entry point for the in-browser page: run a replay from plain settings, return plain data.

The page runs this module under Pyodide in a web worker. Settings arrive as JSON from the page's
form. Traffic is either generated here or read from a file the page has written into Pyodide's
in-memory filesystem, so nothing the user provides leaves the browser.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .report import recommendation_notes, render_markdown
from .service import ConfigError, ProgressCallback, ReplayOptions, replay
from .traces.generator import AgenticWorkload

SHIPPED_PROFILE = "t4-qwen3-4b-vllm0.30.0-ctx"
GENERATED_TRACE = "generated.jsonl"


def run(
    settings: dict[str, Any],
    progress: ProgressCallback | None = None,
    *,
    work_dir: str | Path = ".",
) -> dict[str, Any]:
    """Run one replay described by the page's settings.

    Returns:
        On success, {"report", "markdown", "notes", "generated", "command"}: the report as
        `kvsim replay --format json` writes it, its Markdown rendering, the sentences that
        qualify the recommendation, a summary of generated traffic (None for a file), and the
        `kvsim` commands that reproduce the run. For invalid settings, {"error": message}.
    """
    try:
        trace_path, trace_format, generated = _prepare_traffic(settings.get("traffic") or {},
                                                               Path(work_dir))
        options = _replay_options(settings, trace_format)
        report = replay(trace_path, options, progress=progress)
    except ConfigError as exc:
        return {"error": str(exc)}
    payload = report.as_dict()
    return {
        "report": payload,
        "markdown": render_markdown(payload),
        "notes": [{"text": n.text, "warning": n.warning} for n in recommendation_notes(payload)],
        "generated": generated,
        "command": reproduce_command(settings),
    }


def run_json(settings_json: str, progress: ProgressCallback | None = None,
             work_dir: str = ".") -> str:
    """JSON-in, JSON-out wrapper of `run`, which keeps None as null across the Pyodide boundary."""
    return json.dumps(run(json.loads(settings_json), progress, work_dir=work_dir))


def reproduce_command(settings: dict[str, Any]) -> str:
    """Return the shell commands that run the same replay with the `kvsim` command line."""
    traffic = settings.get("traffic") or {}
    lines = []
    if traffic.get("source", "generate") == "generate":
        generate = ["kvsim", "generate", "--out", GENERATED_TRACE,
                    "--sessions", str(traffic.get("sessions", 100)),
                    "--turns", str(traffic.get("turns", 6)),
                    "--sharing", str(traffic.get("sharing", 0.6)),
                    "--rate", str(traffic.get("rate", 0.25)),
                    "--seed", str(traffic.get("seed", 0))]
        if int(traffic.get("system_prompts", 1)) != 1:
            generate += ["--system-prompts", str(traffic["system_prompts"])]
        lines.append(shlex.join(generate))
        replay_args = ["kvsim", "replay", GENERATED_TRACE]
    else:
        replay_args = ["kvsim", "replay", traffic.get("name") or Path(traffic["path"]).name,
                       "--format-in", traffic.get("format", "vllm-jsonl")]
    profile = settings.get("profile") or {}
    if profile.get("source", "shipped") == "shipped":
        replay_args += ["--profile", profile.get("name", SHIPPED_PROFILE)]
    else:
        replay_args += ["--profile", profile.get("name") or Path(profile["path"]).name]
    replay_args += ["--cost-model", "default",
                    "--policies", str(settings.get("policies", "all")),
                    "--replicas", str(settings.get("replicas", "1-8")),
                    "--seeds", str(settings.get("seeds", 5))]
    if settings.get("ttft_slo_ms") is not None:
        replay_args += ["--ttft-slo-ms", str(settings["ttft_slo_ms"])]
    replay_args += ["--tpot-slo-ms", str(settings.get("tpot_slo_ms", 100))]
    if settings.get("gpu_price_per_hour_usd") is not None:
        replay_args += ["--gpu-price-usd", str(settings["gpu_price_per_hour_usd"])]
    if not settings.get("edge_check", True):
        replay_args.append("--no-edge-check")
    if not settings.get("spread", True):
        replay_args.append("--no-spread")
    lines.append(shlex.join(replay_args))
    return "\n".join(lines)


def _prepare_traffic(traffic: dict[str, Any],
                     work_dir: Path) -> tuple[Path, str, dict[str, Any] | None]:
    """Return the trace file to replay, its format, and a summary when it was generated."""
    if traffic.get("source", "generate") == "file":
        if not traffic.get("path"):
            raise ConfigError("choose a trace file")
        return Path(traffic["path"]), traffic.get("format", "vllm-jsonl"), None
    try:
        workload = AgenticWorkload(
            n_sessions=_positive_int(traffic.get("sessions", 100), "sessions"),
            turns_per_session=_positive_int(traffic.get("turns", 6), "turns per session"),
            prefix_sharing_ratio=float(traffic.get("sharing", 0.6)),
            arrival_rate=float(traffic.get("rate", 0.25)),
            seed=int(traffic.get("seed", 0)),
            n_shared_prefixes=_positive_int(traffic.get("system_prompts", 1), "system prompts"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / GENERATED_TRACE
    summary = workload.write_jsonl(str(path))
    return path, "vllm-jsonl", summary


def _replay_options(settings: dict[str, Any], trace_format: str) -> ReplayOptions:
    profile = settings.get("profile") or {}
    if profile.get("source", "shipped") == "shipped":
        profile_spec = profile.get("name", SHIPPED_PROFILE)
    elif profile.get("path"):
        profile_spec = str(profile["path"])
    else:
        raise ConfigError("choose a profile file")
    try:
        return ReplayOptions(
            policies=str(settings.get("policies", "all")),
            replicas=str(settings.get("replicas", "1-8")),
            seeds=_positive_int(settings.get("seeds", 5), "seeds"),
            trace_format=trace_format,
            cost_model="default",
            profile=profile_spec,
            ttft_slo_ms=_optional_float(settings.get("ttft_slo_ms")),
            tpot_slo_ms=float(settings.get("tpot_slo_ms", 100)),
            gpu_price_per_hour_usd=_optional_float(settings.get("gpu_price_per_hour_usd")),
            edge_check=bool(settings.get("edge_check", True)),
            spread=bool(settings.get("spread", True)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a whole number") from None
    if number < 1 or number != float(value):
        raise ConfigError(f"{name} must be a whole number of at least 1")
    return number


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
