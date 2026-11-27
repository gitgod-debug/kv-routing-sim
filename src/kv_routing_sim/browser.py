"""Entry point for the in-browser page: run a replay from plain settings, return plain data.

The page runs this module under Pyodide in a web worker. Settings arrive as JSON from the page's
form. Traffic is either generated here or read from a file the page has written into Pyodide's
in-memory filesystem, so nothing the user provides leaves the browser.
"""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any

from .report import recommendation_notes, render_markdown, to_json
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
    """JSON-in, JSON-out wrapper of `run`, which keeps None as null across the Pyodide boundary.

    The output is strict JSON, which the page's JSON.parse accepts: a NaN or an infinity is null.
    """
    return to_json(run(json.loads(settings_json), progress, work_dir=work_dir))


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
        if traffic.get("token_scale") not in (None, ""):
            replay_args += ["--token-scale", str(traffic["token_scale"])]
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
    params = {
        "n_sessions": _whole(traffic.get("sessions", 100), "sessions", least=1),
        "turns_per_session": _whole(traffic.get("turns", 6), "turns per session", least=1),
        "prefix_sharing_ratio": _number(traffic.get("sharing", 0.6), "sharing"),
        "arrival_rate": _number(traffic.get("rate", 0.25), "arrival rate"),
        "seed": _whole(traffic.get("seed", 0), "seed", least=0),
        "n_shared_prefixes": _whole(traffic.get("system_prompts", 1), "system prompts", least=1),
    }
    try:
        workload = AgenticWorkload(**params)
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
    traffic = settings.get("traffic") or {}
    return ReplayOptions(
        policies=str(settings.get("policies", "all")),
        replicas=str(settings.get("replicas", "1-8")),
        seeds=_whole(settings.get("seeds", 5), "seeds", least=1),
        trace_format=trace_format,
        cost_model="default",
        profile=profile_spec,
        ttft_slo_ms=_optional_number(settings.get("ttft_slo_ms"), "the first-token limit"),
        tpot_slo_ms=_number(settings.get("tpot_slo_ms", 100), "the per-token limit"),
        gpu_price_per_hour_usd=_optional_number(settings.get("gpu_price_per_hour_usd"),
                                                "the GPU price"),
        edge_check=bool(settings.get("edge_check", True)),
        spread=bool(settings.get("spread", True)),
        token_scale=_optional_number(traffic.get("token_scale"), "the token scale"),
    )


def _number(value: Any, name: str) -> float:
    """A finite number from a setting, with a message of its own when it is missing or not one."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError(f"enter a number for {name}")
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number")
    return number


def _whole(value: Any, name: str, *, least: int) -> int:
    """A whole number of at least `least`: 3.7 is refused, never run as 3."""
    number = _number(value, name)
    if number != int(number) or number < least:
        raise ConfigError(f"{name} must be a whole number of at least {least}")
    return int(number)


def _optional_number(value: Any, name: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _number(value, name)
