"""The replay workflow: size a fleet of replicas for a trace and recommend a routing policy.

This is the application layer shared by `kvsim replay` and the in-browser page. It validates the
options, loads the trace, timing profile and cost model, sweeps policies and replica counts,
recommends a configuration, and optionally checks how firm the replica count is and how close it
sits to saturation. It raises `ConfigError` for bad input and never prints or exits, so each front
end presents the same report in its own way.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import policies as policy_registry
from .eval.cost import CostModel, UnfilledCostModel
from .eval.edge import edge_check
from .eval.recommend import recommend
from .eval.spread import replica_spread
from .eval.sweep import RunCallback, sweep
from .sim.cost_model import (
    UNCALIBRATED_DEFAULT,
    CostProfile,
    replicate_profiles,
    resolve_profile,
)
from .traces import adapters
from .traces.schema import NormalizedRequest, TraceStats

__all__ = [
    "DEFAULT_TPOT_SLO_MS",
    "ConfigError",
    "ProgressCallback",
    "ReplayOptions",
    "ReplayReport",
    "apply_overrides",
    "describe_slo",
    "load_cost_model",
    "load_profile",
    "parse_replicas",
    "read_trace",
    "replay",
    "resolve_policies",
]

DEFAULT_TPOT_SLO_MS = 100.0
"""p95 time per output token allowed when the cost model sets none: MLPerf's pair for a 2 s TTFT."""

SPREAD_UNAVAILABLE = (
    "This profile has no per-repeat fits, so the spread of these replica counts cannot be shown. "
    "For scale: three calibrations of the same T4 agreed on a count exactly at 71% of points and "
    "within one replica at 96%. A profile written by `kvsim calibrate` includes its own spread; "
    "even then, allow one replica more."
)

ProgressCallback = Callable[[str, int, int], None]
"""Receives (stage, runs done, runs in the stage); stages are "sweep", "spread" and "edge"."""


class ConfigError(ValueError):
    """Invalid options, or an input file that cannot be used."""


@dataclass(frozen=True)
class ReplayOptions:
    """Settings for one replay; the defaults match `kvsim replay`."""

    policies: str = "all"
    replicas: str = "1-8"
    seeds: int = 5
    block_size: int = 16
    trace_format: str = "vllm-jsonl"
    cost_model: str | None = None
    profile: str | None = None
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float = DEFAULT_TPOT_SLO_MS
    gpu_price_per_hour_usd: float | None = None
    edge_check: bool = True
    spread: bool = True


@dataclass
class ReplayReport:
    """Everything a replay produced, in the shape `kvsim replay --format json` writes."""

    trace: dict[str, Any]
    profile: dict[str, Any]
    points: list[dict[str, Any]]
    recommendation: dict[str, Any]
    slo: str | None
    replica_spread: dict[str, Any] | None
    replica_spread_note: str | None
    edge: dict[str, Any] | None

    @property
    def calibrated(self) -> bool:
        return bool(self.recommendation.get("calibrated"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace,
            "profile": self.profile,
            "points": self.points,
            "recommendation": self.recommendation,
            "slo": self.slo,
            "replica_spread": self.replica_spread,
            "replica_spread_note": self.replica_spread_note,
            "edge": self.edge,
        }


def parse_replicas(spec: str) -> list[int]:
    """Parse "1-8" or "1,2,4,8" into a list of positive replica counts."""
    spec = spec.strip()
    try:
        if "-" in spec and "," not in spec:
            lo, hi = spec.split("-", 1)
            counts = list(range(int(lo), int(hi) + 1))
        else:
            counts = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise ConfigError(f"could not parse replicas {spec!r}; use '1-8' or '1,2,4'") from None
    if not counts or any(n < 1 for n in counts):
        raise ConfigError(f"replicas {spec!r} must be one or more positive integers")
    return counts


def resolve_policies(spec: str) -> list[str]:
    """Return the policy names for "all" or a comma-separated list, rejecting unknown names."""
    try:
        return policy_registry.resolve(spec)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


def load_cost_model(spec: str | None) -> CostModel | None:
    """Load a cost model from a file, or the shipped one for "default"; None without a spec."""
    if not spec:
        return None
    try:
        return CostModel.from_yaml(spec)
    except UnfilledCostModel as exc:
        raise ConfigError(str(exc)) from None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"could not read cost model {spec!r}: {exc}") from None


def load_profile(spec: str | None) -> CostProfile:
    """Load a timing profile from a file or a shipped name; the uncalibrated default without one."""
    if not spec:
        return UNCALIBRATED_DEFAULT
    try:
        return resolve_profile(spec)
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"could not read profile {spec!r}: {exc}") from None


def apply_overrides(
    cost: CostModel | None,
    *,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float,
    gpu_price_per_hour_usd: float | None = None,
) -> tuple[CostModel | None, bool, bool]:
    """Apply the option-level SLO limits and GPU price to the cost model.

    A first-token limit and a GPU price replace the cost model's. A per-token limit applies only
    when the cost model sets none, and 0 leaves the per-token half off.

    Returns:
        The cost model, whether the first-token limit came from the options, and whether the
        per-token limit did.
    """
    if cost is None:
        return None, False, False
    ttft_from_options = ttft_slo_ms is not None
    if ttft_from_options:
        if ttft_slo_ms <= 0:
            raise ConfigError("the first-token limit must be a positive number of milliseconds")
        cost = dataclasses.replace(cost, ttft_p95_slo_ms=float(ttft_slo_ms))
    if gpu_price_per_hour_usd is not None:
        if gpu_price_per_hour_usd <= 0:
            raise ConfigError("the GPU price must be a positive number of US dollars per hour")
        cost = dataclasses.replace(cost, price_per_hour_usd=float(gpu_price_per_hour_usd))
    tpot_from_options = cost.tpot_p95_slo_ms is None and tpot_slo_ms > 0
    if tpot_from_options:
        cost = dataclasses.replace(cost, tpot_p95_slo_ms=float(tpot_slo_ms))
    return cost, ttft_from_options, tpot_from_options


def describe_slo(
    cost: CostModel | None, *, ttft_from_options: bool = False, tpot_from_options: bool = False
) -> str | None:
    """State the SLO that "meets SLO" was decided on, or None when there is no cost model."""
    if cost is None:
        return None
    if cost.tpot_p95_slo_ms is None:
        sentence = (f"SLO: p95 time to first token ≤ {cost.ttft_p95_slo_ms:,.0f} ms. The time per "
                    "output token is shown but not part of it (`--tpot-slo-ms 0`): read these "
                    "replica counts as a lower bound for anything that has to stream at a usable "
                    "speed.")
    else:
        sentence = (f"SLO: p95 time to first token ≤ {cost.ttft_p95_slo_ms:,.0f} ms and p95 time "
                    f"per output token ≤ {cost.tpot_p95_slo_ms:,.0f} ms.")
    if ttft_from_options:
        sentence += " The first-token limit was set with `--ttft-slo-ms`."
    if tpot_from_options:
        sentence += (" The per-token limit is this command's (`--tpot-slo-ms`; 0 sizes on the "
                     "first token alone); the cost model sets none.")
    return sentence


def read_trace(
    path: str | Path, trace_format: str, block_size: int
) -> tuple[list[NormalizedRequest], dict[str, Any]]:
    """Read a trace with the named adapter; return its requests and the reader's statistics."""
    if trace_format not in adapters.REGISTRY:
        raise ConfigError(f"unknown trace format {trace_format!r}; available: "
                          f"{', '.join(sorted(adapters.REGISTRY))}")
    stats = TraceStats()
    try:
        trace = list(adapters.get(trace_format).read(str(path), block_size=block_size,
                                                     stats=stats))
    except OSError as exc:
        raise ConfigError(f"could not read trace {str(path)!r}: {exc}") from None
    if not trace:
        raise ConfigError(f"no usable requests in {str(path)!r} ({stats.skipped_lines} line(s) "
                          f"skipped: {stats.skip_reasons})")
    return trace, stats.as_dict()


def replay(
    trace_path: str | Path,
    options: ReplayOptions | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> ReplayReport:
    """Replay a trace through every requested policy and replica count, and recommend one.

    Raises:
        ConfigError: an option is invalid, or the trace, profile or cost model cannot be used.
    """
    options = options or ReplayOptions()
    names = resolve_policies(options.policies)
    replica_counts = parse_replicas(options.replicas)
    if options.seeds < 1:
        raise ConfigError("seeds must be at least 1")
    seeds = tuple(range(options.seeds))
    cost, ttft_from_options, tpot_from_options = apply_overrides(
        load_cost_model(options.cost_model),
        ttft_slo_ms=options.ttft_slo_ms,
        tpot_slo_ms=options.tpot_slo_ms,
        gpu_price_per_hour_usd=options.gpu_price_per_hour_usd,
    )
    profile = load_profile(options.profile)
    if options.block_size != profile.block_size:
        raise ConfigError(f"block size {options.block_size} does not match the profile's block "
                          f"size {profile.block_size} ({profile.name})")
    trace, info = read_trace(trace_path, options.trace_format, options.block_size)

    points, _ = sweep(
        trace,
        policies=names,
        replica_counts=replica_counts,
        seeds=seeds,
        profile=profile,
        cost=cost,
        trace_info=info,
        on_run=_stage(progress, "sweep", len(names) * len(replica_counts) * len(seeds)),
    )
    rec = recommend(points, **({"slo": cost.slo_label} if cost is not None else {}))

    spread, spread_note = None, None
    if cost is not None and profile.calibrated and options.spread:
        replicates = replicate_profiles(options.profile)
        if replicates:
            spread = replica_spread(
                trace,
                policies=names,
                replica_counts=replica_counts,
                seeds=seeds,
                cost=cost,
                pooled_points=points,
                replicates=replicates,
                trace_info=info,
                on_run=_stage(progress, "spread",
                              len(replicates) * len(names) * len(replica_counts) * len(seeds)),
            )
        else:
            spread_note = SPREAD_UNAVAILABLE

    edge = None
    if cost is not None and rec.chosen is not None and rec.chosen.meets_slo and options.edge_check:
        edge = edge_check(trace, chosen=rec.chosen, seeds=seeds, profile=profile, cost=cost,
                          trace_info=info, on_run=_stage(progress, "edge", len(seeds)))

    return ReplayReport(
        trace=info,
        profile=profile.provenance(),
        points=[p.as_dict() for p in points],
        recommendation=rec.as_dict(),
        slo=describe_slo(cost, ttft_from_options=ttft_from_options,
                         tpot_from_options=tpot_from_options),
        replica_spread=spread,
        replica_spread_note=spread_note,
        edge=edge,
    )


def _stage(progress: ProgressCallback | None, stage: str, total: int) -> RunCallback | None:
    """Adapt a progress callback to the per-run callback of one stage."""
    if progress is None:
        return None
    done = 0
    progress(stage, done, total)

    def on_run() -> None:
        nonlocal done
        done += 1
        progress(stage, done, total)

    return on_run
