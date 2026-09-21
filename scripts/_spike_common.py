"""Shared helpers for the Phase 0 spike scripts.

Spikes are throwaway measurements, but they still obey the reproducibility contract in
`docs/06-methodology.md` §3: every results file carries date, git SHA, seed, hardware and package
versions, so a number can always be traced back to the machine and the code that produced it.

Naming: spike scripts are `spike_*.py`. The `run_*.py` scripts that regenerate README tables come in
Phase 1 and are held to a stricter standard (seeded, config-driven, byte-identical on re-run).

Adapted from the sibling project `colpali-economics` so both repos' artifacts read the same way.
Two fields are new here, and both exist because of ADR-005:

  * `hardware.compute_capability` — a T4 and a GTX 1650 are both 7.5, an A100 is 8.0, an H100
    is 9.0, and which one produced a number decides whether bf16/FP8 and the FlashAttention
    backend were in play at all. A results file that does not say is not interpretable.
  * `hardware_path` — single-replica / MIG / memory-split. TTFT percentiles from the memory-split
    fallback are contaminated by test-rig SM contention and must never be attributed to a routing
    policy, so the path travels with every number rather than living in someone's memory.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "artifacts" / "results"

# Recorded in every header. Pinned identities matter more than breadth here.
TRACKED_PACKAGES = (
    "vllm",
    "torch",
    "numpy",
    "pydantic",
    "httpx",
    "prometheus-client",
    "matplotlib",
)

# The valid values for `hardware_path`. See ADR-005 for what each one permits reporting.
HARDWARE_PATHS = ("single-replica", "multi-gpu", "mig", "memory-split", "cpu-only")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # a missing git or a non-repo is reported as "unknown", never fatal
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _nvidia_smi_query(fields: str) -> list[str]:
    """Query nvidia-smi for one row per GPU. Returns [] if it is not there or fails.

    Read through nvidia-smi rather than torch because the spike scripts run on a box where the
    driver and MIG topology are the thing under test, and torch may not be importable yet.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode != 0:
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _mig_instances() -> list[str]:
    """MIG instance UUIDs, if MIG is enabled. Empty list is a valid, meaningful answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15, check=False
        )
        return [ln.strip() for ln in out.stdout.splitlines() if "MIG" in ln]
    except (OSError, subprocess.SubprocessError):
        return []


def _hardware() -> dict[str, Any]:
    try:
        usable_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        usable_cpus = os.cpu_count()

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "usable_cpus": usable_cpus,
    }

    gpus = _nvidia_smi_query("name,memory.total,compute_cap,driver_version")
    if gpus:
        info["gpus"] = gpus
        # compute_cap is the third CSV field; it decides bf16/FP8 and the attention backend.
        first = [f.strip() for f in gpus[0].split(",")]
        if len(first) >= 3:
            info["gpu_name"] = first[0]
            info["gpu_memory_total"] = first[1]
            info["compute_capability"] = first[2]
        if len(first) >= 4:
            info["driver_version"] = first[3]
    else:
        info["gpus"] = []

    mig = _mig_instances()
    info["mig_enabled"] = bool(mig)
    if mig:
        info["mig_instances"] = mig

    mig_mode = _nvidia_smi_query("mig.mode.current")
    if mig_mode:
        info["mig_mode_current"] = mig_mode

    return info


def peak_rss_gib() -> float:
    """Peak resident set size of this process, in GiB.

    `ru_maxrss` is kilobytes on Linux (bytes on macOS). This project's hardware is Linux, and the
    unit is named in the key so a reader never has to guess.
    """
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def results_header(
    script: str,
    seed: int,
    hardware_path: str = "single-replica",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The header every results file starts with (docs/06-methodology.md §3).

    `git_dirty` is recorded because a number produced from an uncommitted tree is not
    re-derivable, and the file should say so rather than imply otherwise.

    `hardware_path` is required by ADR-005: it decides which numbers in the file may be reported.
    An unrecognised value raises, because silently accepting a typo here would let a contaminated
    TTFT percentile look like a clean one.
    """
    if hardware_path not in HARDWARE_PATHS:
        raise ValueError(f"hardware_path must be one of {HARDWARE_PATHS}, got {hardware_path!r}")

    header = {
        "script": script,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": seed,
        "hardware_path": hardware_path,
        "hardware": _hardware(),
        "packages": _package_versions(),
    }
    if extra:
        header.update(extra)
    return header


def write_results(filename: str, payload: dict[str, Any]) -> Path:
    """Write a results file to artifacts/results/ and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def write_raw(filename: str, text: str) -> Path:
    """Commit a raw artifact verbatim — e.g. a /metrics scrape.

    The parsed summary is what we read; the raw scrape is what makes the parse falsifiable. The
    Calibration Report commits both, and this is why (`docs/06-methodology.md` §4).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path
