"""Shared helpers for the spike scripts: the results-file header and the writers.

Every results file starts with a header of date, git SHA, seed, hardware and package versions
(docs/06-methodology.md §3). Two header fields come from ADR-005: `hardware.compute_capability`
decides whether bf16/FP8 and the FlashAttention backend were available, and `hardware_path`
records whether the memory-split fallback was used, whose TTFT percentiles include test-rig SM
contention and must not be attributed to a routing policy.
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

# Package versions recorded in every header.
TRACKED_PACKAGES = (
    "vllm",
    "torch",
    "numpy",
    "pydantic",
    "httpx",
    "prometheus-client",
    "matplotlib",
)

# Valid values for `hardware_path`. ADR-005 says which numbers each one permits reporting.
HARDWARE_PATHS = ("single-replica", "multi-gpu", "mig", "memory-split", "cpu-only")

# Only these paths can change a number, so only changes under them mark a header dirty. Counting
# docs or untracked notes elsewhere in the tree would leave the flag always on.
PROVENANCE_PATHS = ("src", "scripts", "configs", "notebooks", "pyproject.toml")


def _git(*args: str, strip: bool = True) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # a missing git or a non-repo is reported as "unknown", never fatal
        )
        # `status --porcelain` lines start with a meaningful space (" D" = deleted, unstaged),
        # so those callers keep the leading whitespace.
        return out.stdout.strip() if strip else out.stdout.rstrip()
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
    """Query nvidia-smi for one row per GPU; return [] if it is missing or fails.

    Uses nvidia-smi rather than torch because the driver and MIG topology are what the spikes
    test, and torch may not be importable yet.
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
    """MIG instance UUIDs if MIG is enabled, else an empty list."""
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

    `ru_maxrss` is in kilobytes on Linux (bytes on macOS); this assumes Linux.
    """
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def results_header(
    script: str,
    seed: int,
    hardware_path: str = "single-replica",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the header every results file starts with (docs/06-methodology.md §3).

    `git_dirty` records whether tracked or untracked files under `PROVENANCE_PATHS` differ from
    HEAD, in which case the numbers cannot be re-derived from a commit. `hardware_path` (ADR-005)
    decides which numbers may be reported, so an unknown value raises instead of passing a typo.
    """
    if hardware_path not in HARDWARE_PATHS:
        raise ValueError(f"hardware_path must be one of {HARDWARE_PATHS}, got {hardware_path!r}")

    status = _git("status", "--porcelain", "--", *PROVENANCE_PATHS, strip=False)
    header = {
        "script": script,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status),
        "seed": seed,
        "hardware_path": hardware_path,
        "hardware": _hardware(),
        "packages": _package_versions(),
    }
    if status:
        # Which paths were uncommitted, so a reader can tell a deleted notebook from edited code.
        header["git_dirty_paths"] = status.splitlines()[:50]
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
    """Write a raw artifact, such as a /metrics scrape, verbatim to artifacts/results/.

    Keeping the raw text beside the parsed summary lets the parse be checked
    (docs/06-methodology.md §4).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path
