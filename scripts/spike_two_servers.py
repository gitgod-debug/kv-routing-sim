#!/usr/bin/env python3
"""Phase-0 spike: can TWO vLLM servers coexist on this one GPU, and by which path?

The spec planned the 2-replica validation on Kaggle's dual-T4 session and flagged, correctly, that
nobody had ever checked whether that works. (ADR-005 briefly moved this to A100/H100 + MIG; reversed
2026-09-22 — Kaggle 2x T4 is the hardware, and two PHYSICAL GPUs beat MIG.) Paths, in order:
MIG instances, with a memory-split fallback. This script answers which of the three states we are in
BEFORE Phase 6 depends on the answer.

  path = "mig"           two MIG instances. Partitions SMs *and* memory.
                         → hit rate AND TTFT percentiles are both reportable.
  path = "memory-split"  two servers on one un-partitioned card, ~45% memory each.
                         → hit rate is reportable; ⛔ TTFT percentiles are NOT. The two servers
                           contend for the same SMs, so a latency difference measured here is an
                           artifact of the test rig, not an effect of the routing policy. Reporting
                           it as a routing result would be the exact error this project exists to
                           be trusted about.
  path = None            neither works. Phase 6 takes its documented fallback: a single-replica
                         calibration plus a reproducible account of why 2 replicas failed. That
                         write-up is itself an OSS-PR premise.

DELIBERATELY STDLIB-ONLY, for the same reason as `spike_vllm_metrics.py`: the environment is the
thing under test.

Usage:
    python scripts/spike_two_servers.py --model Qwen/Qwen3-0.6B
    python scripts/spike_two_servers.py --model <model> --force-path memory-split
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _spike_common import results_header, write_results
except ImportError:
    results_header = None  # type: ignore[assignment]


def http_get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        # fixed localhost URL, not user input
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)


def mig_uuids() -> list[str]:
    """MIG instance UUIDs from `nvidia-smi -L`.

    Lines look like:
      GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-...)
        MIG 3g.40gb  Device  0: (UUID: MIG-abc123...)
    We want the MIG-… UUIDs, which are what CUDA_VISIBLE_DEVICES accepts.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return []
    uuids = []
    for line in out.stdout.splitlines():
        if "MIG-" in line and "UUID:" in line:
            frag = line.split("UUID:", 1)[1].strip().rstrip(")").strip()
            if frag.startswith("MIG-"):
                uuids.append(frag)
    return uuids


def mig_mode() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def wait_until_ready(port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if http_get(f"http://127.0.0.1:{port}/health", timeout=5.0)[0] == 200:
            return True
        if http_get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)[0] == 200:
            return True
        time.sleep(3.0)
    return False


def start(
    model: str, port: int, device: str | None, mem_util: float | None, extra: list[str]
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    import os

    env = dict(os.environ)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    argv = ["vllm", "serve", model, "--port", str(port), "--enable-prefix-caching"]
    if mem_util is not None:
        argv += ["--gpu-memory-utilization", str(mem_util)]
    argv += extra
    proc = subprocess.Popen(  # argv is built here, never shell-interpolated
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    return proc, {"argv": argv, "CUDA_VISIBLE_DEVICES": device, "port": port}


def try_pair(
    model: str,
    devices: tuple[str | None, str | None],
    mem_util: float | None,
    ports: tuple[int, int],
    timeout: float,
    extra: list[str],
) -> dict[str, Any]:
    """Start two servers concurrently and report whether BOTH became ready."""
    procs: list[subprocess.Popen[bytes]] = []
    launched: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    try:
        for dev, port in zip(devices, ports, strict=True):
            proc, info = start(model, port, dev, mem_util, extra)
            procs.append(proc)
            launched.append(info)
            print(f"  launched: {' '.join(info['argv'])}  (dev={dev})", flush=True)
        result["launched"] = launched

        # ⚠️ Both must be ready SIMULTANEOUSLY. Sequential readiness proves nothing: the whole
        # point is concurrent contention for one arrival stream (see ADR-005's rejected
        # "two sequential runs, stitched" alternative).
        ready = {}
        for port in ports:
            ready[port] = wait_until_ready(port, timeout)
            print(f"  port {port} ready: {ready[port]}", flush=True)
        # Re-check the first one: a second server can OOM the first out from under us, and that
        # failure only shows up on a re-check.
        still_ready = {port: wait_until_ready(port, 15.0) for port in ports}
        result["ready_first_pass"] = ready
        result["ready_after_both_up"] = still_ready
        result["both_coexist"] = all(still_ready.values())

        for proc, info in zip(procs, launched, strict=True):
            if proc.poll() is not None:
                info["exited_early_returncode"] = proc.returncode
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                out, _ = proc.communicate(timeout=60)
                if proc.returncode not in (0, -15):  # -15 is our own SIGTERM
                    result.setdefault("server_log_tails", []).append(
                        out.decode("utf-8", errors="replace")[-3000:]
                    )
            except subprocess.TimeoutExpired:
                proc.kill()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--ports", type=int, nargs=2, default=[8000, 8001])
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--memory-split-util", type=float, default=0.45)
    ap.add_argument(
        "--force-path",
        choices=["mig", "memory-split"],
        help="skip detection and test only this path",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-arg", action="append", default=[])
    args = ap.parse_args()

    uuids = mig_uuids()
    findings: dict[str, Any] = {
        "mig_mode_current": mig_mode(),
        "mig_instances_found": uuids,
        "attempts": {},
    }
    print(f"MIG mode: {findings['mig_mode_current']}; MIG instances: {len(uuids)}", flush=True)

    chosen: str | None = None

    want_mig = args.force_path in (None, "mig") and len(uuids) >= 2
    if want_mig:
        print("\n=== attempting path: mig ===", flush=True)
        findings["attempts"]["mig"] = try_pair(
            args.model,
            (uuids[0], uuids[1]),
            None,  # each MIG instance has its own memory; do not also split it
            tuple(args.ports),
            args.startup_timeout,
            args.extra_arg,
        )
        if findings["attempts"]["mig"].get("both_coexist"):
            chosen = "mig"
    elif args.force_path != "memory-split":
        print(
            f"\n⚠️  fewer than 2 MIG instances ({len(uuids)}). Either MIG is off, or it is on with "
            "one instance. See the MIG setup commands in docs/07-environment.md — enabling "
            "it needs root and no running processes.",
            flush=True,
        )

    if chosen is None and args.force_path in (None, "memory-split"):
        print("\n=== attempting path: memory-split (fallback) ===", flush=True)
        findings["attempts"]["memory-split"] = try_pair(
            args.model,
            (None, None),  # same visible device for both
            args.memory_split_util,
            tuple(args.ports),
            args.startup_timeout,
            args.extra_arg,
        )
        if findings["attempts"]["memory-split"].get("both_coexist"):
            chosen = "memory-split"

    findings["chosen_path"] = chosen
    findings["reportable"] = {
        "mig": {"hit_rate": True, "ttft_percentiles": True},
        "memory-split": {"hit_rate": True, "ttft_percentiles": False},
        None: {"hit_rate": False, "ttft_percentiles": False},
    }[chosen]

    payload: dict[str, Any] = {}
    if results_header is not None:
        payload.update(
            results_header(
                "scripts/spike_two_servers.py",
                seed=args.seed,
                hardware_path=chosen or "single-replica",
                extra={"model": args.model},
            )
        )
    payload["findings"] = findings

    print("\n" + json.dumps({k: v for k, v in findings.items() if k != "attempts"}, indent=2))
    if results_header is not None:
        print(f"\nwrote {write_results('spike_two_servers.json', payload)}")

    if chosen == "mig":
        print("\n✅ path = MIG. Phase 6 may report hit rate AND TTFT percentiles.", flush=True)
        return 0
    if chosen == "memory-split":
        print(
            "\n⚠️  path = memory-split. Phase 6 may report hit rate ONLY.\n"
            "⛔ Do NOT publish TTFT percentiles from this path — the SM contention is a test-rig\n"
            "   artifact, not a routing effect (ADR-005). TTFT comes from the single-replica run.",
            flush=True,
        )
        return 0
    print(
        "\n⛔ Neither path worked. Phase 6 takes its documented fallback: a single-replica\n"
        "   calibration plus a reproducible account of why 2 replicas failed. Write the cause\n"
        "   down: such an account may be worth reporting upstream.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
