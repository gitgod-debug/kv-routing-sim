#!/usr/bin/env python3
"""Phase-0 spike: does a live vLLM expose a MOVING prefix-cache hit rate, and on this hardware?

⛔ This script gates the whole project. `CHECKPOINTS.md` Phase 0: no simulator code is written until
it passes. Every assumption underneath the calibration plan — that prefix caching behaves as
documented on this GPU, that the two counters exist and move, that the `vllm:kv_block_*` histograms
are reachable — has never been executed once, and a cache model designed against documentation is
how `Mike-Ew/kv-cache-routing` ended up citing a metric that does not exist
(see `docs/04-prior-art.md`).

WHAT IT CHECKS, in order of how badly a "no" would hurt:

  1. `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` are PRESENT.          (ADR-008)
  2. Both MOVE when a shared-prefix workload is sent. Present-but-flat is a silent
     failure mode and the one most likely to be mistaken for success.
  3. `vllm:prefix_cache_hit_rate` is ABSENT. We expect absent; if it is ever present
     on some version, ADR-008 is wrong and must be amended. Checked, not assumed.
  4. The hit rate with prefix caching ON is materially higher than with it OFF. This
     is the only arm that proves the mechanism rather than the plumbing.
  5. The three `vllm:kv_block_*` histograms appear under `--kv-cache-metrics-sample`.
     ⚠️ Hypothesis H4 is gated on this. If absent, amend H4 with a date — do not keep
     the narrative.

DELIBERATELY STDLIB-ONLY. No httpx, no prometheus-client, no import of this project's own package,
so the file can be scp'd to an unfamiliar box on its own and run against whatever is there. The
thing under test is the environment; a spike that needs its environment installed first cannot test
it. (`_spike_common.py` is the one optional import, for the provenance header.)

Usage:
    python scripts/spike_vllm_metrics.py --model Qwen/Qwen3-0.6B
    python scripts/spike_vllm_metrics.py --model <model> --no-off-arm   # skip the caching-off run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _spike_common import results_header, write_raw, write_results
except ImportError:  # running the file in isolation on a bare box
    results_header = None  # type: ignore[assignment]

# ── The metric families under test. Exact spelling; these are grep targets on a screen. ──────────
#
# ⚠️ prometheus_client appends `_total` to counter families, so the scraped line is very likely
# `vllm:prefix_cache_queries_total{...}`. Match the family with an OPTIONAL `_total` — hard-coding
# either spelling is how this check returns a confident false negative.
COUNTERS = ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits")
MUST_NOT_EXIST = "vllm:prefix_cache_hit_rate"  # went away with V0. See ADR-008.
KV_BLOCK_HISTOGRAMS = (
    "vllm:kv_block_lifetime_seconds",
    "vllm:kv_block_idle_before_evict_seconds",
    "vllm:kv_block_reuse_gap_seconds",
)
OTHER_WANTED = (
    "vllm:time_to_first_token_seconds",
    "vllm:num_requests_waiting",
    "vllm:request_prefill_kv_computed_tokens",
)

SHARED_SYSTEM_PROMPT = (
    "You are a meticulous operations assistant embedded in a document-processing pipeline. "
    "You have access to the following tools, and you must reason about which to call before "
    "answering. Tool 1: search_invoices(query, date_range) returns matching invoice records. "
    "Tool 2: fetch_page(document_id, page_number) returns the rendered text of one page. "
    "Tool 3: escalate(reason) hands the case to a human reviewer with a written justification. "
    "Always state your reasoning before calling a tool, never call more than one tool per turn, "
    "and never fabricate a document id that did not come from a previous tool result. "
    "If a tool returns an error, explain what you would try next rather than retrying blindly. "
) * 4  # long enough to span many 16-token blocks, which is what makes sharing measurable


def http_get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        # fixed localhost URL, not user input
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # fixed localhost URL, not user input
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)


def parse_metric_sum(scrape: str, family: str) -> float | None:
    """Sum every sample of one metric family, tolerating an optional `_total` and any labels.

    Returns None if the family is absent — which is a different answer from 0.0, and the
    difference is the whole point of check 1 vs check 2.
    """
    pattern = re.compile(
        r"^" + re.escape(family) + r"(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+.\-]+)\s*$",
        re.MULTILINE,
    )
    values = [float(m.group(1)) for m in pattern.finditer(scrape)]
    return sum(values) if values else None


def family_present(scrape: str, family: str) -> bool:
    """Is this family in the scrape at all — as a sample, or only as HELP/TYPE metadata?

    A histogram appears as `_bucket`/`_count`/`_sum` lines, so match the family as a line prefix
    rather than as a whole metric name.
    """
    return bool(re.search(r"^" + re.escape(family), scrape, re.MULTILINE))


def flag_supported(flag: str) -> bool | None:
    """Does `vllm serve --help` mention this flag? None if vllm could not be run at all.

    This is the direct answer to the third row of the spec's feasibility risk table: the version
    that actually installs may not be the version the plan was written against.
    """
    try:
        out = subprocess.run(
            ["vllm", "serve", "--help"], capture_output=True, text=True, timeout=180, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return flag in (out.stdout + out.stderr)


def wait_until_ready(port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status, _ = http_get(f"http://127.0.0.1:{port}/health", timeout=5.0)
        if status == 200:
            return True
        status, _ = http_get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
        if status == 200:
            return True
        time.sleep(3.0)
    return False


def start_server(
    model: str, port: int, prefix_caching: bool, kv_sample: str | None, extra: list[str]
) -> tuple[subprocess.Popen[bytes], list[str]]:
    """Launch one vLLM server. Returns the process and the exact argv, which is README content.

    ⚠️ In vLLM V1 prefix caching is ON by default, so the "off" arm needs the explicit negation —
    omitting `--enable-prefix-caching` does NOT turn it off. Getting that backwards would make the
    off/on comparison meaningless while still producing two plausible numbers.
    """
    argv = ["vllm", "serve", model, "--port", str(port)]
    argv += ["--enable-prefix-caching"] if prefix_caching else ["--no-enable-prefix-caching"]
    if kv_sample:
        argv += ["--kv-cache-metrics-sample", kv_sample]
    argv += extra
    # argv is built here, never shell-interpolated
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc, argv


def send_shared_prefix_workload(port: int, model: str, n_requests: int) -> dict[str, Any]:
    """Send N requests that share a long prefix and differ only in the tail.

    That is the whole mechanism under test: identical leading blocks, so every request after the
    first should hit cache on the shared span. If hits do not move here, they will not move
    anywhere.
    """
    url = f"http://127.0.0.1:{port}/v1/completions"
    latencies: list[float] = []
    errors: list[str] = []
    for i in range(n_requests):
        prompt = f"{SHARED_SYSTEM_PROMPT}\n\nUser question {i}: summarise the case in one line."
        t0 = time.monotonic()
        status, body = http_post_json(
            url, {"model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0.0}
        )
        latencies.append(time.monotonic() - t0)
        if status != 200:
            errors.append(f"request {i}: status {status}: {body[:200]}")
    return {
        "requests_sent": n_requests,
        "errors": errors,
        "wall_clock_s": round(sum(latencies), 3),
        "mean_request_s": round(sum(latencies) / max(len(latencies), 1), 4),
    }


def run_arm(
    arm: str, args: argparse.Namespace, prefix_caching: bool, kv_sample: str | None, port: int
) -> dict[str, Any]:
    """One server lifecycle: start, scrape, load, scrape, stop. Returns the arm's findings."""
    print(f"\n=== arm: {arm} (prefix_caching={prefix_caching}) ===", flush=True)
    proc, argv = start_server(args.model, port, prefix_caching, kv_sample, args.extra_arg)
    print("  argv:", " ".join(argv), flush=True)
    arm_result: dict[str, Any] = {"arm": arm, "argv": argv, "prefix_caching": prefix_caching}
    try:
        if not wait_until_ready(port, args.startup_timeout):
            arm_result["error"] = f"server not ready within {args.startup_timeout}s"
            # The server log is the only evidence of WHY, so keep the tail of it.
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=30)
                arm_result["server_log_tail"] = out.decode("utf-8", errors="replace")[-4000:]
            except subprocess.TimeoutExpired:
                proc.kill()
            return arm_result

        status, before = http_get(f"http://127.0.0.1:{port}/metrics", timeout=20.0)
        arm_result["metrics_status"] = status
        arm_result["before"] = {f: parse_metric_sum(before, f) for f in COUNTERS}

        arm_result["workload"] = send_shared_prefix_workload(port, args.model, args.requests)

        time.sleep(2.0)  # let the metrics logger flush
        _, after = http_get(f"http://127.0.0.1:{port}/metrics", timeout=20.0)
        arm_result["after"] = {f: parse_metric_sum(after, f) for f in COUNTERS}

        # ── The five checks ──────────────────────────────────────────────────────────────────
        present = {f: parse_metric_sum(after, f) is not None for f in COUNTERS}
        moved = {
            f: (
                arm_result["after"][f] is not None
                and arm_result["before"][f] is not None
                and arm_result["after"][f] > arm_result["before"][f]
            )
            for f in COUNTERS
        }
        q = arm_result["after"]["vllm:prefix_cache_queries"]
        h = arm_result["after"]["vllm:prefix_cache_hits"]
        arm_result["checks"] = {
            "counters_present": present,
            "counters_moved": moved,
            # Expected False. True would mean ADR-008 is wrong on this version — amend it.
            "hit_rate_metric_exists": family_present(after, MUST_NOT_EXIST),
            "kv_block_histograms_present": {
                f: family_present(after, f) for f in KV_BLOCK_HISTOGRAMS
            },
            "other_wanted_present": {f: family_present(after, f) for f in OTHER_WANTED},
        }
        # The derived hit rate. ⛔ Two counters, never a gauge (ADR-008).
        arm_result["derived_hit_rate"] = (
            round(h / q, 6) if (q not in (None, 0) and h is not None) else None
        )

        raw_name = f"spike_vllm_metrics_{arm}.metrics.txt"
        if results_header is not None:
            write_raw(raw_name, after)
            arm_result["raw_scrape"] = f"artifacts/results/{raw_name}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return arm_result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="HF model id to serve, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--requests", type=int, default=64, help="shared-prefix requests per arm")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--kv-cache-metrics-sample", default="1.0", help="'' to omit the flag entirely")
    ap.add_argument(
        "--no-off-arm",
        action="store_true",
        help="skip the prefix-caching-OFF arm (halves GPU time, weakens the evidence)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra vllm serve args, repeatable (e.g. --extra-arg --max-model-len "
        "--extra-arg 4096)",
    )
    args = ap.parse_args()

    # Probe the flag BEFORE spending a server startup on it.
    kv_flag_ok = flag_supported("--kv-cache-metrics-sample")
    print(f"--kv-cache-metrics-sample supported by this vllm: {kv_flag_ok}", flush=True)
    if kv_flag_ok is None:
        print("⛔ could not run `vllm serve --help` — is the [calib] extra installed?", flush=True)
        return 2
    kv_sample = (
        args.kv_cache_metrics_sample if (kv_flag_ok and args.kv_cache_metrics_sample) else None
    )
    if not kv_flag_ok:
        print(
            "⚠️  flag absent on this version: the three vllm:kv_block_* histograms will be "
            "unavailable, so HYPOTHESIS H4 IS UNTESTABLE AS WRITTEN. Amend docs/09-hypotheses.md "
            "with a date rather than keeping the narrative.",
            flush=True,
        )

    arms = [run_arm("caching_on", args, True, kv_sample, args.port)]
    if not args.no_off_arm:
        arms.append(run_arm("caching_off", args, False, kv_sample, args.port + 1))

    on = next(a for a in arms if a["arm"] == "caching_on")
    off = next((a for a in arms if a["arm"] == "caching_off"), None)

    # ── The gate verdict. Stated explicitly so the exit code is not the only signal. ─────────
    checks = on.get("checks", {})
    gate = {
        "counters_present": all(checks.get("counters_present", {}).values()) or False,
        "counters_moved": all(checks.get("counters_moved", {}).values()) or False,
        "hit_rate_metric_correctly_absent": checks.get("hit_rate_metric_exists") is False,
        "kv_block_histograms_present": (
            all(checks.get("kv_block_histograms_present", {}).values())
            if checks.get("kv_block_histograms_present")
            else False
        ),
        "caching_on_beats_off": (
            on.get("derived_hit_rate") is not None
            and off is not None
            and off.get("derived_hit_rate") is not None
            and on["derived_hit_rate"] > off["derived_hit_rate"]
        )
        if off is not None
        else None,
    }
    # H4's gate is reported, not required — the exit gate in CHECKPOINTS.md is counters present
    # AND moving. Conflating "nice to have" with "blocking" is how a gate stops meaning anything.
    gate["PHASE_0_METRIC_GATE"] = gate["counters_present"] and gate["counters_moved"]

    payload: dict[str, Any] = {}
    if results_header is not None:
        payload.update(
            results_header(
                "scripts/spike_vllm_metrics.py",
                seed=args.seed,
                hardware_path="single-replica",
                extra={
                    "model": args.model,
                    "kv_cache_metrics_sample_flag_supported": kv_flag_ok,
                    "requests_per_arm": args.requests,
                },
            )
        )
    payload["gate"] = gate
    payload["arms"] = arms

    print("\n" + json.dumps(gate, indent=2))
    if results_header is not None:
        path = write_results("spike_vllm_metrics.json", payload)
        print(f"\nwrote {path}")
        print(
            "⛔ commit BOTH the .json and the .metrics.txt raw scrapes — "
            "the raw scrape is the evidence."
        )
    else:
        print(json.dumps(payload, indent=2))

    if not gate["PHASE_0_METRIC_GATE"]:
        print(
            "\n⛔ PHASE 0 GATE FAILED. Do not start simulator code. Fix or document the cause, "
            "and log the friction in the upstream log — it may be worth reporting upstream.",
            flush=True,
        )
        return 1
    print("\n✅ metric gate passed. Next: scripts/spike_two_servers.py (coexistence).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
