#!/usr/bin/env python3
"""Run a notebook on Kaggle through its API, and bring back what it wrote.

    python scripts/kaggle_run.py notebooks/c3_calibrate_t4.ipynb --slug kvsim-c3-calibrate-t4 \
        --out artifacts/results/c3_kaggle_t4_raw

Pushing a notebook starts a run of it — the same thing as "Save & Run All" in the browser. This
pushes it as a PRIVATE notebook with a GPU (Kaggle's T4 ×2 machine) and internet on, waits for the
run to end, then downloads every file the notebook wrote and its execution log.

Needs the `kaggle` package (`pip install kaggle`) and a Kaggle API token where that tool looks for
one: the KAGGLE_API_TOKEN variable, or ~/.kaggle/access_token. This script never reads, prints or
stores the token, and nothing in this repository may contain one.

⚠️ A registered notebook is run ONCE. A second push is a second experiment: say so in
docs/09-hypotheses.md before doing it, and keep both results.
⚠️ Every run spends the account's weekly GPU quota (`kaggle quota` shows what is left).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import tempfile
import time

T4 = "NvidiaTeslaT4"  # Kaggle's name for the "GPU T4 ×2" machine
ENDED = ("COMPLETE", "ERROR", "CANCEL")  # substrings of the statuses that are final


def _api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("error: the `kaggle` package is not installed "
                         "(pip install kaggle)") from None
    api = KaggleApi()
    api.authenticate()
    return api


def push(api, notebook: pathlib.Path, slug: str, accelerator: str | None) -> str:
    """Push `notebook` as a private kernel and start it. Returns its reference, owner/slug."""
    owner = api.get_config_value(api.CONFIG_NAME_USER)
    ref = f"{owner}/{slug}"
    meta = {
        "id": ref, "title": slug, "code_file": notebook.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": accelerator is not None,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(notebook, pathlib.Path(tmp) / notebook.name)
        (pathlib.Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        response = api.kernels_push(tmp, None, accelerator)
    if getattr(response, "error", None):
        raise SystemExit(f"error: Kaggle refused the push: {response.error}")
    print(f"pushed {ref} · version {getattr(response, 'version_number', '?')} · "
          f"{getattr(response, 'url', '')} · private · accelerator {accelerator or 'none'}",
          flush=True)
    return ref


def wait(api, ref: str, poll_s: float, timeout_s: float, *, just_pushed: bool = False,
         settle_s: float = 120.0) -> tuple[str, str]:
    """Poll until the run ends. Returns (status, failure message).

    Right after a push Kaggle can still answer with the PREVIOUS run's final status. So after a
    push, a final status counts only once this run was seen queued or running — or after
    `settle_s`, for a run too short to be caught in between."""
    t0, last, seen_active = time.time(), None, not just_pushed
    while True:
        try:
            r = api.kernels_status(ref)
            status, why, answered = str(r.status), getattr(r, "failure_message", "") or "", True
        except Exception as e:  # noqa: BLE001 - a dropped connection must not lose a GPU run
            status, why, answered = f"(status call failed: {type(e).__name__})", "", False
        if status != last:
            print(f"  {time.time() - t0:6.0f} s  {status}", flush=True)
            last = status
        # Only Kaggle's own answer can end the wait: "ConnectionError" also contains "ERROR".
        final = answered and any(word in status.upper() for word in ENDED)
        seen_active = seen_active or (answered and not final)
        if final and (seen_active or time.time() - t0 > settle_s):
            return status, why
        if time.time() - t0 > timeout_s:
            return (f"{status} (gave up waiting after {timeout_s:.0f} s; the run may still "
                    "go on)"), why
        time.sleep(poll_s)


def readable_log(raw: str) -> str:
    """Kaggle's execution log is JSON rows of {stream_name, time, data}. This is what the notebook
    PRINTED, in order, as plain text — with pip's download progress bars dropped: they are most of
    the bytes and say nothing. Nothing else is removed. A log in another format is returned as is.
    """
    try:
        rows = json.loads(raw)
        text = "".join(r.get("data", "") for r in rows if r.get("stream_name") == "stdout")
    except (ValueError, AttributeError, TypeError):
        return raw
    keep = [ln for ln in text.splitlines() if not re.search("━|\x1b\\[2K", ln)]
    return "\n".join(keep) + "\n"


def fetch(api, ref: str, out: pathlib.Path) -> list[str]:
    """Download every output file of the latest run, and its log."""
    out.mkdir(parents=True, exist_ok=True)
    files, _ = api.kernels_output(ref, str(out), force=True, quiet=True)
    # The run's own log is <slug>.log at the top of the output — NOT any file that ends in .log:
    # the notebook may have written logs of its own (an engine's, say) into a folder.
    log = out / f"{ref.split('/')[-1]}.log"
    if not log.exists():
        try:
            text = api.kernels_logs(ref)
        except Exception as e:  # noqa: BLE001 - the files matter more than the log
            text = f"(the log could not be fetched: {type(e).__name__}: {e})"
        log = out / "kaggle_execution.log"
        log.write_text(text if isinstance(text, str) else json.dumps(text))
        files = [*files, str(log)]
    (out / "notebook_stdout.txt").write_text(readable_log(log.read_text(errors="replace")))
    return [*files, str(out / "notebook_stdout.txt")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("notebook", type=pathlib.Path)
    ap.add_argument("--slug", required=True, help="the notebook's name on Kaggle (letters, - )")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="where to put what it wrote")
    ap.add_argument("--accelerator", default=T4, help=f"default {T4}; 'none' for a CPU run")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between status checks")
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="stop waiting after this")
    ap.add_argument("--fetch-only", action="store_true",
                    help="start nothing: wait for the run already pushed under --slug, and fetch")
    args = ap.parse_args()
    api = _api()
    if args.fetch_only:
        ref = f"{api.get_config_value(api.CONFIG_NAME_USER)}/{args.slug}"
    else:
        ref = push(api, args.notebook, args.slug,
                   None if args.accelerator.lower() == "none" else args.accelerator)
    status, why = wait(api, ref, args.poll, args.timeout, just_pushed=not args.fetch_only)
    files = fetch(api, ref, args.out)
    print(f"{ref}: {status}" + (f" — {why}" if why else ""))
    for f in files:
        print("  ", f)
    (args.out / "kaggle_run.json").write_text(json.dumps(
        # The notebook's name only: whose Kaggle account ran it is not the repository's business.
        {"kernel": ref.split("/")[-1], "status": status, "failure_message": why,
         "notebook": str(args.notebook),
         "accelerator": args.accelerator, "files": [pathlib.Path(f).name for f in files]},
        indent=1) + "\n")
    return 0 if "COMPLETE" in status.upper() else 1


if __name__ == "__main__":
    sys.exit(main())
