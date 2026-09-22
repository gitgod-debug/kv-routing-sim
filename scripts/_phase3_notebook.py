"""Builds notebooks/phase3_calibration.ipynb. Called by scripts/phase3_prepare.py — not run alone.

The cells are plain Python kept here as raw strings, so the notebook is regenerated rather than
hand-edited: every constant it carries comes from configs/phase3_workloads.yaml, and its data cell
comes from the same run that made the predictions. `@@NAME@@` markers are filled in `build_notebook`.

The notebook is SELF-CONTAINED on purpose: the GPU box is a Kaggle kernel with no checkout of this
repo, so it may use only `vllm` and the standard library.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


def _md(text: str) -> dict[str, Any]:
    lines = textwrap.dedent(text).strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def _code(src: str) -> dict[str, Any]:
    lines = textwrap.dedent(src).strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }


INTRO = """
# Phase 3 — the Real-Engine Calibration

**What this is.** The experiment that turns kv-routing-sim's cache model from *"implements vLLM's
algorithm"* into *"predicts vLLM's numbers"*. One vLLM **@@VLLM@@** server — **@@MODEL@@ on one
Tesla T4** — receives three workloads, three runs each. Both of the simulator's cache models made
their predictions **before this notebook ever ran**: they are embedded below, hashed, and committed
to the repo as `artifacts/results/phase3_prereg/predictions.json`.

| | What is sent | `vllm` model predicts | `lru` model predicts | Why it is here |
|---|---|---|---|---|
| **W2** — control | @@N@@ requests one at a time, **1**-token outputs | hit rate **@@W2_V@@** | **@@W2_L@@** | the models agree: this tests the accounting itself |
| **W1** — discriminating | the same @@N@@, **64**-token outputs | **@@W1_V@@** | **@@W1_L@@** | they disagree by **@@W1_GAP@@ points**: decode blocks crowd a small cache — or they don't |
| **W3** — concurrent | the same trace, each request at its own arrival time | **preemptions > 0** | **0 preemptions**, hit @@W3_L@@ | admission and preemption under memory pressure |

**⭐ How to read the result.** For one-at-a-time requests on a fixed-size cache, vLLM is
deterministic, and so are both models. So on W1 and W2 a mismatch is not noise to average away — it
is a **model error**, and the per-request comparison names the request. Three runs per workload
test that determinism as well.

The cache is deliberately small. `--num-gpu-blocks-override @@BLOCKS@@` (one is vLLM's null block,
so @@USABLE@@ are usable) makes capacity an exact, controlled input — instead of a measurement that
drifted 4.8% between identical launches in Phase 0 — and makes the cache overflow, which is where
the two models part ways.
"""

HOW_TO_RUN = """
## How to run it (Kaggle)

1. **Settings → Accelerator: GPU T4 x2**, **Internet: on**. One T4 is used; the other stays idle.
2. **Run All** (or *Save Version → Save & Run All* to run unattended). About **40 minutes**:
   install 5–8 · first engine start 5–10 (downloads ~8 GB) · W2 ~2 · W1 ~15 · sweep ~2 · W3 ~3.
3. When it finishes, download **`phase3_results.zip`** (right sidebar → *Output*, or the file
   browser under `/kaggle/working`).
4. Back in the repo: unzip it into `artifacts/results/phase3_kaggle_t4/` and run
   `python scripts/phase3_analyze.py`.

**If a cell stops with ⛔,** its message says why. Download the zip anyway — the partial results
and the server log are what diagnose it. The last cell stops the server; run it in any case.

**Do not edit the data cell.** It carries the exact token ids the predictions were computed on, and
the notebook refuses to run on anything else. Changing the experiment means editing
`configs/phase3_workloads.yaml` and re-running `scripts/phase3_prepare.py` — which re-makes the
predictions, as it must.

**Self-contained:** no repo checkout, no dataset, no HF token (Qwen3-4B is ungated). Only `vllm`
and the standard library.

## What a T4 can and cannot tell us

Every number W1 and W2 predict is a **cache count**, and cache counts are hardware-independent: the
same trace gives the same counts on any GPU. Timings from this run are **T4 timings** (fp16,
TRITON_ATTN, no FlashAttention-2) and are labelled that way wherever they are used.
"""

CONFIG_CELL = r'''
# ── Generated from configs/phase3_workloads.yaml. Do not edit here: edit the config and re-run
#    scripts/phase3_prepare.py, or the embedded predictions stop meaning anything.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
NUM_GPU_BLOCKS = @@BLOCKS@@          # --num-gpu-blocks-override; one is vLLM's null block
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
RUNS_PER_WORKLOAD = @@RUNS@@
SWEEP_BATCH_SIZES = @@SWEEP_BATCHES@@
SWEEP_OUTPUT_TOKENS = @@SWEEP_OUT@@

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_INDEX = 0                 # one T4 is the replica; the second stays idle until Phase 6
GPU_MEM_UTIL = 0.85
PORT = 8000
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "phase3_results"
'''

ENV_CELL = r'''
import json
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

pathlib.Path(OUT_DIR).mkdir(exist_ok=True)


class Stop(Exception):
    """Halts Run-All with a reason. Download phase3_results.zip anyway."""


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout, check=False).stdout.strip()
    except Exception as e:  # an unfamiliar box: any binary may be missing
        return f"<failed: {e}>"


ENV = {
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "cpu_count": os.cpu_count(),
    "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version "
               "--format=csv,noheader"),
    "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
}
for k, v in ENV.items():
    print(f"{k:10s} {v}")
if "T4" not in ENV["gpus"]:
    print("\n⚠️  Not a Tesla T4. Cache counts stay valid (they are hardware-independent), but "
          "every\n   timing from this run must be labelled with the GPU above, not as a T4 figure.")
'''

INSTALL_CELL = r'''
# PINNED: the predictions were made for this version, and Phase 0 v2 passed every check on it.
!pip install -q vllm==@@VLLM@@
'''

AUDIT_CELL = r'''
import importlib.metadata

import vllm

VLLM_VERSION = vllm.__version__
if VLLM_VERSION != VLLM_PINNED:
    raise Stop(f"⛔ vllm {VLLM_VERSION} is imported, but the predictions were made for "
               f"{VLLM_PINNED}. Restart the kernel and run from the top.")


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


ENV["packages"] = {p: _version(p) for p in ("vllm", "torch", "transformers", "triton")}
print("packages:", ENV["packages"])

ROOT = pathlib.Path(vllm.__file__).parent
SOURCES = {}
for p in ROOT.rglob("*.py"):
    try:
        SOURCES[str(p.relative_to(ROOT))] = p.read_text(errors="replace")
    except OSError:
        pass


def found_in(needle):
    return next((f for f, text in SOURCES.items() if needle in text), None)


# (what the simulator relies on, the exact source text in vLLM that implements it)
CITED = [
    ("null block reserved at start-up", "self.null_block = self.free_block_queue.popleft()"),
    ("uncached blocks reused first (prepend)", "prepend_n"),
    ("finished requests freed tail-first", "Free blocks in reverse order"),
    ("lookup capped at prompt_length - 1", "max_cache_hit_length = request.num_tokens - 1"),
    ("queries counted at admission only",
     "Record at admission so unscheduled lookups are not counted"),
    ("eviction event fired before hash check", "on_block_evicted(block)"),
    ("block access history bounded to 4", "deque(maxlen=4)"),
    ("reset also clears block metrics", "self.metrics_collector.reset()"),
    ("async scheduler caches on each output",
     "request, request.num_computed_tokens - request.num_output_placeholders"),
    ("full-prompt admission check, default on", "scheduler_reserve_full_isl: bool = True"),
    ("admission watermark, default 0", "watermark: float = Field(default=0.0"),
]
# (what this notebook sends or reads, the source text that serves it)
USED = [
    ("metric vllm:prefix_cache_queries", "vllm:prefix_cache_queries"),
    ("metric vllm:prefix_cache_hits", "vllm:prefix_cache_hits"),
    ("metric vllm:prompt_tokens_by_source", "vllm:prompt_tokens_by_source"),
    ("metric vllm:num_preemptions", "vllm:num_preemptions"),
    ("metric vllm:kv_block_lifetime_seconds", "vllm:kv_block_lifetime_seconds"),
    ("metric vllm:kv_block_idle_before_evict_seconds", "vllm:kv_block_idle_before_evict_seconds"),
    ("metric vllm:kv_block_reuse_gap_seconds", "vllm:kv_block_reuse_gap_seconds"),
    ("metric vllm:num_requests_running", "vllm:num_requests_running"),
    ("flag --num-gpu-blocks-override", "num_gpu_blocks_override"),
    ("flag --enable-prompt-tokens-details", "enable_prompt_tokens_details"),
    ("request field ignore_eos", "ignore_eos"),
    ("POST /reset_prefix_cache", '"/reset_prefix_cache"'),
    ("GET /server_info", '"/server_info"'),
]
AUDIT = {}
print(f"\nvllm {VLLM_VERSION}: {len(SOURCES):,} source files read")
for group, rows in (("the simulator cites", CITED), ("this notebook uses", USED)):
    print(f"── {group}")
    for what, needle in rows:
        AUDIT[what] = found_in(needle)
        print(f"  {'OK' if AUDIT[what] else '--'}  {what:44s} {AUDIT[what] or 'NOT FOUND'}")
AUDIT["vllm:prefix_cache_hit_rate (must NOT exist)"] = found_in('"vllm:prefix_cache_hit_rate"')
print(f"  {'OK' if not AUDIT['vllm:prefix_cache_hit_rate (must NOT exist)'] else '!!'}  "
      "vllm:prefix_cache_hit_rate does not exist (ADR-008)")

missing_used = [w for w, _ in USED if AUDIT[w] is None]
missing_cited = [w for w, _ in CITED if AUDIT[w] is None]
if missing_used:
    raise Stop(f"⛔ this vLLM lacks what the notebook needs: {missing_used}")
if missing_cited:
    print(f"\n⚠️  not found verbatim: {missing_cited}. Run anyway — the counts will show whether "
          "the behaviour itself changed.")
'''

DATA_CELL = r'''
import base64
import hashlib
import lzma

TRACE_SHA256 = "@@FP@@"
PREDICTIONS_SHA256 = "@@PRED_SHA@@"
BLOB = (
@@BLOB@@
)


def _canonical(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


DATA = json.loads(lzma.decompress(base64.b64decode(BLOB)))
TRACE, SWEEP_PROMPTS, PRED = DATA["trace"], DATA["decode_sweep_prompts"], DATA["predictions"]
_sent = {"trace": [[r["request_id"], r["arrival_ts"], r["session_id"], r["ids"]] for r in TRACE],
         "decode_sweep_prompts": SWEEP_PROMPTS}
if hashlib.sha256(_canonical(_sent)).hexdigest() != TRACE_SHA256:
    raise Stop("⛔ the embedded trace does not match its fingerprint: the data cell was edited")
if hashlib.sha256(_canonical(PRED)).hexdigest() != PREDICTIONS_SHA256:
    raise Stop("⛔ the embedded predictions do not match their hash: the data cell was edited")

_ids = [t for r in TRACE for t in r["ids"]]
print(f"trace ✓  {len(TRACE)} requests · {len(_ids):,} prompt tokens · lengths "
      f"{sorted({len(r['ids']) for r in TRACE})} · ids in [{min(_ids)}, {max(_ids)}]")
print(f"fingerprint ✓ {TRACE_SHA256[:16]}…   predictions ✓ {PREDICTIONS_SHA256[:16]}…")
print("\nPre-registered — committed to the repo before this run:")
for _name, _w in PRED["workloads"].items():
    _p = _w["predictions"]
    if _w["mode"] == "sequential":
        print(f"  {_name:14s} vllm model hit {_p['vllm']['hit_rate']:.4f}  "
              f"(computed {_p['vllm']['local_compute']:,})   lru model hit "
              f"{_p['lru']['hit_rate']:.4f}  (computed {_p['lru']['local_compute']:,})")
    else:
        _c = _p["timing_independent_claims"]
        print(f"  {_name:14s} vllm model: preemptions {_c['vllm']['preemptions']}   "
              f"lru model: 0 preemptions, hit {_c['lru']['hit_rate']:.4f}")
'''

SERVER_CELL = r'''
BASE = f"http://127.0.0.1:{PORT}"
SERVER = {}
RESET_MODE = "endpoint"  # falls back to "restart" if the dev endpoint is not served
ENGINE_ARGV = [
    "vllm", "serve", MODEL, "--port", str(PORT),
    "--dtype", "half",  # T4 has no bf16
    "--block-size", str(BLOCK_SIZE),
    "--num-gpu-blocks-override", str(NUM_GPU_BLOCKS),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
    "--gpu-memory-utilization", str(GPU_MEM_UTIL),
    "--enable-prefix-caching",
    "--kv-cache-metrics", "--kv-cache-metrics-sample", "1.0",
    "--enable-prompt-tokens-details",
    "--seed", "0",
]


def port_in_use():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def start_server(tag):
    # ⚠️ Refuse to launch while ANYTHING listens on the port. Otherwise a stale server (a cell
    # re-run without the clean-up) answers /health before the new process has had time to die
    # of "address already in use", and the run measures the wrong engine. Found by the dry-run:
    # checking our process first narrows that race but does not close it. A few seconds' grace
    # lets our own just-stopped server release the port.
    for _ in range(30):
        if not port_in_use():
            break
        time.sleep(1)
    else:
        raise Stop(f"⛔ something is already serving on port {PORT} — a server from an earlier "
                   "run? Run the clean-up cell (or !pkill -f 'vllm serve') and start again.")
    # VLLM_SERVER_DEV_MODE serves /reset_prefix_cache and /server_info.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX), VLLM_SERVER_DEV_MODE="1")
    log_path = pathlib.Path(OUT_DIR) / f"server_{tag}.log"
    logf = open(log_path, "w")  # noqa: SIM115 - must outlive this cell
    proc = subprocess.Popen(ENGINE_ARGV, stdout=logf, stderr=subprocess.STDOUT, env=env)
    SERVER.clear()
    SERVER.update(proc=proc, logf=logf, log_path=str(log_path))
    print("launched:", " ".join(ENGINE_ARGV), f"  (GPU {GPU_INDEX})")


def log_text():
    return pathlib.Path(SERVER["log_path"]).read_text(errors="replace") if SERVER else ""


def http(method, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, repr(e)


def wait_ready(timeout=STARTUP_TIMEOUT_S, every=30):
    """Ready = OUR process is alive AND the port answers, checked in that order: a stale server
    left on the port must never pass for ours (found by the Phase-0 dry-run)."""
    t0, last = time.time(), 0.0
    while time.time() - t0 < timeout:
        if SERVER["proc"].poll() is not None:
            print(f"  !! the server EXITED (returncode {SERVER['proc'].returncode}). Log tail:")
            print(log_text()[-3000:])
            if "address already in use" in log_text().lower():
                print("  -> another server owns the port: run the clean-up cell, or "
                      "!pkill -f 'vllm serve'")
            return False
        if http("GET", "/health", timeout=5)[0] == 200:
            print(f"  ready in {time.time() - t0:.0f}s")
            return True
        if time.time() - t0 - last >= every:
            last = time.time() - t0
            tail = log_text().strip().splitlines()
            print(f"    ...{last:.0f}s  {tail[-1][:110] if tail else '(log empty)'}")
        time.sleep(5)
    print(f"  !! not ready within {timeout}s")
    return False


def stop_server():
    if not SERVER:
        return
    SERVER["proc"].terminate()
    try:
        SERVER["proc"].wait(timeout=90)
    except subprocess.TimeoutExpired:
        SERVER["proc"].kill()
    SERVER["logf"].close()
    SERVER.clear()


def scrape():
    return http("GET", "/metrics", timeout=30)[1]


def metric(text, family, **labels):
    """Sum a family over matching labels. None = ABSENT, which is not the same as 0.
    prometheus_client appends _total to counters, so both spellings match."""
    lab = "".join(f'(?=[^}}]*{k}="{v}")' for k, v in labels.items())
    pat = re.compile(r"^" + re.escape(family) + r"(?:_total)?\{" + lab + r"[^}]*\}\s+(\S+)",
                     re.MULTILINE)
    vals = [float(x) for x in pat.findall(text)]
    return sum(vals) if vals else None


COUNTERS = {  # our name -> (metric family, labels)
    "prefix_cache_queries": ("vllm:prefix_cache_queries", {}),
    "prefix_cache_hits": ("vllm:prefix_cache_hits", {}),
    "local_compute": ("vllm:prompt_tokens_by_source", {"source": "local_compute"}),
    "local_cache_hit": ("vllm:prompt_tokens_by_source", {"source": "local_cache_hit"}),
    "preemptions": ("vllm:num_preemptions", {}),
    "hist_lifetime_count": ("vllm:kv_block_lifetime_seconds_count", {}),
    "hist_idle_count": ("vllm:kv_block_idle_before_evict_seconds_count", {}),
    "hist_reuse_gap_count": ("vllm:kv_block_reuse_gap_seconds_count", {}),
    "prompt_tokens": ("vllm:prompt_tokens", {}),
    "generation_tokens": ("vllm:generation_tokens", {}),
}


def read_counters(text):
    return {k: metric(text, fam, **lab) for k, (fam, lab) in COUNTERS.items()}


def settled_scrape(max_wait=180):
    """Scrape once nothing is running or waiting AND the counters have stopped moving. Stats
    reach /metrics a moment after the tokens they describe; an early read under-counts."""
    t0, prev = time.time(), None
    while True:
        text = scrape()
        busy = ((metric(text, "vllm:num_requests_running") or 0)
                + (metric(text, "vllm:num_requests_waiting") or 0))
        now = read_counters(text)
        if busy == 0 and now == prev:
            return text
        if time.time() - t0 > max_wait:
            raise Stop(f"⛔ the engine never settled within {max_wait}s (busy={busy})")
        prev = now
        time.sleep(1.0)


def deltas(before, after):
    b, a = read_counters(before), read_counters(after)
    return {k: None if a[k] is None else round(a[k] - (b[k] or 0.0)) for k in COUNTERS}


def fresh_cache(tag):
    """A cold cache before every run. The reset clears the hashes AND the block-metrics collector
    (block_pool.py reset_prefix_cache -> metrics_collector.reset()), so for every count predicted
    here a reset engine equals a freshly started one — without a 3-5 minute restart."""
    global RESET_MODE
    if RESET_MODE == "endpoint":
        for _ in range(15):
            code, body = http("POST", "/reset_prefix_cache")
            try:
                ok = code == 200 and json.loads(body).get("success") is True
            except ValueError:
                ok = False
            if ok:
                return
            if code in (404, 405):
                print("  /reset_prefix_cache is not served here: restarting the engine per run")
                RESET_MODE = "restart"
                break
            time.sleep(2)  # success=false: a finishing request still holds blocks
        else:
            raise Stop("⛔ the engine refused to reset its prefix cache 15 times")
    stop_server()
    start_server(tag)
    if not wait_ready():
        raise Stop("⛔ the restart failed — the log tail is above")


print("server helpers ready")
'''

CLIENT_CELL = r'''
def stream_one(ids, max_tokens):
    """One streamed completion: when it was sent, when each chunk arrived, and the usage block.
    The engine reports its cached-token count per request, so a mismatch can be pinned to the
    exact request that diverged."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"t_send": time.perf_counter(), "t_first": None, "t_last": None, "chunks": 0,
           "usage": None, "finish_reason": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter()
                obj = json.loads(data)
                if obj.get("choices"):
                    rec["chunks"] += 1
                    if rec["t_first"] is None:
                        rec["t_first"] = now
                    rec["t_last"] = now
                    reason = obj["choices"][0].get("finish_reason")
                    if reason:
                        rec["finish_reason"] = reason
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
    except Exception as e:
        rec["error"] = repr(e)
    rec["t_done"] = time.perf_counter()
    return rec


def record(r, rec, out, t_zero, scheduled=False):
    """Flatten one response into what is saved and compared. Send times are kept relative to the
    run's start for EVERY mode: the analysis re-simulates each run at the times it really ran."""
    u = rec["usage"] or {}
    n = u.get("completion_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    first, last = rec["t_first"], rec["t_last"]
    return {
        "request_id": r["request_id"],
        "valid": (rec["error"] is None and rec["finish_reason"] == "length" and n == out
                  and u.get("prompt_tokens") == len(r["ids"]) and cached is not None),
        "error": rec["error"],
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": n,
        "cached_tokens": cached,
        "finish_reason": rec["finish_reason"],
        "chunks": rec["chunks"],
        "ttft_ms": None if first is None else (first - rec["t_send"]) * 1e3,
        "decode_interval_ms": (last - first) / (n - 1) * 1e3 if first and n and n > 1 else None,
        "total_ms": (rec["t_done"] - rec["t_send"]) * 1e3,
        "send_offset_s": rec["t_send"] - t_zero,
        "scheduled_offset_s": r["arrival_ts"] if scheduled else None,
    }


def send_sequential(out):
    """One at a time: each request is sent only after the previous one has fully returned."""
    recs, t0, t_zero = [], time.time(), time.perf_counter()
    for i, r in enumerate(TRACE, 1):
        recs.append(record(r, stream_one(r["ids"], out), out, t_zero))
        if i % 20 == 0 or i == len(TRACE):
            el = time.time() - t0
            print(f"    {i:>3}/{len(TRACE)}  {el:5.0f}s elapsed, "
                  f"~{el / i * (len(TRACE) - i):4.0f}s to go")
    return recs


def send_concurrent(out, lead_s=2.0):
    """Each request fired at its own arrival time, open-loop: a slow response never delays the
    next send. Threads, because every stream blocks on its own socket."""
    t_zero = time.perf_counter() + lead_s
    sent = []
    with ThreadPoolExecutor(max_workers=len(TRACE)) as pool:
        for r in TRACE:
            delay = t_zero + r["arrival_ts"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            sent.append((r, pool.submit(stream_one, r["ids"], out)))
        recs = [record(r, f.result(), out, t_zero, scheduled=True) for r, f in sent]
    lag = max(x["send_offset_s"] - x["scheduled_offset_s"] for x in recs)
    print(f"    all {len(recs)} sent; worst lag behind the schedule {lag * 1e3:.0f} ms")
    return recs


def fmt(v):
    return "—" if v is None else f"{v:,}"


def nearest_rank(sorted_vals, q):
    """The simulator's percentile rule (sim/metrics.py): a value some request actually saw."""
    if not sorted_vals:
        return None
    return sorted_vals[max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1))))]


def compare_sequential(name, measured, recs):
    preds = PRED["workloads"][name]["predictions"]
    keys = PRED["compared_counters"]
    engine_cached = {x["request_id"]: x["cached_tokens"] for x in recs}
    q, h = measured["prefix_cache_queries"], measured["prefix_cache_hits"]
    hr = h / q if q else float("nan")
    print(f"    {'':24s} {'engine':>9s}   {'vllm model':>12s}    {'lru model':>12s}")
    for k in keys:
        cells = []
        for m in ("vllm", "lru"):
            p = preds[m].get(k)
            mark = " " if p is None else ("✓" if measured.get(k) == p else "✗")
            cells.append(f"{fmt(p):>12s} {mark}")
        print(f"    {k:24s} {fmt(measured.get(k)):>9s}   {cells[0]}   {cells[1]}")
    print(f"    {'hit rate':24s} {hr:9.4f}   {preds['vllm']['hit_rate']:12.4f}     "
          f"{preds['lru']['hit_rate']:12.4f}")
    verdict = {}
    for m in ("vllm", "lru", "vllm_if_one_more_token_committed"):
        p = preds[m]
        per_req = p["per_request_cached_tokens"]
        verdict[m] = {
            "counters_exact": all(p.get(k) is None or measured.get(k) == p[k] for k in keys),
            "requests_matching": sum(engine_cached.get(rid) == v for rid, v in per_req.items()),
            "requests": len(per_req),
            # from the exact counts, not the rounded hit_rate, so an exact match reads +0.00
            "hit_rate_error_pts": round(
                (hr - p["prefix_cache_hits"] / p["prefix_cache_queries"]) * 100, 4),
            "mismatched_requests": [[rid, engine_cached.get(rid), v] for rid, v in per_req.items()
                                    if engine_cached.get(rid) != v][:10],
        }
    for m in ("vllm", "lru"):
        v = verdict[m]
        exact = v["counters_exact"] and v["requests_matching"] == v["requests"]
        print(f"    ==> {m:4s} model {'EXACT   ' if exact else 'MISMATCH'}  counters "
              f"{'✓' if v['counters_exact'] else '✗'} · requests {v['requests_matching']}/"
              f"{v['requests']} · hit-rate error {v['hit_rate_error_pts']:+.2f} pts")
        if v["mismatched_requests"] and m == "vllm":
            print("        first mismatches [request, engine, predicted]:",
                  v["mismatched_requests"][:4])
    alt = verdict["vllm_if_one_more_token_committed"]
    if (preds["alternative_distinguishable"] and alt["counters_exact"]
            and not verdict["vllm"]["counters_exact"]):
        print("    ⚠️  the engine matches the pre-registered +1-token ALTERNATIVE: it commits one "
              "token more per request than the model")
    return verdict


def compare_concurrent(name, measured, recs):
    c = PRED["workloads"][name]["predictions"]["timing_independent_claims"]
    q, h = measured["prefix_cache_queries"], measured["prefix_cache_hits"]
    hr = h / q if q else float("nan")
    ttft = sorted(x["ttft_ms"] for x in recs if x["ttft_ms"] is not None)
    pct = {f"p{round(p * 100)}_ms": nearest_rank(ttft, p) for p in (0.5, 0.95, 0.99)}
    pre = measured["preemptions"]
    print(f"    engine: hit rate {hr:.4f} · computed {fmt(measured['local_compute'])} · "
          f"preemptions {fmt(pre)} · TTFT p50/p95/p99 "
          + " / ".join(f"{v:,.0f}" for v in pct.values() if v is not None) + " ms")
    lru_holds = pre == 0
    print(f"    lru model claimed 0 preemptions        → {'holds' if lru_holds else 'REFUTED'}")
    print(f"    lru model claimed hit rate {c['lru']['hit_rate']:.4f}    → engine {hr:.4f} "
          f"({(hr - c['lru']['hit_rate']) * 100:+.2f} pts)")
    vllm_holds = None
    if c["vllm"]["preemptions"] == "> 0":
        vllm_holds = bool(pre)
        print(f"    vllm model claimed preemptions > 0     → "
              f"{'holds' if vllm_holds else 'REFUTED'}")
    return {"hit_rate": hr, "ttft": pct, "lru_zero_preemptions_holds": lru_holds,
            "vllm_preemptions_claim_holds": vllm_holds,
            "lru_hit_rate_error_pts": round((hr - c["lru"]["hit_rate"]) * 100, 4)}


RUNS = []  # every run, appended as it finishes; each is also saved to its own file at once


def run_workload(name, i):
    w = PRED["workloads"][name]
    out, tag = w["output_tokens"], f"{name}_r{i}"
    print(f"\n── {name} · run {i}/{RUNS_PER_WORKLOAD} · {w['mode']} · {out}-token outputs")
    fresh_cache(tag)
    before = settled_scrape()
    t0 = time.time()
    recs = send_sequential(out) if w["mode"] == "sequential" else send_concurrent(out)
    wall = time.time() - t0
    after = settled_scrape()
    pathlib.Path(OUT_DIR, f"metrics_{tag}_before.txt").write_text(before)
    pathlib.Path(OUT_DIR, f"metrics_{tag}_after.txt").write_text(after)
    measured = deltas(before, after)
    invalid = [x["request_id"] for x in recs if not x["valid"]]
    sanity = {
        "prompt_tokens_total": [measured["prompt_tokens"], sum(len(r["ids"]) for r in TRACE)],
        "generation_tokens_total": [measured["generation_tokens"], out * len(TRACE)],
    }
    sane = all(a == b for a, b in sanity.values())
    if invalid:
        print(f"    ⛔ {len(invalid)} INVALID responses (error, wrong length or no usage), "
              f"e.g. {invalid[:3]}")
    if not sane:
        print(f"    ⚠️  engine totals differ from what was sent: {sanity}")
    compare = compare_sequential if w["mode"] == "sequential" else compare_concurrent
    # Validity is per REQUEST (right length, right prompt, usage present). The engine totals
    # above are reported, not gating: they test our reading of two counters, not the run.
    run = {"workload": name, "run": i, "mode": w["mode"], "output_tokens": out,
           "wall_s": round(wall, 3), "reset_mode": RESET_MODE, "valid": not invalid,
           "invalid_requests": invalid, "sanity": sanity, "measured": measured,
           "comparison": compare(name, measured, recs), "requests": recs}
    pathlib.Path(OUT_DIR, f"run_{tag}.json").write_text(json.dumps(run, indent=1))
    RUNS.append(run)
    return run


print("client helpers ready")
'''

START_CELL = r'''
start_server("engine")
if not wait_ready():
    raise Stop("⛔ the engine did not start — the log tail is above. Download the zip anyway.")

LOG = log_text()
_m = re.search(r"KV cache size: ([\d,]+) tokens", LOG)
CAPACITY = {
    "kv_cache_size_tokens_logged": int(_m.group(1).replace(",", "")) if _m else None,
    # max_concurrency (512 / (2048 / 16) = 4.0) x max_model_len (2048) = 8,192
    "expected_tokens": NUM_GPU_BLOCKS * BLOCK_SIZE,
    "override_logged": f"num_gpu_blocks_override={NUM_GPU_BLOCKS}" in LOG,
    "kv_cache_metrics_enabled": "kv_cache_metrics=True" in LOG,
}
for k, v in CAPACITY.items():
    print(f"  {k:30s} {v}")
if CAPACITY["kv_cache_size_tokens_logged"] != CAPACITY["expected_tokens"]:
    raise Stop(f"⛔ the engine logged {CAPACITY['kv_cache_size_tokens_logged']} tokens of KV "
               f"cache, not {CAPACITY['expected_tokens']:,}: every prediction assumes exactly "
               f"{NUM_GPU_BLOCKS} blocks. Check the log for the override line.")
if not CAPACITY["kv_cache_metrics_enabled"]:
    print("⚠️  kv_cache_metrics=True is not in the config dump: the kv_block_* counts will read "
          "as absent")

_code, _body = http("GET", "/server_info?config_format=json", timeout=180)
if _code != 200:
    _code, _body = http("GET", "/server_info", timeout=180)
pathlib.Path(OUT_DIR, "server_info.json").write_text(_body)
_a = re.search(r'async_scheduling"?\s*[:=]\s*"?(\w+)', _body)
ASYNC_SCHEDULING = _a.group(1) if _a else None
print(f"  server_info: HTTP {_code}, saved · async_scheduling = {ASYNC_SCHEDULING}")

# Warm-up: the first requests after start-up pay one-off costs. Throwaway requests on ids that
# share nothing with the trace (every run below starts from a reset anyway) — and a check, before
# any measured run, that each request reports what the comparison needs.
_warm = [record({"request_id": f"warm{j}", "ids": ids[:200]}, stream_one(ids[:200], 8), 8,
                time.perf_counter()) for j, ids in enumerate(SWEEP_PROMPTS[:3])]
if not all(x["valid"] for x in _warm):
    raise Stop("⛔ a warm-up response is unusable — wrong length (is ignore_eos honoured?), an "
               "error, or no usage.prompt_tokens_details.cached_tokens (is "
               f"--enable-prompt-tokens-details in effect?): {_warm[0]}")
_absent = [k for k, v in read_counters(scrape()).items() if v is None]
if any(not k.startswith("hist_") for k in _absent):
    raise Stop(f"⛔ /metrics lacks counters the comparison reads: {_absent}")
if _absent:
    print(f"⚠️  histogram families absent: {_absent} — the H4 counts will read as missing")
print("  warm-up done: responses carry cached_tokens and exact lengths; every counter present")
'''

W2_CELL = r'''
for _i in range(1, RUNS_PER_WORKLOAD + 1):
    run_workload("W2_short", _i)
'''

W1_CELL = r'''
for _i in range(1, RUNS_PER_WORKLOAD + 1):
    run_workload("W1_long", _i)
'''

SWEEP_CELL = r'''
SWEEP = []
print(f"── decode sweep · batch sizes {SWEEP_BATCH_SIZES} · {SWEEP_OUTPUT_TOKENS}-token outputs")
for _b in SWEEP_BATCH_SIZES:
    fresh_cache(f"sweep_b{_b}")
    _before = settled_scrape()
    _prompts, _t_zero = SWEEP_PROMPTS[:_b], time.perf_counter()
    with ThreadPoolExecutor(max_workers=_b) as _pool:
        _futs = [_pool.submit(stream_one, ids, SWEEP_OUTPUT_TOKENS) for ids in _prompts]
        _recs = [record({"request_id": f"sweep{_b}-{j}", "ids": ids}, f.result(),
                        SWEEP_OUTPUT_TOKENS, _t_zero)
                 for j, (ids, f) in enumerate(zip(_prompts, _futs))]
    _d = deltas(_before, settled_scrape())
    _iv = sorted(x["decode_interval_ms"] for x in _recs if x["valid"] and x["decode_interval_ms"])
    _med = _iv[len(_iv) // 2] if _iv else None
    SWEEP.append({"batch": _b, "median_decode_interval_ms": _med, "measured": _d,
                  "valid": len(_iv) == _b and _d["preemptions"] == 0, "requests": _recs})
    print(f"    batch {_b:>2}: {_med or float('nan'):6.1f} ms per decode step · "
          f"preemptions {fmt(_d['preemptions'])}")
pathlib.Path(OUT_DIR, "decode_sweep.json").write_text(json.dumps(SWEEP, indent=1))
'''

W3_CELL = r'''
for _i in range(1, RUNS_PER_WORKLOAD + 1):
    run_workload("W3_concurrent", _i)
'''

VERDICT_CELL = r'''
VERDICT = {}
print(f"{'workload':15s} {'valid':>7s}   {'vllm model':28s} {'lru model':28s}")
for _name, _w in PRED["workloads"].items():
    _runs = [r for r in RUNS if r["workload"] == _name]
    _ok = [r for r in _runs if r["valid"]]
    _row = {"runs": len(_runs), "valid_runs": len(_ok)}
    if _w["mode"] == "sequential":
        for _m in ("vllm", "lru"):
            _ex = sum(1 for r in _ok if r["comparison"][_m]["counters_exact"]
                      and r["comparison"][_m]["requests_matching"]
                      == r["comparison"][_m]["requests"])
            _err = [r["comparison"][_m]["hit_rate_error_pts"] for r in _ok]
            _row[_m] = {"exact_runs": _ex, "hit_rate_error_pts": _err}
        _cells = [f"exact {_row[m]['exact_runs']}/{len(_ok)}  err "
                  + ",".join(f"{e:+.2f}" for e in _row[m]["hit_rate_error_pts"])
                  for m in ("vllm", "lru")]
        _distinct = {json.dumps(r["measured"], sort_keys=True) for r in _ok}
        _row["runs_identical"] = len(_distinct) == 1
    else:
        _v = sum(1 for r in _ok if r["comparison"]["vllm_preemptions_claim_holds"])
        _l = sum(1 for r in _ok if r["comparison"]["lru_zero_preemptions_holds"])
        _row |= {"vllm_claim_holds_runs": _v, "lru_claim_holds_runs": _l}
        _cells = [f"preemptions>0 holds {_v}/{len(_ok)}", f"0 preemptions holds {_l}/{len(_ok)}"]
    VERDICT[_name] = _row
    print(f"{_name:15s} {len(_ok):>3}/{len(_runs):<3}   {_cells[0]:28s} {_cells[1]:28s}")

RESULTS = {
    "notebook": "notebooks/phase3_calibration.ipynb",
    "trace_fingerprint_sha256": TRACE_SHA256,
    "predictions_sha256": PREDICTIONS_SHA256,
    "model": MODEL,
    "engine_argv": ENGINE_ARGV,
    "environment": ENV,
    "metric_and_source_audit": AUDIT,
    "capacity": CAPACITY,
    "async_scheduling": ASYNC_SCHEDULING,
    "verdict": VERDICT,
    "runs": RUNS,
    "decode_sweep": SWEEP,
}
pathlib.Path(OUT_DIR, "phase3.json").write_text(json.dumps(RESULTS, indent=1, default=str))
_zip = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
print(f"\nsaved {OUT_DIR}/ ({len(list(pathlib.Path(OUT_DIR).iterdir()))} files) and "
      f"{pathlib.Path(_zip).name} ({pathlib.Path(_zip).stat().st_size / 1e6:.1f} MB)")
print("⭐ Download phase3_results.zip and bring it back — then run the clean-up cell.")
'''

CLEANUP_CELL = r'''
stop_server()
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))
'''


def _blob_lines(blob: str, width: int = 96) -> str:
    return "\n".join(f'    "{blob[i:i + width]}"' for i in range(0, len(blob), width))


def build_notebook(
    cfg: dict[str, Any], blob: str, fingerprint: str, pred_sha: str, predictions: dict[str, Any]
) -> str:
    e = cfg["engine"]
    sweep = cfg["calibration"]["decode_batch_sweep"]
    w = predictions["workloads"]
    w1, w2 = w["W1_long"]["predictions"], w["W2_short"]["predictions"]
    w3 = w["W3_concurrent"]["predictions"]["timing_independent_claims"]
    fill = {
        "MODEL": e["model"],
        "VLLM": e["vllm"],
        "BLOCK_SIZE": str(e["block_size"]),
        "BLOCKS": str(e["num_gpu_blocks"]),
        "USABLE": str(e["num_gpu_blocks"] - 1),
        "MAX_MODEL_LEN": str(e["max_model_len"]),
        "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "RUNS": str(cfg["runs_per_workload"]),
        "SWEEP_BATCHES": repr(sweep["batch_sizes"]),
        "SWEEP_OUT": str(sweep["output_tokens"]),
        "N": str(len(w1["vllm"]["per_request_cached_tokens"])),
        "W1_V": f"{w1['vllm']['hit_rate']:.4f}",
        "W1_L": f"{w1['lru']['hit_rate']:.4f}",
        "W1_GAP": f"{abs(w1['vllm']['hit_rate'] - w1['lru']['hit_rate']) * 100:.1f}",
        "W2_V": f"{w2['vllm']['hit_rate']:.4f}",
        "W2_L": f"{w2['lru']['hit_rate']:.4f}",
        "W3_L": f"{w3['lru']['hit_rate']:.4f}",
        "FP": fingerprint,
        "PRED_SHA": pred_sha,
        "BLOB": _blob_lines(blob),
    }

    def sub(text: str) -> str:
        for k, v in fill.items():
            text = text.replace(f"@@{k}@@", v)
        if "@@" in text:
            raise ValueError(f"unfilled marker in: {text[text.index('@@') - 40:][:80]!r}")
        return text

    cells = [
        _md(sub(INTRO)),
        _md(sub(HOW_TO_RUN)),
        _md("## 0 — Configuration"),
        _code(sub(CONFIG_CELL)),
        _md("## 1 — Environment"),
        _code(sub(ENV_CELL)),
        _md("## 2 — Install vLLM, pinned"),
        _code(sub(INSTALL_CELL)),
        _md(
            """
            ## 3 — Audit: does the installed engine contain the code the model cites?

            The cache model cites vLLM's source line by line. This reads the installed package
            and finds each cited behaviour by its exact text — so the report can say the engine
            that was measured is the engine that was modelled, not merely the same version number.
            """
        ),
        _code(sub(AUDIT_CELL)),
        _md(
            """
            ## 4 — The trace and the pre-registered predictions (embedded — do not edit)

            Generated by `scripts/phase3_prepare.py` from `configs/phase3_workloads.yaml`. The
            fingerprint covers every token id this notebook sends; the second hash covers the
            predictions. Both are also in the committed prereg file, so anyone can check that this
            run compared against what was committed, and nothing else.
            """
        ),
        _code(sub(DATA_CELL)),
        _md("## 5 — Helpers: the engine, `/metrics`, the reset"),
        _code(sub(SERVER_CELL)),
        _md("## 6 — Helpers: the client, the runs, the comparison"),
        _code(sub(CLIENT_CELL)),
        _md(
            """
            ## 7 — Start the engine, and verify what it is

            Capacity is checked from the engine's own log, never assumed: every prediction is for
            exactly this many blocks, so a different number stops the run here rather than
            producing numbers that compare against the wrong thing.
            """
        ),
        _code(sub(START_CELL)),
        _md(
            """
            ## 8 — W2, the control: 1-token outputs, one request at a time

            No request ever holds a decode block, so this isolates the accounting itself — the
            token unit, the partial-block rule, the lookup cap, the free-queue order. It also
            carries a pre-registered alternative: if the engine committed one more token per
            request than its source says, W2 would show it.
            """
        ),
        _code(sub(W2_CELL)),
        _md(
            """
            ## 9 — W1, the discriminating workload: 64-token outputs (≈ 15 minutes)

            Identical requests, but each now holds its decode blocks while it runs — and vLLM
            caches the full ones. The `vllm` model says they crowd useful prefix blocks out of a
            small cache; the `lru` model does not model them at all. They cannot both be right.
            """
        ),
        _code(sub(W1_CELL)),
        _md(
            """
            ## 10 — Decode sweep: a calibration input, not a prediction

            W1 and W2 only ever run a batch of one. The timing model's per-sequence decode cost
            needs larger batches, so this measures the decode step at several batch sizes, on
            prompts that share nothing with the trace — it cannot leak W3's answer into W3's
            prediction. Context length matches W3's (~450–650 tokens).
            """
        ),
        _code(sub(SWEEP_CELL)),
        _md(
            """
            ## 11 — W3: the same trace, concurrent

            Each request at its own arrival time. Under memory pressure the `vllm` model admits
            by free blocks and preempts; the `lru` model has no mechanism to preempt at all. Its
            timings are T4 timings. The full W3 comparison — the fitted timing model against these
            runs — is made afterwards by `scripts/phase3_analyze.py`, by the pre-registered
            procedure.
            """
        ),
        _code(sub(W3_CELL)),
        _md("## 12 — Verdict, and save everything"),
        _code(sub(VERDICT_CELL)),
        _md("## 13 — Clean up\n\n⚠️ **Run this.** A forgotten `vllm serve` holds the GPU."),
        _code(sub(CLEANUP_CELL)),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
