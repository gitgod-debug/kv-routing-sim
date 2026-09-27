"""Builds notebooks/phase6_sim2real.ipynb. Called by scripts/phase6_prepare.py — not run alone.

The notebook is SELF-CONTAINED on purpose: the GPU box is a Kaggle kernel with no checkout of this
repo, so it may use only `vllm` and the standard library. It REGENERATES the traffic from the same
generator settings instead of carrying megabytes of token ids, and refuses to run unless what it
generated has the exact fingerprint the predictions were made on. `@@NAME@@` markers are filled in
`build_notebook`.
"""

from __future__ import annotations

import json
from typing import Any

from _phase3_notebook import AUDIT_CELL, INSTALL_CELL, _code, _md

INTRO = """
# Phase 6 — sim-to-real on two live replicas

**What this is.** The check that kv-routing-sim's routing results hold on a real fleet. Two vLLM
**@@VLLM@@** servers — **@@MODEL@@, one per Tesla T4** — receive the same agentic traffic three times,
routed by round-robin, session-sticky and radix-affinity. The traffic is H2b's: one shared system
prompt (**K1**) and 64 tenant prompts (**K64**). The simulator predicted every run's cache hit rate
**before this notebook ever ran**; the predictions are embedded below, hashed, and committed to the
repo (`artifacts/results/phase6_prereg/predictions.json`).

**What decides it** (registered in `configs/phase6_sim2real.yaml`):
- **S1** — each run's measured hit rate within **10%** of its prediction;
- **S2** — policies predicted **≥ 0.02** apart come out in the same order. The pair that carries the
  headline is K64: radix-affinity (**@@K64_A@@**) above session-sticky (**@@K64_S@@**).

Hit rates are cache counts, and cache counts are hardware-independent. Timings are T4 timings (fp16,
TRITON_ATTN) and are labelled so.
"""

HOW_TO_RUN = """
**How to run.** Kaggle → *Settings* → Accelerator **GPU T4 ×2**, Internet **on** → *Run All*. About
40 minutes: ~10 to install and start two servers, then six runs of ~3–4 minutes. At the end, download
**`phase6_results.zip`** from the Output panel and bring it back. If a cell stops with ⛔, download the
zip anyway: it holds both server logs.
"""

CONFIG_CELL = r'''
# ── Generated from configs/phase6_sim2real.yaml. Do not edit here: edit the config and re-run
#    scripts/phase6_prepare.py, or the embedded predictions stop meaning anything.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
NUM_GPU_BLOCKS = @@BLOCKS@@          # --num-gpu-blocks-override; one is vLLM's null block
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
N_REPLICAS = @@REPLICAS@@
ROUTER_SEED = @@SEED@@
POLICIES = @@POLICIES@@
S1_MAX_RELATIVE_ERROR = @@S1@@
S2_MIN_GAP = @@S2@@

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_MEM_UTIL = 0.85
PORTS = [8000, 8001]          # replica i listens on PORTS[i] and runs on GPU i
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "phase6_results"
'''

ENV_CELL = r'''
import hashlib
import json
import os
import pathlib
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

pathlib.Path(OUT_DIR).mkdir(exist_ok=True)


class Stop(Exception):
    """Halts Run-All with a reason. Download phase6_results.zip anyway."""


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
_n_gpus = len([ln for ln in ENV["gpus"].splitlines() if ln.strip() and "failed" not in ln])
if _n_gpus < N_REPLICAS:
    raise Stop(f"⛔ {_n_gpus} GPU(s) visible, {N_REPLICAS} needed: one server per GPU is the "
               "whole point. Kaggle: Settings → Accelerator → GPU T4 ×2.")
if "T4" not in ENV["gpus"]:
    print("\n⚠️  Not a Tesla T4. Hit rates stay valid (cache counts are hardware-independent), "
          "but every timing must be labelled with the GPU above, not as a T4 figure.")
'''

DATA_CELL = r'''
# ── The traffic, regenerated — a port of kv_routing_sim/traces/generator.py (AgenticWorkload) and
#    traces/hashing.py (chain_hashes). The fingerprints prove it made the exact ids predicted on.
WORKLOADS = json.loads(r"""@@WORKLOADS@@""")
FINGERPRINTS = json.loads(r"""@@FINGERPRINTS@@""")
PRED = json.loads(r"""@@PRED@@""")
PREDICTIONS_SHA256 = "@@PRED_SHA@@"

_SHARED_BASE, _PREFIX_STRIDE, _SESSION_BASE, _TURN_BASE = 0, 100_000, 10_000_000, 500_000_000
_MIX_MUL, _MIX_ADD, _MASK64 = 0x9E3779B97F4A7C15, 0x632BE59BD9B4E019, 0xFFFFFFFFFFFFFFFF
_LOW, _HIGH = 1000, 100_000  # ENGINE_SAFE_IDS: ordinary text tokens in Qwen3's vocabulary


def _engine_id(raw):
    return _LOW + ((raw * _MIX_MUL + _MIX_ADD) & _MASK64) % (_HIGH - _LOW)


def generate(w):
    """AgenticWorkload._iter_with_tokens + prefix_assignment, sorted by arrival like generate()."""
    turns, ctx = w["turns_per_session"], w["session_context_tokens"]
    user_n, asst_n, r = w["user_turn_tokens"], w["assistant_turn_tokens"], w["prefix_sharing_ratio"]
    mean_private = ctx + user_n + (user_n + asst_n) * (turns - 1) / 2.0
    shared_len = 0 if r <= 0.0 else max(0, round(r / (1.0 - r) * mean_private))
    k_total = w.get("n_shared_prefixes", 1)
    prefixes = [[_SHARED_BASE + k * _PREFIX_STRIDE + i for i in range(shared_len)]
                for k in range(k_total)]
    if k_total == 1:
        assign = [0] * w["n_sessions"]
    else:
        ks = range(k_total)
        weights = [1.0 / (k + 1) ** w.get("prefix_zipf_s", 1.0) for k in ks]
        assign = random.Random(f"shared-prefix-{w['seed']}").choices(ks, weights=weights,
                                                                     k=w["n_sessions"])
    rng, clock, out = random.Random(w["seed"]), 0.0, []
    for s in range(w["n_sessions"]):
        clock += rng.expovariate(w["arrival_rate"])
        shared = prefixes[assign[s]]
        private = [_SESSION_BASE + s * 100_000 + i for i in range(ctx)]
        history = []
        for t in range(turns):
            base = _TURN_BASE + s * 100_000 + t * 1_000
            user = [base + i for i in range(user_n)]
            out.append({"request_id": f"s{s}-t{t}",
                        "arrival_ts": round(clock + t * w["think_time_s"], 6),
                        "session_id": f"s{s}",
                        "ids": [_engine_id(x) for x in shared + private + history + user]})
            history = history + user + [base + 500 + i for i in range(asst_n)]
    out.sort(key=lambda q: q["arrival_ts"])
    return out


def _h64(data):
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def chain_hashes(ids, block_size=BLOCK_SIZE, salt=0):
    """traces/hashing.py: one hash per FULL block, each chained to its parent."""
    out, parent = [], _h64(salt.to_bytes(8, "big"))
    for i in range(len(ids) // block_size):
        body = b"".join(int(t).to_bytes(8, "big", signed=True)
                        for t in ids[i * block_size:(i + 1) * block_size])
        parent = _h64(parent.to_bytes(8, "big") + body)
        out.append(parent)
    return out


def _canonical(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


TRACES = {}
for _name, _w in WORKLOADS.items():
    _tr = generate(_w)
    _fp = hashlib.sha256(_canonical([[q["request_id"], q["arrival_ts"], q["session_id"], q["ids"]]
                                     for q in _tr])).hexdigest()
    if _fp != FINGERPRINTS[_name]:
        raise Stop(f"⛔ {_name}: the regenerated trace does not match its fingerprint — this "
                   "Python's random module or the generator port differs from the one predicted on")
    for _q in _tr:
        _q["blocks"] = chain_hashes(_q["ids"])
    TRACES[_name] = _tr
    print(f"{_name:4s} ✓  {len(_tr)} requests · prompts {min(len(q['ids']) for q in _tr)}–"
          f"{max(len(q['ids']) for q in _tr)} tokens · fingerprint {_fp[:16]}…")
if hashlib.sha256(_canonical(PRED)).hexdigest() != PREDICTIONS_SHA256:
    raise Stop("⛔ the embedded predictions do not match their hash: the data cell was edited")
print(f"predictions ✓ {PREDICTIONS_SHA256[:16]}…\n\nPre-registered hit rates (simulator, 2 replicas):")
for _name, _by in PRED["runs"].items():
    print(f"  {_name:4s} " + "   ".join(f"{p} {v['hit_rate']:.4f}" for p, v in _by.items()))
'''

SERVERS_CELL = r'''
SERVERS = {}  # replica -> {"proc", "logf", "log_path"}
RESET_MODE = "endpoint"  # falls back to "restart" if the dev endpoint is not served


def engine_argv(i):
    return [
        "vllm", "serve", MODEL, "--port", str(PORTS[i]),
        "--dtype", "half",  # T4 has no bf16
        "--block-size", str(BLOCK_SIZE),
        "--num-gpu-blocks-override", str(NUM_GPU_BLOCKS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--seed", "0",
    ]


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server(i):
    # ⚠️ Refuse to launch while ANYTHING listens on the port: a stale server would answer /health
    # before ours had time to die of "address already in use" (found by Phase 3's dry-run).
    for _ in range(30):
        if not port_in_use(PORTS[i]):
            break
        time.sleep(1)
    else:
        raise Stop(f"⛔ something already serves on port {PORTS[i]} — a server from an earlier "
                   "run? Run the clean-up cell (or !pkill -f 'vllm serve') and start again.")
    # One physical GPU per replica: complete isolation, so TTFT is attributable (ADR-005).
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), VLLM_SERVER_DEV_MODE="1")
    log_path = pathlib.Path(OUT_DIR) / f"server_{i}.log"
    logf = open(log_path, "w")  # noqa: SIM115 - must outlive this cell
    proc = subprocess.Popen(engine_argv(i), stdout=logf, stderr=subprocess.STDOUT, env=env)
    SERVERS[i] = {"proc": proc, "logf": logf, "log_path": str(log_path)}
    print(f"launched replica {i}: GPU {i}, port {PORTS[i]}")


def log_text(i):
    return pathlib.Path(SERVERS[i]["log_path"]).read_text(errors="replace") if i in SERVERS else ""


def http(i, method, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORTS[i]}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, repr(e)


def wait_ready(i, timeout=STARTUP_TIMEOUT_S, every=30):
    """Ready = OUR process is alive AND the port answers, checked in that order."""
    t0, last = time.time(), 0.0
    while time.time() - t0 < timeout:
        if SERVERS[i]["proc"].poll() is not None:
            print(f"  !! replica {i} EXITED (returncode {SERVERS[i]['proc'].returncode}). Log tail:")
            print(log_text(i)[-3000:])
            return False
        if http(i, "GET", "/health", timeout=5)[0] == 200:
            print(f"  replica {i} ready in {time.time() - t0:.0f}s")
            return True
        if time.time() - t0 - last >= every:
            last = time.time() - t0
            tail = log_text(i).strip().splitlines()
            print(f"    ...{last:.0f}s  {tail[-1][:110] if tail else '(log empty)'}")
        time.sleep(5)
    print(f"  !! replica {i} not ready within {timeout}s")
    return False


def stop_server(i):
    if i not in SERVERS:
        return
    SERVERS[i]["proc"].terminate()
    try:
        SERVERS[i]["proc"].wait(timeout=90)
    except subprocess.TimeoutExpired:
        SERVERS[i]["proc"].kill()
    SERVERS[i]["logf"].close()
    del SERVERS[i]


def stop_all():
    for i in list(SERVERS):
        stop_server(i)


def metric(text, family, **labels):
    """Sum a family over matching labels. None = ABSENT, which is not the same as 0."""
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
    "prompt_tokens": ("vllm:prompt_tokens", {}),
    "generation_tokens": ("vllm:generation_tokens", {}),
}


def read_counters(text):
    return {k: metric(text, fam, **lab) for k, (fam, lab) in COUNTERS.items()}


def settled_scrape(i, max_wait=300):
    """Scrape once nothing runs or waits AND the counters stopped moving: stats reach /metrics a
    moment after the tokens they describe, so an early read under-counts."""
    t0, prev = time.time(), None
    while True:
        text = http(i, "GET", "/metrics", timeout=30)[1]
        busy = ((metric(text, "vllm:num_requests_running") or 0)
                + (metric(text, "vllm:num_requests_waiting") or 0))
        now = read_counters(text)
        if busy == 0 and now == prev:
            return text
        if time.time() - t0 > max_wait:
            raise Stop(f"⛔ replica {i} never settled within {max_wait}s (busy={busy})")
        prev = now
        time.sleep(1.0)


def deltas(before, after):
    b, a = read_counters(before), read_counters(after)
    return {k: None if a[k] is None else round(a[k] - (b[k] or 0.0)) for k in COUNTERS}


def fresh_caches():
    """A cold cache on BOTH replicas before every run, by /reset_prefix_cache (a restart if the
    endpoint is not served)."""
    global RESET_MODE
    for i in range(N_REPLICAS):
        if RESET_MODE == "endpoint":
            for _ in range(15):
                code, body = http(i, "POST", "/reset_prefix_cache")
                try:
                    ok = code == 200 and json.loads(body).get("success") is True
                except ValueError:
                    ok = False
                if ok:
                    break
                if code in (404, 405):
                    print("  /reset_prefix_cache is not served: restarting the engines per run")
                    RESET_MODE = "restart"
                    break
                time.sleep(2)  # success=false: a finishing request still holds blocks
            else:
                raise Stop(f"⛔ replica {i} refused to reset its prefix cache 15 times")
        if RESET_MODE == "restart":
            stop_server(i)
            start_server(i)
            if not wait_ready(i):
                raise Stop(f"⛔ replica {i} failed to restart — the log tail is above")


print("server helpers ready")
'''

START_CELL = r'''
CAPACITY = {}
for _i in range(N_REPLICAS):  # one at a time: the first start downloads the weights for both
    start_server(_i)
    if not wait_ready(_i):
        raise Stop(f"⛔ replica {_i} did not start — the log tail is above. Download the zip.")
    _log = log_text(_i)
    _m = re.search(r"KV cache size: ([\d,]+) tokens", _log)
    CAPACITY[_i] = {
        "kv_cache_size_tokens_logged": int(_m.group(1).replace(",", "")) if _m else None,
        "expected_tokens": NUM_GPU_BLOCKS * BLOCK_SIZE,
        "override_logged": f"num_gpu_blocks_override={NUM_GPU_BLOCKS}" in _log,
    }
    print(f"  replica {_i}: {CAPACITY[_i]}")
    if CAPACITY[_i]["kv_cache_size_tokens_logged"] != CAPACITY[_i]["expected_tokens"]:
        raise Stop(f"⛔ replica {_i} logged {CAPACITY[_i]['kv_cache_size_tokens_logged']} tokens "
                   f"of KV cache, not {CAPACITY[_i]['expected_tokens']:,}: every prediction "
                   f"assumes exactly {NUM_GPU_BLOCKS} blocks per replica.")
'''

CLIENT_CELL = r'''
def stream_one(i, ids, max_tokens):
    """One streamed completion to replica i: send time, first-chunk time, and the usage block."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORTS[i]}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"t_send": time.perf_counter(), "t_first": None, "usage": None,
           "finish_reason": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("choices"):
                    if rec["t_first"] is None:
                        rec["t_first"] = time.perf_counter()
                    reason = obj["choices"][0].get("finish_reason")
                    if reason:
                        rec["finish_reason"] = reason
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
    except Exception as e:
        rec["error"] = repr(e)
    rec["t_done"] = time.perf_counter()
    return rec


# ── The routers: ports of kv_routing_sim/policies/*.py, same algorithm, same tie-breaking stream.
#    `out[i]` is replica i's in-flight requests — the simulator's running + queued.
def argmin_random(values, rng):
    lo = min(values)
    cand = [i for i, v in enumerate(values) if v == lo]
    return cand[0] if len(cand) == 1 else rng.choice(cand)


class RoundRobin:
    def __init__(self, n):
        self.n, self.cursor = n, 0

    def choose(self, out, q, rng):
        i, self.cursor = self.cursor, (self.cursor + 1) % self.n
        return i

    def observe(self, i, q):
        pass


class SessionSticky:
    def __init__(self, n, max_sessions=1_000_000):
        self.n, self.assigned, self.max_sessions = n, {}, max_sessions

    def choose(self, out, q, rng):
        sid = q["session_id"]
        if sid is None:
            return argmin_random(out, rng)
        if sid in self.assigned:
            return self.assigned[sid]
        chosen = argmin_random(out, rng)
        if len(self.assigned) < self.max_sessions:
            self.assigned[sid] = chosen
        return chosen

    def observe(self, i, q):
        pass


class RadixAffinity:
    """Its OWN map of block -> replicas it dispatched to (a bitmask), never the true caches."""

    def __init__(self, n, router_map_blocks=500_000, load_penalty=0.0):
        self.n, self.owners, self.cap, self.alpha = n, OrderedDict(), router_map_blocks, load_penalty

    def _match(self, i, blocks):
        bit, k = 1 << i, 0
        for h in blocks:
            if not (self.owners.get(h, 0) & bit):
                break
            k += 1
        return k

    def choose(self, out, q, rng):
        blocks = q["blocks"]
        if not blocks:
            return argmin_random(out, rng)
        first = self.owners.get(blocks[0], 0)
        if first == 0:
            return argmin_random(out, rng)
        scores = [self._match(i, blocks) if first & (1 << i) else 0 for i in range(self.n)]
        best = max(scores)
        if best == 0:
            return argmin_random(out, rng)
        if self.alpha > 0.0:
            adj = [s - self.alpha * out[i] for i, s in enumerate(scores)]
            top = max(adj)
            cand = [i for i, a in enumerate(adj) if a == top]
        else:
            cand = [i for i, s in enumerate(scores) if s == best]
        if len(cand) == 1:
            return cand[0]
        return cand[argmin_random([out[i] for i in cand], rng)]

    def observe(self, i, q):
        bit = 1 << i
        for h in q["blocks"]:
            if h in self.owners:
                self.owners[h] |= bit
                self.owners.move_to_end(h)
            else:
                self.owners[h] = bit
                if len(self.owners) > self.cap:
                    self.owners.popitem(last=False)


ROUTERS = {"round-robin": RoundRobin, "session-sticky": SessionSticky,
           "radix-affinity": RadixAffinity}


def nearest_rank(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1))))]


def run_one(name, policy, output_tokens, lead_s=2.0):
    """One (workload, policy): reset both caches, replay the trace open-loop at its arrival times,
    route each request as the simulator did, and read both engines' counters."""
    trace = TRACES[name]
    fresh_caches()
    before = [settled_scrape(i) for i in range(N_REPLICAS)]
    router, rng = ROUTERS[policy](N_REPLICAS), random.Random(ROUTER_SEED)
    lock, inflight, routed = threading.Lock(), [0] * N_REPLICAS, []

    def work(i, q):
        rec = stream_one(i, q["ids"], output_tokens)
        with lock:
            inflight[i] -= 1
        return rec

    t_zero = time.perf_counter() + lead_s
    with ThreadPoolExecutor(max_workers=len(trace)) as pool:
        for q in trace:
            delay = t_zero + q["arrival_ts"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with lock:
                i = router.choose(list(inflight), q, rng)
                router.observe(i, q)
                inflight[i] += 1
            routed.append((q, i, pool.submit(work, i, q)))
        results = [(q, i, f.result()) for q, i, f in routed]
    after = [settled_scrape(i) for i in range(N_REPLICAS)]
    per_replica = [deltas(b, a) for b, a in zip(before, after)]
    recs = []
    for q, i, rec in results:
        u = rec["usage"] or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        recs.append({
            "request_id": q["request_id"], "replica": i,
            "valid": (rec["error"] is None and rec["finish_reason"] == "length"
                      and u.get("completion_tokens") == output_tokens
                      and u.get("prompt_tokens") == len(q["ids"]) and cached is not None),
            "error": rec["error"], "cached_tokens": cached,
            "ttft_ms": None if rec["t_first"] is None else (rec["t_first"] - rec["t_send"]) * 1e3,
            "send_offset_s": rec["t_send"] - t_zero, "scheduled_offset_s": q["arrival_ts"],
        })
    hits = sum(d["prefix_cache_hits"] or 0 for d in per_replica)
    queries = sum(d["prefix_cache_queries"] or 0 for d in per_replica)
    measured = hits / queries if queries else None
    predicted = PRED["runs"][name][policy]["hit_rate"]
    rel = None if measured is None else (measured - predicted) / predicted
    ttfts = sorted(r["ttft_ms"] for r in recs if r["ttft_ms"] is not None)
    run = {
        "workload": name, "policy": policy, "per_replica_counters": per_replica,
        "measured_hit_rate": measured, "predicted_hit_rate": predicted, "relative_error": rel,
        "s1_holds": rel is not None and abs(rel) <= S1_MAX_RELATIVE_ERROR,
        "requests_per_replica": [sum(1 for r in recs if r["replica"] == i)
                                 for i in range(N_REPLICAS)],
        "predicted_requests_per_replica": PRED["runs"][name][policy]["requests_per_replica"],
        "preemptions": sum(d["preemptions"] or 0 for d in per_replica),
        "predicted_preemptions": PRED["runs"][name][policy]["preemptions"],
        "ttft_p50_ms": nearest_rank(ttfts, 0.50), "ttft_p95_ms": nearest_rank(ttfts, 0.95),
        "invalid_requests": [r["request_id"] for r in recs if not r["valid"]][:20],
        "worst_send_lag_ms": max(r["send_offset_s"] - r["scheduled_offset_s"] for r in recs) * 1e3,
        "requests": recs,
    }
    print(f"  {name:4s} {policy:15s} hit {measured:.4f} (predicted {predicted:.4f}, "
          f"{rel:+.1%}) {'✅' if run['s1_holds'] else '⛔'} · requests {run['requests_per_replica']} "
          f"(predicted {run['predicted_requests_per_replica']}) · preemptions {run['preemptions']} "
          f"(predicted {run['predicted_preemptions']}) · p95 TTFT {run['ttft_p95_ms'] or 0:.0f} ms"
          f" · {len(run['invalid_requests'])} invalid · send lag ≤ {run['worst_send_lag_ms']:.0f} ms")
    return run


print("routers and client ready")
'''

WARMUP_CELL = r'''
# Throwaway requests on ids that share nothing with the traces (every run below starts from a reset)
# — and a check, before any measured run, that each response reports what the comparison needs.
for _i in range(N_REPLICAS):
    _ids = [1000 + 7 * j + _i for j in range(200)]
    _rec = stream_one(_i, _ids, 8)
    _u = _rec["usage"] or {}
    if (_rec["error"] or _rec["finish_reason"] != "length" or _u.get("completion_tokens") != 8
            or (_u.get("prompt_tokens_details") or {}).get("cached_tokens") is None):
        raise Stop(f"⛔ replica {_i}'s warm-up response is unusable — is ignore_eos honoured, is "
                   f"--enable-prompt-tokens-details in effect? {_rec}")
    _absent = [k for k, v in read_counters(http(_i, "GET", "/metrics")[1]).items() if v is None]
    if _absent:
        raise Stop(f"⛔ replica {_i}'s /metrics lacks counters the comparison reads: {_absent}")
print("warm-up done: both replicas report cached_tokens, exact lengths and every counter")
'''

RUN_CELL = r'''
RUNS = []
for _name in WORKLOADS:
    for _policy in POLICIES:
        RUNS.append(run_one(_name, _policy, WORKLOADS[_name]["output_tokens"]))
'''

VERDICT_CELL = r'''
S2 = {}
for _name in WORKLOADS:
    _m = {r["policy"]: r["measured_hit_rate"] for r in RUNS if r["workload"] == _name}
    _p = {p: PRED["runs"][_name][p]["hit_rate"] for p in POLICIES}
    _pairs = []
    for _a in POLICIES:
        for _b in POLICIES:
            if _a < _b and abs(_p[_a] - _p[_b]) >= S2_MIN_GAP:
                _same = (_m[_a] - _m[_b]) * (_p[_a] - _p[_b]) > 0
                _pairs.append({"pair": [_a, _b], "predicted_gap": round(_p[_a] - _p[_b], 4),
                               "measured_gap": round(_m[_a] - _m[_b], 4), "same_order": _same})
    S2[_name] = {"pairs": _pairs, "holds": all(x["same_order"] for x in _pairs)}
    for x in _pairs:
        print(f"  S2 {_name:4s} {x['pair'][0]:>15s} − {x['pair'][1]:<15s} predicted "
              f"{x['predicted_gap']:+.4f}  measured {x['measured_gap']:+.4f}  "
              f"{'✅' if x['same_order'] else '⛔'}")
S1_ALL = all(r["s1_holds"] for r in RUNS)
VERDICT = {"s1_all_runs": S1_ALL, "s2": S2,
           "phase6_holds": S1_ALL and all(v["holds"] for v in S2.values())}
print(f"\nS1 (every run within {S1_MAX_RELATIVE_ERROR:.0%}): {'HOLDS' if S1_ALL else 'FAILS'}")
print(f"Phase 6: {'HOLDS' if VERDICT['phase6_holds'] else 'FALSIFIED'}")

RESULTS = {
    "notebook": "notebooks/phase6_sim2real.ipynb",
    "fingerprints": FINGERPRINTS,
    "predictions_sha256": PREDICTIONS_SHA256,
    "model": MODEL,
    "engine_argv": [engine_argv(i) for i in range(N_REPLICAS)],
    "environment": ENV,
    "metric_and_source_audit": AUDIT,
    "capacity": CAPACITY,
    "reset_mode": RESET_MODE,
    "verdict": VERDICT,
    "runs": RUNS,
}
pathlib.Path(OUT_DIR, "phase6.json").write_text(json.dumps(RESULTS, indent=1, default=str))
_zip = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
print(f"\nsaved {OUT_DIR}/ and {pathlib.Path(_zip).name} "
      f"({pathlib.Path(_zip).stat().st_size / 1e6:.1f} MB)")
print("⭐ Download phase6_results.zip and bring it back — then run the clean-up cell.")
'''

CLEANUP_CELL = r'''
stop_all()
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))
'''


def build_notebook(cfg: dict[str, Any], workloads: dict[str, Any], fingerprints: dict[str, str],
                   pred: dict[str, Any], pred_sha: str) -> str:
    e = cfg["engine"]
    k64 = pred["runs"]["K64"]
    fill = {
        "MODEL": e["model"],
        "VLLM": e["vllm"],
        "BLOCK_SIZE": str(e["block_size"]),
        "BLOCKS": str(e["num_gpu_blocks"]),
        "MAX_MODEL_LEN": str(e["max_model_len"]),
        "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "REPLICAS": str(cfg["replicas"]),
        "SEED": str(cfg["router_seed"]),
        "POLICIES": repr(list(cfg["policies"])),
        "S1": repr(cfg["s1_max_relative_error"]),
        "S2": repr(cfg["s2_min_gap"]),
        "K64_A": f"{k64['radix-affinity']['hit_rate']:.4f}",
        "K64_S": f"{k64['session-sticky']['hit_rate']:.4f}",
        "WORKLOADS": json.dumps(workloads),
        "FINGERPRINTS": json.dumps(fingerprints),
        "PRED": json.dumps(pred, sort_keys=True),
        "PRED_SHA": pred_sha,
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
        _md("## 1 — Environment: two GPUs, one server each"),
        _code(sub(ENV_CELL)),
        _md("## 2 — Install vLLM, pinned"),
        _code(INSTALL_CELL.replace("@@VLLM@@", e["vllm"])),
        _md("## 3 — Audit: does the installed engine contain the code the model cites?"),
        _code(AUDIT_CELL),
        _md(
            """
            ## 4 — The traffic and the pre-registered predictions (embedded — do not edit)

            The traffic is regenerated here from the generator's settings, and must match the
            fingerprint of the trace the simulator predicted on — or the notebook stops.
            """
        ),
        _code(sub(DATA_CELL)),
        _md("## 5 — Two servers"),
        _code(sub(SERVERS_CELL)),
        _code(sub(START_CELL)),
        _md(
            """
            ## 6 — The routers and the client

            The three routers are ports of the simulator's policies — the same algorithm and the
            same tie-breaking random stream — so a difference between prediction and measurement
            is the engine's, not a different router's.
            """
        ),
        _code(sub(CLIENT_CELL)),
        _code(sub(WARMUP_CELL)),
        _md("## 7 — Six runs: two workloads × three routing policies"),
        _code(sub(RUN_CELL)),
        _md("## 8 — The verdict, and the results file"),
        _code(sub(VERDICT_CELL)),
        _md("## 9 — Clean-up (run after downloading the zip)"),
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
