"""Builds notebooks/c2_timing_t4.ipynb. Called by scripts/c2_prepare.py — not run alone.

C2 measures what the timing model needs and Phase 6 exposed it lacks: decode and prefill cost against
context length (parts A and B), whether CPU contention slows the engine (part C), and two held-out
traffic runs the fitted model must predict (part D). The notebook only MEASURES; the verdict is
computed afterwards by scripts/c2_analyze.py, committed before the run.

Self-contained like Phase 6's (vLLM + the standard library), and it reuses Phase 6's cells as they
are — environment, traffic port, servers, warm-up — so the two runs share one tested harness.
`@@NAME@@` markers are filled in `build_notebook`.
"""

from __future__ import annotations

import json
from typing import Any

from _phase3_notebook import AUDIT_CELL, INSTALL_CELL, _code, _md
from _phase6_notebook import DATA_CELL, ENV_CELL, SERVERS_CELL, START_CELL, WARMUP_CELL

INTRO = """
# C2 — the timing model on a real T4

**Why.** Phase 6 matched every hit rate on two live T4s, but latency came out 5–222× the prediction:
decode steps ran 2.7–4.0× what the fitted profile says. Two suspects: the profile has no term for
context length (Phase 6's prompts are ~1,700 tokens), or CPU contention (two engines and a client on
4 vCPUs). This notebook separates them, and measures what a context-aware model is fitted on.

| Part | What it measures | Replica 1 |
|---|---|---|
| **A** | decode step against context length × batch size — @@N_DECODE@@ cells × @@DECODE_REPS@@ | idle |
| **B** | prefill time with and without a cached prefix — @@N_PREFILL@@ cells × @@PREFILL_REPS@@ | idle |
| **C** | the same decode cells alone and with replica 1 and a client busy (engine-side latency) | busy / idle |
| **D** | two held-out runs of Phase 6's traffic on one engine, which the fitted model must predict | idle |

**Registered before the run** (`configs/c2_timing.yaml`, `docs/09-hypotheses.md`): **C2a** — the
decode step at 1,536 tokens is ≥ 1.5× the step at 128 (batch 8); **C2b** — contention slows the
engine by ≥ 20%; **C2c** — the context-aware model, fitted on A and B only, predicts D's p50 and p95
TTFT within 2×. Predicted: C2a holds, C2b fails, C2c holds. This notebook decides nothing: it
measures, and `scripts/c2_analyze.py` decides.
"""

HOW_TO_RUN = """
**How to run.** Kaggle → *Settings* → Accelerator **GPU T4 ×2**, Internet **on** → *Run All*. About
an hour: ~12 minutes to install and start two servers, then parts A–D. Results are saved after every
part, so if a cell stops with ⛔, download **`c2_results.zip`** anyway — the parts before it are in
it. At the end, download `c2_results.zip` from the Output panel and bring it back.
"""

CONFIG_CELL = r'''
import json

# ── Generated from configs/c2_timing.yaml. Do not edit here: edit the config and re-run
#    scripts/c2_prepare.py, or the registration stops describing this notebook.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
NUM_GPU_BLOCKS = @@BLOCKS@@          # --num-gpu-blocks-override; one is vLLM's null block
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
N_REPLICAS = @@REPLICAS@@
MEASURE = @@MEASURE@@                 # every measured part runs on this replica
DESIGN = json.loads(r"""@@DESIGN@@""")

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_MEM_UTIL = 0.85
PORTS = [8000, 8001]          # replica i listens on PORTS[i] and runs on GPU i
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "c2_results"
'''

CLIENT_CELL = r'''
from statistics import median


def stream_one(i, ids, max_tokens):
    """One streamed completion to replica i: send time, the arrival time of EVERY chunk (one chunk
    per token on vLLM 0.30 — checked in the warm-up below), and the usage block."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORTS[i]}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"t_send": time.perf_counter(), "t_first": None, "t_chunks": [], "usage": None,
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
                    t = time.perf_counter()
                    rec["t_chunks"].append(t)
                    if rec["t_first"] is None:
                        rec["t_first"] = t
                    reason = obj["choices"][0].get("finish_reason")
                    if reason:
                        rec["finish_reason"] = reason
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
    except Exception as e:
        rec["error"] = repr(e)
    rec["t_done"] = time.perf_counter()
    return rec


def rand_ids(rng, n):
    """Ordinary text tokens, random: two prompts share no 16-token block by chance."""
    return [rng.randrange(1000, 100_000) for _ in range(n)]


def usable(rec, prompt_len, out):
    u = rec["usage"] or {}
    return (rec["error"] is None and rec["finish_reason"] == "length"
            and u.get("completion_tokens") == out and u.get("prompt_tokens") == prompt_len)


def itl_totals(text):
    return (metric(text, "vllm:inter_token_latency_seconds_sum"),
            metric(text, "vllm:inter_token_latency_seconds_count"))


def itl_mean(before, after):
    """Engine-side mean inter-token latency between two scrapes (vLLM measures it itself)."""
    (s0, c0), (s1, c1) = itl_totals(before), itl_totals(after)
    if None in (s0, c0, s1, c1) or c1 - c0 <= 0:
        return None
    return (s1 - s0) / (c1 - c0)


def reset_replica(i):
    """A cold cache on ONE replica (the other may be busy in part C). Timing needs no reset —
    prompts are random — so a refusal is recorded, never fatal."""
    for _ in range(15):
        code, body = http(i, "POST", "/reset_prefix_cache")
        try:
            if code == 200 and json.loads(body).get("success") is True:
                return True
        except ValueError:
            pass
        if code in (404, 405):
            return False
        time.sleep(2)
    return False


def nearest_rank(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1))))]


def measure_decode(i, L, b, seed):
    """b requests of L random tokens at once. A request's decode step = mean gap from token `skip`
    to its last; the cell's = the median over requests. Invalid if any request failed or anything
    was preempted — a preempted request's gaps would time recomputation, not decode."""
    out, skip = DESIGN["decode"]["output_tokens"], DESIGN["decode"]["skip_tokens"]
    rng = random.Random(seed)
    prompts = [rand_ids(rng, L) for _ in range(b)]
    was_reset = reset_replica(i)
    before = settled_scrape(i)
    with ThreadPoolExecutor(max_workers=b) as pool:
        recs = list(pool.map(lambda ids: stream_one(i, ids, out), prompts))
    after = settled_scrape(i)
    d = deltas(before, after)
    steps = []
    for r in recs:
        t = r["t_chunks"]
        if usable(r, L, out) and len(t) == out:
            steps.append((t[-1] - t[skip]) / (out - 1 - skip))
    return {
        "L": L, "b": b, "seed": seed, "reset": was_reset,
        "valid": len(steps) == b and (d["preemptions"] or 0) == 0,
        "step_s": median(steps) if steps else None,
        "request_steps_s": [round(x, 6) for x in steps],
        "ttft_s": [round(r["t_first"] - r["t_send"], 6) for r in recs if r["t_first"]],
        "server_itl_s": itl_mean(before, after),
        "preemptions": d["preemptions"], "failed": b - len(steps),
    }


def measure_prefill(i, c, n, seed):
    """A warm request of c tokens, then c + n tokens with the same first c: TTFT of the second."""
    rng = random.Random(seed)
    prefix, new = rand_ids(rng, c), rand_ids(rng, n)
    reset_replica(i)
    settled_scrape(i)
    if c:
        warm = stream_one(i, prefix, 1)
        if not usable(warm, c, 1):
            return {"c": c, "n": n, "seed": seed, "valid": False, "error": warm["error"]}
        settled_scrape(i)
    rec = stream_one(i, prefix + new, 1)
    u = rec["usage"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "c": c, "n": n, "seed": seed, "cached_tokens": cached, "prompt_tokens": c + n,
        "ttft_s": None if rec["t_first"] is None else round(rec["t_first"] - rec["t_send"], 6),
        "valid": usable(rec, c + n, 1) and cached is not None and rec["t_first"] is not None,
        "error": rec["error"],
    }


# Part C's background load: a SEPARATE process streaming to replica 1, as Phase 6's client did —
# its CPU use is the point. It stops when the stop file appears.
BG_SCRIPT = r"""
import json, random, sys, threading, time, urllib.request, pathlib
port, conc, shared_n, private_n, out, stop, model = sys.argv[1:8]
conc, shared_n, private_n, out = int(conc), int(shared_n), int(private_n), int(out)
shared = [1000 + (7 * j) % 90000 for j in range(shared_n)]
done = [0]
def worker(k):
    rng = random.Random(k)
    while not pathlib.Path(stop).exists():
        ids = shared + [rng.randrange(1000, 100000) for _ in range(private_n)]
        body = {"model": model, "prompt": ids, "max_tokens": out, "temperature": 0.0,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for raw in resp:
                    line = raw.strip()
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        json.loads(line[5:].strip())
            done[0] += 1
        except Exception:
            time.sleep(0.5)
threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(conc)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(json.dumps({"completed": done[0]}))
"""


def start_background():
    bg = DESIGN["cpu"]["background"]
    script = pathlib.Path(OUT_DIR) / "bg_load.py"
    script.write_text(BG_SCRIPT)
    stop = pathlib.Path(OUT_DIR) / "bg.stop"
    stop.unlink(missing_ok=True)
    log = open(pathlib.Path(OUT_DIR) / "bg_load.log", "a")  # noqa: SIM115 - outlives the call
    proc = subprocess.Popen(
        [sys.executable, str(script), str(PORTS[bg["replica"]]), str(bg["concurrency"]),
         str(bg["shared_prefix_tokens"]), str(bg["private_tokens"]), str(bg["output_tokens"]),
         str(stop), MODEL], stdout=log, stderr=subprocess.STDOUT)
    return {"proc": proc, "stop": stop, "log": log}


def stop_background(bg):
    bg["stop"].touch()
    try:
        bg["proc"].wait(timeout=600)
    except subprocess.TimeoutExpired:
        bg["proc"].kill()
    bg["log"].close()
    settled_scrape(DESIGN["cpu"]["background"]["replica"], max_wait=600)


def run_holdout(name):
    """Phase 6's traffic on ONE engine, open-loop at its arrival times."""
    trace, out = TRACES[name], WORKLOADS[name]["output_tokens"]
    reset_replica(MEASURE)
    before = settled_scrape(MEASURE)
    t_zero = time.perf_counter() + 2.0
    with ThreadPoolExecutor(max_workers=len(trace)) as pool:
        futs = []
        for q in trace:
            delay = t_zero + q["arrival_ts"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futs.append((q, pool.submit(stream_one, MEASURE, q["ids"], out)))
        results = [(q, f.result()) for q, f in futs]
    after = settled_scrape(MEASURE, max_wait=900)
    d = deltas(before, after)
    recs = []
    for q, rec in results:
        u = rec["usage"] or {}
        recs.append({
            "request_id": q["request_id"],
            "valid": usable(rec, len(q["ids"]), out),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_ms": None if rec["t_first"] is None else (rec["t_first"] - rec["t_send"]) * 1e3,
            "send_offset_s": rec["t_send"] - t_zero, "scheduled_offset_s": q["arrival_ts"],
        })
    ttfts = sorted(r["ttft_ms"] for r in recs if r["ttft_ms"] is not None)
    hits, queries = d["prefix_cache_hits"] or 0, d["prefix_cache_queries"] or 0
    run = {
        "workload": name, "counters": d, "hit_rate": hits / queries if queries else None,
        "preemptions": d["preemptions"], "server_itl_s": itl_mean(before, after),
        "ttft_p50_ms": nearest_rank(ttfts, 0.50), "ttft_p95_ms": nearest_rank(ttfts, 0.95),
        "invalid": sum(1 for r in recs if not r["valid"]),
        "worst_send_lag_ms": max(r["send_offset_s"] - r["scheduled_offset_s"] for r in recs) * 1e3,
        "requests": recs,
    }
    pred = PRED["runs"][name]["context-free profile"]
    print(f"  {name:12s} hit {run['hit_rate']:.4f} (context-free model {pred['hit_rate']:.4f}) · "
          f"p50 {run['ttft_p50_ms'] or 0:.0f} ms, p95 {run['ttft_p95_ms'] or 0:.0f} ms (context-free "
          f"model {pred['ttft_p50_ms']:.0f} / {pred['ttft_p95_ms']:.0f}) · preemptions "
          f"{run['preemptions']} · {run['invalid']} invalid · send lag ≤ {run['worst_send_lag_ms']:.0f} ms")
    return run


RESULTS = {"notebook": "notebooks/c2_timing_t4.ipynb", "design": DESIGN,
           "fingerprints": FINGERPRINTS, "predictions_sha256": PREDICTIONS_SHA256}


def save():
    """Write everything measured so far, and the zip: a later failure keeps the earlier parts."""
    RESULTS.update({"environment": ENV, "metric_and_source_audit": AUDIT, "capacity": CAPACITY,
                    "engine_argv": [engine_argv(i) for i in range(N_REPLICAS)]})
    pathlib.Path(OUT_DIR, "c2.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    return shutil.make_archive(OUT_DIR, "zip", OUT_DIR)


print("client, parts and saving ready")
'''

CHECK_CELL = r'''
# The design reads decode steps from chunk arrival times: that needs one chunk per token.
_rec = stream_one(MEASURE, [1000 + j for j in range(64)], 16)
if not usable(_rec, 64, 16) or len(_rec["t_chunks"]) != 16:
    raise Stop(f"⛔ 16 tokens arrived in {len(_rec['t_chunks'])} chunks: part A's step timing "
               "assumes one chunk per token")
_itl = itl_totals(http(MEASURE, "GET", "/metrics")[1])
if None in _itl:
    print("⚠️  vllm:inter_token_latency_seconds is absent: part C's engine-side test cannot be "
          "decided (the client-side steps are still recorded)")
print(f"one chunk per token ✓ · engine-side inter-token latency {'✓' if None not in _itl else '—'}")
'''

PART_A_CELL = r'''
DECODE = []
_d = DESIGN["decode"]
for _rep in range(_d["reps"]):
    for _L, _b in _d["grid"]:
        _m = measure_decode(MEASURE, _L, _b, seed=f"A-{_L}-{_b}-{_rep}")
        DECODE.append({"rep": _rep, **_m})
        _itl = _m["server_itl_s"]
        print(f"  A rep {_rep}  L {_L:>5}  b {_b:>3}  step {(_m['step_s'] or 0) * 1e3:7.1f} ms  "
              f"engine ITL {(_itl or 0) * 1e3:7.1f} ms  preemptions {_m['preemptions']}  "
              f"{'✓' if _m['valid'] else '⛔ invalid'}")
RESULTS["decode"] = DECODE
print(f"saved {save()}")
'''

PART_B_CELL = r'''
PREFILL = []
_p = DESIGN["prefill"]
for _rep in range(_p["reps"]):
    for _c, _n in _p["grid"]:
        _m = measure_prefill(MEASURE, _c, _n, seed=f"B-{_c}-{_n}-{_rep}")
        PREFILL.append({"rep": _rep, **_m})
        print(f"  B rep {_rep}  cached {_c:>5} (engine: {_m.get('cached_tokens')})  new {_n:>5}  "
              f"TTFT {(_m.get('ttft_s') or 0) * 1e3:7.1f} ms  {'✓' if _m['valid'] else '⛔ invalid'}")
RESULTS["prefill"] = PREFILL
print(f"saved {save()}")
'''

PART_C_CELL = r'''
CPU = []
_c = DESIGN["cpu"]
for _rep in range(_c["reps"]):
    for _cond in ("alone", "contended"):
        _bg = None
        if _cond == "contended":
            _bg = start_background()
            time.sleep(_c["settle_s"])
        try:
            for _L, _b in _c["cells"]:
                _m = measure_decode(MEASURE, _L, _b, seed=f"C-{_L}-{_b}-{_rep}-{_cond}")
                CPU.append({"rep": _rep, "condition": _cond, **_m})
                print(f"  C rep {_rep}  {_cond:9s}  L {_L:>5}  b {_b:>3}  engine ITL "
                      f"{(_m['server_itl_s'] or 0) * 1e3:7.1f} ms  client step "
                      f"{(_m['step_s'] or 0) * 1e3:7.1f} ms  {'✓' if _m['valid'] else '⛔ invalid'}")
        finally:
            if _bg is not None:
                stop_background(_bg)
RESULTS["cpu"] = CPU
print(f"saved {save()}")
'''

PART_D_CELL = r'''
HOLDOUT = []
for _name in WORKLOADS:
    HOLDOUT.append(run_holdout(_name))
RESULTS["holdout"] = HOLDOUT
_zip = save()
print(f"\nsaved {_zip} ({pathlib.Path(_zip).stat().st_size / 1e6:.1f} MB)")
print("⭐ Download c2_results.zip and bring it back — then run the clean-up cell.")
'''

CLEANUP_CELL = r'''
stop_all()
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))
'''


def build_notebook(cfg: dict[str, Any], workloads: dict[str, Any], fingerprints: dict[str, str],
                   pred: dict[str, Any], pred_sha: str) -> str:
    e = cfg["engine"]
    design = {k: cfg[k] for k in ("decode", "prefill", "cpu")}
    fill = {
        "MODEL": e["model"], "VLLM": e["vllm"], "BLOCK_SIZE": str(e["block_size"]),
        "BLOCKS": str(e["num_gpu_blocks"]), "MAX_MODEL_LEN": str(e["max_model_len"]),
        "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "REPLICAS": str(cfg["replicas"]), "MEASURE": str(cfg["measure_replica"]),
        "DESIGN": json.dumps(design), "WORKLOADS": json.dumps(workloads),
        "FINGERPRINTS": json.dumps(fingerprints), "PRED": json.dumps(pred, sort_keys=True),
        "PRED_SHA": pred_sha, "N_DECODE": str(len(cfg["decode"]["grid"])),
        "DECODE_REPS": str(cfg["decode"]["reps"]), "N_PREFILL": str(len(cfg["prefill"]["grid"])),
        "PREFILL_REPS": str(cfg["prefill"]["reps"]),
    }

    def sub(text: str) -> str:
        for k, v in fill.items():
            text = text.replace(f"@@{k}@@", v)
        if "@@" in text:
            raise ValueError(f"unfilled marker in: {text[text.index('@@') - 40:][:80]!r}")
        return text

    data_cell = DATA_CELL.replace("Pre-registered hit rates (simulator, 2 replicas):",
                                  "The context-free profile's predictions (one replica), reported:")
    cells = [
        _md(sub(INTRO)),
        _md(HOW_TO_RUN),
        _md("## 0 — Configuration"),
        _code(sub(CONFIG_CELL)),
        _md("## 1 — Environment: two GPUs, one server each"),
        _code(sub(ENV_CELL)),
        _md("## 2 — Install vLLM, pinned"),
        _code(INSTALL_CELL.replace("@@VLLM@@", e["vllm"])),
        _md("## 3 — Audit: does the installed engine contain the code the model cites?"),
        _code(AUDIT_CELL),
        _md("## 4 — Part D's held-out traffic (embedded — do not edit)"),
        _code(sub(data_cell)),
        _md("## 5 — Two servers"),
        _code(sub(SERVERS_CELL)),
        _code(sub(START_CELL)),
        _md("## 6 — The client, the four parts, and saving"),
        _code(sub(CLIENT_CELL)),
        _code(sub(WARMUP_CELL)),
        _code(CHECK_CELL),
        _md("## 7 — Part A: decode step against context length × batch size (replica 1 idle)"),
        _code(PART_A_CELL),
        _md("## 8 — Part B: prefill with and without a cached prefix (replica 1 idle)"),
        _code(PART_B_CELL),
        _md("## 9 — Part C: CPU contention — the same cells alone and with replica 1 busy"),
        _code(PART_C_CELL),
        _md("## 10 — Part D: held-out traffic on one engine"),
        _code(PART_D_CELL),
        _md("## 11 — Clean-up (run after downloading the zip)"),
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
