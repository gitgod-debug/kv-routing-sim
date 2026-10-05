"""Builds notebooks/c4_per_token_t4.ipynb. Called by scripts/c4_prepare.py — not run alone.

C4 measures what T1 and T2 could not check from data on disk:
  * decode steps of batches that SHARE a cached prefix — so the batch's summed context goes several
    times past the KV cache, where every calibration cell so far stopped (part 2);
  * per-request token times on traffic around a 100 ms per-token limit, client-side and from the
    engine's own histogram, whose buckets have a boundary at exactly 0.1 s (part 3);
  * `kvsim calibrate` with its repeats, several times in one session (parts 1 and 4).
The notebook only MEASURES; the verdicts are computed afterwards by scripts/c4_analyze.py,
committed before the run.

As little new harness as possible: the environment, traffic port, server and warm-up cells are
Phase 6's, the pre-flight and the command's cell are C3's, the wheel cell is C1's, and the
streaming client and the unique-prompt decode cell are C2's as they are. New here, and exercised
by scripts/c4_dryrun.py: the shared-prefix decode cell, the histogram reader, and a held-out run
that keeps every request's token times. `@@NAME@@` markers are filled in `build_notebook`.
"""

from __future__ import annotations

import json
from typing import Any

from _c1_notebook import WHEEL_CELL
from _c2_notebook import CHECK_CELL
from _c2_notebook import CLIENT_CELL as C2_CLIENT_CELL
from _c3_notebook import CALIBRATE_CELL as C3_CALIBRATE_CELL
from _c3_notebook import PREFLIGHT_CELL, _between, _start_cell, _swap
from _phase3_notebook import AUDIT_CELL, INSTALL_CELL, _code, _md
from _phase6_notebook import DATA_CELL, ENV_CELL, SERVERS_CELL, WARMUP_CELL

INTRO = """
# C4 — the per-token model where it matters, on a real T4

**Why.** Two results of 2026-10-05 could not be checked from data on disk.
- **T1** found the simulator's per-token latency within 7% of a live engine on three runs — and
  1.5–1.7× too slow on the two at the knee of the load curve. There, the model's decode cost is a
  straight line drawn several times past the largest batch any calibration ever measured.
- **T2** sized the fleet under a target with both halves (first token *and* time per token) — in a
  regime, below 100 ms per token, where no traffic had ever been measured.

| Part | What it measures |
|---|---|
| **1** | `kvsim calibrate`, with its repeats (seed @@SEED0@@) |
| **2** | decode steps of batches that **share a cached prefix** — @@N_SHARED@@ batch sizes × @@SHARED_REPS@@, up to a summed context @@S_MAX_X@@× the largest ever fitted — and @@N_CONTROL@@ unique-prompt controls |
| **2b** | the long gap every decoding sequence sees when one more request arrives and is prefilled — @@N_STALL@@ cells × @@STALL_REPS@@ |
| **3** | @@N_WORKLOADS@@ held-out workloads around the per-token limit, every request's token times kept, the engine's own histograms read before and after |
| **4** | `kvsim calibrate` again (seed@@SEEDS_REST@@), after the traffic |

**Registered before the run** (`configs/c4_per_token.yaml`, `docs/09-hypotheses.md`):

@@RULES_TABLE@@

The command comes from the package's wheel, **embedded in this notebook** (sha256
`@@WHEEL_SHA_SHORT@@…`, built from commit `@@GIT_SHA@@`). This notebook decides nothing: it measures,
and `scripts/c4_analyze.py` decides.
@@AMENDMENTS@@"""

HOW_TO_RUN = """
**How to run.** `python scripts/kaggle_run.py notebooks/c4_per_token_t4.ipynb --slug
kvsim-c4-per-token-t4 --out <folder>` — or by hand: Kaggle → *Settings* → Accelerator **GPU T4 ×2**
(one T4 is used), Internet **on** → *Run All*. About @@MINUTES@@ minutes. Results are saved after
every part, so a stop with ⛔ still leaves **`c4_results.zip`** with the parts before it. Run it
once: a second run is a second experiment.
"""

CONFIG_CELL = r'''
import json

# ── Generated from configs/c4_per_token.yaml. Do not edit here: edit the config and re-run
#    scripts/c4_prepare.py, or the registration stops describing this notebook.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
NUM_GPU_BLOCKS = @@BLOCKS@@          # --num-gpu-blocks-override; one is vLLM's null block
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
N_REPLICAS = @@REPLICAS@@
MEASURE = @@MEASURE@@                 # everything runs against this replica
CAL = json.loads(r"""@@CAL@@""")     # the command's settings; its timing cells are its defaults
DESIGN = json.loads(r"""@@DESIGN@@""")
TPOT_LIMIT_S = @@TPOT_LIMIT_S@@      # the per-token half of the target, in seconds
WHEEL_SHA256 = "@@WHEEL_SHA@@"

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_MEM_UTIL = 0.85
PORTS = [8000]                # replica i listens on PORTS[i] and runs on GPU i
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "c4_results"
'''

NEW_CELL = r'''
import zipfile

TPOT_FAMILY = "vllm:request_time_per_output_token_seconds"
ITL_FAMILY = "vllm:inter_token_latency_seconds"


def hist(text, family):
    """A histogram family from /metrics: cumulative counts by upper bound, and sum and count."""
    pat = re.compile(r"^" + re.escape(family) + r'_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+(\S+)',
                     re.MULTILINE)
    buckets = {}
    for le, v in pat.findall(text):
        buckets[le] = buckets.get(le, 0.0) + float(v)
    return {"buckets": buckets, "sum": metric(text, family + "_sum"),
            "count": metric(text, family + "_count")}


def hist_delta(before, after, family):
    """What a histogram family gained between two scrapes. None if the engine does not have it."""
    b, a = hist(before, family), hist(after, family)
    if not a["buckets"] or a["count"] is None:
        return None
    return {"buckets": {le: v - b["buckets"].get(le, 0.0) for le, v in a["buckets"].items()},
            "sum": a["sum"] - (b["sum"] or 0.0), "count": a["count"] - (b["count"] or 0.0)}


def share_within(h, limit_s):
    """The share of a histogram's observations at or under `limit_s` — exact when `limit_s` is one
    of its bucket bounds, None when it is not (no interpolation: that would invent a number)."""
    if not h or not h["count"]:
        return None
    for le, v in h["buckets"].items():
        if le != "+Inf" and abs(float(le) - limit_s) < 1e-12:
            return v / h["count"]
    return None


def measure_decode_shared(i, prefix_len, suffix_len, b, seed):
    """b requests that SHARE a cached prefix — the prefix, then a unique suffix each — sent together.

    The prefix is cached by one request first, so every one of the b finds it. Its blocks are then
    stored once, however many sequences use them: far more sequences fit than with unique prompts,
    while each still attends over its whole context. That is the regime a shared system prompt
    puts an engine in, and the one no unique-prompt cell can reach. Timed as part A of C2 was: a
    request's decode step = the mean gap from token `skip` to its last; the cell's = the median.
    Invalid if any request failed, anything was preempted, or a request did not hit the prefix."""
    out, skip = DESIGN["decode"]["output_tokens"], DESIGN["decode"]["skip_tokens"]
    rng = random.Random(seed)
    prefix = rand_ids(rng, prefix_len)
    prompts = [prefix + rand_ids(rng, suffix_len) for _ in range(b)]
    was_reset = reset_replica(i)
    settled_scrape(i)
    warm = stream_one(i, prefix, 1)
    if not usable(warm, prefix_len, 1):
        return {"prefix": prefix_len, "suffix": suffix_len, "b": b, "seed": seed, "valid": False,
                "step_s": None, "error": warm["error"]}
    before = settled_scrape(i)
    with ThreadPoolExecutor(max_workers=b) as pool:
        recs = list(pool.map(lambda ids: stream_one(i, ids, out), prompts))
    after = settled_scrape(i)
    d = deltas(before, after)
    steps, cached = [], []
    for r in recs:
        t = r["t_chunks"]
        if usable(r, prefix_len + suffix_len, out) and len(t) == out:
            steps.append((t[-1] - t[skip]) / (out - 1 - skip))
            cached.append(((r["usage"] or {}).get("prompt_tokens_details") or {})
                          .get("cached_tokens"))
    want = (prefix_len // BLOCK_SIZE) * BLOCK_SIZE
    return {
        "prefix": prefix_len, "suffix": suffix_len, "b": b, "seed": seed, "reset": was_reset,
        "valid": (len(steps) == b and (d["preemptions"] or 0) == 0
                  and all(c == want for c in cached)),
        "step_s": median(steps) if steps else None,
        "request_steps_s": [round(x, 6) for x in steps],
        "cached_tokens_seen": sorted({str(c) for c in cached}), "cached_tokens_expected": want,
        "server_itl_s": itl_mean(before, after),
        "preemptions": d["preemptions"], "failed": b - len(steps),
    }


def measure_stall(i, prefix_len, suffix_len, b, new_tokens, delay_s, seed):
    """What ONE arriving request's prefill costs the sequences that are already decoding.

    b requests that share a cached prefix start decoding. `delay_s` later one more request arrives:
    the same prefix, then `new_tokens` unique tokens. The step that prefills it also carries
    everyone else's next token — so each running sequence sees one long gap. The simulator's rule
    is that this step costs prefill + decode, added; no cell has measured that until now.

    Per running sequence: the gap that ends when the late request's first token arrives, and the
    median of its other gaps. Invalid if a request failed, anything was preempted, the late request
    missed the prefix, or it did not land while the others were mid-decode."""
    out, skip = DESIGN["stall"]["output_tokens"], DESIGN["decode"]["skip_tokens"]
    rng = random.Random(seed)
    prefix = rand_ids(rng, prefix_len)
    prompts = [prefix + rand_ids(rng, suffix_len) for _ in range(b)]
    late = prefix + rand_ids(rng, new_tokens)
    was_reset = reset_replica(i)
    settled_scrape(i)
    warm = stream_one(i, prefix, 1)
    if not usable(warm, prefix_len, 1):
        return {"b": b, "new_tokens": new_tokens, "seed": seed, "valid": False,
                "error": warm["error"]}
    before = settled_scrape(i)
    with ThreadPoolExecutor(max_workers=b + 1) as pool:
        futs = [pool.submit(stream_one, i, ids, out) for ids in prompts]
        time.sleep(delay_s)
        late_fut = pool.submit(stream_one, i, late, 1)
        recs, late_rec = [f.result() for f in futs], late_fut.result()
    after = settled_scrape(i)
    d = deltas(before, after)
    want = (prefix_len // BLOCK_SIZE) * BLOCK_SIZE
    late_cached = ((late_rec["usage"] or {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens")
    t_late = late_rec["t_first"]
    at, usual, position = [], [], []
    for r in recs:
        t = r["t_chunks"]
        if not (usable(r, prefix_len + suffix_len, out) and len(t) == out and t_late):
            continue
        # The token of this sequence that arrived with the late request's first token: the one
        # whose arrival time is nearest to it.
        k = min(range(1, out), key=lambda j: abs(t[j] - t_late))
        gaps = [t[j] - t[j - 1] for j in range(max(skip, 1), out) if j != k]
        at.append(t[k] - t[k - 1])
        usual.append(median(gaps))
        position.append(k)
    ok = (len(at) == b and (d["preemptions"] or 0) == 0 and usable(late_rec, len(late), 1)
          and late_cached == want and all(skip < k < out - 4 for k in position))
    return {
        "prefix": prefix_len, "suffix": suffix_len, "b": b, "new_tokens": new_tokens,
        "delay_s": delay_s, "seed": seed, "reset": was_reset, "valid": ok,
        "gap_at_arrival_s": median(at) if at else None,
        "usual_gap_s": median(usual) if usual else None,
        "token_position": [min(position), max(position)] if position else None,
        "late_ttft_s": None if t_late is None else t_late - late_rec["t_send"],
        "late_cached_tokens": late_cached, "preemptions": d["preemptions"],
        "failed": b - len(at),
    }


def run_holdout(name):
    """Held-out traffic on ONE engine, open-loop at its arrival times — C2's part D, and now
    every request's token times are kept: its first token, its last, and how many came between.
    The engine's own two per-token histograms are read before and after."""
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
        u, t = rec["usage"] or {}, rec["t_chunks"]
        ok = usable(rec, len(q["ids"]), out)
        recs.append({
            "request_id": q["request_id"], "valid": ok,
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_ms": None if rec["t_first"] is None else (rec["t_first"] - rec["t_send"]) * 1e3,
            # Time per output token, as MLPerf and vLLM define it: (last − first) ÷ (tokens − 1).
            # Only when every token came as its own chunk — else the count of gaps is not known.
            "tpot_ms": (t[-1] - t[0]) / (len(t) - 1) * 1e3 if ok and len(t) == out > 1 else None,
            "chunks": len(t),
            "send_offset_s": rec["t_send"] - t_zero, "scheduled_offset_s": q["arrival_ts"],
        })
    ttfts = sorted(r["ttft_ms"] for r in recs if r["ttft_ms"] is not None)
    tpots = sorted(r["tpot_ms"] for r in recs if r["tpot_ms"] is not None)
    hits, queries = d["prefix_cache_hits"] or 0, d["prefix_cache_queries"] or 0
    tpot_hist = hist_delta(before, after, TPOT_FAMILY)
    run = {
        "workload": name, "counters": d, "hit_rate": hits / queries if queries else None,
        "preemptions": d["preemptions"], "server_itl_s": itl_mean(before, after),
        "ttft_p50_ms": nearest_rank(ttfts, 0.50), "ttft_p95_ms": nearest_rank(ttfts, 0.95),
        "tpot_p50_ms": nearest_rank(tpots, 0.50), "tpot_p95_ms": nearest_rank(tpots, 0.95),
        "client_share_within_limit": (sum(1 for x in tpots if x <= TPOT_LIMIT_S * 1e3)
                                      / len(tpots) if tpots else None),
        "engine_tpot_hist": tpot_hist, "engine_itl_hist": hist_delta(before, after, ITL_FAMILY),
        "engine_share_within_limit": share_within(tpot_hist, TPOT_LIMIT_S),
        "invalid": sum(1 for r in recs if not r["valid"]),
        "without_tpot": sum(1 for r in recs if r["tpot_ms"] is None),
        "worst_send_lag_ms": max(r["send_offset_s"] - r["scheduled_offset_s"] for r in recs) * 1e3,
        "requests": recs,
    }
    pred = PRED["runs"][name]["profile A"]
    share = run["engine_share_within_limit"]
    print(f"  {name:10s} hit {run['hit_rate']:.4f} ({pred['hit_rate']:.4f}) · TTFT p95 "
          f"{run['ttft_p95_ms'] or 0:.0f} ms ({pred['ttft_p95_ms']:.0f}) · ms per token p50 "
          f"{run['tpot_p50_ms'] or 0:.1f}, p95 {run['tpot_p95_ms'] or 0:.1f} "
          f"({pred['tpot_p95_ms']:.1f}) · at or under {TPOT_LIMIT_S * 1e3:.0f} ms: client "
          f"{(run['client_share_within_limit'] or 0):.1%}, engine "
          f"{'—' if share is None else f'{share:.1%}'} ({pred['share_within_limit']:.1%}) · "
          f"preemptions {run['preemptions']} · {run['invalid']} invalid · send lag ≤ "
          f"{run['worst_send_lag_ms']:.0f} ms     [in brackets: registered for profile A]")
    return run


RESULTS = {"notebook": "notebooks/c4_per_token_t4.ipynb", "design": DESIGN,
           "fingerprints": FINGERPRINTS, "predictions_sha256": PREDICTIONS_SHA256,
           "wheel_sha256": WHEEL_SHA256, "tpot_limit_s": TPOT_LIMIT_S,
           "calibrate": [], "decode_shared": [], "decode_control": [], "stall": [],
           "holdout": []}


def save():
    """Write everything measured so far, and the zip: a later failure keeps the earlier parts."""
    RESULTS.update({"environment": ENV, "preflight": PREFLIGHT, "metric_and_source_audit": AUDIT,
                    "capacity": CAPACITY,
                    "engine_argv": [engine_argv(i) for i in range(N_REPLICAS)]})
    pathlib.Path(OUT_DIR, "c4.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    with zipfile.ZipFile(f"{OUT_DIR}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pathlib.Path(OUT_DIR).glob("*")):
            if f.suffix != ".whl":  # the wheel is in the notebook already
                z.write(f)
    return f"{OUT_DIR}.zip"


print("client, the command, the decode cells, the held-out run and saving ready")
'''

HIST_CHECK_CELL = r'''
# Part 3 reads the engine's own per-request histogram, and its verdict on the per-token limit
# needs a bucket bound at exactly that limit. Both are checked now, on the engine as it runs.
_h = hist(http(MEASURE, "GET", "/metrics")[1], TPOT_FAMILY)
_bounds = sorted(float(le) for le in _h["buckets"] if le != "+Inf")
RESULTS["tpot_histogram_bounds"] = _bounds
if not _bounds:
    print(f"⚠️  {TPOT_FAMILY} is absent: the engine-side share cannot be read "
          "(every request's client-side token times are still recorded)")
elif not any(abs(b - TPOT_LIMIT_S) < 1e-12 for b in _bounds):
    print(f"⚠️  no bucket bound at {TPOT_LIMIT_S} s (bounds: {_bounds}): the engine-side share "
          "cannot be read exactly, and is not interpolated")
else:
    print(f"{TPOT_FAMILY}: {len(_bounds)} bounds, one at {TPOT_LIMIT_S} s ✓")
'''

PART_1_CELL = r'''
RESULTS["calibrate"].append(run_calibrate(CAL["seeds"][0]))
print(f"saved {save()}")
'''

PART_2_CELL = r'''
_s = DESIGN["decode_shared"]
for _rep in range(_s["reps"]):
    for _L, _b in _s["controls"]:
        _m = measure_decode(MEASURE, _L, _b, seed=f"ctl-{_L}-{_b}-{_rep}")
        RESULTS["decode_control"].append({"rep": _rep, **_m})
        print(f"  control rep {_rep}  unique prompts  L {_L:>5}  b {_b:>3}  step "
              f"{(_m['step_s'] or 0) * 1e3:7.1f} ms  {'✓' if _m['valid'] else '⛔ invalid'}")
    for _b in _s["batches"]:
        _m = measure_decode_shared(MEASURE, _s["prefix_tokens"], _s["suffix_tokens"], _b,
                                   seed=f"shared-{_b}-{_rep}")
        RESULTS["decode_shared"].append({"rep": _rep, **_m})
        print(f"  shared  rep {_rep}  prefix {_s['prefix_tokens']} + {_s['suffix_tokens']}  "
              f"b {_b:>3}  step {(_m['step_s'] or 0) * 1e3:7.1f} ms  preemptions "
              f"{_m.get('preemptions')}  cached {_m.get('cached_tokens_seen')}  "
              f"{'✓' if _m['valid'] else '⛔ invalid'}")
    print(f"saved {save()}")
'''

PART_2B_CELL = r'''
_t = DESIGN["stall"]
for _rep in range(_t["reps"]):
    for _b, _n, _delay in _t["cells"]:
        _m = measure_stall(MEASURE, _s["prefix_tokens"], _s["suffix_tokens"], _b, _n, _delay,
                           seed=f"stall-{_b}-{_n}-{_rep}")
        RESULTS["stall"].append({"rep": _rep, **_m})
        print(f"  stall rep {_rep}  b {_b:>3}  + {_n} new tokens after {_delay} s:  gap "
              f"{(_m.get('gap_at_arrival_s') or 0) * 1e3:7.1f} ms against the usual "
              f"{(_m.get('usual_gap_s') or 0) * 1e3:6.1f} ms  at token {_m.get('token_position')}"
              f"  late TTFT {(_m.get('late_ttft_s') or 0) * 1e3:6.1f} ms  "
              f"{'✓' if _m['valid'] else '⛔ invalid'}")
print(f"saved {save()}")
'''

PART_3_CELL = r'''
for _name in WORKLOADS:
    RESULTS["holdout"].append(run_holdout(_name))
    print(f"saved {save()}")
'''

PART_4_CELL = r'''
for _seed in CAL["seeds"][1:]:
    RESULTS["calibrate"].append(run_calibrate(_seed))
    print(f"saved {save()}")
_zip = save()
print(f"\nsaved {_zip} ({pathlib.Path(_zip).stat().st_size / 1e6:.1f} MB)")
print("⭐ c4_results.zip is complete — then the clean-up cell.")
'''

CLEANUP_CELL = r'''
stop_all()
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))
'''


def _preflight_cell() -> str:
    """C3's pre-flight, with the two sentences that name C3 naming this run instead."""
    cell = PREFLIGHT_CELL.replace("c3_results.zip", "c4_results.zip")
    cell = _swap(cell, "C3 is registered ", "This run is registered ")
    return _swap(cell, "the profile C3c tests is a T4 profile", "its profiles are T4 profiles")


def _client_cell() -> str:
    """C2's streaming client and its unique-prompt decode cell, byte for byte; C3's cell that runs
    the command, byte for byte; then what is new here."""
    helpers = _between(C2_CLIENT_CELL, "from statistics import median", "def measure_prefill(")
    command = _between(C3_CALIBRATE_CELL, "def run_calibrate(seed):", "RESULTS = {")
    return "\n" + helpers + command + NEW_CELL.lstrip("\n")


def build_notebook(cfg: dict[str, Any], workloads: dict[str, Any], fingerprints: dict[str, str],
                   pred: dict[str, Any], pred_sha: str, *, wheel_b64: str, wheel_sha: str,
                   wheel_name: str, git_sha: str, rules_table: str, minutes: int,
                   s_max_x: float, amendments: list[dict[str, Any]] = ()) -> str:
    e, c, s = cfg["engine"], cfg["calibrate"], cfg["decode_shared"]
    design = {"decode": {"output_tokens": s["output_tokens"], "skip_tokens": s["skip_tokens"]},
              "decode_shared": s, "stall": cfg["stall"]}
    fill = {
        "MODEL": e["model"], "VLLM": e["vllm"], "BLOCK_SIZE": str(e["block_size"]),
        "BLOCKS": str(e["num_gpu_blocks"]), "MAX_MODEL_LEN": str(e["max_model_len"]),
        "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "REPLICAS": str(cfg["replicas"]), "MEASURE": str(cfg["measure_replica"]),
        "CAL": json.dumps(c), "DESIGN": json.dumps(design), "WORKLOADS": json.dumps(workloads),
        "TPOT_LIMIT_S": repr(cfg["slo"]["tpot_p95_ms"] / 1000.0),
        "FINGERPRINTS": json.dumps(fingerprints), "PRED": json.dumps(pred, sort_keys=True),
        "PRED_SHA": pred_sha, "WHEEL_B64": wheel_b64, "WHEEL_SHA": wheel_sha,
        "WHEEL_SHA_SHORT": wheel_sha[:12], "WHEEL_NAME": wheel_name, "GIT_SHA": git_sha,
        "SEED0": str(c["seeds"][0]),
        "SEEDS_REST": ("s " if len(c["seeds"]) > 2 else " ") + ", ".join(
            str(x) for x in c["seeds"][1:]),
        "N_SHARED": str(len(s["batches"])), "SHARED_REPS": str(s["reps"]),
        "N_CONTROL": str(len(s["controls"])), "S_MAX_X": f"{s_max_x:.0f}",
        "N_STALL": str(len(cfg["stall"]["cells"])), "STALL_REPS": str(cfg["stall"]["reps"]),
        "N_WORKLOADS": str(len(workloads)), "RULES_TABLE": rules_table, "MINUTES": str(minutes),
        "AMENDMENTS": "".join(
            f"\n**Amended {a['date_utc'][:10]}** (commit `{a['git_sha']}`): {a['why']} Nothing "
            "registered changed — the same wheel, traffic, predictions and rules.\n"
            for a in amendments),
    }

    def sub(text: str) -> str:
        for k, v in fill.items():
            text = text.replace(f"@@{k}@@", v)
        left = text.replace("@@KVSIM-WHEEL@@", "")  # the wheel cell's own tag: the dry-run skips it
        if "@@" in left:
            raise ValueError(f"unfilled marker in: {left[left.index('@@') - 40:][:80]!r}")
        return text

    data_cell = _swap(DATA_CELL, "Pre-registered hit rates (simulator, 2 replicas):",
                      "Registered for profile A, one replica (hit rate; the rest is in PRED):")
    cells = [
        _md(sub(INTRO)),
        _md(sub(HOW_TO_RUN)),
        _md("## 0 — Configuration"),
        _code(sub(CONFIG_CELL)),
        _md("## 1 — Environment"),
        _code(sub(ENV_CELL)),
        _md("## 2 — Install vLLM, pinned — and kv-routing-sim, from the wheel embedded here"),
        _code(INSTALL_CELL.replace("@@VLLM@@", e["vllm"])),
        _md("## 2b — Pre-flight: can this box's software start the engine at all?"),
        _code(_preflight_cell()),
        _code(sub(WHEEL_CELL)),
        _md("## 3 — Audit: does the installed engine contain the code the model cites?"),
        _code(AUDIT_CELL),
        _md("## 4 — Part 3's held-out traffic and the registered predictions (embedded — do not "
            "edit)"),
        _code(sub(data_cell)),
        _md("## 5 — One server"),
        _code(sub(SERVERS_CELL)),
        _code(sub(_start_cell().replace("c3_results.zip", "c4_results.zip"))),
        _md("## 6 — The client, the command, the decode cells, the held-out run, and saving"),
        _code(sub(_client_cell())),
        _code(sub(WARMUP_CELL)),
        _code(CHECK_CELL),
        _code(HIST_CHECK_CELL),
        _md("## 7 — Part 1: `kvsim calibrate`, with its repeats"),
        _code(PART_1_CELL),
        _md("## 8 — Part 2: decode steps of batches that share a cached prefix"),
        _code(PART_2_CELL),
        _md("## 8b — Part 2b: what one arriving request costs the sequences already decoding"),
        _code(PART_2B_CELL),
        _md("## 9 — Part 3: held-out traffic around the per-token limit"),
        _code(PART_3_CELL),
        _md("## 10 — Part 4: `kvsim calibrate` again, after the traffic"),
        _code(PART_4_CELL),
        _md("## 11 — Clean-up"),
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
