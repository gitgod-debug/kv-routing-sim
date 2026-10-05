"""Builds notebooks/c3_calibrate_t4.ipynb. Called by scripts/c3_prepare.py — not run alone.

C3 is the context-aware `kvsim calibrate`'s first run on a real GPU. The notebook starts one vLLM
server on a T4 with C2's flags, runs the command against it from the package's own wheel (embedded,
as in C1), sends three held-out workloads the way C2's part D did, and runs the command once more.
It only MEASURES; the verdict is computed afterwards by scripts/c3_analyze.py, committed before the
run.

Nothing that MEASURES is new harness: the environment, traffic port, server and warm-up cells are
Phase 6's, the wheel cell is C1's, and the streaming client and the held-out run are cut out of
C2's client cell as they are — so C3's traffic is measured by the code that measured C2's.

One cell is C3's own, added 2026-10-05 after the first attempt stopped before any measurement
(docs/09): PREFLIGHT_CELL. It checks, in about a minute, that the box's software can start vLLM at
all, repairs the one fault that attempt met, and makes every early stop leave a zip. `@@NAME@@`
markers are filled in `build_notebook`.
"""

from __future__ import annotations

import json
from typing import Any

from _c1_notebook import WHEEL_CELL
from _c2_notebook import CLIENT_CELL as C2_CLIENT_CELL
from _phase3_notebook import AUDIT_CELL, INSTALL_CELL, _code, _md
from _phase6_notebook import DATA_CELL, ENV_CELL, SERVERS_CELL, START_CELL, WARMUP_CELL

INTRO = """
# C3 — the context-aware `kvsim calibrate` on a real T4

**Why.** `kvsim calibrate` is how a user gets numbers for their own GPU. C1 proved its cache check
on a real T4, with its first, straight-line timing fit. C2 then showed that line up to 7× optimistic
and validated a context-aware model — fitted by an analysis script on 101 measured points. The
command now fits that model itself, from fewer cells of its own, and that has only ever met a mock
engine. This is its first real one.

| Part | What happens |
|---|---|
| **1** | `kvsim calibrate` against the engine, every setting at its default (seed @@SEED0@@) |
| **2** | @@N_WORKLOADS@@ held-out workloads on the same engine — traffic no fit has seen |
| **3** | the command once more (seed @@SEED1@@), after the traffic — reported, not tested |

**Registered before the run** (`configs/c3_calibrate.yaml`, `docs/09-hypotheses.md`):

| | Rule | Predicted |
|---|---|---|
| **C3a** | the command exits 0, its check is exact (@@CHECK_HITS@@ of @@CHECK_QUERIES@@ tokens, all @@CHECK_N@@ requests) and it reads the engine's @@BLOCKS_TEXT@@ blocks | @@PRED_A@@ |
| **C3b** | the profile it writes predicts part 2's p50 and p95 TTFT within @@MAX_B@@× | @@PRED_B@@ |
| **C3c** | the profile C2 fitted on 2026-09-30 predicts them within @@MAX_C@@× — its numbers are embedded below | @@PRED_C@@ |

The command comes from the package's wheel, **embedded in this notebook** (sha256
`@@WHEEL_SHA_SHORT@@…`, built from commit `@@GIT_SHA@@`). This notebook decides nothing: it measures,
and `scripts/c3_analyze.py` decides.
@@AMENDMENTS@@"""

HOW_TO_RUN = """
**How to run.** Kaggle → *Settings* → Accelerator **GPU T4 ×2** (one T4 is used), Internet **on** →
*Run All*. About 50 minutes: ~12 to install and start the server, ~5 for the command, ~25 of traffic,
~5 for the command again. Results are saved after every part, so if a cell stops with ⛔, download
**`c3_results.zip`** anyway — the parts before it are in it. At the end, download `c3_results.zip`
from the Output panel and bring it back. Run it once: a second run is a second experiment.

Section 2b checks the box's software in about a minute, before anything long. If it stops with ⛔,
nothing was measured: bring back the zip and that cell's output.
"""

CONFIG_CELL = r'''
import json

# ── Generated from configs/c3_calibrate.yaml. Do not edit here: edit the config and re-run
#    scripts/c3_prepare.py, or the registration stops describing this notebook.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
NUM_GPU_BLOCKS = @@BLOCKS@@          # --num-gpu-blocks-override; one is vLLM's null block
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
N_REPLICAS = @@REPLICAS@@
MEASURE = @@MEASURE@@                 # the command and the traffic both run against this replica
CAL = json.loads(r"""@@CAL@@""")     # the command's settings; its timing cells are its defaults
WHEEL_SHA256 = "@@WHEEL_SHA@@"

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_MEM_UTIL = 0.85
PORTS = [8000]                # replica i listens on PORTS[i] and runs on GPU i
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "c3_results"
'''

PREFLIGHT_CELL = r'''
# ── Pre-flight (added 2026-10-05, docs/09). Can this box's software start vLLM at all? Answered in
#    about a minute, before the 8 GB download — and repaired where the repair is known.
#
#    Why it exists: `pip install vllm` brings the torch build vLLM pins, but KEEPS any package whose
#    version number already satisfies a pin. Kaggle's image ships torchaudio 2.11.0 built for another
#    CUDA than that torch — the very version vLLM pins — so pip keeps it, and then nothing that
#    imports transformers can start: vLLM stops with "PyTorch and TorchAudio were compiled with
#    different CUDA versions". The repair puts back the build from PyPI, which is what pip installs
#    on a clean box. vLLM itself, its version and its flags are not touched.
import importlib
import importlib.metadata
import zipfile

PREFLIGHT = {"repairs": []}
_FAMILY = ("torchaudio", "torchvision", "torchcodec")  # none is needed by a text model; any can break

# Run in a FRESH interpreter: is the module installed, and does importing it work?
_PROBE = """
import importlib, importlib.metadata as md, importlib.util, json, sys
mod = sys.argv[1]
if importlib.util.find_spec(mod) is None:
    print("PROBE " + json.dumps({"installed": False}))
    sys.exit(0)
try:
    ver = md.version(mod)
except Exception:
    ver = None
try:
    importlib.import_module(mod)
    out = {"installed": True, "version": ver, "ok": True}
except BaseException as e:
    out = {"installed": True, "version": ver, "ok": False,
           "error": (type(e).__name__ + ": " + str(e))[:500]}
print("PROBE " + json.dumps(out))
"""
_TORCH = """
import json, torch
ok = torch.cuda.is_available()
print("PROBE " + json.dumps({"version": torch.__version__, "cuda_build": torch.version.cuda,
                             "cuda_available": ok,
                             "device": torch.cuda.get_device_name(0) if ok else None}))
"""


def _run(cmd, timeout=1200):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout + p.stderr
    except Exception as e:  # a missing binary or a timeout is an answer too
        return -1, repr(e)


def _answer(code, *args):
    """The PROBE line a fresh interpreter printed, or why there is none."""
    rc, out = _run([sys.executable, "-c", code, *args], 600)
    line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("PROBE ")), None)
    if line is None:
        return {"installed": True, "version": None, "ok": False,
                "error": f"no answer (exit {rc}): {out[-400:]}"}
    return json.loads(line[6:])


def _probe(mod):
    return _answer(_PROBE, mod)


def _pinned(mod, requires=None):
    """The exact version vLLM's own metadata requires of `mod`, or None if it pins none."""
    if requires is None:
        try:
            requires = importlib.metadata.requires("vllm") or []
        except importlib.metadata.PackageNotFoundError:
            return None
    for req in requires:
        m = re.match(r"\s*" + re.escape(mod) + r"\s*==\s*([0-9][0-9A-Za-z.]*)\s*(;.*)?$", req)
        if m and "extra" not in (m.group(2) or ""):
            return m.group(1)
    return None


def repair(mod, probe=None, run=None, pinned=None):
    """A package that is installed but cannot be imported: put back the build vLLM pins, from PyPI
    — what pip installs on a clean box. If that does not import either, remove it: a text model
    needs none of these, and transformers skips what is not installed."""
    probe, run, pinned = probe or _probe, run or _run, pinned or _pinned
    was = probe(mod)
    if not was["installed"] or was["ok"]:
        return was
    pip, steps, want = [sys.executable, "-m", "pip"], [], pinned(mod)
    if want:
        steps.append(pip + ["install", "-q", "--no-deps", "--force-reinstall", "--index-url",
                            "https://pypi.org/simple", f"{mod}=={want}"])
    steps.append(pip + ["uninstall", "-y", "-q", mod])
    now = was
    for cmd in steps:
        rc, out = run(cmd)
        now = probe(mod)
        PREFLIGHT["repairs"].append({"package": mod, "was": was, "did": " ".join(cmd[3:]),
                                     "returncode": rc, "now": now, "output_tail": out[-400:]})
        if not now["installed"] or now["ok"]:
            break
    return now


def diag_zip():
    """What an early stop leaves behind: the environment, every package's version, and whatever
    is in the results folder — the engine's whole log, if it got that far."""
    PREFLIGHT["pip_freeze"] = _run([sys.executable, "-m", "pip", "freeze"], 300)[1].splitlines()
    pathlib.Path(OUT_DIR, "preflight.json").write_text(
        json.dumps({"environment": ENV, "preflight": PREFLIGHT}, indent=1, default=str))
    with zipfile.ZipFile(f"{OUT_DIR}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pathlib.Path(OUT_DIR).glob("*")):
            if f.suffix != ".whl":
                z.write(f)
    return f"{OUT_DIR}.zip"


# ── run it
importlib.invalidate_caches()
PREFLIGHT["torch"] = _answer(_TORCH)
print("torch      ", PREFLIGHT["torch"])
if not PREFLIGHT["torch"].get("cuda_available"):
    diag_zip()
    raise Stop("⛔ torch cannot use a GPU in this environment — the reason is above. Nothing was "
               "measured. Bring back c3_results.zip and this cell's output.")
if "T4" not in (PREFLIGHT["torch"].get("device") or ""):
    diag_zip()
    raise Stop(f"⛔ the GPU is {PREFLIGHT['torch'].get('device')!r}, not a Tesla T4. C3 is registered "
               "for one T4, and the profile C3c tests is a T4 profile: nothing is measured on "
               "other hardware. Kaggle: Settings → Accelerator → GPU T4 ×2.")
PREFLIGHT["packages"] = {}
for _mod in _FAMILY:
    PREFLIGHT["packages"][_mod] = repair(_mod)
    print(f"{_mod:11s}", PREFLIGHT["packages"][_mod])
for _r in PREFLIGHT["repairs"]:
    print(f"  repaired {_r['package']}: was {_r['was'].get('version')} — "
          f"{(_r['was'].get('error') or '')[:120]}\n    did: pip {_r['did']}\n    now: {_r['now']}")
_rc, _out = _run(["vllm", "serve", "--help"], 900)
PREFLIGHT["vllm_cli"] = {"returncode": _rc, "output_tail": _out[-3000:]}
if _rc != 0:
    print(_out[-3000:])
    diag_zip()
    raise Stop("⛔ `vllm serve --help` fails in this environment, so no engine can start — the "
               "reason is above. Nothing was measured. Bring back c3_results.zip and this "
               "cell's output.")
diag_zip()
print(f"vllm        its command line starts ✓ · {len(PREFLIGHT['repairs'])} repair step(s) · "
      "pre-flight saved")
'''

CALIBRATE_CELL = r'''


def run_calibrate(seed):
    """`kvsim calibrate` as a user runs it, from the embedded wheel: its own process, every timing
    cell at its default. Never raises — a failure is a result, and part 2 must still run."""
    out = pathlib.Path(OUT_DIR) / f"calibrated_profile_seed{seed}.json"
    out.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "kv_routing_sim.cli", "calibrate",
           "--engine", f"http://127.0.0.1:{PORTS[MEASURE]}", "--model", MODEL, "--out", str(out),
           "--seed", str(seed), "--check-sessions", str(CAL["check_sessions"]),
           "--check-turns", str(CAL["check_turns"]), "--max-error", str(CAL["max_error"]),
           "--token-range", f"{CAL['token_range'][0]},{CAL['token_range'][1]}"]
    print("$", " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = None, str(e.stdout or ""), f"timed out after {e.timeout} s"
    wall = round(time.time() - t0, 1)
    pathlib.Path(OUT_DIR, f"calibrate_seed{seed}_stdout.txt").write_text(stdout)
    pathlib.Path(OUT_DIR, f"calibrate_seed{seed}_stderr.txt").write_text(stderr)
    print(stdout)
    print(stderr)
    profile = json.loads(out.read_text()) if out.exists() else None
    chk = ((profile or {}).get("_source") or {}).get("check") or {}
    reg, pred = PRED["check"][str(seed)], chk.get("predicted") or {}
    rec = {
        "seed": seed, "returncode": rc, "wall_clock_s": wall,
        "profile_file": out.name if profile else None,
        "calibrated": bool(profile and profile.get("calibrated")),
        "read_blocks": profile and profile.get("kv_cache_blocks"),
        "check_measured": chk.get("measured"), "check_predicted": pred,
        "cached_tokens_equal": chk.get("cached_tokens_equal"),
        "cached_tokens_compared": chk.get("cached_tokens_compared"),
        # The embedded code must predict exactly the registered numbers; if not, the wheel is not
        # the code that made the prediction, and nothing here can be compared.
        "prediction_matches_registration": (pred.get("hits"), pred.get("queries"))
        == (reg["hits"], reg["queries"]),
    }
    print(f"exit code {rc} · {wall} s · calibrated {rec['calibrated']} · check requests equal "
          f"{rec['cached_tokens_equal']}/{rec['cached_tokens_compared']} (registered "
          f"{reg['requests']}) · blocks read {rec['read_blocks']}")
    return rec


RESULTS = {"notebook": "notebooks/c3_calibrate_t4.ipynb", "fingerprints": FINGERPRINTS,
           "predictions_sha256": PREDICTIONS_SHA256, "wheel_sha256": WHEEL_SHA256,
           "calibrate": [], "holdout": []}


def save():
    """Write everything measured so far, and the zip: a later failure keeps the earlier parts."""
    RESULTS.update({"environment": ENV, "preflight": PREFLIGHT, "metric_and_source_audit": AUDIT,
                    "capacity": CAPACITY,
                    "engine_argv": [engine_argv(i) for i in range(N_REPLICAS)]})
    pathlib.Path(OUT_DIR, "c3.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    with zipfile.ZipFile(f"{OUT_DIR}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pathlib.Path(OUT_DIR).glob("*")):
            if f.suffix != ".whl":  # the wheel is in the notebook already
                z.write(f)
    return f"{OUT_DIR}.zip"


print("client, the command, the held-out run and saving ready")
'''

PART_1_CELL = r'''
RESULTS["calibrate"].append(run_calibrate(CAL["seeds"][0]))
print(f"saved {save()}")
'''

PART_2_CELL = r'''
for _name in WORKLOADS:
    RESULTS["holdout"].append(run_holdout(_name))
    print(f"saved {save()}")
'''

PART_3_CELL = r'''
for _seed in CAL["seeds"][1:]:
    RESULTS["calibrate"].append(run_calibrate(_seed))
_zip = save()
print(f"\nsaved {_zip} ({pathlib.Path(_zip).stat().st_size / 1e6:.1f} MB)")
print("⭐ Download c3_results.zip and bring it back — then run the clean-up cell.")
'''

CLEANUP_CELL = r'''
stop_all()
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv"))
'''


def _between(text: str, start: str, end: str) -> str:
    """A slice of another notebook's cell, from `start` up to `end`. Both must be there."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"error: C2's client cell changed ({start!r} … {end!r}); update "
                         "_c3_notebook.py")
    return text[text.index(start):text.index(end)]


def _swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"error: expected exactly one {old!r} in a reused cell; update "
                         "_c3_notebook.py")
    return text.replace(old, new)


def _start_cell() -> str:
    """Phase 6's start cell — but here a stop comes before `save()` exists, so each one writes the
    diagnostics zip first, and its message says so."""
    cell = _swap(START_CELL,
                 'raise Stop(f"⛔ replica {_i} did not start — the log tail is above. Download '
                 'the zip.")',
                 'diag_zip()\n        raise Stop(f"⛔ replica {_i} did not start — the log tail is '
                 'above. Nothing was measured. "\n                   "Bring back c3_results.zip: '
                 'it holds the whole log.")')
    return _swap(cell, "        raise Stop(f\"⛔ replica {_i} logged ",
                 "        diag_zip()\n        raise Stop(f\"⛔ replica {_i} logged ")


def _client_cell() -> str:
    """C2's streaming client and its held-out run, byte for byte — but for the label of the
    prediction printed beside each run, which here is C2's profile, not the context-free one."""
    helpers = _between(C2_CLIENT_CELL, "from statistics import median", "def measure_decode(")
    holdout = _between(C2_CLIENT_CELL, "def run_holdout(name):", "RESULTS = {")
    holdout = _swap(holdout, '["context-free profile"]', '["C2 profile"]')
    holdout = _swap(holdout, "(context-free model {pred['hit_rate']",
                    "(C2's profile {pred['hit_rate']")
    holdout = _swap(holdout, 'ms (context-free "\n          f"model {pred',
                    'ms (C2\'s profile "\n          f"{pred')
    return "\n" + helpers + holdout + CALIBRATE_CELL.lstrip("\n")


def build_notebook(cfg: dict[str, Any], workloads: dict[str, Any], fingerprints: dict[str, str],
                   pred: dict[str, Any], pred_sha: str, *, wheel_b64: str, wheel_sha: str,
                   wheel_name: str, git_sha: str, amendments: list[dict[str, Any]] = ()) -> str:
    e, c, rules, p = cfg["engine"], cfg["calibrate"], cfg["rules"], cfg["predictions"]
    check0 = pred["check"][str(c["seeds"][0])]
    fill = {
        "MODEL": e["model"], "VLLM": e["vllm"], "BLOCK_SIZE": str(e["block_size"]),
        "BLOCKS": str(e["num_gpu_blocks"]), "BLOCKS_TEXT": f"{e['num_gpu_blocks']:,}",
        "MAX_MODEL_LEN": str(e["max_model_len"]),
        "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "REPLICAS": str(cfg["replicas"]), "MEASURE": str(cfg["measure_replica"]),
        "CAL": json.dumps(c), "WORKLOADS": json.dumps(workloads),
        "FINGERPRINTS": json.dumps(fingerprints), "PRED": json.dumps(pred, sort_keys=True),
        "PRED_SHA": pred_sha, "WHEEL_B64": wheel_b64, "WHEEL_SHA": wheel_sha,
        "WHEEL_SHA_SHORT": wheel_sha[:12], "WHEEL_NAME": wheel_name, "GIT_SHA": git_sha,
        "SEED0": str(c["seeds"][0]), "SEED1": str(c["seeds"][1]),
        "N_WORKLOADS": str(len(workloads)), "CHECK_HITS": f"{check0['hits']:,}",
        "CHECK_QUERIES": f"{check0['queries']:,}", "CHECK_N": str(check0["requests"]),
        "MAX_B": f"{rules['c3b_max_factor']:g}", "MAX_C": f"{rules['c3c_max_factor']:g}",
        "PRED_A": p["c3a"], "PRED_B": p["c3b"], "PRED_C": p["c3c"],
        # What was changed after registration, and that nothing registered was: said on page one.
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
                      "Registered for C2's profile, one replica (hit rate; p50 and p95 are in PRED):")
    cells = [
        _md(sub(INTRO)),
        _md(HOW_TO_RUN),
        _md("## 0 — Configuration"),
        _code(sub(CONFIG_CELL)),
        _md("## 1 — Environment"),
        _code(sub(ENV_CELL)),
        _md("## 2 — Install vLLM, pinned — and kv-routing-sim, from the wheel embedded here"),
        _code(INSTALL_CELL.replace("@@VLLM@@", e["vllm"])),
        _md("## 2b — Pre-flight: can this box's software start the engine at all?"),
        _code(PREFLIGHT_CELL),
        _code(sub(WHEEL_CELL)),
        _md("## 3 — Audit: does the installed engine contain the code the model cites?"),
        _code(AUDIT_CELL),
        _md("## 4 — Part 2's held-out traffic and the registered predictions (embedded — do not "
            "edit)"),
        _code(sub(data_cell)),
        _md("## 5 — One server"),
        _code(sub(SERVERS_CELL)),
        _code(sub(_start_cell())),
        _md("## 6 — The client, the command, the held-out run, and saving"),
        _code(sub(_client_cell())),
        _code(sub(WARMUP_CELL)),
        _md("## 7 — Part 1: `kvsim calibrate` (the tested run)"),
        _code(PART_1_CELL),
        _md("## 8 — Part 2: held-out traffic on the same engine"),
        _code(PART_2_CELL),
        _md("## 9 — Part 3: `kvsim calibrate` again, another seed (reported, not tested)"),
        _code(PART_3_CELL),
        _md("## 10 — Clean-up (run after downloading the zip)"),
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
