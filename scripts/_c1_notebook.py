"""Builds notebooks/c1_calibrate_t4.ipynb. Called by scripts/c1_prepare.py — not run alone.

C1 is `kvsim calibrate`'s first run on a real GPU. The notebook starts one vLLM server on a T4 and
runs the command against it — the command from the package's own wheel, embedded below, so the
kernel needs no checkout of this repo. The server helpers are Phase 3's, reused as they are, except
that the block override is removed: here the engine allocates its natural capacity, and C1b checks
that the command reads it. `@@NAME@@` markers are filled in `build_notebook`.
"""

from __future__ import annotations

import json
from typing import Any

from _phase3_notebook import AUDIT_CELL, ENV_CELL, INSTALL_CELL, SERVER_CELL, _code, _md

INTRO = """
# C1 — `kvsim calibrate` on a real GPU

**What this is.** The first run of `kvsim calibrate` against a live engine. It was built after H6
showed that a replica count measured on one GPU does not transfer to another, so users have to
measure their own — and until now it had only been tested against a mock engine. One vLLM
**@@VLLM@@** server, **@@MODEL@@ on one Tesla T4**, with the engine's natural KV capacity.

The rules were committed before this notebook ran (`configs/c1_calibrate.yaml`, and a dated entry in
`docs/09-hypotheses.md`). **C1 holds only if both hold:**

| | Rule | Prediction |
|---|---|---|
| **C1a** | `kvsim calibrate` exits 0: its cache check is within 5% **and** all @@CHECK_N@@ requests' cached tokens equal the model's | exact: hit rate **@@PRED_HIT@@** (@@PRED_HITS@@ of @@PRED_QUERIES@@ tokens) |
| **C1b** | the KV capacity it reads equals the engine log's `GPU KV cache size` ÷ @@BLOCK_SIZE@@ | equal |

Reported, not tested: the fitted timings against Phase 3's, the capacity against 1,537, and the
wall-clock time.

The command comes from the package's wheel, **embedded in this notebook** (sha256 `@@WHEEL_SHA_SHORT@@…`,
built from commit `@@GIT_SHA@@`), so this runs exactly the committed code.
"""

HOW_TO_RUN = """
### How to run (≈ 20 minutes)
1. Kaggle → *Accelerator: GPU T4 ×2* (one T4 is used) → *Internet: on*.
2. *Run All*. The first cells install vLLM and download ~8 GB of weights.
3. Download **`c1_results.zip`** from the output panel and bring it back to the repo.

If a cell stops with ⛔, the zip is still written by the verdict cell when it can be; bring it back
anyway — a stop is a result too.
"""

CONFIG_CELL = r'''
# ── Generated from configs/c1_calibrate.yaml by scripts/c1_prepare.py. Do not edit here.
MODEL = "@@MODEL@@"
VLLM_PINNED = "@@VLLM@@"
BLOCK_SIZE = @@BLOCK_SIZE@@
MAX_MODEL_LEN = @@MAX_MODEL_LEN@@
MAX_NUM_SEQS = @@MAX_NUM_SEQS@@
MAX_NUM_BATCHED_TOKENS = @@MAX_NUM_BATCHED_TOKENS@@
GPU_MEM_UTIL = @@GPU_MEM_UTIL@@
SEED = @@SEED@@
CHECK_SESSIONS = @@CHECK_SESSIONS@@
CHECK_TURNS = @@CHECK_TURNS@@
TOKEN_RANGE = @@TOKEN_RANGE@@
MAX_ERROR = @@MAX_ERROR@@
PRED = json.loads(r"""@@PRED@@""")          # the registered prediction for the cache check
PHASE3 = json.loads(r"""@@PHASE3@@""")      # Phase 3's fitted T4 profile, for the report

# ── Box settings. Safe to change: nothing predicted depends on them.
GPU_INDEX = 0
PORT = 8000
STARTUP_TIMEOUT_S = 1800      # the first start downloads ~8 GB of weights and compiles
OUT_DIR = "c1_results"
'''

WHEEL_CELL = r'''
# @@KVSIM-WHEEL@@ — kv-routing-sim, from its own wheel. Checked against the registered hash first.
import base64
import hashlib

_WHEEL = base64.b64decode("""@@WHEEL_B64@@""")
if hashlib.sha256(_WHEEL).hexdigest() != "@@WHEEL_SHA@@":
    raise Stop("⛔ the embedded wheel does not match its registered hash")
_path = pathlib.Path(OUT_DIR) / "@@WHEEL_NAME@@"
_path.write_bytes(_WHEEL)
print(sh(f"{sys.executable} -m pip install -q {_path}", timeout=600) or "installed")
print(sh(f"{sys.executable} -m kv_routing_sim.cli calibrate --help | head -1"))
'''

START_CELL = r'''
start_server("c1")
if not wait_ready():
    raise Stop("⛔ the engine did not start — the log tail is above")
_m = re.search(r"GPU KV cache size: ([\d,]+) tokens", log_text())
LOG_BLOCKS = int(_m.group(1).replace(",", "")) // BLOCK_SIZE if _m else None
print(f"engine log: {LOG_BLOCKS:,} blocks of {BLOCK_SIZE} tokens" if LOG_BLOCKS
      else "⚠️  no 'GPU KV cache size' line in the engine log: C1b cannot be decided")
'''

RUN_CELL = r'''
PROFILE_PATH = pathlib.Path(OUT_DIR) / "calibrated_profile.json"
CMD = [sys.executable, "-m", "kv_routing_sim.cli", "calibrate", "--engine", BASE,
       "--model", MODEL, "--out", str(PROFILE_PATH), "--seed", str(SEED),
       "--check-sessions", str(CHECK_SESSIONS), "--check-turns", str(CHECK_TURNS),
       "--max-error", str(MAX_ERROR), "--token-range", f"{TOKEN_RANGE[0]},{TOKEN_RANGE[1]}"]
print("$", " ".join(CMD))
_t0 = time.time()
_proc = subprocess.run(CMD, capture_output=True, text=True, timeout=3600, check=False)
WALL_S = round(time.time() - _t0, 1)
RC = _proc.returncode
pathlib.Path(OUT_DIR, "calibrate_stdout.txt").write_text(_proc.stdout)
pathlib.Path(OUT_DIR, "calibrate_stderr.txt").write_text(_proc.stderr)
print(_proc.stdout)
print(_proc.stderr)
print(f"exit code {RC} · {WALL_S} s")
'''

VERDICT_CELL = r'''
import zipfile

PROFILE = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else None
_chk = (PROFILE or {}).get("_source", {}).get("check", {})
_eq, _cmp, _n = (_chk.get("cached_tokens_equal"), _chk.get("cached_tokens_compared"),
                 _chk.get("requests"))
C1A = bool(RC == 0 and PROFILE and PROFILE["calibrated"] and _eq == _cmp == _n)
C1B = bool(PROFILE and LOG_BLOCKS and PROFILE["kv_cache_blocks"] == LOG_BLOCKS)
_pred = _chk.get("predicted", {})
C1 = {
    "holds": C1A and C1B,
    "c1a": C1A,
    "c1b": C1B,
    # The embedded code must predict exactly the registered numbers; if not, the wheel is not the
    # code that made the prediction, and nothing here can be compared.
    "prediction_matches_registration": (_pred.get("hits"), _pred.get("queries"))
    == (PRED["hits"], PRED["queries"]),
    "returncode": RC,
    "measured": _chk.get("measured"),
    "predicted": _pred,
    "registered_prediction": PRED,
    "cached_tokens_equal": _eq,
    "cached_tokens_compared": _cmp,
    "log_blocks": LOG_BLOCKS,
    "read_blocks": PROFILE and PROFILE["kv_cache_blocks"],
    "reported": {
        "fitted": {k: PROFILE and PROFILE[k] for k in
                   ("prefill_base_s", "prefill_per_token_s", "decode_base_s", "decode_per_seq_s")},
        "phase3": {k: PHASE3[k] for k in
                   ("prefill_base_s", "prefill_per_token_s", "decode_base_s", "decode_per_seq_s",
                    "kv_cache_blocks")},
        "wall_clock_s": WALL_S,
    },
    "env": ENV,
    "audit": AUDIT,
}
pathlib.Path(OUT_DIR, "c1.json").write_text(json.dumps(C1, indent=1, default=str))
with zipfile.ZipFile("c1_results.zip", "w", zipfile.ZIP_DEFLATED) as _z:
    for _f in sorted(pathlib.Path(OUT_DIR).glob("*")):
        if not _f.name.endswith(".whl"):
            _z.write(_f)
print(f"C1a  calibrated, every request equal ... {'✅' if C1A else '⛔'}  "
      f"({_eq}/{_cmp} equal, rc {RC})")
print(f"C1b  capacity read = engine log ........ {'✅' if C1B else '⛔'}  "
      f"({C1['read_blocks']} read, {LOG_BLOCKS} in the log)")
print(f"prediction matches the registration ... {C1['prediction_matches_registration']}")
print(f"\n==> C1 {'HOLDS' if C1['holds'] else 'DOES NOT HOLD'} — bring back c1_results.zip")
'''

CLEANUP_CELL = r'''
stop_server()
print("server stopped")
'''


def _server_cell() -> str:
    """Phase 3's server helpers, without the block override: here capacity is the engine's own."""
    line = '    "--num-gpu-blocks-override", str(NUM_GPU_BLOCKS),\n'
    if line not in SERVER_CELL:
        raise SystemExit("error: Phase 3's server cell changed; update _c1_notebook.py")
    return SERVER_CELL.replace(line, "")


def build_notebook(cfg: dict[str, Any], *, wheel_b64: str, wheel_sha: str, wheel_name: str,
                   git_sha: str, prediction: dict[str, Any],
                   phase3: dict[str, Any]) -> dict[str, Any]:
    e, c = cfg["engine"], cfg["calibrate"]
    fill = {
        "MODEL": e["model"], "VLLM": e["vllm"], "BLOCK_SIZE": str(e["block_size"]),
        "MAX_MODEL_LEN": str(e["max_model_len"]), "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "MAX_NUM_BATCHED_TOKENS": str(e["max_num_batched_tokens"]),
        "GPU_MEM_UTIL": str(e["gpu_memory_utilization"]), "SEED": str(c["seed"]),
        "CHECK_SESSIONS": str(c["check_sessions"]), "CHECK_TURNS": str(c["check_turns"]),
        "TOKEN_RANGE": json.dumps(c["token_range"]), "MAX_ERROR": str(c["max_error"]),
        "PRED": json.dumps(prediction), "PHASE3": json.dumps(phase3),
        "WHEEL_B64": wheel_b64, "WHEEL_SHA": wheel_sha, "WHEEL_SHA_SHORT": wheel_sha[:12],
        "WHEEL_NAME": wheel_name, "GIT_SHA": git_sha,
        "CHECK_N": str(prediction["requests"]), "PRED_HIT": f"{prediction['hit_rate']:.4f}",
        "PRED_HITS": f"{prediction['hits']:,}", "PRED_QUERIES": f"{prediction['queries']:,}",
    }

    def f(text: str) -> str:
        for k, v in fill.items():
            text = text.replace(f"@@{k}@@", v)
        if "@@" in text.replace("@@KVSIM-WHEEL@@", ""):
            raise SystemExit(f"error: an unfilled marker is left: {text[text.index('@@'):][:40]}")
        return text

    cells = [
        _md(f(INTRO)),
        _md(HOW_TO_RUN),
        _code(f("import json\n" + CONFIG_CELL)),
        _code(ENV_CELL),
        _code(f(INSTALL_CELL)),
        _code(f(WHEEL_CELL)),
        _code(AUDIT_CELL),
        _code(_server_cell()),
        _code(START_CELL),
        _code(RUN_CELL),
        _code(VERDICT_CELL),
        _code(CLEANUP_CELL),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
