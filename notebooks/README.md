# notebooks/ — for the GPU box

The GPU box is a **Jupyter environment**, so the Phase-0 gate runs as a notebook rather than as the
`scripts/spike_*.py` CLIs. Same checks, same output format; it just survives in a kernel instead of a
shell.

---

## Status

✅ **Phase 0 is complete** — v1 (2026-09-21) and v2 (2026-09-22) passed every check, including an
**exact** 2-replica sim-to-real prediction (1,088 tokens, 0 error). Results:
[`phase0_kaggle_t4/`](../artifacts/results/phase0_kaggle_t4/) ·
[`phase0_v2_kaggle_t4/`](../artifacts/results/phase0_v2_kaggle_t4/).

✅ **Phase 3 ran on 2026-09-22 and passed** — the block-pool model matched every pre-registered count on
both sequential workloads, request by request, and predicted the concurrent one within 0.3% on hit
rate. Results: [`phase3_kaggle_t4/`](../artifacts/results/phase3_kaggle_t4/) ·
[`calibration_report.md`](../artifacts/results/phase3_kaggle_t4/calibration_report.md). Its section
below is kept as the record of how it was run.

---

## ▶ Run next — two notebooks, one Kaggle session

Both are registered: their rules and predictions were committed before any GPU run. Both are
self-contained — they need no copy of this repo on the GPU box.

**On Kaggle, for each notebook:** *File → Import notebook* → pick the `.ipynb` → *Settings:
Accelerator **GPU T4 ×2**, Internet **on*** → *Run All* → when it finishes, download the results zip
from the output panel and put it in this repo (under `artifacts/results/`), then ask for the write-up.

| Notebook | What it tests | Time | Bring back |
|---|---|---|---|
| [`phase6_sim2real.ipynb`](phase6_sim2real.ipynb) | **Phase 6:** the finding's mechanism on **two live replicas** — three routers, one shared prompt and 64 tenant prompts; hit rates against predictions | ~40 min | `phase6_results.zip` |
| [`c1_calibrate_t4.ipynb`](c1_calibrate_t4.ipynb) | **C1:** the first run of `kvsim calibrate` on a real GPU — its cache check must be exact (48 of 48 requests) and the capacity it reads must equal the engine's log | ~20 min | `c1_results.zip` |

If a cell stops with ⛔, still download the zip if it was written — a stop is a result too, and the
reason is in the output.

---

## ▶ Phase 3 — `phase3_calibration.ipynb`

**What it does.** Starts one vLLM 0.30.0 server (Qwen3-4B, one T4) with a deliberately small,
**exactly controlled** cache (`--num-gpu-blocks-override 512`), sends it three workloads three times
each, and scores every run against predictions both cache models made **before the run**:

| Workload | Sent | `vllm` model | `lru` model |
|---|---|---|---|
| **W2** control | 120 requests one at a time, 1-token outputs | hit rate **0.6090** | **0.6073** |
| **W1** discriminating | the same 120, 64-token outputs | **0.4883** | **0.6073** — 11.9 points apart |
| **W3** concurrent | the same trace at its own arrival times | preemptions **> 0** | **0** preemptions |

W1 and W2 are exact predictions, not estimates: for one-at-a-time requests vLLM is deterministic, so
every counter — queries, hits, computed tokens, the three `kv_block_*` histogram counts, and each
request's own `cached_tokens` — must match to the token, or the model is wrong about something
specific. The per-request comparison says which request.

**Where and how.** Kaggle, **Accelerator: GPU T4 x2** (one is used), **Internet: on**. Upload the
one `.ipynb`, **Run All**. Nothing else: no repo checkout, no dataset, no HF token (Qwen3-4B is
ungated). The trace (57,960 prompt tokens as token ids) and the predictions are **embedded in the
notebook** and fingerprinted — it refuses to run if they were edited.

**How long.** ≈ 40 min: `pip install` 5–8 · first engine start 5–10 (≈ 8 GB of weights) · W2 ~2 ·
W1 ~15 · decode sweep ~2 · W3 ~3.

**Bring back** `phase3_results.zip` (right sidebar → *Output*, or `/kaggle/working`). Then:

```bash
mkdir -p artifacts/results/phase3_kaggle_t4
unzip phase3_results.zip -d artifacts/results/phase3_kaggle_t4
.venv_kv/bin/python scripts/phase3_analyze.py        # writes calibration_report.md + fitted_profile.json
```

Commit the whole folder, **raw scrapes included** — the JSON is a parse; the scrapes are the evidence.

**If a cell stops with ⛔** it says why; download the zip anyway (the server log is in it) and run
the clean-up cell. It checks, before any measured request: the vLLM version, that the installed
engine contains every source line the model cites, the fingerprints, the engine's KV capacity (from
its own log), that each response carries `cached_tokens` and exact lengths, and that nothing else
is already serving on the port.

**Changing the experiment** means editing `../configs/phase3_workloads.yaml` and re-running
`scripts/phase3_prepare.py` — which re-makes the predictions too — then
`scripts/phase3_dryrun.py`, which runs the notebook against a mock engine through four scenarios.
Never edit the `.ipynb` by hand; it is generated.

---

## Phase 0 — `phase0_spike_vllm.ipynb` *(done; kept as the record of what ran)*

### Which file to run

**`phase0_spike_vllm.ipynb` — now v2.** Same self-contained rules. It finishes the three things v1
could not, all of which were notebook bugs rather than findings about vLLM:

| v1 got wrong | v2 does instead |
|---|---|
| compared an undefined hit rate with caching off | compares prefill tokens actually computed (`local_compute`) |
| used `--kv-cache-metrics-sample` to enable histograms | passes **`--kv-cache-metrics`** (the real enable flag) and reads the answer from the engine's config dump, never from `--help` |
| died on the 40,960-token default context | `--max-model-len 4096` |
| put both servers on GPU 0 of a two-GPU box | one server per physical GPU |

⭐ **And v2 is the first multi-replica sim-to-real check.** Its round-robin arm has a prediction from our
simulator, committed before it runs: **1,088 prefill tokens computed, hit rate 0.966862**. The notebook
compares and reports the error itself.

**Run it on the same Kaggle dual-T4 box.** Every question v2 asks is hardware-independent, and two
physical GPUs are the best possible 2-replica setup.

**It is self-contained.** It does not import `kv_routing_sim`, needs no repo checkout, no dataset, and
no HF token. Only `vllm` plus the standard library. Copy the one `.ipynb` to the box and run it top to
bottom.

### Which model

| Stage | Model | Why |
|---|---|---|
| **Phase 0 spike** | **`Qwen/Qwen3-0.6B`** | Ungated (verified 2026-09-21 — no token, no licence click), ~1.5 GB, starts in a minute or two. 28 layers / 8 KV heads / head_dim 128 → **112 KiB of KV per token**. The spike only needs the counters to exist and move; a big model would just cost you time. |
| **Phase 3 calibration** | **`Qwen/Qwen3-4B`** | Ungated. ⛔ **Not Qwen3-8B** — its 15.26 GiB of fp16 weights exceed a T4's 13.5 GiB usable. 4B fits with 6 GiB left for cache, and its KV cache is **architecturally identical to 8B's** (36 layers, 8 KV heads, 144 KiB/token) — so for calibrating a *cache* model it is a stand-in, not a downgrade. |

⛔ **Not `meta-llama/*`.** Verified `gated=manual`: licence acceptance plus a token. A pointless
dependency for a spike, and it fails at 3am with an unhelpful 401.

Both shapes are in `src/kv_routing_sim/cache/capacity.py`, read from each model's own `config.json`
rather than guessed.

### What data do I need?

**None.** The notebook builds its own prompts as **token id lists**, which is deliberate: sending ids
instead of text means the prompt length is exact rather than an estimate, so the blocks-vs-tokens
determination in §7 is a clean measurement and no tokenizer is involved.

### How long

≈ 25–35 min. `pip install vllm` 3–8 min · model download 1–2 min · each server start 1–3 min (there are
four starts) · the workloads are seconds.

### What to bring back

The notebook writes everything to `phase0_results/`:

```
phase0_spike.json          the verdict + full provenance
metrics_caching_on.txt     ⭐ the RAW scrape — this is the evidence, commit it
metrics_caching_off.txt    the comparison arm
server_8000.log            for diagnosing a failed start
server_8001.log
```

Copy that folder into `../artifacts/results/` and commit it. **The raw `.txt` scrapes matter as much as
the JSON** — the JSON is a parse, the scrape is the thing parsed.

### What it decides

| Answer | What changes |
|---|---|
| Counters present **and moving** | ✅ Phase 0 passes; simulator work is unblocked |
| Counters present but **flat** | ⛔ Gate fails. The silent failure mode — do not proceed |
| `queries` counts **tokens**, not blocks | Divide vLLM by `block_size` before any comparison, and amend **ADR-008** with a date. A 16× error otherwise, in the number this project calls its credential |
| No `kv_block_*` histograms | **H4 is untestable as written.** Amend `docs/09-hypotheses.md` with a date — do not keep the narrative |
| Two servers won't coexist | Phase 6 takes its documented fallback: single-replica calibration plus a reproducible account of why |

⚠️ **Log every snag in the upstream log as it happens.** That log is the `guidellm` PR's seed, and
it is free because the friction happens anyway. It is also unrecoverable afterwards.

⚠️ **Run the last cell.** On rented hardware a forgotten `vllm serve` bills until the box dies.
