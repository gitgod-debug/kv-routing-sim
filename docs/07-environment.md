# 07 — Environment: two machines, and the calibration runbook

**Recorded 2026-09-20.** Re-verify if either machine changes.

This project runs on **two** machines with a hard split of duties, and confusing them is how a number gets
published against the wrong hardware.

---

## Machine 1 — the dev box (local). Everything except the engine.

Measured, not assumed:

| Component | Value | What it means for this project |
|---|---|---|
| **GPU** | NVIDIA GTX 1650, **4 GB VRAM**, **compute capability 7.5**, driver 595.71.05 | ⭐ **7.5 is the T4's exact compute capability** — so anything that is a *Turing* question can be answered here for free. 4 GB is too small for a useful calibration model, so it is not the calibration box. |
| **CPU** | 12 cores | The simulator is CPU work and this is plenty. ⚠️ Any claim about *user-facing* sim runtime must be pinned (`taskset`) to be honest — a 12-core sweep time is not a laptop sweep time. |
| **RAM** | 14 GB total, ~7 GB available | **This is the constraint that shapes `traces/`.** Traces stream; the cache model holds hashes, not tokens (ADR-002). The stated memory ceiling is measured here. |
| **Disk** | 726 GB free on `/data` | Comfortable. |
| **Python** | 3.13.12 (miniconda) | Core deps (numpy/pyyaml/pydantic/matplotlib) are all fine on 3.13. |
| **Platform** | `Linux-6.17.0-41-generic-x86_64-with-glibc2.42` | |
| **Git** | 2.51.0 | Repo initialized 2026-09-20 |

### Consequences

1. **All simulator work happens here** — traces, cache model, policies, sweeps, Pareto curves, the
   recommender. No GPU needed, by design (ADR-009).
2. **The 14 GB RAM is a feature.** It forces the streaming discipline the "what breaks at 10× and 100×"
   README section depends on. Develop the memory ceiling here and it will be a real number.
3. ⚠️ **`vllm` is NOT installed on this box and should not be.** 4 GB VRAM cannot host a calibration-grade
   model, and installing it locally invites a stray import into core code, which ADR-009 exists to prevent.

### Python environment

Project-local `.venv`, never `base`/`ml_env` — sharing an interpreter with other work is how a pin
silently drifts.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # then run everything as .venv/bin/python ...
```

⚠️ **The project `.venv` is not yet created as of 2026-09-20.** Record the *actually installed*
versions in a table here the moment it is — the pins in `pyproject.toml` are intentions, and only an
install produces facts.

### ✅ What the core install actually pulls — measured 2026-09-20, in a throwaway venv

ADR-009 claims `pip install kv-routing-sim` never drags in a GPU dependency. That was **verified, not
assumed**: `pip install -e .` into a clean 3.13.12 venv resolved to **19 packages, none of them
torch, CUDA or vLLM**:

```
PyYAML 6.0.3 · annotated-types 0.8.0 · contourpy 1.4.0 · cycler 0.12.1 · fonttools 4.65.0
kiwisolver 1.5.1 · matplotlib 3.11.2 · numpy 2.5.3 · packaging 26.3 · pillow 12.3.0
pydantic 2.13.5 · pydantic_core 2.46.5 · pyparsing 3.3.2 · python-dateutil 2.9.0.post0
six 1.17.0 · typing-inspection 0.4.4 · typing_extensions 4.16.0        (+ pip, kv-routing-sim)
```

The `kvsim` console script resolves and exits 2 with its Phase-0 message, so the declared entry
point is real rather than aspirational. **The free-CPU HF Space in Phase 7 is therefore reachable**;
that deployment target only exists because of this property.

---

## Machine 2 — the calibration box. The engine, and nothing else.

| | |
|---|---|
| **Hardware** | **A100 or H100** — one GPU at a time |
| **Reachability** | **Not reachable from the dev session.** Scripts are written here, run there, artifacts copied back. |
| **Duties** | Phase 0 spike · Phase 3 single-replica calibration · Phase 6 two-replica validation |
| **Two replicas** | Two **MIG** instances on the one card; fallback = two servers at `--gpu-memory-utilization 0.45` each (ADR-005) |

**Why this replaced the spec's Kaggle dual-T4 plan, and why it is an upgrade rather than a substitution:
see ADR-005.** Briefly: A100 is SM80 (bf16, native FlashAttention backend), H100 is SM90 (adds FP8); T4 is
SM75 with none of those and vLLM falls back to a different attention backend there. Kaggle remains a
documented fallback, not the plan.

⚠️ **Record, on the first run, into this file:** the exact GPU, driver, CUDA version, the **vLLM version
that actually installed** (the pin is 0.29.0 — if the box gives something else, ADR-008's metric-name audit
must be re-run against *its* `loggers.py`), and whether MIG is available and enabled.

### ⛔ The rule that outranks convenience

> **No simulator code is written before the Phase-0 spike passes on this box.**

The spec is explicit about why: every assumption in the calibration plan — that prefix caching behaves as
documented, that the counters exist and move, that two servers can coexist — has *never been executed
once*. The sibling project caught two hard blockers cheaply because a Phase-0 spike ran before any design
depended on them. This project had no Phase-0 equivalent until now.

---

## The runbook — Phase 0, on the calibration box

Copy the repo (or just `scripts/`) to the box, then:

```bash
# 0. Record what we are actually on. Goes into the results header automatically,
#    but read it yourself too — the GPU and driver are README content.
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
nvidia-smi -L
nvidia-smi --query-gpu=mig.mode.current --format=csv     # is MIG even on?

# 1. Install the engine. ONLY on this box.
python3 -m venv .venv && .venv/bin/pip install -e ".[calib]"
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"   # ⚠️ record this number

# 2. ⛔ Re-run the metric-name audit against the version that ACTUALLY installed (ADR-008).
#    If it is not 0.29.0, do not skip this.
.venv/bin/python -c "import vllm,pathlib;p=pathlib.Path(vllm.__file__).parent/'v1/metrics/loggers.py';print(p);print('hit_rate as a metric name:', 'prefix_cache_hit_rate' in p.read_text())"
grep -rn "prefix_cache_hits\|prefix_cache_queries\|kv_block_" "$(.venv/bin/python -c 'import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent/"v1/metrics")')" | head -40

# 3. The spike itself: one server, shared-prefix workload, scrape /metrics before and after,
#    assert the two counters exist and MOVE, and compare prefix-caching off vs on.
.venv/bin/python scripts/spike_vllm_metrics.py --model <small-instruct-model> --port 8000

# 4. The coexistence check: two servers, MIG if available, memory-split otherwise.
.venv/bin/python scripts/spike_two_servers.py --model <small-instruct-model>
```

Then **copy `artifacts/results/*.json` and `*.metrics.txt` back into the repo and commit them.** The raw
scrape is committed, not just the parsed summary — it is the calibration evidence.

### MIG setup, for Phase 6 (needs root; do it once)

```bash
sudo nvidia-smi -i 0 -mig 1                 # enable MIG on GPU 0 (no processes may be running)
sudo nvidia-smi mig -i 0 -cgi 3g.40gb,3g.40gb -C   # two instances, each with compute instances
nvidia-smi -L                               # read the MIG-… UUIDs

# then one server per instance:
CUDA_VISIBLE_DEVICES=MIG-<uuid-a> vllm serve <model> --port 8000 --enable-prefix-caching &
CUDA_VISIBLE_DEVICES=MIG-<uuid-b> vllm serve <model> --port 8001 --enable-prefix-caching &
```

⚠️ **Profile names are card-specific** (`3g.40gb` is an A100-80GB profile). Run
`nvidia-smi mig -lgip` and use what that lists. If MIG cannot be enabled — no root, or a rented box with
it disabled — take the memory-split fallback and **record which path was used**, because it changes which
numbers may be reported (ADR-005: hit rate yes, TTFT percentiles no).

---

## Flags the calibration must name in the README

Not optional — the README's Calibration Report section is judged on this being specific:

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-cache-metrics-sample <rate> \     # ⚠️ gates the three vllm:kv_block_* histograms (H4)
  --block-size 16 \
  --max-num-seqs <N> \
  --gpu-memory-utilization <U>
```

⛔ **"vLLM or SGLang" is not an acceptable README sentence.** Say which engine, which version, which GPU,
which flags. Vagueness there reads as *never actually ran it* — which is precisely the inference a reader
drew about `Mike-Ew/kv-cache-routing` (see `04-prior-art.md`).

---

## Deployment targets (Phase 7)

| Need | Where | Constraint |
|---|---|---|
| Public demo | **HF Space, free CPU** | 2 vCPU / 16 GB. The simulator is CPU work, so this is a natural fit — and it only works because of ADR-009. |
| Docs site | Render free | 512 MB, sleeps when idle. No model weights, ever. |
| Package | PyPI | `pip install kv-routing-sim` + semver + changelog |
