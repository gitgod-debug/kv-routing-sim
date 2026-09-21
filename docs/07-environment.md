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
point is real rather than aspirational. **The static demo page in Phase 7 is therefore reachable** —
and so would a CPU Space be, if one were free (it is not; see ADR-011);
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

> ### ⭐ The box is a JUPYTER environment, so run the notebook
> **[`../notebooks/phase0_spike_vllm.ipynb`](../notebooks/phase0_spike_vllm.ipynb)** — self-contained,
> no repo checkout, no dataset, no HF token. Model: **`Qwen/Qwen3-0.6B`** (ungated). See
> [`../notebooks/README.md`](../notebooks/README.md).
>
> The shell commands below are the same checks for a box where you do have a terminal. Either path is
> fine; the notebook is the one that fits the hardware we actually have.

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

## ⭐ What runs where — and whose GPU

*(Added 2026-09-20. It was not written down anywhere, and "does the deployed thing use the user's GPU?"
is the first question a reader asks. The answer is no, and the reason is the whole design.)*

**No GPU is involved in anything that is deployed. Not ours, not the user's.** The GPU appears exactly
once, on the calibration box, offline, before anything is published.

```
  MACHINE 2 — the A100/H100          run ONCE, offline (Phase 3 + Phase 6).
  ┌──────────────────────────┐       Produces ~20 numbers and some committed JSON.
  │ live vLLM + /metrics     │       Then the GPU is finished, forever.
  └──────────────────────────┘
               │  the fitted cost function + the calibration profile are
               │  committed to the repo as static artifacts
               ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Everything below is pure CPU Python. No weights, no CUDA, no engine.    │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  PyPI  `pip install kv-routing-sim`                                      │
  │    → runs on the USER'S OWN machine or CI. Their trace never leaves it.  │
  │    → this is the real usage path.                                        │
  │                                                                          │
  │  Static demo page — GitHub Pages + a free Static Space                   │
  │    → precomputed grid + sliders. Instant, never sleeps, no compute.      │
  │      A SHOP WINDOW, not the product. (ADR-011, corrected 2026-09-21:     │
  │      free Gradio Spaces now need PRO.)                                   │
  │                                                                          │
  │  GitHub Pages (static)                                                   │
  │    → docs only. No compute. (Render was cut 2026-09-20 — see doc 10.)    │
  └──────────────────────────────────────────────────────────────────────────┘
```

**Why this works at all:** the simulator replays a trace through a dict-based trie. Replaying 100k
requests through six policies is *milliseconds* of CPU work — there is nothing for a GPU to do. That is
the point of a simulator: **pay the GPU cost once, then answer "what if" thousands of times for free.**

**A user never grants GPU access and never needs a GPU.** They hand over a **log file**, not a fleet. And
because prompts are hashed at the adapter boundary (ADR-002), a real trace can be processed without its
prompt text ever entering the simulator.

> ### ⛔ The one thing not to get wrong about the demo: traces are user content
> A Space that invites strangers to upload a production serving log is asking them to send their users'
> prompts to a third-party host. **The Space demo runs on bundled sample traces and the synthetic
> generator only.** Real traces belong to the local CLI, on the user's own machine, where they already
> are. If an upload box ever appears in the Space, hashing must happen client-side or it must not ship.

### What a user gets without a GPU — and what they don't

See **ADR-010** for the full decision. In short: the **policy ranking** and the **crossover boundary**
transfer (pending **H6**); **absolute ms and ₹** do not, and need `kvsim calibrate` on the user's own
hardware. ⛔ Never blend a calibrated error bar with an uncalibrated recommendation.

### ✅ ANSWERED 2026-09-20 — see the deployment notes

**Render is CUT; docs go to GitHub Pages.** The reasoning below is kept as the record of why.

Recorded as open rather than answered, because it is a positioning decision and not a technical one:

- **PyPI** — clearly justified. It is how anyone actually uses the tool.
- **HF Space** — justified as **distribution, not as a product surface**: it exists so a reader arriving
  from the README or a Show HN post believes the repo is real without installing anything. That is a
  legitimate job. It should be *called* that, in the README, rather than implied to be a service.
- **Render docs site** — ⚠️ **weakest of the three, and possibly cut.** A 512 MB container that sleeps,
  serving a static docs page, is strictly worse than GitHub Pages at the same price. It is on the list
  because the project plan's deployment list has it, not because this project needs it.
  **Decide at Phase 7:** either it hosts something genuinely dynamic (a thin API that serves the
  recommender over HTTP, which would be a real production artifact), or it is cut and the docs go to Pages.

---

## Deployment targets (Phase 7)

| Need | Where | Constraint |
|---|---|---|
| Public demo | **Static page** — GitHub Pages **+** a free Static Space | ⚠️ **Corrected 2026-09-21 (ADR-011): a free Gradio Space no longer exists** — "Gradio and Docker Spaces … require a paid plan to create: PRO for personal accounts." Only Static is free. A precomputed grid + sliders needs no compute, **never sleeps**, and is reproducible from a committed script. **Bundled + generated traces only.** |
| Docs site | **GitHub Pages** (static) | ✅ Render **CUT** — free web services spin down after **15 min** idle, take **~1 min** to wake, and the 750 hours are a monthly cap. Returns only if we build the thin HTTP recommender. |
| Package | PyPI | `pip install kv-routing-sim` + semver + changelog. **The real usage path.** |
