# notebooks/ — for the GPU box

The GPU box is a **Jupyter environment**, so the Phase-0 gate runs as a notebook rather than as the
`scripts/spike_*.py` CLIs. Same checks, same output format; it just survives in a kernel instead of a
shell.

---

## Which file to run

**`phase0_spike_vllm.ipynb`** — and nothing else, yet. It is the gate: no simulator number is
trustworthy until it passes (`../CHECKPOINTS.md` Phase 0).

**It is self-contained.** It does not import `kv_routing_sim`, needs no repo checkout, no dataset, and
no HF token. Only `vllm` plus the standard library. Copy the one `.ipynb` to the box and run it top to
bottom.

## Which model

| Stage | Model | Why |
|---|---|---|
| **Phase 0 spike** | **`Qwen/Qwen3-0.6B`** | Ungated (verified 2026-09-21 — no token, no licence click), ~1.5 GB, starts in a minute or two. 28 layers / 8 KV heads / head_dim 128 → **112 KiB of KV per token**. The spike only needs the counters to exist and move; a big model would just cost you time. |
| **Phase 3 calibration** | **`Qwen/Qwen3-8B`** | Also ungated. 36 layers → **144 KiB/token**, ~16 GB of bf16 weights, comfortable on an A100/H100. The cost coefficients have to come from a model shape people actually serve. |

⛔ **Not `meta-llama/*`.** Verified `gated=manual`: licence acceptance plus a token. A pointless
dependency for a spike, and it fails at 3am with an unhelpful 401.

Both shapes are in `src/kv_routing_sim/cache/capacity.py`, read from each model's own `config.json`
rather than guessed.

## What data do I need?

**None.** The notebook builds its own prompts as **token id lists**, which is deliberate: sending ids
instead of text means the prompt length is exact rather than an estimate, so the blocks-vs-tokens
determination in §7 is a clean measurement and no tokenizer is involved.

## How long

≈ 25–35 min. `pip install vllm` 3–8 min · model download 1–2 min · each server start 1–3 min (there are
four starts) · the workloads are seconds.

## What to bring back

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

## What it decides

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
