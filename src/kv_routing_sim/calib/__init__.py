"""Live-engine calibration.

  * `calibrate.py` — `kvsim calibrate`: probe a running vLLM, fit the timing profile (prefill per
                     uncached token, decode step per batch size), and check the cache model against
                     the engine's own prefix-cache counters. Built 2026-09-28, after H6 showed that
                     a replica count measured on our T4 does not transfer to other hardware.

The hit rate is always DERIVED: `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`, two
counters. ⛔ `vllm:prefix_cache_hit_rate` does not exist (ADR-008).

The pre-registered Phase 3 calibration itself — predicted against measured on a live T4 — lives in
`scripts/` and `notebooks/`, because it runs on a GPU box with vLLM installed; its report is
`artifacts/results/phase3_kaggle_t4/calibration_report.md`. Nothing in this package imports vLLM: it
talks to an engine over HTTP with the standard library, so `pip install kv-routing-sim` never needs
a GPU wheel (ADR-009).

⛔ The predicted-vs-measured result is published whatever it says. A large, explained error is a
result; a quietly re-tuned model that matches is fraud (docs/06-methodology.md §4).
"""
