"""Live-engine calibration — the credential. 📐 designed, not built.

  * `metrics_scrape.py` — scrape vLLM `/metrics`, parse the exposition format, and DERIVE the hit
                          rate from `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`.
                          ⛔ `vllm:prefix_cache_hit_rate` does not exist (ADR-008).
  * `compare.py`        — predicted vs measured per workload shape → the residual, with error bars.
  * `report.py`         — renders the Real-Engine Calibration Report: engine, version, flags,
                          hardware, and every `/metrics` field consumed, by name.

⚠️ This is the ONLY GPU-touching module. It imports `vllm`/`httpx` LAZILY, inside functions, and
raises a message naming the `[calib]` extra (ADR-009) — `pip install kv-routing-sim` must never
need a GPU wheel, or the CPU-only pitch and the free-CPU HF Space both break.

⛔ The predicted-vs-measured table is published whatever it says. A large, explained error is a
result; a quietly re-tuned model that matches is fraud (docs/06-methodology.md §4).
"""
