---
title: kv-routing-sim
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: LLM request routing, simulated; cache model checked on vLLM
---

# kv-routing-sim

A copy of the project's static pages: the demo grid (`index.html`) and the in-browser runner
(`run.html`), which runs the simulator inside your browser on generated traffic or on a log you
choose. A log is read locally and not uploaded.

The numbers are for one GPU type (Tesla T4), one model (Qwen3-4B) and vLLM 0.30.0. The code, every
registered test with its outcome, and the evidence behind each number are in the repository:
https://github.com/gitgod-debug/kv-routing-sim

These files are built by `python scripts/build_site.py --space --out _space` in that repository.
