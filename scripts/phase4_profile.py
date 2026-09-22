#!/usr/bin/env python3
"""The T4 profile Phase 4 sweeps on: Phase 3's FITTED timings + the engine's own NATURAL capacity.

    python scripts/phase4_profile.py      → configs/profiles/t4-qwen3-4b-vllm0.30.0.json

Phase 3 forced the cache down to 512 blocks so that its predictions could be exact. A deployment of
the same engine would not. Before applying the override, the engine computed the capacity it would
have used by itself, and logged it:

    Overriding num_gpu_blocks=1537 with num_gpu_blocks_override=512

That is what one T4 really offers Qwen3-4B with these flags — read from the engine, not estimated.
(`cache/capacity.py`'s arithmetic says ~2,394 for the same card and utilization: 56% too many,
because vLLM also reserves memory for activations and CUDA graphs. Read it from the engine.)

Timing does not depend on how many blocks exist, so the fitted timings carry over unchanged. The
capacity DOES depend on the flags, so they are recorded with it.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import asdict, replace

from kv_routing_sim.cache.capacity import SHAPES, kv_blocks_for
from kv_routing_sim.sim.cost_model import CostProfile

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results" / "phase3_kaggle_t4"
OUT = REPO / "configs" / "profiles" / "t4-qwen3-4b-vllm0.30.0.json"


def main() -> int:
    fitted = CostProfile(**json.loads((RESULTS / "fitted_profile.json").read_text()))
    log = (RESULTS / "server_engine.log").read_text(errors="replace")
    m = re.search(r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)", log)
    if not m:
        raise SystemExit("⛔ the engine log has no 'Overriding num_gpu_blocks=' line")
    natural = int(m.group(1))
    kv = re.search(r"Available KV cache memory: ([\d.]+) GiB", log)
    argv = json.loads((RESULTS / "phase3.json").read_text())["engine_argv"]

    def flag(name: str) -> str:
        return argv[argv.index(name) + 1]

    util = float(flag("--gpu-memory-utilization"))
    estimate = kv_blocks_for(SHAPES["qwen3-4b"], 15.0, gpu_memory_utilization=util)
    prof = replace(fitted, name="t4-qwen3-4b-vllm0.30.0", kv_cache_blocks=natural)

    doc = {
        "_what": (
            "Phase 3's fitted T4 timings with the engine's own natural KV capacity. Timings are "
            "T4 figures; capacity depends on the flags below."
        ),
        "_source": {
            "timings": "artifacts/results/phase3_kaggle_t4/fitted_profile.json",
            "capacity": (
                "artifacts/results/phase3_kaggle_t4/server_engine.log: "
                f"'{m.group(0)}'"
                + (f"; available KV cache memory {kv.group(1)} GiB" if kv else "")
            ),
            "capacity_depends_on": {
                "--gpu-memory-utilization": util,
                "--max-model-len": int(flag("--max-model-len")),
                "--max-num-batched-tokens": int(flag("--max-num-batched-tokens")),
                "--max-num-seqs": int(flag("--max-num-seqs")),
                "gpu": "Tesla T4, 15,360 MiB",
            },
            "capacity_py_estimate_for_comparison": estimate,
            "known_limits": [
                "decode fit R^2 0.58: one sequence decodes slower (63.6 ms) than 2-8 do",
                (
                    "the step model charges a decode step on top of each prefill: ~+20 ms TTFT "
                    "on an idle replica"
                ),
            ],
            "generated_by": "scripts/phase4_profile.py",
        },
        **asdict(prof),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"natural capacity {natural} blocks ({natural * prof.block_size:,} tokens); "
          f"capacity.py would have said {estimate} ({(estimate / natural - 1) * 100:+.0f}%)")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
