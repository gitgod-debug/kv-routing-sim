#!/usr/bin/env python3
"""Build a probe notebook with only the C3 notebook's environment, install and pre-flight cells.

    python scripts/c3_probe.py OUT.ipynb
    python scripts/kaggle_run.py OUT.ipynb --slug kvsim-c3-preflight-probe --out SOMEWHERE

The probe starts no engine and measures nothing. In about ten minutes on the GPU box's current
image, it checks that the pre-flight leaves software that can start vLLM, before a full run spends
an hour. The cells are the builder's own (`_c3_notebook.PREFLIGHT_CELL`, Phase 6's environment
cell, Phase 3's install cell), so the probe runs what the registered notebook runs. The vLLM
version comes from configs/c3_calibrate.yaml.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _c3_notebook import PREFLIGHT_CELL
from _phase3_notebook import INSTALL_CELL, _code, _md
from _phase6_notebook import ENV_CELL

CONFIG_CELL = '''
N_REPLICAS = 1
OUT_DIR = "c3_results"
'''

SNAPSHOT_CELL = r'''
# What the box has installed right now — the torch family, and what vLLM leans on.
_NAMES = "torch|torchaudio|torchvision|torchcodec|transformers|tokenizers|vllm|numpy|numba|triton"
print(sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E '^({_NAMES}) '"))
'''

REPORT_CELL = r'''
print(json.dumps({k: v for k, v in PREFLIGHT.items() if k != "pip_freeze"}, indent=1)[:6000])
print(sorted(p.name for p in pathlib.Path(OUT_DIR).glob("*")),
      pathlib.Path(f"{OUT_DIR}.zip").exists())
'''


def build(vllm: str) -> str:
    cells = [
        _md("# C3 pre-flight probe — no engine is started, nothing is measured\n\n"
            "The environment, install and pre-flight cells of `c3_calibrate_t4.ipynb`, alone."),
        _code(CONFIG_CELL),
        _code(ENV_CELL),
        _md("## The box as Kaggle ships it"),
        _code(SNAPSHOT_CELL),
        _md("## Install vLLM, pinned — as the registered notebook does"),
        _code(INSTALL_CELL.replace("@@VLLM@@", vllm)),
        _code(SNAPSHOT_CELL),
        _md("## The pre-flight cell"),
        _code(PREFLIGHT_CELL),
        _md("## What it left"),
        _code(SNAPSHOT_CELL),
        _code(REPORT_CELL),
    ]
    return json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load((REPO / "configs" / "c3_calibrate.yaml").read_text())
    out = pathlib.Path(sys.argv[1])
    out.write_text(build(str(cfg["engine"]["vllm"])), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
