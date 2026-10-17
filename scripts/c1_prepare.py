#!/usr/bin/env python3
"""Prepare the C1 run: build the wheel, predict the cache check, and write the notebook.

    python scripts/c1_prepare.py
        # configs/c1_calibrate.yaml -> artifacts/results/c1_prereg/prediction.json
        #                           -> notebooks/c1_calibrate_t4.ipynb

The notebook embeds the package's wheel so the GPU box runs exactly the committed code, which is
why this refuses to run on a dirty tree. The prediction is the simulator's hit rate for the
seed-determined check workload that `kvsim calibrate` sends; the workload is far smaller than any
T4 cache, so nothing is evicted and capacity does not enter it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import tempfile

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _c1_notebook import build_notebook
from _spike_common import results_header

from kv_routing_sim.calib.calibrate import check_workload, predict_check
from kv_routing_sim.sim.cost_model import load_profile

CONFIG = REPO / "configs" / "c1_calibrate.yaml"
OUT = REPO / "artifacts" / "results" / "c1_prereg" / "prediction.json"
NOTEBOOK = REPO / "notebooks" / "c1_calibrate_t4.ipynb"
# Everything that ends up in the wheel or the notebook.
CODE = ("src", "pyproject.toml", "configs/c1_calibrate.yaml", "scripts/_c1_notebook.py",
        "scripts/_phase3_notebook.py", "scripts/c1_prepare.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from uncommitted code (the notebook then says so); never for the "
                         "registered run")
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit(f"error: commit first — the wheel must be the committed code:\n{dirty}")

    cfg = yaml.safe_load(CONFIG.read_text())
    c = cfg["calibrate"]
    header = results_header("scripts/c1_prepare.py", seed=c["seed"], hardware_path="cpu-only")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps", "-q",
                        "-w", tmp], check=True)
        whl = next(pathlib.Path(tmp).glob("*.whl"))
        data, name = whl.read_bytes(), whl.name
    sha = hashlib.sha256(data).hexdigest()

    profile = load_profile(REPO / cfg["phase3_profile"])
    work = check_workload(random.Random(f"check-{c['seed']}"), c["check_sessions"],
                          c["check_turns"], tuple(c["token_range"]))
    sim = predict_check(work, profile)
    prediction = {
        "requests": len(work),
        "queries": sim.cache_queries,
        "hits": sim.cache_hits,
        "hit_rate": round(sim.hit_rate, 6),
        "cached_tokens": [sim.per_request[f"c{i}"]["cached_tokens"] for i in range(len(work))],
    }
    phase3 = {k: v for k, v in json.loads((REPO / cfg["phase3_profile"]).read_text()).items()
              if not k.startswith("_")}

    git = header["git_sha"] + ("-dirty" if dirty else "")
    nb = build_notebook(cfg, wheel_b64=base64.b64encode(data).decode(), wheel_sha=sha,
                        wheel_name=name, git_sha=git, prediction=prediction, phase3=phase3)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**header, "config": cfg,
                               "wheel": {"name": name, "sha256": sha, "bytes": len(data),
                                         "built_from": git},
                               "prediction": prediction}, indent=1) + "\n")
    print(f"wheel {name} · {len(data) / 1024:.0f} KiB · sha256 {sha[:16]}… · from {git}")
    print(f"prediction: hit rate {prediction['hit_rate']} ({prediction['hits']:,} of "
          f"{prediction['queries']:,} tokens, {prediction['requests']} requests)")
    print(f"wrote {OUT.relative_to(REPO)} and {NOTEBOOK.relative_to(REPO)} "
          f"({NOTEBOOK.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
