"""Data shipped inside the package, so `pip install kv-routing-sim` works without a clone.

  * `profiles/*.json` — measured timing + capacity profiles, loadable by NAME:
                        `kvsim replay trace.jsonl --profile t4-qwen3-4b-vllm0.30.0`
  * `cost_model.yaml` — the cited cost model: `--cost-model default`

⛔ These are COPIES of `configs/profiles/` and `configs/cost_model.yaml`, which stay the source of
truth for the repo's scripts. `tests/test_packaging.py` fails if a copy drifts by one byte.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable


def root() -> Traversable:
    return resources.files(__name__)


def profiles() -> dict[str, Traversable]:
    """Every shipped profile, by name (the file name without `.json`)."""
    return {p.name.removesuffix(".json"): p for p in (root() / "profiles").iterdir()
            if p.name.endswith(".json")}


def cost_model() -> Traversable:
    return root() / "cost_model.yaml"
