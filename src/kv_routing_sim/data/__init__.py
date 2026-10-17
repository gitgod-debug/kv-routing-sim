"""Data files shipped with the package, so it works without a clone of the repository.

`profiles/*.json` are measured timing profiles, loadable by name
(`--profile t4-qwen3-4b-vllm0.30.0`), and `cost_model.yaml` is the cited cost model
(`--cost-model default`). They are copies of `configs/profiles/` and `configs/cost_model.yaml`,
which remain the source of truth; `tests/test_packaging.py` fails if a copy differs.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable


def root() -> Traversable:
    return resources.files(__name__)


def profiles() -> dict[str, Traversable]:
    """Return every shipped profile by name (its file name without `.json`)."""
    return {p.name.removesuffix(".json"): p for p in (root() / "profiles").iterdir()
            if p.name.endswith(".json")}


def cost_model() -> Traversable:
    return root() / "cost_model.yaml"
