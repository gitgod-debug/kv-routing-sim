#!/usr/bin/env python3
"""Assemble the static site: the pages, the demo grids, the docs, and a wheel for the runner.

    python scripts/build_site.py --out _site
    python scripts/build_site.py --space --out _space   # the same, as a Hugging Face static Space

The GitHub Pages workflow runs this; run it locally and serve the output directory to preview
the site exactly as it is published (python -m http.server --directory _site). `--space` adds the
README.md that a Space reads its settings from; pushing it is a separate step, done by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_docs

GRIDS = ("demo_grid.json", "demo_grid_ctx.json")


PACKAGE_INPUTS = ("pyproject.toml", "README.md", "LICENSE")


def build_wheel(destination: Path) -> Path:
    """Build the package's wheel from a clean copy of its sources; return the copied wheel."""
    with tempfile.TemporaryDirectory() as scratch:
        source, wheel_dir = Path(scratch) / "source", Path(scratch) / "wheels"
        source.mkdir()
        for name in PACKAGE_INPUTS:
            shutil.copy2(REPO / name, source / name)
        shutil.copytree(REPO / "src", source / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--quiet", "--wheel-dir",
             str(wheel_dir), str(source)],
            check=True,
        )
        wheels = list(wheel_dir.glob("kv_routing_sim-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one wheel, found {[w.name for w in wheels]}")
        return Path(shutil.copy2(wheels[0], destination))


def build_site(out: Path, *, space: bool = False) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO / "site", out)
    for grid in GRIDS:
        shutil.copy2(REPO / "artifacts" / "results" / grid, out / grid)
    wheel = build_wheel(out)
    (out / "wheel.json").write_text(json.dumps({"wheel": wheel.name}) + "\n", encoding="utf-8")
    pages = build_docs.build(out / "docs")
    if space:
        shutil.copy2(REPO / "space" / "README.md", out / "README.md")
    print(f"built {out} with {wheel.name} and {len(pages)} docs pages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "_site")
    parser.add_argument("--space", action="store_true",
                        help="add the README.md of a Hugging Face static Space")
    args = parser.parse_args()
    build_site(args.out.resolve(), space=args.space)


if __name__ == "__main__":
    main()
