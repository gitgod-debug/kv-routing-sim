#!/usr/bin/env python3
"""The README's figure — the flip — drawn from the committed demo grid.

    python scripts/make_figures.py  # -> docs/figures/flip-*.svg and paper/figures/flip.pdf

Needs the [analysis] extra (matplotlib). It reads `artifacts/results/demo_grid.json`, which
`scripts/build_grid.py` writes only when it reproduces Phase 5's H2b table exactly, so the figure
shows the experiment's numbers and runs nothing itself. Same grid in, same bytes out.

Left: the fewest T4s that hold the 2 s p95 TTFT target, as distinct tenant prompts multiply.
Right: the prefix-cache hit rate on a fixed fleet — the mechanism behind the left panel.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
GRID = REPO / "artifacts" / "results" / "demo_grid.json"
OUT = REPO / "docs" / "figures"
PAPER = REPO / "paper" / "figures" / "flip.pdf"  # the preprint's copy: pdfLaTeX does not read SVG
LOAD = 2  # sessions/s: the README table's column
REPLICAS = 4  # the fleet the hit rates are read on, as in Phase 5's write-up

THEMES = {
    "light": {"fg": "#1d1d1b", "muted": "#5f5d57", "grid": "#dedad2"},
    "dark": {"fg": "#ecebe6", "muted": "#a8a59c", "grid": "#3a3934"},
}


def _series(grid: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(grid label, legend text, colour, marker) for each line, in drawing order."""
    star = grid["alpha_star"]
    return [
        ("round-robin", "round-robin", "#8d8a82", "o"),
        ("session-sticky", "session-sticky", "#1f9d6a", "s"),
        (f"radix-affinity-balanced@{star}", f"prefix-affinity, load weight {star} (registered)",
         "#3a78d8", "D"),
        ("radix-affinity-balanced@1", "prefix-affinity, load weight 1 (found after the test)",
         "#c2542d", "^"),
    ]


def _not_met(grid: dict[str, Any]) -> int:
    """Drawn one above the largest fleet: no replica count in the grid holds the target."""
    return max(grid["axes"]["replicas"]) + 1


def _needed(grid: dict[str, Any], k: int, load: int, policy: str) -> int:
    ax = grid["axes"]
    cells = grid["cells"][k][load][ax["policies"].index(policy)]
    return next((n for n, c in zip(ax["replicas"], cells, strict=True) if c[4] == 1),
                _not_met(grid))


def draw(grid: dict[str, Any], theme: str) -> plt.Figure:
    t = THEMES[theme]
    ax_ = grid["axes"]
    li, ri = ax_["loads"].index(LOAD), ax_["replicas"].index(REPLICAS)
    ks = ax_["prompts"]
    cap = grid["usable_blocks_per_replica"]
    xt = [f"{k}\n{grid['working_set_blocks'][str(k)] / cap:.2f}×" for k in ks]
    xs = range(len(ks))
    series = _series(grid)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=100)
    fig.patch.set_alpha(0)
    for a in (left, right):
        a.set_facecolor("none")
        a.set_xticks(list(xs), xt)
        a.set_xlabel("distinct system prompts\nworking set ÷ one T4's KV cache", color=t["muted"])
        a.tick_params(colors=t["muted"])
        a.grid(axis="y", color=t["grid"], linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(t["grid"])

    n = len(series)
    for j, (policy, _text, colour, marker) in enumerate(series):
        dx = (j - (n - 1) / 2) * 0.05  # a sideways nudge, so equal values do not hide each other
        x = [i + dx for i in xs]
        need = [_needed(grid, i, li, policy) for i in xs]
        left.plot(x, need, color=colour, marker=marker, linewidth=2, markersize=6)
        hit = [grid["cells"][i][li][ax_["policies"].index(policy)][ri][0] for i in xs]
        right.plot(x, hit, color=colour, marker=marker, linewidth=2, markersize=6)

    left.set_title(f"T4s needed at {LOAD} sessions/s (p95 TTFT ≤ 2 s)", color=t["fg"], fontsize=11)
    top = _not_met(grid)
    vals = list(range(1, top)) if top <= 9 else list(range(1, top - 1, 3))
    left.set_yticks([*vals, top], [str(v) for v in vals] + [f"> {top - 1}"])  # not held
    left.set_ylim(0.5, top + 0.5)
    right.set_title(f"Prefix-cache hit rate on {REPLICAS} T4s at {LOAD} sessions/s", color=t["fg"],
                    fontsize=11)
    right.set_ylim(0.45, 1.0)

    handles = [plt.Line2D([], [], color=c, marker=m, linewidth=2) for _p, _t, c, m in series]
    fig.legend(handles, [text for _p, text, _c, _m in series], loc="lower center", ncol=2,
               frameon=False, labelcolor=t["fg"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def main() -> int:
    plt.rcParams["svg.hashsalt"] = "kv-routing-sim"  # stable ids: same grid, same bytes
    plt.rcParams["svg.fonttype"] = "none"  # text stays text: smaller, and in the reader's font
    plt.rcParams["pdf.fonttype"] = 42  # TrueType in the PDF, not Type 3, as arXiv prefers
    grid = json.loads(GRID.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        fig = draw(grid, theme)
        path = OUT / f"flip-{theme}.svg"
        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KiB)")
    PAPER.parent.mkdir(parents=True, exist_ok=True)
    fig = draw(grid, "light")
    fig.savefig(PAPER, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {PAPER.relative_to(REPO)} ({PAPER.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
