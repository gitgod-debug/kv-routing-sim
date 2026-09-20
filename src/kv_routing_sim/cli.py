"""The `kvsim` CLI. 📐 designed, not built.

Target contract (from the spec, and it is a packaging promise as much as a UX one):

    kvsim replay trace.jsonl --policies all --replicas 1-8 --format json|md|html

Error semantics (ADR-004): an unparseable policy or sweep config FAILS LOUDLY AT STARTUP. It must
never fail silently mid-sweep, because a sweep that dies at hour three having written half a Pareto
curve is worse than one that refuses to start.

This module exists in Phase 0 so the console-script entry point declared in `pyproject.toml` is
real. A declared entry point that does not import is a broken install, and it would be discovered
by a user rather than by us.
"""

from __future__ import annotations

import sys

_PHASE_0_MESSAGE = """\
kvsim is not implemented yet — this is the Phase 0 scaffold.

The build is gated: no simulator code is written until the Phase-0 spike passes against a live
inference engine. See CHECKPOINTS.md for where the build is, and docs/00-START-HERE.md for what
this project is.

To run the gating spike (on a GPU box, with the [calib] extra installed):
    python scripts/spike_vllm_metrics.py --model <model>
"""


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 2 (usage/unavailable), never 0 — there is nothing to succeed at yet."""
    del argv  # no arguments are parsed while there is nothing to parse them for
    print(_PHASE_0_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
