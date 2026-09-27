"""Rank agreement between two orderings — the statistic H6 is decided on (docs/09-hypotheses.md).

Kendall's tau-b, written out rather than imported: it is a few lines, the core install stays free
of scipy (ADR-009), and a reader can check the tie handling without leaving this file.

⭐ tau-B, NOT tau-a, BECAUSE THIS PROJECT'S RANKINGS ARE FULL OF TIES. The three load-balancing
policies often need exactly the same number of replicas, and "not met at 8 replicas" ties every
policy that fails. tau-a would count those ties as disagreement and understate agreement; tau-b
takes them out of the denominator instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b between two equal-length sequences. None when either one is all ties.

        tau_b = (concordant - discordant) / sqrt(pairs not tied in x * pairs not tied in y)

    ⛔ None, not 0 or 1: a constant sequence has no order, so there is nothing to agree or
    disagree with. Making up a number for it would tilt the result toward whichever way the
    convention leaned.
    """
    if len(x) != len(y):
        raise ValueError(f"tau-b needs equal lengths, got {len(x)} and {len(y)}")
    concordant = discordant = only_x_tied = only_y_tied = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue  # tied in both: in neither count
            if dx == 0:
                only_x_tied += 1
            elif dy == 0:
                only_y_tied += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    not_tied_x = concordant + discordant + only_y_tied
    not_tied_y = concordant + discordant + only_x_tied
    denom = math.sqrt(not_tied_x * not_tied_y)
    return None if denom == 0 else (concordant - discordant) / denom
