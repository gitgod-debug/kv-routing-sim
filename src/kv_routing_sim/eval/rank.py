"""Rank agreement between two orderings, as Kendall's tau-b.

It is written out rather than imported so that the core install needs no scipy (ADR-009). Tau-b, not
tau-a, because policy rankings are full of ties (several policies often need the same number of
replicas), and tau-b removes ties from the denominator instead of counting them as disagreement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return Kendall's tau-b between two equal-length sequences, or None if either is all ties.

        tau_b = (concordant - discordant) / sqrt(pairs not tied in x * pairs not tied in y)

    A constant sequence has no order to agree or disagree with, so it gets None rather than an
    arbitrary 0 or 1.
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
