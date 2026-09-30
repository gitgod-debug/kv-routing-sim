"""The context-aware timing model's fit — the procedure C2 registered and validated on a real T4.

    decode(b, S)  = d0 + d1·b + d2·S          b sequences in the step, S = their summed context
    prefill(n, c) = p0 + p1·n + p2·n·(c + n/2)  n new tokens after c cached ones

Least squares with every coefficient ≥ 0: every subset of free coefficients is solved and the
feasible one with the least squared error wins — a negative "per token" cost would make a longer
prompt faster. C2 fitted this on micro-benchmarks and it predicted held-out traffic within 1.41×
(configs/c2_timing.yaml, docs/09-hypotheses.md 2026-09-30); `kvsim calibrate` fits the same model.

The same arithmetic as scripts/c2_analyze.py — which is a registered script and stays byte-for-byte
as it ran — kept here so the installed package needs no scripts/. Standard library only.
"""

from __future__ import annotations

import statistics


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def nnls(rows: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    """Least squares with every coefficient ≥ 0, by solving every subset of free coefficients.
    Exhaustive and deterministic: ties go to the subset enumerated first."""
    k = len(rows[0])
    best: tuple[float, list[float]] | None = None
    for mask in range(1 << k):
        cols = [j for j in range(k) if mask >> j & 1]
        coef = [0.0] * k
        if cols:
            ata = [[sum(r[i] * r[j] for r in rows) for j in cols] for i in cols]
            aty = [sum(r[i] * v for r, v in zip(rows, y, strict=True)) for i in cols]
            sol = _solve(ata, aty)
            if sol is None or any(s < 0 for s in sol):
                continue
            for j, s in zip(cols, sol, strict=True):
                coef[j] = s
        sse = sum((sum(c * x for c, x in zip(coef, r, strict=True)) - v) ** 2
                  for r, v in zip(rows, y, strict=True))
        if best is None or sse < best[0]:
            best = (sse, coef)
    assert best is not None
    return best[1], best[0]


def r_squared(rows: list[list[float]], y: list[float], coef: list[float]) -> float:
    mean = statistics.fmean(y)
    tot = sum((v - mean) ** 2 for v in y)
    res = sum((sum(c * x for c, x in zip(coef, r, strict=True)) - v) ** 2
              for r, v in zip(rows, y, strict=True))
    return 1.0 - res / tot if tot else 1.0


def fit_decode(cells: list[dict], mid_tokens: float) -> dict[str, float]:
    """cells: {"batch", "context", "step_s"}; S = b·(context + mid_tokens), the batch's mean
    summed context over the tokens that were timed."""
    rows = [[1.0, float(c["batch"]), c["batch"] * (c["context"] + mid_tokens)] for c in cells]
    y = [c["step_s"] for c in cells]
    coef, _ = nnls(rows, y)
    return {"base_s": coef[0], "per_seq_s": coef[1], "per_ctx_token_s": coef[2],
            "points": len(cells), "r2": r_squared(rows, y, coef)}


def fit_prefill(points: list[dict], dec: dict[str, float]) -> dict[str, float]:
    """points: {"cached_tokens", "uncached_tokens", "ttft_s"}. The simulator's TTFT of a lone
    request is prefill plus one decode step, so that step is taken out of each TTFT first."""
    rows, y = [], []
    for p in points:
        c, n = p["cached_tokens"], p["uncached_tokens"]
        one_step = dec["base_s"] + dec["per_seq_s"] + dec["per_ctx_token_s"] * (c + n)
        rows.append([1.0, float(n), n * (c + n / 2.0)])
        y.append(p["ttft_s"] - one_step)
    coef, _ = nnls(rows, y)
    return {"base_s": coef[0], "per_token_s": coef[1], "attn_s": coef[2], "points": len(points),
            "r2": r_squared(rows, y, coef)}
