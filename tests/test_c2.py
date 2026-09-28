"""C2's analysis (scripts/c2_analyze.py) — committed before the run, so it is tested before the run.

The two fits must recover known coefficients from clean data, and the constrained least squares must
never return a negative cost: a negative "per token" coefficient would make a longer prompt faster.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c2_analyze", REPO / "scripts" / "c2_analyze.py")
c2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c2)

DESIGN = {"decode": {"skip_tokens": 8, "output_tokens": 64}}


def test_nnls_solves_an_exact_system():
    rows = [[1.0, x, x * x] for x in (1.0, 2.0, 3.0, 5.0, 8.0)]
    y = [1 + 2 * x + 3 * x * x for x in (1.0, 2.0, 3.0, 5.0, 8.0)]
    coef, sse = c2.nnls(rows, y)
    assert coef == pytest.approx([1.0, 2.0, 3.0])
    assert sse == pytest.approx(0.0, abs=1e-9)


def test_nnls_never_returns_a_negative_cost():
    """Data that falls with x: unconstrained least squares gives a negative slope; the registered
    rule clamps it to zero and refits the rest."""
    rows = [[1.0, x] for x in (1.0, 2.0, 3.0, 4.0)]
    coef, _ = c2.nnls(rows, [5.0, 4.0, 3.0, 2.0])
    assert coef[1] == 0.0
    assert coef[0] == pytest.approx(3.5)  # the mean, once the slope is fixed at zero


def test_the_decode_fit_recovers_a_known_context_term():
    d0, d1, d2 = 0.04, 0.002, 3e-6
    mid = (8 + 64) / 2
    decode = [{"L": L, "b": b, "valid": True, "step_s": d0 + d1 * b + d2 * b * (L + mid)}
              for L in (128, 512, 1024, 1536) for b in (1, 2, 4, 8)]
    decode.append({"L": 1984, "b": 8, "valid": False, "step_s": 99.0})  # invalid: must be ignored
    fit = c2.fit_decode(decode, DESIGN, with_context=True)
    assert (fit["base_s"], fit["per_seq_s"], fit["per_ctx_token_s"]) == pytest.approx((d0, d1, d2))
    assert fit["points"] == 16 and fit["r2"] == pytest.approx(1.0)
    flat = c2.fit_decode(decode, DESIGN, with_context=False)
    assert flat["per_ctx_token_s"] == 0.0 and flat["r2"] < fit["r2"]


def test_the_prefill_fit_takes_one_decode_step_out_of_ttft():
    """TTFT = prefill(n, c) + decode(1, c + n): the fit must return prefill's own coefficients."""
    dec = {"base_s": 0.04, "per_seq_s": 0.002, "per_ctx_token_s": 3e-6}
    p0, p1, p2 = 0.005, 7e-4, 2e-7
    prefill = []
    for c in (0, 256, 768):
        for n in (64, 256, 704):
            ttft = (p0 + p1 * n + p2 * n * (c + n / 2)
                    + dec["base_s"] + dec["per_seq_s"] + dec["per_ctx_token_s"] * (c + n))
            prefill.append({"valid": True, "cached_tokens": c, "prompt_tokens": c + n,
                            "ttft_s": ttft})
    fit = c2.fit_prefill(prefill, dec, with_context=True)
    assert (fit["base_s"], fit["per_token_s"], fit["attn_s"]) == pytest.approx((p0, p1, p2))


def test_a_missing_cell_leaves_a_rule_undecided():
    assert c2._median_step([], 1536, 8) is None
    assert c2.factor(None, 100.0) is None and c2.factor(50.0, 100.0) == pytest.approx(2.0)
