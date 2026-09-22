"""Statistics, the cost model, and the guards that stop a number acquiring unearned authority."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from kv_routing_sim.eval.bootstrap import (
    bootstrap_ci_grouped,
    mean,
    paired_bootstrap_delta,
)
from kv_routing_sim.eval.cost import CostModel, UnfilledCostModel, successful_requests
from kv_routing_sim.eval.pareto import Point, frontier
from kv_routing_sim.eval.recommend import recommend


def _p(policy, n, hit, p95, inr, meets=True, calibrated=False):
    return Point(policy, n, hit, p95, p95 / 2, inr, 10.0, 1.0, meets, calibrated)


def test_grouped_bootstrap_gives_wider_intervals_than_ungrouped_would():
    """Sessions are the independent unit. Resampling requests pretends dependence away.

    Two sessions that are internally identical but differ from each other carry real uncertainty;
    a request-level bootstrap would mix them and report near-zero width.
    """
    groups = [[1.0] * 20, [9.0] * 20]
    ci = bootstrap_ci_grouped(groups, mean, n_resamples=500, seed=0)
    assert ci.unit == "session"
    assert ci.hi - ci.lo > 3.0  # genuinely uncertain, as it should be
    assert ci.lo <= ci.point <= ci.hi


def test_paired_bootstrap_is_tighter_than_two_separate_intervals():
    """Shared trace difficulty cancels in the delta. This is why every row gets a paired CI."""
    a = [[x] for x in range(1, 41)]
    b = [[x + 2.0] for x in range(1, 41)]
    delta = paired_bootstrap_delta(a, b, mean, n_resamples=500, seed=0)
    assert delta.point == pytest.approx(-2.0)
    assert delta.hi - delta.lo < 0.2  # the difference is known far better than either level
    assert delta.hi < 0  # and is significantly negative


def test_paired_bootstrap_requires_matched_groups():
    with pytest.raises(ValueError, match="matched groups"):
        paired_bootstrap_delta([[1.0]], [[1.0], [2.0]], mean)


def test_cost_model_refuses_to_run_on_an_unfilled_config(tmp_path):
    """A model that treats an unknown FX rate as zero produces a confident, free-looking number."""
    p = tmp_path / "cost.yaml"
    p.write_text(
        textwrap.dedent("""
            fx: {usd_to_inr: <TBD>, source: <TBD>, date: <TBD>}
            compute: {sku: <TBD>, price_per_hour_usd: <TBD>}
            slo: {ttft_p95_ms: <TBD>}
        """)
    )
    with pytest.raises(UnfilledCostModel, match="still <TBD>"):
        CostModel.from_yaml(p)


def test_every_shipped_cost_constant_has_a_source_and_a_date():
    """configs/cost_model.yaml stayed <TBD> until each constant had a source and a date (filled
    2026-09-22). This keeps it that way: a constant without both is refused."""
    import datetime

    import yaml

    cm = CostModel.from_yaml("configs/cost_model.yaml")
    assert cm.sku == "g4dn.xlarge"  # one Tesla T4: the GPU the profile was measured on
    raw = yaml.safe_load(pathlib.Path("configs/cost_model.yaml").read_text())
    for section in ("fx", "compute", "slo"):
        assert str(raw[section]["source"]).startswith("https://"), section
        assert isinstance(raw[section]["date"], datetime.date), section


def test_one_tbd_left_is_refused(tmp_path):
    shipped = pathlib.Path("configs/cost_model.yaml").read_text()
    p = tmp_path / "cost.yaml"
    p.write_text(shipped.replace("price_per_hour_usd: 0.526", "price_per_hour_usd: <TBD>"))
    with pytest.raises(UnfilledCostModel, match="price_per_hour_usd"):
        CostModel.from_yaml(p)


def test_cost_arithmetic(tmp_path):
    p = tmp_path / "cost.yaml"
    p.write_text(
        textwrap.dedent("""
            fx: {usd_to_inr: 80.0, source: test, date: 2026-09-20}
            compute: {sku: test-a100, price_per_hour_usd: 2.0, source: test, date: 2026-09-20}
            slo: {ttft_p95_ms: 200.0, source: test, date: 2026-09-20}
        """)
    )
    m = CostModel.from_yaml(p)
    # 4 replicas x $2/h x 80 = INR 640/h; at 3600 successful req/h that is INR 640/3600 per req.
    got = m.inr_per_1k_requests(n_replicas=4, successful_requests_per_hour=3600.0)
    assert got == pytest.approx(
        640.0 / 3600.0 * 1000.0
    )
    assert m.inr_per_1k_requests(n_replicas=4, successful_requests_per_hour=0.0) == float("inf")


def test_successful_requests_counts_only_slo_meeting_ones():
    assert successful_requests([10.0, 500.0, 50.0], slo_p95_ms=100.0) == 2


def test_frontier_drops_dominated_points():
    good = _p("a", 2, 0.9, 50.0, 10.0)
    dominated = _p("b", 2, 0.5, 90.0, 20.0)
    other = _p("c", 4, 0.95, 80.0, 5.0)
    front = frontier([good, dominated, other])
    assert dominated not in front
    assert good in front and other in front


def test_recommend_distinguishes_no_slo_from_failed_slo():
    """Three different situations. Saying 'nothing met the SLO' when none was set is false."""
    no_slo = [_p("round-robin", 2, 0.7, 40.0, float("inf"), meets=None)]
    assert "no cost model supplied" in recommend(no_slo).reason

    failed = [_p("round-robin", 2, 0.7, 4000.0, 9.0, meets=False)]
    assert "NO configuration met" in recommend(failed).reason

    ok = [_p("round-robin", 2, 0.7, 40.0, 9.0, meets=True)]
    assert "cheapest configuration meeting" in recommend(ok).reason


def test_recommendation_from_an_uncalibrated_sweep_carries_the_tier_warning():
    rec = recommend([_p("round-robin", 2, 0.7, 40.0, 9.0, calibrated=False)])
    d = rec.as_dict()
    assert d["calibrated"] is False
    assert "UNCALIBRATED" in d["WARNING"]
    assert "POLICY and REPLICA COUNT" in d["WARNING"]


def test_recommendation_prefers_fewer_replicas_at_equal_cost():
    """H5's shape: the win should show up as needing fewer GPUs, and ties must not hide it."""
    cheap_small = _p("radix-affinity-balanced", 2, 0.8, 50.0, 5.0)
    cheap_big = _p("round-robin", 8, 0.7, 50.0, 5.0)
    assert recommend([cheap_big, cheap_small]).chosen is cheap_small
