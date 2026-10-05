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
from kv_routing_sim.eval.rank import kendall_tau_b
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


def test_kendall_tau_b_matches_worked_examples():
    """H6 is decided on this statistic, and its tie handling is the part that matters: rankings by
    replica count are full of ties."""
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # one pair tied only in x, one tied only in y, one concordant: 1 / sqrt(2 x 2)
    assert kendall_tau_b([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    # ties shared by both orderings are not disagreement
    assert kendall_tau_b([3, 3, 3, 1, 2], [5, 5, 5, 1, 2]) == pytest.approx(1.0)


def test_kendall_tau_b_is_undefined_when_there_is_nothing_to_order():
    """A constant ranking has no order to agree with: None, never a made-up 0 or 1."""
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) is None
    assert kendall_tau_b([2, 2], [2, 2]) is None


def test_kendall_tau_b_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        kendall_tau_b([1, 2], [1, 2, 3])


def test_recommendation_reports_ties_on_the_same_replica_count():
    """Between policies that hold the SLO on the same replicas, the rupee gap is noise (H6 found a
    pick made on 0.1%). The pick is kept; the tie is reported, not hidden."""
    sticky = _p("session-sticky", 3, 0.9, 500.0, 4.7842)
    balanced = _p("radix-affinity-balanced", 3, 0.9, 600.0, 4.7796)
    rr = _p("round-robin", 4, 0.8, 500.0, 6.0)
    rec = recommend([sticky, balanced, rr])
    assert rec.chosen is balanced
    assert rec.tied_with == ("session-sticky",)
    assert rec.as_dict()["tied_with"] == ["session-sticky"]
    assert recommend([rr]).tied_with == ()


# ── the SLO's second half: time per output token ─────────────────────────────────────────────

_COST = """
    fx: {{usd_to_inr: 80.0, source: test, date: 2026-09-20}}
    compute: {{sku: test, price_per_hour_usd: 2.0, source: test, date: 2026-09-20}}
    slo: {{ttft_p95_ms: 2000.0{tpot}, source: test, date: 2026-09-20}}
"""


def _cost(tmp_path, tpot: str = "") -> CostModel:
    p = tmp_path / f"cost{len(tpot)}.yaml"
    p.write_text(textwrap.dedent(_COST.format(tpot=tpot)))
    return CostModel.from_yaml(p)


def test_the_per_token_half_of_the_slo_is_off_unless_the_cost_model_sets_it(tmp_path):
    """Every result registered up to 2026-10-05 is on the first-token limit alone. A cost model
    without `tpot_p95_ms` must keep meaning exactly that — whatever the tokens' speed."""
    one = _cost(tmp_path)
    assert one.tpot_p95_slo_ms is None and one.slo_label == "p95 TTFT SLO"
    assert one.meets_slo(1_999.0, 10_000.0) and not one.meets_slo(2_001.0, 1.0)
    shipped = CostModel.from_yaml("configs/cost_model.yaml")
    assert shipped.tpot_p95_slo_ms is None  # the shipped default: first token only


def test_with_both_halves_a_configuration_must_meet_both(tmp_path):
    two = _cost(tmp_path, ", tpot_p95_ms: 100")
    assert two.tpot_p95_slo_ms == 100.0 and "time per output token" in two.slo_label
    assert two.meets_slo(1_500.0, 100.0)  # at the limit is within it, as for TTFT
    assert not two.meets_slo(1_500.0, 100.1)  # a prompt first token does not excuse slow ones
    assert not two.meets_slo(2_500.0, 50.0)
    # No request had a second token: there is no per-token latency to miss a limit with.
    assert two.meets_slo(1_500.0, float("nan"))


def test_a_successful_request_meets_every_half_that_is_set():
    ttft = [100.0, 100.0, 3_000.0, 100.0]
    tpot = [50.0, 150.0, 50.0, float("nan")]  # the last: one output token, no interval
    assert successful_requests(ttft, 2_000.0) == 3
    assert successful_requests(ttft, 2_000.0, tpot, None) == 3  # no limit set: as before
    assert successful_requests(ttft, 2_000.0, tpot, 100.0) == 2  # 50 ms and the single-token one


def test_a_point_carries_the_per_token_p95_and_decides_the_slo_on_it_only_when_asked(tmp_path):
    from kv_routing_sim.eval.pareto import to_point
    from kv_routing_sim.eval.sweep import average_points
    from kv_routing_sim.sim.cost_model import CostProfile
    from kv_routing_sim.sim.engine import run
    from kv_routing_sim.traces.schema import NormalizedRequest

    # Decode steps of 150 ms: the first token is prompt, the rest are slow.
    slow = CostProfile(name="slow-decode", prefill_base_s=0.0, prefill_per_token_s=0.0001,
                       decode_base_s=0.150, decode_per_seq_s=0.0, kv_cache_blocks=5_000,
                       block_size=16, max_num_seqs=16)
    trace = [NormalizedRequest(f"r{i}", float(i), (i * 10 + 1, i * 10 + 2), 32, 5, f"s{i}")
             for i in range(8)]
    r = run(trace, policy="round-robin", n_replicas=1, profile=slow)
    one, two = _cost(tmp_path), _cost(tmp_path, ", tpot_p95_ms: 100")
    first_only, both = to_point(r, one), to_point(r, two)
    assert first_only.p95_tpot_ms == both.p95_tpot_ms == pytest.approx(150.0)
    assert first_only.p95_ttft_ms == pytest.approx(153.2)  # 3.2 ms of prefill + one step
    assert first_only.meets_slo is True and both.meets_slo is False
    assert to_point(r, None).meets_slo is None  # no cost model: still "no SLO", not a failure
    # Every request met the first-token limit and none the per-token one: the cost denominator
    # follows the SLO, so under both halves nothing here was served successfully.
    assert first_only.inr_per_1k != float("inf") and both.inr_per_1k == float("inf")
    assert first_only.as_dict()["p95_tpot_ms"] == 150.0
    assert average_points([both, both]).p95_tpot_ms == pytest.approx(150.0)
    # A point built without it (older code, hand-made points) says "not measured", not 0.
    assert _p("x", 1, 0.5, 10.0, 1.0).as_dict()["p95_tpot_ms"] is None


def test_the_recommendations_reason_names_the_slo_it_was_decided_on():
    ok = [_p("round-robin", 2, 0.7, 40.0, 9.0, meets=True)]
    assert recommend(ok).reason == "cheapest configuration meeting the p95 TTFT SLO"
    two = recommend(ok, slo="SLO (p95 TTFT and p95 time per output token)")
    assert two.reason.endswith("meeting the SLO (p95 TTFT and p95 time per output token)")
    failed = [_p("round-robin", 2, 0.7, 4000.0, 9.0, meets=False)]
    assert "NO configuration met the SLO (p95" in recommend(
        failed, slo="SLO (p95 TTFT and p95 time per output token)").reason
