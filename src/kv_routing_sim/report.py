"""Render a replay report as Markdown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SLO_MARKS = {True: "yes", False: "no", None: "—"}
"""How the "meets SLO" column reads; the dash means no SLO was defined."""


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report, as returned by `ReplayReport.as_dict()`, as Markdown."""
    lines: list[str] = []
    lines += _recommendation_section(report)
    lines += _configurations_section(report)
    lines += _spread_section(report)
    if report.get("trace"):
        lines += ["## Trace", "", "```json", json.dumps(report["trace"], indent=2), "```", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class Note:
    """A sentence that qualifies the recommendation; a warning calls for caution or action."""

    text: str
    warning: bool = False


def recommendation_notes(report: dict[str, Any]) -> list[Note]:
    """Return the sentences that qualify the recommendation, in display order."""
    rec = report.get("recommendation") or {}
    chosen = rec.get("recommended")
    if not chosen:
        return []
    notes = []
    if rec.get("tied_with"):
        tied = ", ".join(rec["tied_with"])
        notes.append(Note(f"tied on replicas with: {tied} (between these the rupee gap is noise)"))
    # A cost difference only means something between two configurations that both meet the SLO.
    baseline = rec.get("baseline") or {}
    if (rec.get("inr_delta_pct_vs_baseline") is not None and chosen.get("meets_slo")
            and baseline.get("meets_slo") and baseline.get("policy") != chosen["policy"]):
        notes.append(Note(f"vs round-robin: {rec['inr_delta_pct_vs_baseline']:+.1f}% cost, "
                          f"{rec['replica_delta_vs_baseline']:+d} replicas"))
    elif chosen.get("meets_slo") and baseline.get("meets_slo") is False:
        notes.append(Note("round-robin met the SLO on none of the replica counts swept, so there "
                          "is no cost to compare with."))
    spread = report.get("replica_spread") or {}
    firmness = (spread.get("replicas") or {}).get(chosen["policy"])
    if firmness:
        notes.append(Note(_firmness_line(firmness, spread), warning=not firmness["firm"]))
    edge = report.get("edge")
    if edge:
        notes.append(Note(_edge_line(edge), warning=edge["verdict"] in ("knee", "edge")))
    return notes


def _recommendation_section(report: dict[str, Any]) -> list[str]:
    rec = report.get("recommendation") or {}
    chosen = rec.get("recommended")
    if not chosen:
        return []
    lines = [
        "## Recommendation",
        "",
        f"**{chosen['policy']} on {chosen['n_replicas']} replica(s)** — {rec.get('reason', '')}",
        "",
    ]
    for note in recommendation_notes(report):
        lines += [note.text, ""]
    if rec.get("WARNING"):
        lines += [f"> Warning: {rec['WARNING']}", ""]
    return lines


def _configurations_section(report: dict[str, Any]) -> list[str]:
    points = report.get("points") or []
    if not points:
        return []
    lines = [
        "## All configurations",
        "",
        ("| policy | replicas | hit rate | p50 TTFT ms | p95 TTFT ms | p95 ms per token | "
         "INR/1k | imbalance | meets SLO |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for p in points:
        inr = "—" if p["inr_per_1k_requests"] is None else f"{p['inr_per_1k_requests']:.3f}"
        tpot = "—" if p.get("p95_tpot_ms") is None else f"{p['p95_tpot_ms']:.1f}"
        lines.append(
            f"| {p['policy']} | {p['n_replicas']} | {p['hit_rate']:.4f} | "
            f"{p['p50_ttft_ms']:.1f} | {p['p95_ttft_ms']:.1f} | {tpot} | {inr} | "
            f"{p['load_imbalance']:.3f} | {SLO_MARKS[p['meets_slo']]} |"
        )
    lines.append("")
    if report.get("slo"):
        lines += [report["slo"], ""]
    return lines


def _spread_section(report: dict[str, Any]) -> list[str]:
    spread = report.get("replica_spread")
    if not (spread and spread.get("replicas")):
        note = report.get("replica_spread_note")
        return [f"> Note: {note}", ""] if note else []
    top = spread["swept_up_to"]
    lines = [
        "## How firm are the replica counts?",
        "",
        (f"The calibration measured every timing cell {spread['repeats']} times. The same "
         "sizing, on the fit of each repeat alone:"),
        "",
        "| policy | all repeats together | each repeat alone |",
        "|---|---:|---|",
    ]
    for policy, row in spread["replicas"].items():
        repeats = " · ".join(_count(n, top) for n in row["repeats"])
        lines.append(f"| {policy} | {_count(row['pooled'], top)} | {repeats} |")
    lines += [
        "",
        ("This is how much one calibration's own repeats disagree: a floor on the uncertainty, "
         "not a confidence interval. On a live T4, another calibration run within the hour gave "
         "a count outside a run's own range in 18% of cases, and outside it by more than one "
         "replica in 5%, so size for the top of the range, plus one. The range says nothing "
         "about another day or another software image: three calibrations of the same T4 "
         "agreed exactly on 71% of counts and within one replica on 96%."),
        "",
    ]
    return lines


def _count(n: int | None, swept_up_to: int) -> str:
    return f"not met on {swept_up_to}" if n is None else str(n)


def _firmness_line(row: dict[str, Any], spread: dict[str, Any]) -> str:
    """Describe how the recommended policy's replica count varies across calibration repeats."""
    top, repeats = spread["swept_up_to"], spread["repeats"]
    if row["firm"]:
        return (f"How firm: the same count on each of the calibration's {repeats} repeats taken "
                "alone; still allow one more, because another calibration run can differ by a "
                "replica.")
    return (f"How firm: {_count(row['min'], top)} to {_count(row['max'], top)} replicas across "
            f"the calibration's {repeats} repeats taken alone; size for the larger, plus one.")


def _edge_line(edge: dict[str, Any]) -> str:
    """Describe how close to saturation the recommended configuration sits."""
    pct = round((edge["slowdown"] - 1) * 100)

    def moved(label: str, pair: list, factor: float | None) -> str | None:
        if factor is None:
            return None
        return f"{label} {pair[0]:,.0f} to {pair[1]:,.0f} ms (×{factor:.2f})"

    changes = [m for m in (moved("p95 TTFT", edge["p95_ttft_ms"], edge["ttft_factor"]),
                           moved("p95 time per token", edge["p95_tpot_ms"], edge["tpot_factor"]))
               if m]
    what = "; ".join(changes)
    verdict = edge["verdict"]
    if verdict == "knee":
        knee_pct = round((edge["knee_factor"] - 1) * 100)
        return (f"At the knee: with timings {pct}% slower, {what}. A {pct}% change in the timings "
                f"moves a p95 by more than {knee_pct}%, so the fleet is close to saturation, "
                "where the simulator's latencies were 1.5–1.7× off on a live GPU. Do not trust "
                "this count: size at least one replica higher, and measure.")
    if verdict == "edge":
        return (f"At the edge: with timings {pct}% slower this configuration no longer meets the "
                f"SLO ({what}). Two calibrations of the same GPU differ by that much: allow one "
                "replica more.")
    if verdict == "not_met":
        return f"With timings {pct}% slower: {what}."
    return f"Not at the edge: with timings {pct}% slower it still meets the SLO ({what})."
