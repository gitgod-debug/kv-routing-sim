"""Render a replay report as Markdown, or as JSON that every parser accepts."""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from typing import Any

from .traces.adapters import agentx

SLO_MARKS = {True: "yes", False: "no", None: "—"}
"""How the "meets SLO" column reads; the dash means no SLO was defined."""

AGENTX_TOKEN_SCALE = inspect.signature(agentx.read).parameters["token_scale"].default
"""The AgentX reader's default token scale, read from the reader so that the two cannot differ."""


def to_json(payload: Any, *, indent: int | None = None) -> str:
    """Serialize as strict JSON: a NaN or an infinity is written as null.

    Python writes them as NaN and Infinity, which JavaScript's JSON.parse and most other parsers
    reject.
    """
    return json.dumps(_finite(payload), indent=indent, allow_nan=False)


def _finite(node: Any) -> Any:
    if isinstance(node, float):
        return node if math.isfinite(node) else None
    if isinstance(node, dict):
        return {k: _finite(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_finite(v) for v in node]
    return node


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
    prompts = report.get("prompt_range")
    if prompts and prompts["requests_longer"]:
        share = 100 * prompts["requests_longer"] / prompts["requests"]
        notes.append(Note(
            f"Outside the measured range: {prompts['requests_longer']:,} of "
            f"{prompts['requests']:,} prompts ({share:.0f}%) are longer than "
            f"{prompts['longest_measured']:,} tokens, the longest this profile was timed on (the "
            f"longest here has {prompts['longest_here']:,}). Their timings are extrapolated.",
            warning=True))
    scaling = (report.get("trace") or {}).get("scaling") or {}
    if scaling.get("token_scale", 1) != 1:
        block_size = (report.get("profile") or {}).get("block_size", 16)
        notes.append(Note(_scale_line(scaling, block_size), warning=True))
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
         "a count outside a run's own range in 16% of cases, and outside it by more than one "
         "replica in 5%, so size for the top of the range, plus one. The range says nothing "
         "about another day or another software image. Three calibrations of one GPU type all "
         "gave a count at 91 of 144 points: the same one at 62 and within one replica at 86. At 9 "
         "more points they disagreed on whether the target could be held at all."),
        "",
    ]
    return lines


def _count(n: int | None, swept_up_to: int) -> str:
    return f"not met on {swept_up_to}" if n is None else str(n)


def _ratio(scale: float) -> str:
    """0.03125 as 1/32. A scale that is not one over a whole number stays a decimal."""
    k = round(1 / scale)
    return f"1/{k}" if scale < 1 and abs(k * scale - 1) < 1e-9 else f"{scale:g}"


def _scale_line(scaling: dict[str, Any], block_size: int) -> str:
    """Say that the reader scaled every token count, and how to read the trace at another scale."""
    scale = scaling["token_scale"]
    line = f"Token counts were scaled by {_ratio(scale)} when the trace was read"
    if "concurrency" not in scaling:  # Mooncake or Bailian: the scale was asked for
        return f"{line} (--token-scale). Use --token-scale 1 for the trace at its recorded size."
    if scale == AGENTX_TOKEN_SCALE:
        line += ": the AgentX reader's default, so that it fits a T4."
    else:
        line += " (--token-scale)."
    # One 64-token AgentX block must fit in one simulator block, so the scale is at most
    # block_size / 64.
    largest = block_size / 64
    if largest >= 1:
        return f"{line} Use --token-scale 1 for the trace at its recorded size."
    return (f"{line} With {block_size}-token blocks the largest scale is {_ratio(largest)} "
            f"(--token-scale {largest:g}). The trace at its recorded size needs a profile with "
            "64-token blocks.")


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
