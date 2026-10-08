"""The `kvsim` command line.

    kvsim replay    trace.jsonl --policies all --replicas 1-8 --format md \
                    --profile t4-qwen3-4b-vllm0.30.0-ctx --cost-model default
    kvsim generate  --sharing 0.6 --out trace.jsonl
    kvsim crossover --sharing 0.1,0.3,0.5,0.7,0.9 --rates 4,8,16 --profile <profile.json>
    kvsim calibrate --engine http://localhost:8000 --out configs/profiles/mine.json

Error semantics (ADR-004): a bad config or an unknown policy FAILS LOUDLY AT STARTUP, before any
work begins. A sweep that dies at hour three having written half a Pareto curve is worse than one
that refuses to start.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from . import policies as policy_pkg
from .eval.cost import CostModel, UnfilledCostModel
from .eval.edge import edge_check
from .eval.recommend import recommend
from .eval.spread import replica_spread
from .eval.sweep import crossover_sweep, sweep
from .sim.cost_model import (
    UNCALIBRATED_DEFAULT,
    CostProfile,
    replicate_profiles,
    resolve_profile,
)
from .traces import adapters
from .traces.generator import AgenticWorkload
from .traces.schema import TraceStats

UNCALIBRATED_BANNER = (
    "\n  ⚠️  UNCALIBRATED COST PROFILE — no live engine has been measured.\n"
    "      Usable:     the policy RANKING and the replica count (ADR-010 tier 1, pending H6).\n"
    "      NOT usable: the millisecond and rupee figures. They are shapes, not predictions.\n"
    "      Fix:        kvsim calibrate --engine <your vLLM URL> --out mine.json, then\n"
    "                  --profile mine.json\n"
)


def _parse_replicas(spec: str) -> list[int]:
    """'1-8' or '1,2,4,8' -> [ints]. Fails loudly."""
    spec = spec.strip()
    try:
        if "-" in spec and "," not in spec:
            lo, hi = spec.split("-", 1)
            out = list(range(int(lo), int(hi) + 1))
        else:
            out = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"error: could not parse --replicas {spec!r}; use '1-8' or '1,2,4'"
        ) from None
    if not out or any(n < 1 for n in out):
        raise SystemExit(f"error: --replicas {spec!r} must be one or more positive integers")
    return out


def _load_cost(path: str | None) -> CostModel | None:
    if not path:
        return None
    try:
        return CostModel.from_yaml(path)
    except UnfilledCostModel as exc:
        raise SystemExit(f"error: {exc}") from None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: could not read cost model {path!r}: {exc}") from None


def _load_profile(path: str | None) -> CostProfile:
    """The timing + capacity profile. Without one: the uncalibrated guess, with a banner."""
    if not path:
        return UNCALIBRATED_DEFAULT
    try:
        return resolve_profile(path)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: could not read profile {path!r}: {exc}") from None


def _emit(payload: dict[str, Any], fmt: str, out: str | None) -> None:
    text = json.dumps(payload, indent=2) if fmt == "json" else _markdown(payload)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def _slo_mark(meets: bool | None) -> str:
    """✅ / ❌ / — . The dash means no SLO was defined, which is not the same as failing one."""
    return "—" if meets is None else ("✅" if meets else "❌")


def _count(n: int | None, swept_up_to: int) -> str:
    return f"not met on {swept_up_to}" if n is None else str(n)


def _firmness(row: dict[str, Any], spread: dict[str, Any]) -> str:
    """One sentence beside the recommendation: its replica count on each repeat's fit."""
    top = spread["swept_up_to"]
    # C4, rule E (2026-10-05): a run's own repeats do NOT cover another run's count often enough
    # — 82% of cases on a live T4, 95% with one replica more each way. So both sentences say it.
    if row["firm"]:
        return (f"how firm: the same count on each of the calibration's {spread['repeats']} "
                "repeats taken alone — still allow one more: another run of the calibration can "
                "differ by a replica")
    return (f"⚠️ how firm: {_count(row['min'], top)} to {_count(row['max'], top)} replicas "
            f"across the calibration's {spread['repeats']} repeats taken alone — size for the "
            "larger, plus one")


# ⭐ THE PER-TOKEN HALF IS ON BY DEFAULT IN `kvsim replay` (2026-10-07). The MLPerf row that the 2 s
# first-token limit is cited from reads "TTFT/TPOT: 2000 ms/100 ms". Sizing on the first token
# alone gave counts about half of what a fleet needs to stream at a usable speed (T2), and the
# per-token model was then checked on a live T4 around this limit (C4: p95 within 7%). The
# shipped cost model still sets the first half only, so that every registered sweep — they read
# that file, not this command — means what it meant.
DEFAULT_TPOT_SLO_MS = 100.0


def _with_tpot_default(cost: CostModel | None, tpot_slo_ms: float) -> tuple[CostModel | None, bool]:
    """The cost model with this command's per-token limit filled in, if it sets none itself.
    Returns it and whether the limit came from the command. 0 (or less) leaves it off."""
    if cost is None or cost.tpot_p95_slo_ms is not None or tpot_slo_ms <= 0:
        return cost, False
    return dataclasses.replace(cost, tpot_p95_slo_ms=float(tpot_slo_ms)), True


def _slo_sentence(cost: CostModel | None, tpot_from_command: bool = False) -> str | None:
    """What "meets SLO" was decided on — and, when the per-token half is off, that it is."""
    if cost is None:
        return None
    if cost.tpot_p95_slo_ms is None:
        return (f"SLO: p95 time to first token ≤ {cost.ttft_p95_slo_ms:,.0f} ms. The time per "
                "output token is shown but NOT part of it (`--tpot-slo-ms 0`): read these replica "
                "counts as a lower bound for anything that has to stream at a usable speed.")
    both = (f"SLO: p95 time to first token ≤ {cost.ttft_p95_slo_ms:,.0f} ms AND p95 time per "
            f"output token ≤ {cost.tpot_p95_slo_ms:,.0f} ms.")
    if tpot_from_command:
        both += (" The per-token limit is this command's (`--tpot-slo-ms`; 0 sizes on the first "
                 "token alone); the cost model sets none.")
    return both


def _edge_line(e: dict[str, Any]) -> str:
    """One line on how close to the edge the recommended configuration sits (eval/edge.py)."""
    pct = round((e["slowdown"] - 1) * 100)

    def moved(label: str, pair: list, factor: float | None) -> str | None:
        if factor is None:
            return None
        return f"{label} {pair[0]:,.0f} → {pair[1]:,.0f} ms (×{factor:.2f})"

    parts = [m for m in (moved("p95 TTFT", e["p95_ttft_ms"], e["ttft_factor"]),
                         moved("p95 time per token", e["p95_tpot_ms"], e["tpot_factor"])) if m]
    what = "; ".join(parts)
    if e["verdict"] == "knee":
        return (f"⛔ at the knee: with timings {pct}% slower, {what}. A {pct}% change in the "
                f"timings moves a p95 by more than {round((e['knee_factor'] - 1) * 100)}%: the "
                "fleet is close to saturation here, where the simulator's latencies were "
                "1.5–1.7× off on a live GPU. Do not trust this count: size at least one replica "
                "higher, and measure.")
    if e["verdict"] == "edge":
        return (f"⚠️ at the edge: with timings {pct}% slower this configuration no longer meets "
                f"the SLO ({what}). Two calibrations of the same GPU differ by that much: allow "
                "one replica more.")
    if e["verdict"] == "not_met":
        return f"with timings {pct}% slower: {what}."
    return f"not at the edge: with timings {pct}% slower it still meets the SLO ({what})."


def _markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    rec = payload.get("recommendation") or {}
    if rec.get("recommended"):
        r = rec["recommended"]
        lines += [
            "## Recommendation",
            "",
            f"**{r['policy']} on {r['n_replicas']} replica(s)** — {rec.get('reason', '')}",
            "",
        ]
        if rec.get("tied_with"):
            lines.append(f"tied on replicas with: {', '.join(rec['tied_with'])} "
                         "(between these the rupee gap is noise)")
            lines.append("")
        if rec.get("inr_delta_pct_vs_baseline") is not None:
            lines.append(
                f"vs round-robin: {rec['inr_delta_pct_vs_baseline']:+.1f}% cost, "
                f"{rec['replica_delta_vs_baseline']:+d} replicas"
            )
            lines.append("")
        firm = ((payload.get("replica_spread") or {}).get("replicas") or {}).get(r["policy"])
        if firm:
            lines += [_firmness(firm, payload["replica_spread"]), ""]
        if payload.get("edge"):
            lines += [_edge_line(payload["edge"]), ""]
        if rec.get("WARNING"):
            lines += [f"> ⚠️ {rec['WARNING']}", ""]

    pts = payload.get("points") or []
    if pts:
        lines += [
            "## All configurations",
            "",
            ("| policy | replicas | hit rate | p50 TTFT ms | p95 TTFT ms | p95 ms per token | "
             "INR/1k | imbalance | meets SLO |"),
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for p in pts:
            inr = "—" if p["inr_per_1k_requests"] is None else f"{p['inr_per_1k_requests']:.3f}"
            tpot = "—" if p.get("p95_tpot_ms") is None else f"{p['p95_tpot_ms']:.1f}"
            lines.append(
                f"| {p['policy']} | {p['n_replicas']} | {p['hit_rate']:.4f} | "
                f"{p['p50_ttft_ms']:.1f} | {p['p95_ttft_ms']:.1f} | {tpot} | {inr} | "
                f"{p['load_imbalance']:.3f} | {_slo_mark(p['meets_slo'])} |"
            )
        lines.append("")
        if payload.get("slo"):
            lines += [payload["slo"], ""]
    spread = payload.get("replica_spread")
    if spread and spread.get("replicas"):
        top = spread["swept_up_to"]
        lines += [
            "## How firm are the replica counts?",
            "",
            (f"The calibration measured every timing cell {spread['repeats']} times. The same "
             "sizing, on the fit of each repeat alone:"),
            "",
            "| policy | all repeats together | each repeat alone |",
            "|---|---:|---|",
        ]
        for policy, row in spread["replicas"].items():
            lines.append(f"| {policy} | {_count(row['pooled'], top)} | "
                         f"{' · '.join(_count(n, top) for n in row['repeats'])} |")
        lines += [
            "",
            ("This is what ONE calibration's own repeats disagree by — a floor on the "
             "uncertainty, not a confidence interval, and a low floor. Measured on a live T4 "
             "(C4, 2026-10-05): another run of the calibration, within the hour, gave a count "
             "outside a run's own range in 18% of cases, and outside it by more than one replica "
             "in 5%. So size for the top of the range, plus one. And it knows nothing about "
             "another day or another software image: three calibrations of our own T4 agreed "
             "exactly on 71% of counts and within one replica on 96%."),
            "",
        ]
    elif payload.get("replica_spread_note"):
        lines += [f"> ⚠️ {payload['replica_spread_note']}", ""]
    if payload.get("trace"):
        lines += ["## Trace", "", "```json", json.dumps(payload["trace"], indent=2), "```", ""]
    return "\n".join(lines)


def cmd_replay(args: argparse.Namespace) -> int:
    names = policy_pkg.resolve(args.policies)
    replicas = _parse_replicas(args.replicas)
    cost, tpot_from_command = _with_tpot_default(_load_cost(args.cost_model), args.tpot_slo_ms)
    profile = _load_profile(args.profile)
    if args.block_size != profile.block_size:
        # Hashing the trace at one block size and simulating a cache of another would compare
        # nothing with nothing — fail before any work (ADR-004).
        raise SystemExit(
            f"error: --block-size {args.block_size} does not match the profile's block size "
            f"{profile.block_size} ({profile.name})"
        )

    stats = TraceStats()
    reader = adapters.get(args.format_in)
    trace = list(reader.read(args.trace, block_size=args.block_size, stats=stats))
    if not trace:
        raise SystemExit(
            f"error: no usable requests in {args.trace!r} "
            f"({stats.skipped_lines} line(s) skipped: {stats.skip_reasons})"
        )
    info = stats.as_dict()

    points, _ = sweep(
        trace,
        policies=names,
        replica_counts=replicas,
        seeds=tuple(range(args.seeds)),
        profile=profile,
        cost=cost,
        trace_info=info,
    )
    rec = recommend(points, **({"slo": cost.slo_label} if cost is not None else {}))
    if not rec.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    # How firm is the count? Only a measured profile and an SLO make the question meaningful.
    spread, spread_note = None, None
    if cost is not None and profile.calibrated and not args.no_spread:
        replicates = replicate_profiles(args.profile)
        if replicates:
            spread = replica_spread(trace, policies=names, replica_counts=replicas,
                                    seeds=tuple(range(args.seeds)), cost=cost,
                                    pooled_points=points, replicates=replicates, trace_info=info)
        else:
            spread_note = (
                "This profile carries no per-repeat fits, so how firm these replica counts are "
                "cannot be shown. For scale: three calibrations of our own T4 agreed on a count "
                "exactly at 71% of points and within one replica at 96%. A profile written by "
                "`kvsim calibrate` shows its own spread — a floor: allow one replica more.")
    # How close to the edge? Only where a configuration was recommended because it met an SLO.
    edge = None
    if (cost is not None and rec.chosen is not None and rec.chosen.meets_slo
            and not args.no_edge_check):
        edge = edge_check(trace, chosen=rec.chosen, seeds=tuple(range(args.seeds)),
                          profile=profile, cost=cost, trace_info=info)
    _emit(
        {
            "trace": info,
            "profile": profile.provenance(),
            "points": [p.as_dict() for p in points],
            "recommendation": rec.as_dict(),
            "slo": _slo_sentence(cost, tpot_from_command),
            "replica_spread": spread,
            "replica_spread_note": spread_note,
            "edge": edge,
        },
        args.format,
        args.out,
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    wl = AgenticWorkload(
        n_sessions=args.sessions,
        turns_per_session=args.turns,
        prefix_sharing_ratio=args.sharing,
        arrival_rate=args.rate,
        seed=args.seed,
    )
    report = wl.write_jsonl(args.out)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def cmd_crossover(args: argparse.Namespace) -> int:
    names = policy_pkg.resolve(args.policies)
    profile = _load_profile(args.profile)
    rows = crossover_sweep(
        policies=names,
        sharing_ratios=[float(x) for x in args.sharing.split(",")],
        arrival_rates=[float(x) for x in args.rates.split(",")],
        n_replicas=args.replicas,
        seeds=tuple(range(args.seeds)),
        n_sessions=args.sessions,
        profile=profile,
    )
    if not profile.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    _emit({"profile": profile.provenance(), "crossover": rows}, "json", args.out)
    return 0


def _ints(spec: str, flag: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise SystemExit(f"error: could not parse {flag} {spec!r}; use e.g. '1,2,4,8'") from None
    if not out:
        raise SystemExit(f"error: {flag} is empty")
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit a profile to a running vLLM, check the cache model against it, write the profile."""
    from .calib.calibrate import CalibrationError, Engine, calibrate

    lo_hi = _ints(args.token_range, "--token-range")
    if len(lo_hi) != 2 or not 0 <= lo_hi[0] < lo_hi[1]:
        raise SystemExit("error: --token-range must be two integers LO,HI with LO < HI")
    if args.decode_tokens < 2:
        raise SystemExit("error: --decode-tokens must be at least 2 (it measures between tokens)")
    try:
        profile, ok = calibrate(
            Engine(args.engine, timeout_s=args.timeout),
            model=args.model, gpu=args.gpu, name=args.name, kv_blocks=args.kv_blocks,
            block_size=args.block_size, max_num_seqs=args.max_num_seqs,
            prefill_lengths=_ints(args.prefill_lengths, "--prefill-lengths"), reps=args.reps,
            prefill_cached=_ints(args.prefill_cached, "--prefill-cached"),
            prefill_new=_ints(args.prefill_new, "--prefill-new"),
            decode_batches=_ints(args.decode_batches, "--decode-batches"),
            decode_contexts=_ints(args.decode_contexts, "--decode-contexts"),
            decode_tokens=args.decode_tokens, check_sessions=args.check_sessions,
            check_turns=args.check_turns, max_error=args.max_error, seed=args.seed,
            token_range=(lo_hi[0], lo_hi[1]),
        )
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")
    if ok:
        print(f"wrote {args.out} — calibrated. Use it: kvsim replay <trace> --profile {args.out}")
        return 0
    chk = profile["_source"]["check"]
    why = (f"it missed this engine's hit rate by {profile['calibration_error']:.2%} (limit "
           f"{args.max_error:.0%})" if profile["calibration_error"] > args.max_error else
           f"{chk['cached_tokens_compared'] - chk['cached_tokens_equal']} of "
           f"{chk['cached_tokens_compared']} requests' cached-token counts differ")
    print(
        f"wrote {args.out} — NOT calibrated: the cache model does not predict this engine; {why}. "
        "Its timings are measured, but its hit rates would not be. Check --block-size and the "
        "engine version; details in _source.check.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kvsim", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("replay", help="replay a trace through routing policies")
    r.add_argument("trace")
    r.add_argument("--policies", default="all",
                   help="comma-separated names, or 'all' for the seven standard policies; the two "
                        "ported published ones are asked for by name: lmetric, smetric-local")
    r.add_argument("--replicas", default="1-8")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--block-size", type=int, default=16)
    r.add_argument("--format-in", default="vllm-jsonl", choices=sorted(adapters.REGISTRY))
    r.add_argument("--cost-model", default=None,
                   help="path to a cost_model.yaml, or 'default' for the shipped, cited one")
    r.add_argument("--profile", default=None,
                   help="a profile .json, or a shipped profile's name: t4-qwen3-4b-vllm0.30.0")
    r.add_argument("--tpot-slo-ms", type=float, default=DEFAULT_TPOT_SLO_MS,
                   help="the SLO's second half: p95 time per output token allowed, in ms, when the "
                        "cost model sets no `tpot_p95_ms` itself (default: %(default)g, the limit "
                        "MLPerf pairs with a 2 s first token). 0 sizes on the first token alone")
    r.add_argument("--format", default="md", choices=("md", "json"))
    r.add_argument("--out", default=None)
    r.add_argument("--no-edge-check", action="store_true",
                   help="skip the check of how close to saturation the recommended configuration "
                        "sits (one extra run per seed, on timings 5%% slower)")
    r.add_argument("--no-spread", action="store_true",
                   help="skip re-sizing on each repeat of the calibration (one extra sweep per "
                        "repeat); the replica counts are then shown without their spread")
    r.set_defaults(func=cmd_replay)

    g = sub.add_parser("generate", help="write a synthetic agentic trace")
    g.add_argument("--out", required=True)
    g.add_argument("--sessions", type=int, default=200)
    g.add_argument("--turns", type=int, default=6)
    g.add_argument("--sharing", type=float, default=0.6)
    g.add_argument("--rate", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("crossover", help="the H2 experiment: sharing x load -> winner")
    c.add_argument("--policies", default="least-loaded,radix-affinity-balanced,session-sticky")
    c.add_argument("--sharing", default="0.1,0.3,0.5,0.7,0.9")
    c.add_argument("--rates", default="4,8,16,32")
    c.add_argument("--replicas", type=int, default=4)
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--sessions", type=int, default=150)
    c.add_argument("--profile", default=None,
                   help="a profile .json, or a shipped profile's name: t4-qwen3-4b-vllm0.30.0")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_crossover)

    cal = sub.add_parser(
        "calibrate", help="fit a timing profile to YOUR running vLLM; check the cache model on it"
    )
    cal.add_argument("--engine", default="http://localhost:8000", help="the vLLM server's URL")
    cal.add_argument("--out", required=True, help="where to write the profile (.json)")
    cal.add_argument("--model", default=None, help="served model name (default: /v1/models)")
    cal.add_argument("--gpu", default=None, help="GPU name for the profile (default: nvidia-smi "
                     "when the engine is local)")
    cal.add_argument("--name", default=None, help="profile name (default: gpu-model-vllmVERSION)")
    cal.add_argument("--kv-blocks", type=int, default=None,
                     help="KV capacity in blocks, if the engine does not report it")
    cal.add_argument("--block-size", type=int, default=None)
    cal.add_argument("--max-num-seqs", type=int, default=None)
    cal.add_argument("--prefill-lengths", default="64,128,256,512,768,1024,1536,2048")
    cal.add_argument("--prefill-cached", default="256,768,1280",
                     help="cached-prefix lengths: new tokens are timed after each (0 = none)")
    cal.add_argument("--prefill-new", default="64,256,704",
                     help="new-token counts timed after each cached prefix")
    cal.add_argument("--reps", type=int, default=3,
                     help="repeats of every timing cell; the profile keeps the fit of each")
    cal.add_argument("--decode-batches", default="1,2,4,8,16")
    cal.add_argument("--decode-contexts", default="32,512,1024",
                     help="prompt lengths for the decode cells: they fit the context term")
    cal.add_argument("--decode-tokens", type=int, default=64)
    cal.add_argument("--check-sessions", type=int, default=16)
    cal.add_argument("--check-turns", type=int, default=3)
    cal.add_argument("--max-error", type=float, default=0.05,
                     help="largest relative hit-rate error that still counts as calibrated")
    cal.add_argument("--token-range", default="1000,30000",
                     help="token ids for the random prompts, LO,HI (must be valid for the model)")
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")
    cal.set_defaults(func=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
