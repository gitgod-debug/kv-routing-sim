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
import json
import sys
from pathlib import Path
from typing import Any

from . import policies as policy_pkg
from .eval.cost import CostModel, UnfilledCostModel
from .eval.recommend import recommend
from .eval.sweep import crossover_sweep, sweep
from .sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile, resolve_profile
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
        if rec.get("WARNING"):
            lines += [f"> ⚠️ {rec['WARNING']}", ""]

    pts = payload.get("points") or []
    if pts:
        lines += [
            "## All configurations",
            "",
            "| policy | replicas | hit rate | p50 ms | p95 ms | INR/1k | imbalance | meets SLO |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for p in pts:
            inr = "—" if p["inr_per_1k_requests"] is None else f"{p['inr_per_1k_requests']:.3f}"
            lines.append(
                f"| {p['policy']} | {p['n_replicas']} | {p['hit_rate']:.4f} | "
                f"{p['p50_ttft_ms']:.1f} | {p['p95_ttft_ms']:.1f} | {inr} | "
                f"{p['load_imbalance']:.3f} | {_slo_mark(p['meets_slo'])} |"
            )
        lines.append("")
    if payload.get("trace"):
        lines += ["## Trace", "", "```json", json.dumps(payload["trace"], indent=2), "```", ""]
    return "\n".join(lines)


def cmd_replay(args: argparse.Namespace) -> int:
    names = policy_pkg.resolve(args.policies)
    replicas = _parse_replicas(args.replicas)
    cost = _load_cost(args.cost_model)
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
    rec = recommend(points)
    if not rec.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    _emit(
        {
            "trace": info,
            "profile": profile.provenance(),
            "points": [p.as_dict() for p in points],
            "recommendation": rec.as_dict(),
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
    r.add_argument("--policies", default="all")
    r.add_argument("--replicas", default="1-8")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--block-size", type=int, default=16)
    r.add_argument("--format-in", default="vllm-jsonl", choices=sorted(adapters.REGISTRY))
    r.add_argument("--cost-model", default=None,
                   help="path to a cost_model.yaml, or 'default' for the shipped, cited one")
    r.add_argument("--profile", default=None,
                   help="a profile .json, or a shipped profile's name: t4-qwen3-4b-vllm0.30.0")
    r.add_argument("--format", default="md", choices=("md", "json"))
    r.add_argument("--out", default=None)
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
    cal.add_argument("--reps", type=int, default=3, help="repeats of each prefill cell")
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
