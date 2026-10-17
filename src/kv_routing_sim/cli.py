"""The `kvsim` command line.

    kvsim replay    trace.jsonl --policies all --replicas 1-8 --format md \
                    --profile t4-qwen3-4b-vllm0.30.0-ctx --cost-model default
    kvsim generate  --sharing 0.6 --out trace.jsonl
    kvsim crossover --sharing 0.1,0.3,0.5,0.7,0.9 --rates 4,8,16 --profile <profile.json>
    kvsim calibrate --engine http://localhost:8000 --out configs/profiles/mine.json

Invalid options fail at startup, before any simulation runs (ADR-004).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .eval.sweep import crossover_sweep
from .report import render_markdown
from .service import (
    DEFAULT_TPOT_SLO_MS,
    ConfigError,
    ReplayOptions,
    load_profile,
    replay,
    resolve_policies,
)
from .traces import adapters
from .traces.generator import AgenticWorkload

UNCALIBRATED_BANNER = """
Warning: uncalibrated cost profile; no live engine has been measured.
  Usable:     the policy ranking and the replica count, as a rough guide for other hardware.
  Not usable: the millisecond and rupee figures, which are shapes, not predictions.
  Fix:        kvsim calibrate --engine <your vLLM URL> --out mine.json, then --profile mine.json
"""


def cmd_replay(args: argparse.Namespace) -> int:
    options = ReplayOptions(
        policies=args.policies,
        replicas=args.replicas,
        seeds=args.seeds,
        block_size=args.block_size,
        trace_format=args.format_in,
        cost_model=args.cost_model,
        profile=args.profile,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        gpu_price_per_hour_usd=args.gpu_price_usd,
        edge_check=not args.no_edge_check,
        spread=not args.no_spread,
    )
    report = replay(args.trace, options)
    if not report.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    payload = report.as_dict()
    _emit(json.dumps(payload, indent=2) if args.format == "json" else render_markdown(payload),
          args.out)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    try:
        workload = AgenticWorkload(
            n_sessions=args.sessions,
            turns_per_session=args.turns,
            prefix_sharing_ratio=args.sharing,
            arrival_rate=args.rate,
            seed=args.seed,
            n_shared_prefixes=args.system_prompts,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    summary = workload.write_jsonl(args.out)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def cmd_crossover(args: argparse.Namespace) -> int:
    names = resolve_policies(args.policies)
    profile = load_profile(args.profile)
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
    payload: dict[str, Any] = {"profile": profile.provenance(), "crossover": rows}
    _emit(json.dumps(payload, indent=2), args.out)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit a profile to a running vLLM, check the cache model against it, and write the profile."""
    from .calib.calibrate import CalibrationError, Engine, calibrate

    lo_hi = _ints(args.token_range, "--token-range")
    if len(lo_hi) != 2 or not 0 <= lo_hi[0] < lo_hi[1]:
        raise ConfigError("--token-range must be two integers LO,HI with LO < HI")
    if args.decode_tokens < 2:
        raise ConfigError("--decode-tokens must be at least 2 (it measures between tokens)")
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
    check = profile["_source"]["check"]
    if profile["calibration_error"] > args.max_error:
        why = (f"it missed this engine's hit rate by {profile['calibration_error']:.2%} (limit "
               f"{args.max_error:.0%})")
    else:
        differing = check["cached_tokens_compared"] - check["cached_tokens_equal"]
        why = (f"{differing} of {check['cached_tokens_compared']} requests' cached-token counts "
               "differ")
    print(
        f"wrote {args.out} — not calibrated: the cache model does not predict this engine; {why}. "
        "Its timings are measured, but its hit rates would not be. Check --block-size and the "
        "engine version; details in _source.check.",
        file=sys.stderr,
    )
    return 1


def _ints(spec: str, flag: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers given for `flag`."""
    try:
        values = tuple(int(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise ConfigError(f"could not parse {flag} {spec!r}; use e.g. '1,2,4,8'") from None
    if not values:
        raise ConfigError(f"{flag} is empty")
    return values


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvsim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

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
    r.add_argument("--ttft-slo-ms", type=float, default=None,
                   help="p95 time to first token allowed, in ms; replaces the cost model's limit")
    r.add_argument("--tpot-slo-ms", type=float, default=DEFAULT_TPOT_SLO_MS,
                   help="p95 time per output token allowed, in ms, when the cost model sets no "
                        "`tpot_p95_ms` itself (default: %(default)g, the limit MLPerf pairs with "
                        "a 2 s first token). 0 sizes on the first token alone")
    r.add_argument("--gpu-price-usd", type=float, default=None,
                   help="price of one GPU per hour in US dollars; replaces the cost model's")
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
    g.add_argument("--system-prompts", type=int, default=1,
                   help="distinct shared system prompts (tenants), drawn Zipf-like per session")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("crossover", help="which policy wins across prefix sharing and load")
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
        "calibrate", help="fit a timing profile to your running vLLM and check the cache model"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    raise SystemExit(main())
