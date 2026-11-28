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
import math
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from .eval.sweep import crossover_sweep
from .report import render_markdown, to_json
from .service import (
    DEFAULT_TPOT_SLO_MS,
    ConfigError,
    ReplayOptions,
    check_fleet,
    load_profile,
    replay,
    resolve_policies,
)
from .traces import adapters
from .traces.generator import AgenticWorkload

PROFILE_HELP = (
    "a profile .json, or a shipped profile's name: t4-qwen3-4b-vllm0.30.0-ctx (checked against "
    "live latency) or t4-qwen3-4b-vllm0.30.0 (the first one, optimistic on long prompts)"
)

UNCALIBRATED_BANNER = """
Warning: uncalibrated cost profile; no live engine has been measured.
  Usable:     as a first look at how the routing policies differ on this traffic.
  Not usable: the millisecond and rupee figures, which are shapes, not predictions; nor the
              policy ranking and the replica count as they stand: both change with the hardware.
  Fix:        kvsim calibrate --engine <your vLLM URL> --out mine.json, then --profile mine.json
"""


def cmd_replay(args: argparse.Namespace) -> int:
    _check_out(args.out)
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
        token_scale=args.token_scale,
    )
    report = replay(args.trace, options)
    if not report.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    payload = report.as_dict()
    _emit(to_json(payload, indent=2) if args.format == "json" else render_markdown(payload),
          args.out)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    _check_out(args.out, required=True)
    try:
        workload = AgenticWorkload(
            n_sessions=args.sessions,
            turns_per_session=args.turns,
            prefix_sharing_ratio=args.sharing,
            arrival_rate=args.rate,
            seed=args.seed,
            n_shared_prefixes=args.system_prompts,
        )
        summary = workload.write_jsonl(args.out)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    except OSError as exc:
        raise ConfigError(f"could not write {args.out!r}: {exc}") from None
    print(to_json(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def cmd_crossover(args: argparse.Namespace) -> int:
    names = resolve_policies(args.policies)
    sharing = _floats(args.sharing, "--sharing")
    if not all(0.0 <= x < 1.0 for x in sharing):
        raise ConfigError("--sharing values must be at least 0 and below 1")
    rates = _floats(args.rates, "--rates")
    if not all(x > 0 for x in rates):
        raise ConfigError("--rates must be numbers above 0, in sessions per second")
    for flag, value in (("--replicas", args.replicas), ("--seeds", args.seeds),
                        ("--sessions", args.sessions)):
        if value < 1:
            raise ConfigError(f"{flag} must be at least 1")
    _check_out(args.out)
    profile = load_profile(args.profile)
    check_fleet(args.replicas, profile)
    try:
        rows = crossover_sweep(
            policies=names,
            sharing_ratios=sharing,
            arrival_rates=rates,
            n_replicas=args.replicas,
            seeds=tuple(range(args.seeds)),
            n_sessions=args.sessions,
            profile=profile,
        )
    except ValueError as exc:  # traffic the generator cannot build, or the profile cannot hold
        raise ConfigError(str(exc)) from None
    if not profile.calibrated:
        print(UNCALIBRATED_BANNER, file=sys.stderr)
    payload: dict[str, Any] = {"profile": profile.provenance(), "crossover": rows}
    _emit(to_json(payload, indent=2), args.out)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit a profile to a running vLLM, check the cache model against it, and write the profile."""
    from .calib.calibrate import CalibrationError, Engine, calibrate

    _check_engine(args.engine)
    # Checked now: the measurements take minutes, and a profile that cannot be written loses them.
    _check_out(args.out, required=True)
    lo_hi = _ints(args.token_range, "--token-range")
    if len(lo_hi) != 2 or not 0 <= lo_hi[0] < lo_hi[1]:
        raise ConfigError("--token-range must be two integers LO,HI with LO < HI")
    if args.decode_tokens < 2:
        raise ConfigError("--decode-tokens must be at least 2 (it measures between tokens)")
    # The same for every other option: a bad value must not surface minutes into the run.
    lists = {flag: _ints(getattr(args, flag.lstrip("-").replace("-", "_")), flag)
             for flag in ("--prefill-lengths", "--prefill-cached", "--prefill-new",
                          "--decode-batches", "--decode-contexts")}
    for flag, values in lists.items():
        if any(v < 1 for v in values):
            raise ConfigError(f"{flag} must be whole numbers of at least 1")
    for flag, value, least in (("--reps", args.reps, 1),
                               ("--check-sessions", args.check_sessions, 1),
                               ("--check-turns", args.check_turns, 1),
                               ("--kv-blocks", args.kv_blocks, 2),
                               ("--block-size", args.block_size, 1),
                               ("--max-num-seqs", args.max_num_seqs, 1)):
        if value is not None and value < least:
            raise ConfigError(f"{flag} must be at least {least}")
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        raise ConfigError("--timeout must be a number of seconds above 0")
    if not (math.isfinite(args.max_error) and args.max_error >= 0):
        raise ConfigError("--max-error must be a number of at least 0")
    try:
        profile, ok = calibrate(
            Engine(args.engine, timeout_s=args.timeout),
            model=args.model, gpu=args.gpu, name=args.name, kv_blocks=args.kv_blocks,
            block_size=args.block_size, max_num_seqs=args.max_num_seqs,
            prefill_lengths=lists["--prefill-lengths"], reps=args.reps,
            prefill_cached=lists["--prefill-cached"],
            prefill_new=lists["--prefill-new"],
            decode_batches=lists["--decode-batches"],
            decode_contexts=lists["--decode-contexts"],
            decode_tokens=args.decode_tokens, check_sessions=args.check_sessions,
            check_turns=args.check_turns, max_error=args.max_error, seed=args.seed,
            token_range=(lo_hi[0], lo_hi[1]),
        )
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(profile, indent=1) + "\n"
    try:
        Path(args.out).write_text(text, encoding="utf-8")
    except OSError as exc:
        # The measurements took minutes. Print the profile so that they are not lost.
        print(text)
        print(f"error: could not write {args.out!r}: {exc}. The profile is printed above: save "
              "it to a file.", file=sys.stderr)
        return 2
    if ok:
        print(f"wrote {args.out} — calibrated. Use it: kvsim replay <trace> --profile {args.out}")
        return 0
    check = profile["_source"]["check"]
    if profile["calibration_error"] is None:
        print(f"wrote {args.out}, not calibrated: the engine reported no prefix-cache hit during "
              "the check, so the cache model could not be compared with it. Its timings are "
              "measured, but its hit rates would not be. Check that prefix caching is on, and use "
              "more --check-sessions or --check-turns. Details are in _source.check.",
              file=sys.stderr)
        return 1
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


def cmd_profiles(args: argparse.Namespace) -> int:
    """List the measured profiles, or say how far a setup is from the nearest one."""
    from .profiles import measured_profiles, nearest, render_distance, render_list, setup_of

    given = {"gpu": args.gpu, "model": args.model, "engine_version": args.engine_version,
             "block_size": args.block_size, "max_num_seqs": args.max_num_seqs}
    if args.engine:
        from .calib.calibrate import CalibrationError, Engine, probe

        _check_engine(args.engine)
        try:
            info = probe(Engine(args.engine), model=args.model, block_size=args.block_size,
                         max_num_seqs=args.max_num_seqs)
        except CalibrationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        read = {"model": info.model, "engine_version": info.version,
                "block_size": info.block_size, "max_num_seqs": info.max_num_seqs}
        given = {k: v if v is not None else read.get(k) for k, v in given.items()}
    if not any(v is not None for v in given.values()):
        print(render_list(measured_profiles()))
        return 0
    yours = setup_of(**given)
    print(render_distance(yours, *nearest(yours)))
    return 0


def _ints(spec: str, flag: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers given for `flag`."""
    try:
        values = tuple(int(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise ConfigError(f"could not parse {flag} {spec!r}; use e.g. '1,2,4,8'") from None
    if not values:
        raise ConfigError(f"{flag} is empty")
    return values


def _floats(spec: str, flag: str) -> tuple[float, ...]:
    """Parse a comma-separated list of finite numbers given for `flag`."""
    try:
        values = tuple(float(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise ConfigError(f"could not parse {flag} {spec!r}. Use numbers such as "
                          "'0.1,0.5'") from None
    if not values:
        raise ConfigError(f"{flag} is empty")
    if not all(math.isfinite(v) for v in values):
        raise ConfigError(f"{flag} must be finite numbers")
    return values


def _check_out(out: str | None, *, required: bool = False) -> None:
    """Refuse an output path that cannot be written, before anything runs."""
    if not out:
        if required:
            raise ConfigError("--out needs a file name")
        return
    path = Path(out)
    if path.is_dir():
        raise ConfigError(f"--out {out!r} is a folder. Give a file name.")
    if not path.parent.exists():
        raise ConfigError(f"--out {out!r}: the folder {str(path.parent)!r} does not exist")
    if not path.parent.is_dir():
        raise ConfigError(f"--out {out!r}: {str(path.parent)!r} is not a folder")
    if not os.access(path if path.exists() else path.parent, os.W_OK):
        raise ConfigError(f"--out {out!r}: no permission to write there")


def _check_engine(url: str) -> None:
    example = "for example http://localhost:8000"
    if not url.lower().startswith(("http://", "https://")):
        raise ConfigError(f"--engine {url!r} must start with http:// or https://, {example}")
    try:
        parts = urllib.parse.urlsplit(url)
        host, _ = parts.hostname, parts.port  # .port raises for a port that is not a number
    except ValueError:
        host = None
    if not host or any(c.isspace() for c in url):
        raise ConfigError(f"--engine {url!r} is not a server address, {example}")


def _emit(text: str, out: str | None) -> None:
    if out:
        try:
            Path(out).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not write {out!r}: {exc}") from None
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
    r.add_argument("--token-scale", type=float, default=None,
                   help="factor on every token count, for the agentx, bailian and mooncake "
                        "formats (default: the reader's own, 1/32 for agentx and 1 otherwise)")
    r.add_argument("--cost-model", default=None,
                   help="path to a cost_model.yaml, or 'default' for the shipped, cited one")
    r.add_argument("--profile", default=None, help=PROFILE_HELP)
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
    c.add_argument("--profile", default=None, help=PROFILE_HELP)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_crossover)

    p = sub.add_parser("profiles", help="the measured profiles, and how far a setup is from them")
    p.add_argument("--gpu", default=None, help='as nvidia-smi names it, e.g. "Tesla T4"')
    p.add_argument("--model", default=None)
    p.add_argument("--engine-version", default=None, help="vLLM's version, e.g. 0.30.0")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--engine", default=None,
                   help="a running vLLM to read the model, version, block size and max_num_seqs "
                        "from; the GPU is not served over HTTP, so give --gpu as well")
    p.set_defaults(func=cmd_profiles)

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
    except MemoryError:
        raise SystemExit("error: this run needs more memory than this machine has. Each replica "
                         "keeps its own cache, so use fewer replicas, a shorter trace or a "
                         "profile with fewer KV blocks.") from None


if __name__ == "__main__":
    raise SystemExit(main())
