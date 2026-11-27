#!/usr/bin/env python3
"""Post hoc: every registered prediction, computed again with the engine of release 0.2.0.

    python scripts/registered_predictions_again.py

Each prediction here was registered before its GPU run, and the engine of release 0.1.2 made it.
Release 0.2.0 fixed that engine (configs/rederive_0_2_0.yaml). This script computes every
prediction twice from the committed code: with the old engine, which must give the registered
numbers exactly, and with the fixed one. It lists each number that the fix moves. The registered
numbers stay as they are, because they are the record of what was predicted.

It also evaluates the check registered in configs/rederive_0_2_0.yaml: the hit rates and cached
tokens of H3, H4, V1, C1 and Phase 6 must not move. This script was written after that result had
been seen, so its report is labelled post hoc.

Writes artifacts/results/engine_0_2_0/registered_predictions.json and report.md.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import sys
from collections.abc import Callable
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _spike_common import results_header

from kv_routing_sim.calib.calibrate import check_workload, predict_check
from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.sim.engine import run, use_engine
from kv_routing_sim.traces.generator import AgenticWorkload

RESULTS = REPO / "artifacts" / "results"
OUT = RESULTS / "engine_0_2_0"
# The fields of the registered check: cache counts, which the fix was not meant to move.
CACHE_WORDS = ("hit", "quer", "cached", "local_compute")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def phase3() -> tuple[Any, Any]:
    prep = _script("phase3_prepare")
    prereg = _json(prep.PREREG)
    return prereg["predictions"], lambda: prep.make_predictions(prereg["config"])[3]


def c1() -> tuple[Any, Any]:
    prereg = _json(RESULTS / "c1_prereg" / "prediction.json")
    cfg = prereg["config"]
    c = cfg["calibrate"]

    def make():
        work = check_workload(random.Random(f"check-{c['seed']}"), c["check_sessions"],
                              c["check_turns"], tuple(c["token_range"]))
        sim = predict_check(work, load_profile(REPO / cfg["phase3_profile"]))
        return {"hits": sim.cache_hits, "queries": sim.cache_queries, "requests": len(work),
                "cached_tokens": [sim.per_request[f"c{i}"]["cached_tokens"]
                                  for i in range(len(work))]}

    p = prereg["prediction"]
    return {k: p[k] for k in ("hits", "queries", "requests", "cached_tokens")}, make


def c2() -> tuple[Any, Any]:
    prereg = _json(RESULTS / "c2_prereg" / "predictions.json")
    profile = load_profile(REPO / prereg["config"]["profile"])

    def make():
        out = {}
        for name, spec in prereg["workloads"].items():
            trace, _ = AgenticWorkload(**spec).generate()
            r = run(trace, policy="round-robin", n_replicas=1, seed=0, profile=profile,
                    cache_model="vllm")
            pct = r.ttft_percentiles()
            out[name] = {"context-free profile": {
                "hit_rate": round(r.hit_rate, 6), "prefix_cache_queries": r.cache_queries,
                "prefix_cache_hits": r.cache_hits, "preemptions": r.preemptions,
                "ttft_p50_ms": pct["p50_ms"], "ttft_p95_ms": pct["p95_ms"]}}
        return out

    return prereg["predictions"]["runs"], make


def c3() -> tuple[Any, Any]:
    prep = _script("c3_prepare")
    prereg = _json(RESULTS / "c3_prereg" / "predictions.json")
    cfg = prereg["config"]
    profile = load_profile(REPO / cfg["profile"])

    def make():
        runs = {name: prep.predict_holdout(list(AgenticWorkload(**spec).iter_with_tokens()),
                                           profile)
                for name, spec in prereg["workloads"].items()}
        checks = {str(s): prep.predict_check(cfg, s) for s in cfg["calibrate"]["seeds"]}
        return {"runs": runs, "check": checks}

    want = {"runs": {n: prereg["predictions"]["runs"][n]["C2 profile"]
                     for n in prereg["workloads"]},
            "check": prereg["predictions"]["check"]}
    return want, make


def c4() -> tuple[Any, Any]:
    prep = _script("c4_prepare")
    prereg = _json(RESULTS / "c4_prereg" / "predictions.json")
    cfg = prereg["config"]
    profile = load_profile(REPO / cfg["profile"])

    def make():
        traces = {n: list(AgenticWorkload(**s).iter_with_tokens())
                  for n, s in prereg["workloads"].items()}
        return prep.predictions(cfg, traces, profile)

    return prereg["predictions"], make


def phase6() -> tuple[Any, Any]:
    prereg = _json(RESULTS / "phase6_prereg" / "predictions.json")
    cfg = yaml.safe_load((REPO / "configs" / "phase6_sim2real.yaml").read_text())
    profile = load_profile(REPO / cfg["profile"])

    def make():
        out = {}
        for name, spec in cfg["workloads"].items():
            trace, _ = AgenticWorkload(**spec).generate()
            out[name] = {}
            for policy in cfg["policies"]:
                r = run(trace, policy=policy, n_replicas=cfg["replicas"],
                        seed=cfg["router_seed"], profile=profile, cache_model="vllm")
                pct = r.ttft_percentiles()
                out[name][policy] = {
                    "hit_rate": round(r.hit_rate, 6), "prefix_cache_queries": r.cache_queries,
                    "prefix_cache_hits": r.cache_hits,
                    "requests_per_replica": r.per_replica_requests,
                    "preemptions": r.preemptions, "ttft_p50_ms": pct["p50_ms"],
                    "ttft_p95_ms": pct["p95_ms"]}
        return out

    return prereg["predictions"]["runs"], make


# Test, where its prediction was registered, and whether the registered check names it.
SOURCES: dict[str, tuple[str, Callable[[], tuple[Any, Any]], bool]] = {
    "Phase 3 (H3, H4; V1 registered the same predictions)":
        ("artifacts/results/phase3_prereg/predictions.json", phase3, True),
    "C1": ("artifacts/results/c1_prereg/prediction.json", c1, True),
    "C2": ("artifacts/results/c2_prereg/predictions.json", c2, False),
    "C3": ("artifacts/results/c3_prereg/predictions.json", c3, False),
    "C4": ("artifacts/results/c4_prereg/predictions.json", c4, False),
    "Phase 6": ("artifacts/results/phase6_prereg/predictions.json", phase6, True),
}


def moved(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    """Every leaf where the two differ, with its path."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [m for k in sorted(set(old) | set(new), key=str)
                for m in moved(old.get(k), new.get(k), f"{path}.{k}")]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new) and \
            any(isinstance(x, (dict, list)) for x in old):
        return [m for i, (a, b) in enumerate(zip(old, new)) for m in moved(a, b, f"{path}[{i}]")]
    return [] if old == new else [{"path": path, "registered": old, "engine_0_2_0": new}]


def is_cache_count(path: str) -> bool:
    last = path.rsplit(".", 1)[-1].lower()
    return any(w in last for w in CACHE_WORDS)


def phase6_rules(fixed: dict[str, Any]) -> dict[str, Any]:
    """Phase 6's registered rules, applied to its two live runs with the fixed engine's numbers."""
    analyze = _script("phase6_analyze")
    cfg = yaml.safe_load(analyze.CONFIG.read_text())
    stored = _json(RESULTS / "phase6_kaggle_t4" / "analysis.json")
    runs = {"run1": analyze.run1(), "run2": analyze.run2(_json(analyze.OUT / "phase6.json"))}
    out = {}
    for name, measured in runs.items():
        v = analyze.verdict(measured, fixed, cfg)
        out[name] = {"s1_max_registered": stored[f"verdict_{name}"]["s1_max"],
                     "s1_max_engine_0_2_0": v["s1_max"], "s2_holds": v["s2_holds"],
                     "holds": v["holds"]}
    return out


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# POST HOC: the registered predictions, computed again with the engine of 0.2.0",
        "",
        ("⚠️ Post hoc. Written on 28 November 2026, after the results below had been seen. "
         "The predictions were registered before their GPU runs, with the engine of release "
         "0.1.2, and they stay as they are. Generated by "
         "`scripts/registered_predictions_again.py`."),
        "",
        ("| Test | the old engine gives the registered numbers | numbers the fix moves "
         "| of them cache counts |"),
        "|---|---|---:|---:|",
    ]
    for name, s in payload["sources"].items():
        cache = sum(is_cache_count(m["path"]) for m in s["moved"])
        lines.append(f"| {name} | {'yes' if s['old_engine_reproduces'] else 'NO'} | "
                     f"{len(s['moved'])} | {cache} |")
    check = payload["registered_check"]
    lines += [
        "",
        "## The check registered in `configs/rederive_0_2_0.yaml`",
        "",
        ("The hit rates and cached tokens of H3, H4, V1, C1 and Phase 6 must not move. "
         f"**{'Holds' if check['holds'] else 'Does not hold'}.** Cache counts that moved:"),
        "",
    ]
    lines += [f"- {m['test']} `{m['path']}`: {m['registered']} registered, "
              f"{m['engine_0_2_0']} with the engine of 0.2.0" for m in check["moved"]] or ["- none"]
    lines += ["", "## Every number the fix moves", ""]
    for name, s in payload["sources"].items():
        for m in s["moved"]:
            lines.append(f"- {name} `{m['path']}`: {json.dumps(m['registered'])} → "
                         f"{json.dumps(m['engine_0_2_0'])}")
    p6 = payload["phase6_rules"]
    lines += [
        "",
        "## Phase 6's rules with the fixed engine's numbers",
        "",
        ("| run | largest hit-rate error, registered numbers | with the engine of 0.2.0 "
         "| rules hold |"),
        "|---|---:|---:|---|",
    ]
    lines += [f"| {r} | {v['s1_max_registered']:.2%} | {v['s1_max_engine_0_2_0']:.2%} | "
              f"{'yes' if v['holds'] else 'no'} |" for r, v in p6.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = results_header("scripts/registered_predictions_again.py", seed=0,
                             hardware_path="cpu-only")
    payload["post_hoc"] = True
    sources: dict[str, Any] = {}
    check: list[dict[str, Any]] = []
    fixed_phase6: dict[str, Any] = {}
    for name, (where, source, checked) in SOURCES.items():
        registered, make = source()
        with use_engine("0.1.2"):
            old = make()
        new = make()
        moves = moved(registered, new)
        sources[name] = {"registered_in": where, "old_engine_reproduces": old == registered,
                         "moved": moves}
        if not sources[name]["old_engine_reproduces"]:
            raise SystemExit(f"error: the engine of 0.1.2 no longer gives {name}'s registered "
                             f"predictions. The code has drifted; stopping.")
        if checked:
            check += [{"test": name, **m} for m in moves if is_cache_count(m["path"])]
        if name == "Phase 6":
            fixed_phase6 = new
        print(f"{name}: {len(moves)} numbers move", flush=True)
    payload["sources"] = sources
    payload["registered_check"] = {"holds": not check, "moved": check}
    payload["phase6_rules"] = phase6_rules(fixed_phase6)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "registered_predictions.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "report.md").write_text(markdown(payload))
    print(f"-> {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
