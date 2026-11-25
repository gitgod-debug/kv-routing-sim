#!/usr/bin/env python3
"""L1: the whole AgentX trace at its recorded size. Cache accounting and routing, no timing.

    python scripts/run_l1.py                   # every run -> artifacts/results/l1_agentx_full/
    python scripts/run_l1.py --keep-finished   # go on with a job that was stopped

Registered in docs/09-hypotheses.md; the design and the rules are in configs/l1_agentx_full.yaml.
Every request finishes before the next one arrives, so the result says which tokens the caches
served and which had to be computed, and nothing about time, GPU counts or cost.

Two counts are kept apart. The simulator's: each request's cached tokens, under a routing policy
and a cache size. And one made directly from the raw file, without the simulator or its trace
reader: how many leading blocks of each prompt the same session had sent before. The second is the
most any cache can serve; what a policy computes beyond the rest is re-prefilled work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import pathlib
import resource
import sys
import time
from collections import defaultdict
from typing import Any

import yaml

from kv_routing_sim.eval.bootstrap import bootstrap_ci_grouped, paired_bootstrap_delta
from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.adapters import agentx
from kv_routing_sim.traces.schema import NormalizedRequest, TraceStats

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "l1_agentx_full.yaml"
OUT = REPO / "artifacts" / "results" / "l1_agentx_full"
LOG = REPO / "artifacts" / "logs" / "l1_runs.jsonl"  # finished runs, so a stopped job can go on

# Set once by `load`, then inherited by the forked workers.
CFG: dict[str, Any] = {}
TRACE: list[NormalizedRequest] = []
SERVABLE: list[int] = []  # per request, the tokens a cache that never forgets would serve


def raw_count(path: pathlib.Path, block: int) -> dict[str, Any]:
    """Count from the raw file what a cache that never forgets would serve each request.

    For each session (a sub-agent is its own), in time order and for its main model only: the
    leading blocks of the prompt that the session sent before, up to the last token, which an
    engine always computes. Walks a tree of the recorded block ids, so a block counts only when
    everything before it was the same too.
    """
    servable: dict[str, list[tuple[int, int]]] = {}
    call_s = live_s = 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            session = json.loads(line)
            main = session["models"][0]
            rows: dict[str, list[tuple[float, dict]]] = defaultdict(list)
            for r in session["requests"]:
                nested = r.get("type") == "subagent"
                sid = f"{session['id']}/{r.get('agent_id', 'subagent')}" if nested \
                    else session["id"]
                for q in r.get("requests", []) if nested else [r]:
                    at = float(r["t"]) + float(q["t"]) if nested else float(q["t"])
                    rows[sid].append((at, q))
            live_s += max((at + float(q.get("api_time") or 0.0)
                           for qs in rows.values() for at, q in qs), default=0.0)
            for sid, qs in rows.items():
                tree: dict = {}
                for _, q in sorted(qs, key=lambda x: x[0]):
                    if q.get("model") != main:
                        continue
                    call_s += float(q.get("api_time") or 0.0)
                    node, known = tree, 0
                    for h in q["hash_ids"]:
                        if h not in node:
                            break
                        node, known = node[h], known + 1
                    for h in q["hash_ids"][known:]:
                        node[h] = node = {}
                    n = int(q["in"])
                    servable.setdefault(sid, []).append(
                        (n, block * min(known, (n - 1) // block)))
    return {"servable": servable, "call_hours": call_s / 3600, "live_hours": live_s / 3600}


def load(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read the trace and the raw count, and check that they describe the same requests."""
    path = REPO / cfg["trace"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != cfg["trace_sha256"]:
        raise SystemExit(f"error: {cfg['trace']} is not the registered file (sha256 {sha})")
    stats, t0 = TraceStats(), time.monotonic()
    TRACE.extend(agentx.read(path, stats=stats, **cfg["adapter"]))
    read_s = time.monotonic() - t0
    raw = raw_count(path, cfg["adapter"]["block_size"])
    per_session = {sid: iter(rows) for sid, rows in raw["servable"].items()}
    for q in TRACE:
        n, cached = next(per_session[q.session_id])
        if n != q.prompt_tokens:
            raise SystemExit(f"error: {q.request_id} has {q.prompt_tokens} prompt tokens in the "
                             f"trace and {n} in the raw count")
        SERVABLE.append(cached)
    if any(next(rows, None) is not None for rows in per_session.values()):
        raise SystemExit("error: the raw count has requests the trace does not")
    CFG.update(cfg)
    return {
        "requests": len(TRACE), "other_model_requests_left_out": stats.skip_reasons,
        "sessions": len({q.session_id.split("/")[0] for q in TRACE}),
        "sessions_with_sub_agents": len({q.session_id for q in TRACE}),
        "block_hashes": sum(len(q.block_hashes) for q in TRACE),
        "prompt_tokens": sum(q.prompt_tokens for q in TRACE),
        "output_tokens": sum(q.output_tokens for q in TRACE),
        "longest_prompt": max(q.prompt_tokens for q in TRACE),
        "servable_tokens": sum(SERVABLE),
        "replayed_hours": round(TRACE[-1].arrival_ts / 3600, 1),
        "recorded_call_hours": round(raw["call_hours"], 1),
        "recorded_session_hours": round(raw["live_hours"], 1),
        "read_s": round(read_s, 1),
        "rss_mib_after_reading": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def replay(policy: str, replicas: int, usable_blocks: int, seed: int):
    """Run the whole trace through one policy and cache size."""
    profile = CostProfile(
        name="no-timing", prefill_base_s=0.0, prefill_per_token_s=0.0,
        decode_base_s=float(CFG["step_s"]), decode_per_seq_s=0.0,
        kv_cache_blocks=usable_blocks + 1,  # one is vLLM's null block
        block_size=CFG["adapter"]["block_size"], max_num_seqs=256)
    t0 = time.monotonic()
    r = run(TRACE, policy=policy, n_replicas=replicas, seed=seed, profile=profile,
            cache_model=CFG["cache_model"], record_per_request=True)
    cached = [r.per_request[q.request_id]["cached_tokens"] for q in TRACE]
    if r.preemptions or r.cache_queries != sum(q.prompt_tokens for q in TRACE) \
            or r.cache_hits != sum(cached):
        raise SystemExit(f"error: {policy}, {usable_blocks} blocks, seed {seed}: the counters and "
                         f"the per-request records disagree ({r.preemptions} preemptions)")
    return r, cached, time.monotonic() - t0


def never_forgets(_: None = None) -> dict[str, Any]:
    """L1a: one replica that never evicts, request by request against the raw count."""
    r, cached, seconds = replay("round-robin", 1, int(CFG["never_forgets_blocks"]), 0)
    return {
        "requests": len(TRACE),
        "requests_equal": sum(a == b for a, b in zip(cached, SERVABLE, strict=True)),
        "hit_tokens": sum(cached), "servable_tokens": sum(SERVABLE),
        "evictions": r.evictions, "seconds": round(seconds, 1),
        "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def grid_run(key: tuple[str, int, int]) -> dict[str, Any]:
    """One policy, cache size and seed: what was computed again, in all and per session."""
    policy, capacity, seed = key
    r, cached, seconds = replay(policy, CFG["replicas"], capacity // CFG["adapter"]["block_size"],
                                seed)
    again: dict[str, int] = defaultdict(int)
    for q, hit, could in zip(TRACE, cached, SERVABLE, strict=True):
        again[q.session_id.split("/")[0]] += could - hit
    per_replica = r.per_replica_requests
    return {
        "policy": policy, "capacity_tokens": capacity, "seed": seed,
        "hit_tokens": sum(cached), "re_prefilled_tokens": sum(again.values()),
        "evictions": r.evictions, "seconds": round(seconds, 1),
        "busiest_replica_share": round(max(per_replica) / sum(per_replica), 4),
        "rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "per_session": dict(again),
    }


def ratio(items: list[tuple[float, int]]) -> float:
    """Re-prefilled tokens over tokens sent: a ratio of sums over the sessions drawn."""
    return sum(a for a, _ in items) / sum(n for _, n in items)


def summarise(cfg: dict[str, Any], runs: list[dict[str, Any]],
              sent: dict[str, int]) -> list[dict[str, Any]]:
    """One row per policy and cache size, with an interval and a paired one against round-robin."""
    sessions = sorted(sent)
    total = sum(sent.values())
    rows, per_session = [], {}
    for capacity in cfg["capacity_tokens"]:
        for policy in cfg["policies"]:
            mine = [r for r in runs
                    if (r["policy"], r["capacity_tokens"]) == (policy, capacity)]
            groups = [[(sum(r["per_session"].get(s, 0) for r in mine) / len(mine), sent[s])]
                      for s in sessions]
            per_session[policy, capacity] = groups
            share = [r["re_prefilled_tokens"] / total for r in mine]
            ci = bootstrap_ci_grouped(groups, ratio, n_resamples=cfg["bootstrap_resamples"])
            paired = paired_bootstrap_delta(groups, per_session["round-robin", capacity], ratio,
                                            n_resamples=cfg["bootstrap_resamples"])
            rows.append({
                "policy": policy, "capacity_tokens": capacity, "seeds": len(mine),
                "hit_rate": sum(r["hit_tokens"] for r in mine) / len(mine) / total,
                "re_prefilled_share": sum(share) / len(share),
                "re_prefilled_share_by_seed": share,
                "re_prefilled_ci": ci.as_dict(),
                "minus_round_robin_ci": paired.as_dict(),
                "evictions": round(sum(r["evictions"] for r in mine) / len(mine)),
                "seconds_max": max(r["seconds"] for r in mine),
                "busiest_replica_share": max(r["busiest_replica_share"] for r in mine),
            })
    return rows


def verdicts(cfg: dict[str, Any], exact: dict[str, Any],
             rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score L1a to L1d by the registered rules."""
    rules = cfg["rules"]
    at = {(r["policy"], r["capacity_tokens"]): r for r in rows}
    caps = cfg["capacity_tokens"]
    minutes = max(r["seconds_max"] for r in rows) / 60
    ratios, points = {}, {}
    for c in caps:
        sticky = at["session-sticky", c]["re_prefilled_share"]
        rr = at["round-robin", c]["re_prefilled_share"]
        ratios[c] = rr / sticky if sticky > 0 else float("inf")
        points[c] = 100 * abs(at["radix-affinity", c]["re_prefilled_share"] - sticky)
    return {
        "L1a": {"requests_equal": exact["requests_equal"], "of": exact["requests"],
                "holds": exact["requests_equal"] == exact["requests"]
                and exact["hit_tokens"] == exact["servable_tokens"]},
        "L1b": {"longest_run_minutes": round(minutes, 1), "limit": rules["l1b_max_minutes"],
                "holds": minutes <= rules["l1b_max_minutes"]},
        "L1c": {"round_robin_over_sticky": {str(c): round(v, 2) for c, v in ratios.items()},
                "at_least": rules["l1c_min_ratio"],
                "holds": all(v >= rules["l1c_min_ratio"] for v in ratios.values())},
        "L1d": {"points_apart": {str(c): round(v, 3) for c, v in points.items()},
                "at_most": rules["l1d_max_points"],
                "holds": all(v <= rules["l1d_max_points"] for v in points.values())},
    }


def report(p: dict[str, Any]) -> str:
    t, e, at_once = p["trace"], p["never_forgets"], p["config"]["adapter"]["concurrency"]
    in_a_call = at_once * t["recorded_call_hours"] / t["recorded_session_hours"]
    mark = {True: "held", False: "did not hold"}
    v = p["verdicts"]
    lines = [
        "# L1: the whole AgentX trace at its recorded size",
        "",
        "Registered before it ran (docs/09-hypotheses.md). Made by `scripts/run_l1.py`; the",
        "numbers are in `l1.json` beside this file. Cache accounting only: each request finishes",
        "before the next arrives, so nothing here is a latency, a GPU count or a cost.",
        "",
        (f"**Trace.** {t['sessions']} coding-agent sessions ({t['sessions_with_sub_agents']} with "
         f"sub-agents), {t['requests']:,} requests to each session's main model, "
         f"{t['prompt_tokens']:,} prompt tokens in {t['block_hashes']:,} blocks of "
         f"{p['config']['adapter']['block_size']} tokens, {t['output_tokens']:,} output tokens; "
         f"the longest prompt has {t['longest_prompt']:,}. "
         f"{at_once} sessions are replayed at a time."),
        (f"**Overlap left out.** Recorded calls take {t['recorded_call_hours']:,} of "
         f"{t['recorded_session_hours']:,} session-hours, so on average {in_a_call:.1f} of "
         f"{at_once} sessions are in a call at any time."),
        (f"**Ceiling.** {t['servable_tokens'] / t['prompt_tokens']:.2%} of the prompt tokens were "
         "sent before by the same session, so a cache that never forgot would serve them. The rest "
         "is new text."),
        "",
        ("| Cache per replica | Policy | Hit rate | Computed again, share of prompt tokens "
         "(95% interval over sessions) | Minus round-robin (paired) | Evictions |"),
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in p["rows"]:
        ci, d = r["re_prefilled_ci"], r["minus_round_robin_ci"]
        paired = "" if r["policy"] == "round-robin" else (
            f"{100 * d['point']:+.1f} points ({100 * d['ci_lo']:+.1f} to "
            f"{100 * d['ci_hi']:+.1f})")
        lines.append(
            f"| {r['capacity_tokens'] / 2**20:g} Mi tokens | {r['policy']} | "
            f"{r['hit_rate']:.1%} | {r['re_prefilled_share']:.2%} ({ci['ci_lo']:.2%} to "
            f"{ci['ci_hi']:.2%}) | {paired} | {r['evictions']:,} |")
    lines += [
        "",
        "## The four predictions",
        "",
        (f"- **L1a {mark[v['L1a']['holds']]}.** With one replica that never evicts, "
         f"{e['requests_equal']:,} of {e['requests']:,} requests' cached tokens equal the count "
         "made from the raw file."),
        (f"- **L1b {mark[v['L1b']['holds']]}.** The longest run of the grid took "
         f"{v['L1b']['longest_run_minutes']} minutes on one core (limit "
         f"{v['L1b']['limit']})."),
        (f"- **L1c {mark[v['L1c']['holds']]}.** Round-robin computes again "
         + ", ".join(f"{x}× at {int(c) / 2**20:g} Mi" for c, x in
                     v["L1c"]["round_robin_over_sticky"].items())
         + f" what session-sticky does (at least {v['L1c']['at_least']}× predicted)."),
        (f"- **L1d {mark[v['L1d']['holds']]}.** Prefix-affinity and session-sticky are "
         + ", ".join(f"{x} points apart at {int(c) / 2**20:g} Mi" for c, x in
                     v["L1d"]["points_apart"].items())
         + f" (at most {v['L1d']['at_most']} predicted)."),
        "",
        (f"Reading the trace took {t['read_s']:.0f} s and {t['rss_mib_after_reading']:,} MiB; the "
         f"replica that never evicts took {e['seconds']:.0f} s and {e['rss_mib']:,} MiB. "
         f"{p['n_runs']} runs in {p['runtime_s'] / 60:.0f} minutes on {p['config']['workers']} "
         "workers."),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep-finished", action="store_true",
                    help="go on with the runs already in the log")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    t0 = time.monotonic()
    facts = load(cfg)
    print(json.dumps(facts), flush=True)
    gc.collect()
    gc.freeze()  # the workers share the trace; keep the collector from touching its pages

    stamp = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    done: dict[str, Any] = {}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_finished and LOG.exists():
        for line in LOG.read_text().splitlines():
            row = json.loads(line)
            if row["config_sha256"] == stamp:
                done[row["key"]] = row["result"]
        print(f"{len(done)} finished runs kept", flush=True)
    else:
        LOG.write_text("")

    def keep(key: str, result: dict[str, Any]) -> None:
        done[key] = result
        with LOG.open("a") as fh:
            fh.write(json.dumps({"config_sha256": stamp, "key": key, "result": result}) + "\n")
        shown = {k: v for k, v in result.items() if k != "per_session"}
        print(f"{time.monotonic() - t0:6.0f} s  {key}  {json.dumps(shown)}", flush=True)

    fork = multiprocessing.get_context("fork")
    if "never_forgets" not in done:
        with fork.Pool(1) as pool:
            keep("never_forgets", pool.apply(never_forgets))
    keys = [(p, c, s) for c in cfg["capacity_tokens"] for p, seeds in cfg["policies"].items()
            for s in seeds]
    todo = [k for k in keys if "|".join(map(str, k)) not in done]
    with fork.Pool(cfg["workers"], maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(grid_run, todo):
            keep(f"{result['policy']}|{result['capacity_tokens']}|{result['seed']}", result)

    sent: dict[str, int] = defaultdict(int)
    for q in TRACE:
        sent[q.session_id.split("/")[0]] += q.prompt_tokens
    runs = [done["|".join(map(str, k))] for k in keys]
    rows = summarise(cfg, runs, sent)
    sys.path.insert(0, str(REPO / "scripts"))
    from _spike_common import results_header

    payload = results_header("scripts/run_l1.py", seed=0, hardware_path="cpu-only")
    payload.update({
        "config": cfg, "trace": facts, "never_forgets": done["never_forgets"], "rows": rows,
        "verdicts": verdicts(cfg, done["never_forgets"], rows),
        "runs": [{k: v for k, v in r.items() if k != "per_session"} for r in runs],
        "n_runs": len(runs) + 1, "runtime_s": round(time.monotonic() - t0, 1),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "l1.json").write_text(json.dumps(payload, indent=1, default=str) + "\n",
                                 encoding="utf-8")
    (OUT / "report.md").write_text(report(payload), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}/l1.json and report.md")
    for name, verdict in payload["verdicts"].items():
        print(f"  {name}: {'held' if verdict['holds'] else 'did not hold'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
