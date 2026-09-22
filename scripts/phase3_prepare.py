#!/usr/bin/env python3
"""Prepare Phase 3: the trace, the pre-registered predictions, and the notebook — from ONE config.

    python scripts/phase3_prepare.py

Writes, all stamped with one fingerprint (sha256 of every token id the notebook will send):
  artifacts/results/phase3_prereg/predictions.json   ← commit this BEFORE the GPU run
  notebooks/phase3_calibration.ipynb                 ← the trace and predictions are embedded in it

⭐ Why one script: the notebook must send EXACTLY the trace the predictions were computed on. If the
trace lived in one place and the predictions in another, a regenerated trace could silently diverge
from the numbers it is compared against. Here it cannot: the notebook re-checks the fingerprint of
the trace it carries, and the hash of the predictions it carries, before it sends a single request.

⭐ And why the notebook never generates anything: it only SENDS token ids produced by this repo's
one generator. A second implementation inside the notebook would drift from the first — the reason
ADR-011 forbids re-implementing the cache model in JavaScript.

Nothing here is fitted. The two cache models run exactly as committed; their predictions are
whatever they are.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import pathlib
import random
import sys
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import yaml

from kv_routing_sim.sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.sim.metrics import RunResult
from kv_routing_sim.traces.generator import ENGINE_SAFE_IDS, AgenticWorkload
from kv_routing_sim.traces.schema import NormalizedRequest

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "phase3_workloads.yaml"
PREREG = REPO / "artifacts" / "results" / "phase3_prereg" / "predictions.json"
NOTEBOOK = REPO / "notebooks" / "phase3_calibration.ipynb"
SEQ_GAP_S = 1000.0  # sequential replay: every request finishes long before the next arrives

# The counters the notebook reads from /metrics and compares, prediction by prediction.
COMPARED = (
    "prefix_cache_queries",
    "prefix_cache_hits",
    "local_compute",
    "preemptions",
    "hist_lifetime_count",
    "hist_idle_count",
    "hist_reuse_gap_count",
)


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def profile(cfg: dict[str, Any], *, slow: float = 1.0) -> CostProfile:
    """The config's capacity and limits on the UNCALIBRATED T4 timings, every timing x `slow`.

    Sequential counts must not move when `slow` does — `predict_sequential` checks that they don't,
    and that check is what makes them pre-registrable as exact integers."""
    e, u = cfg["engine"], UNCALIBRATED_DEFAULT
    return replace(
        u,
        name=f"phase3-prereg-uncalibrated-x{slow:g}",
        prefill_base_s=u.prefill_base_s * slow,
        prefill_per_token_s=u.prefill_per_token_s * slow,
        decode_base_s=u.decode_base_s * slow,
        decode_per_seq_s=u.decode_per_seq_s * slow,
        kv_cache_blocks=e["num_gpu_blocks"],
        block_size=e["block_size"],
        max_num_seqs=e["max_num_seqs"],
    )


def build_trace(cfg: dict[str, Any]) -> list[tuple[NormalizedRequest, list[int]]]:
    return list(AgenticWorkload(**cfg["trace"], output_tokens=1).iter_with_tokens())


def sweep_prompts(cfg: dict[str, Any]) -> list[list[int]]:
    """The decode-sweep prompts: random engine-safe ids, sharing nothing with the trace (a shared
    16-token block would need 16 simultaneous collisions in a 99,000-id range)."""
    c = cfg["calibration"]["decode_batch_sweep"]
    rng = random.Random(c["seed"])
    lo, hi = ENGINE_SAFE_IDS
    return [
        [rng.randrange(lo, hi) for _ in range(c["prompt_tokens"])]
        for _ in range(max(c["batch_sizes"]))
    ]


def fingerprint(trace: list[tuple[NormalizedRequest, list[int]]], prompts: list[list[int]]) -> str:
    """sha256 over everything the notebook sends. The notebook recomputes it the same way."""
    return sha256(
        {
            "trace": [[r.request_id, r.arrival_ts, r.session_id, ids] for r, ids in trace],
            "decode_sweep_prompts": prompts,
        }
    )


def _replay(cfg, reqs, out: int, model: str, *, slow: float = 1.0, sequential: bool) -> RunResult:
    tr = [
        replace(r, output_tokens=out, arrival_ts=i * SEQ_GAP_S if sequential else r.arrival_ts)
        for i, r in enumerate(reqs)
    ]
    return run(
        tr,
        policy="round-robin",
        n_replicas=1,
        profile=profile(cfg, slow=slow),
        cache_model=model,
        track_block_metrics=model == "vllm",
        record_per_request=True,
    )


def _counts(r: RunResult, model: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prefix_cache_queries": r.cache_queries,
        "prefix_cache_hits": r.cache_hits,
        "local_compute": r.cache_queries - r.cache_hits,
        "hit_rate": round(r.hit_rate, 6),
        "preemptions": r.preemptions,
        "real_evictions": r.evictions,
    }
    if model == "vllm":
        ev = r.eviction_events
        out |= {
            # What the engine's three kv_block_* histograms should COUNT, by vLLM's own rule: an
            # event on EVERY recycled tracked block, cached or not (block_pool.py:739 vs 741).
            "hist_lifetime_count": len(ev),
            "hist_idle_count": len(ev),
            "hist_reuse_gap_count": sum(len(e.reuse_gaps_s) for e in ev),
            "hist_events_on_cached_blocks": sum(1 for e in ev if e.was_cached),
        }
    else:
        # None, not 0: the lru model has no eviction events, so it makes NO prediction here. A 0
        # would be a prediction — and a guaranteed-false one.
        out |= dict.fromkeys(("hist_lifetime_count", "hist_idle_count", "hist_reuse_gap_count"))
    return out


def _cached(r: RunResult) -> dict[str, int]:
    return {rid: v["cached_tokens"] for rid, v in r.per_request.items()}


def predict_sequential(cfg, reqs: list[NormalizedRequest], out: int) -> dict[str, Any]:
    preds: dict[str, Any] = {}
    for model in ("vllm", "lru"):
        a = _replay(cfg, reqs, out, model, sequential=True)
        # ⭐ The self-check that makes these numbers pre-registrable: sequential counts must be
        # TIMING-INDEPENDENT. Recompute with every timing 10x slower; demand identical counts.
        b = _replay(cfg, reqs, out, model, slow=10.0, sequential=True)
        ca = _counts(a, model)
        if ca != _counts(b, model) or _cached(a) != _cached(b):
            raise SystemExit(f"⛔ {model}: the sequential prediction depends on timing")
        ca["per_request_cached_tokens"] = _cached(a)
        preds[model] = ca

    # ⭐ The pre-registered ALTERNATIVE. vLLM's source says a request commits prompt + output - 1
    # tokens (the last sampled token is never fed back in; kv_cache_manager.py `allocate_slots` caps
    # caching at `request.num_tokens`, and the async scheduler's catch-up in
    # `_update_request_with_output` reaches the same count before the request is freed). If the
    # engine instead committed ONE MORE token, this is what it would show — so an off-by-one is
    # diagnosed by the run itself, not argued about afterwards.
    alt = _replay(cfg, reqs, out + 1, "vllm", sequential=True)
    ca = _counts(alt, "vllm")
    ca["per_request_cached_tokens"] = _cached(alt)
    preds["vllm_if_one_more_token_committed"] = ca
    main_keys = {k: preds["vllm"][k] for k in (*COMPARED, "per_request_cached_tokens")}
    alt_keys = {k: ca[k] for k in (*COMPARED, "per_request_cached_tokens")}
    preds["alternative_distinguishable"] = main_keys != alt_keys
    return preds


def predict_concurrent(cfg, reqs: list[NormalizedRequest], out: int) -> dict[str, Any]:
    lru = _replay(cfg, reqs, out, "lru", sequential=False)
    lru_slow = _replay(cfg, reqs, out, "lru", slow=10.0, sequential=False)
    lru_seq = _replay(cfg, reqs, out, "lru", sequential=True)
    # The lru model's claims are properties of the model, so they must hold inside it exactly.
    if not (lru.preemptions == lru_slow.preemptions == 0):
        raise SystemExit("⛔ the lru model preempted — it has no mechanism to")
    if not (lru.hit_rate == lru_slow.hit_rate == lru_seq.hit_rate):
        raise SystemExit("⛔ the lru model's concurrent hit rate depends on timing")

    # The vllm model's concurrent behaviour IS timing-dependent, so register only what survives a
    # 4x bracket around the uncalibrated profile — a claim that flips inside the bracket is not one.
    bracket = {s: _replay(cfg, reqs, out, "vllm", slow=s, sequential=False) for s in (0.5, 1, 2)}
    preempts = {f"x{s:g}": r.preemptions for s, r in bracket.items()}
    vllm_claim: dict[str, Any] = {"preemptions_by_timing_scale": preempts}
    if all(v > 0 for v in preempts.values()):
        vllm_claim |= {
            "preemptions": "> 0",
            "why": "memory-bound admission preempts at every timing in the bracket",
        }
    else:
        vllm_claim |= {"preemptions": "not claimed", "why": "timing-dependent inside the bracket"}
    mid = bracket[1]
    return {
        "timing_independent_claims": {
            "lru": {
                "preemptions": 0,
                "hit_rate": round(lru.hit_rate, 6),
                "local_compute": lru.cache_queries - lru.cache_hits,
                "per_request_cached_tokens": _cached(lru),
                "why": (
                    "lru has no memory-based admission, so it never preempts, and its lookups "
                    "happen in arrival order — so its hit rate equals its sequential replay "
                    f"({lru_seq.hit_rate:.6f}) at any timing."
                ),
            },
            "vllm": vllm_claim,
        },
        "illustrative_only_uncalibrated_profile": {
            "vllm": {
                "hit_rate": round(mid.hit_rate, 6),
                "preemptions": mid.preemptions,
                "ttft": mid.ttft_percentiles(),
            },
            "lru": {
                "hit_rate": round(lru.hit_rate, 6),
                "preemptions": lru.preemptions,
                "ttft": lru.ttft_percentiles(),
            },
            "caveat": (
                "UNCALIBRATED timings: not predictions. What is pre-registered for W3 beyond the "
                "claims above is the PROCEDURE (analysis_procedure step 5)."
            ),
        },
    }


ANALYSIS_PROCEDURE = [
    (
        "1. Validity. A run counts only if every request finished with finish_reason 'length', "
        "exactly output_tokens completion tokens and prompt_tokens == len(ids); the engine logged "
        "num_gpu_blocks 512; and the embedded fingerprints matched. Invalid runs are reported, "
        "never silently dropped."
    ),
    (
        "2. Sequential counts (W1, W2): compare each run's /metrics deltas with each model's "
        "prediction, counter by counter, and each request's "
        "usage.prompt_tokens_details.cached_tokens with the per-request prediction. Error = "
        "measured - predicted, in tokens and hit-rate points."
    ),
    "3. Determinism: report whether the three runs of each sequential workload agree exactly.",
    (
        "4. Timing fit — never tuned against W3. decode_base_s + decode_per_seq_s: OLS of the mean "
        "inter-token interval on batch size over the decode sweep. prefill_base_s + "
        "prefill_per_token_s: OLS of client TTFT on uncached tokens (prompt_tokens - cached_tokens)"
        " over every W1 and W2 request, minus the fitted batch-1 decode step (the simulator emits "
        "the first token at the end of the prefill step)."
    ),
    (
        "5. W3: simulate the RECORDED send times with the fitted profile under both cache models; "
        "compare hit rate, local_compute, preemptions and client TTFT p50/p95/p99 with the three "
        "measured runs (mean and range). Published whatever it says."
    ),
    (
        "6. H4: compare the engine's kv_block_* histogram counts, and bucket distributions, with "
        "the vllm model's eviction events emitted under vLLM's own rule."
    ),
]

DECISION_RULES = {
    "vllm model exact on W1 and W2": (
        "the block-pool model is confirmed on both shapes with zero error; the lru model's W1 "
        "error is the size of what the ablation misses (H3 passes; H4's mechanism is identified)"
    ),
    "vllm exact on W2 but not W1": (
        "decode-block handling differs from the model; the per-request diffs and the histogram "
        "counts localise which requests and which blocks"
    ),
    "W2 matches vllm_if_one_more_token_committed": (
        "the engine commits one token more than its source suggests: fix that one rule and "
        "re-run the prediction, nothing else"
    ),
    "neither model matches W2": (
        "partial-block or free-order accounting differs; H3 is reported as FAILED for that shape"
    ),
    "W3 misses by more than its run-to-run range": (
        "report it, and downgrade concurrent claims from predictive to comparative "
        "(CHECKPOINTS Phase 3 gate) — no re-tuning against W3"
    ),
}


def make_predictions(cfg: dict[str, Any], *, verbose: bool = False):
    """Everything pre-registered, from the config alone: (trace, prompts, fingerprint, predictions).

    Deterministic — tests/test_phase3_prereg.py recomputes it and demands the committed file."""
    trace = build_trace(cfg)
    prompts = sweep_prompts(cfg)
    fp = fingerprint(trace, prompts)
    reqs = [r for r, _ in trace]
    bs = cfg["engine"]["block_size"]
    say = print if verbose else (lambda *a, **k: None)
    say(f"trace: {len(trace)} requests, prompts {sorted({r.prompt_tokens for r in reqs})} tokens")
    say(f"fingerprint {fp[:16]}…")

    workloads: dict[str, Any] = {}
    for name, w in cfg["workloads"].items():
        out = w["output_tokens"]
        if w["mode"] == "sequential":
            p = predict_sequential(cfg, reqs, out)
            v, lr = p["vllm"], p["lru"]
            say(
                f"  {name:14s} vllm {v['hit_rate']:.4f} (computed {v['local_compute']:>6,}) | "
                f"lru {lr['hit_rate']:.4f} (computed {lr['local_compute']:>6,}) | "
                f"gap {(v['hit_rate'] - lr['hit_rate']) * 100:+6.2f} pts | "
                f"hist {v['hist_lifetime_count']:>5} | "
                f"+1-token alternative distinguishable: {p['alternative_distinguishable']}"
            )
        else:
            p = predict_concurrent(cfg, reqs, out)
            c = p["timing_independent_claims"]
            say(
                f"  {name:14s} lru: 0 preemptions, hit {c['lru']['hit_rate']:.4f} | vllm: "
                f"preemptions {c['vllm']['preemptions']} {c['vllm']['preemptions_by_timing_scale']}"
            )
        # How many requests END on a block boundary (prompt + output - 1 tokens). Only those could
        # be affected by an off-by-one in the final-token rule; reported, so it is visible.
        on_boundary = sum(1 for r in reqs if (r.prompt_tokens + out - 1) % bs == 0)
        workloads[name] = {
            "mode": w["mode"],
            "output_tokens": out,
            "requests_ending_on_a_block_boundary": on_boundary,
            "predictions": p,
        }

    predictions = {
        "trace_fingerprint_sha256": fp,
        "engine": cfg["engine"],
        "usable_blocks": cfg["engine"]["num_gpu_blocks"] - 1,
        "compared_counters": list(COMPARED),
        "workloads": workloads,
    }
    return trace, prompts, fp, predictions


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    trace, prompts, fp, predictions = make_predictions(cfg, verbose=True)
    pred_sha = sha256(predictions)
    prereg = {
        "what": "Phase 3 pre-registered predictions. Committed BEFORE any GPU run.",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg,
        "trace_fingerprint_sha256": fp,
        "predictions_sha256": pred_sha,
        "n_requests": len(trace),
        "sequential_counts_verified_timing_independent": True,
        "simulator_assumptions": [
            (
                "The simulator schedules synchronously. vLLM 0.30.0 enables async scheduling by "
                "default; for one-at-a-time requests both reach the same final state per request "
                "(async catches up in AsyncScheduler._update_request_with_output before the free), "
                "so W1/W2 predictions do not depend on it. W3 may."
            ),
            (
                "No chunked prefill and no token budget: W1/W2 prompts (<= 583 tokens) are far "
                "below max_num_batched_tokens (8192). W3 may batch several prefills into one step."
            ),
        ],
        "analysis_procedure": ANALYSIS_PROCEDURE,
        "decision_rules": DECISION_RULES,
        "predictions": predictions,
    }
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(prereg, indent=1) + "\n")
    print(f"wrote {PREREG.relative_to(REPO)}  (predictions sha256 {pred_sha[:16]}…)")

    payload = {
        "trace": [
            {"request_id": r.request_id, "arrival_ts": r.arrival_ts, "session_id": r.session_id,
             "ids": ids}
            for r, ids in trace
        ],
        "decode_sweep_prompts": prompts,
        "predictions": predictions,
    }
    # xz, not gzip: a session's turns repeat each other ~50 KB apart in arrival order, beyond
    # gzip's 32 KB window but well inside xz's.
    blob = base64.b64encode(lzma.compress(canonical(payload), preset=9)).decode()

    sys.path.insert(0, str(REPO / "scripts"))
    from _phase3_notebook import build_notebook

    NOTEBOOK.write_text(build_notebook(cfg, blob, fp, pred_sha, predictions))
    print(f"wrote {NOTEBOOK.relative_to(REPO)}  (embedded data {len(blob) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
