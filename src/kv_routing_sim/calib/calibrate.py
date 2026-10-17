"""`kvsim calibrate`: fit a timing profile to a live vLLM engine and check the cache model on it.

    kvsim calibrate --engine http://localhost:8000 --out configs/profiles/mine.json

Timings measured on one GPU do not transfer to another (ADR-010), so the command measures the engine
it is given: a vLLM V1 server with prefix caching, reached over HTTP using only the standard library
(ADR-009). It probes the engine, fits decode and prefill timings with `calib/fit.py`, and checks the
cache model's hit rate against the engine's prefix-cache counters.
"""

from __future__ import annotations

import datetime
import json
import random
import re
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import __version__
from ..sim.cost_model import CostProfile
from ..sim.engine import run
from ..traces.hashing import chain_hashes
from ..traces.schema import NormalizedRequest
from .fit import fit_decode, fit_prefill

# Token ids for the random prompts. Below the special tokens of every common vocabulary (Qwen and
# Llama put theirs above 128,000) and above the byte and control range; --token-range overrides it.
DEFAULT_TOKEN_RANGE = (1_000, 30_000)
DEFAULT_PREFILL_LENGTHS = (64, 128, 256, 512, 768, 1024, 1536, 2048)
# Prefill after a cached prefix: c cached tokens, then n new ones.
DEFAULT_PREFILL_CACHED = (256, 768, 1280)
DEFAULT_PREFILL_NEW = (64, 256, 704)
DEFAULT_DECODE_BATCHES = (1, 2, 4, 8, 16)
DEFAULT_DECODE_CONTEXTS = (32, 512, 1024)  # prompt lengths for the decode cells: the context term
DECODE_SKIP_TOKENS = 8  # gaps are timed from this token on: past every prefill in the batch


class CalibrationError(RuntimeError):
    """A precondition failed; raised before any number is written (ADR-004)."""


def metric(text: str, family: str, **labels: str) -> float | None:
    """Sum a metric family over the samples matching `labels`, or return None if there are none.

    prometheus_client appends `_total` to counters, so both spellings match.
    """
    lab = "".join(f'(?=[^}}]*{k}="{v}")' for k, v in labels.items())
    pat = re.compile(r"^" + re.escape(family) + r"(?:_total)?\{" + lab + r"[^}]*\}\s+(\S+)",
                     re.MULTILINE)
    vals = [float(x) for x in pat.findall(text)]
    return sum(vals) if vals else None


_COUNTERS = {
    "queries": "vllm:prefix_cache_queries",
    "hits": "vllm:prefix_cache_hits",
    "preemptions": "vllm:num_preemptions",
}


def _counters(text: str) -> dict[str, float | None]:
    return {k: metric(text, fam) for k, fam in _COUNTERS.items()}


class Engine:
    """A running vLLM server, reached over HTTP. Every call has a timeout and none retries."""

    def __init__(self, base: str, *, timeout_s: float = 600.0, poll_s: float = 0.5) -> None:
        self.base = base.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_s = poll_s  # between /metrics scrapes while waiting for the engine to settle

    def request(self, method: str, path: str, body: Any = None,
                timeout: float = 30.0) -> tuple[int, str]:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as exc:
            raise CalibrationError(f"cannot reach the engine at {self.base}{path}: {exc}") from None

    def metrics(self) -> str:
        code, body = self.request("GET", "/metrics")
        if code != 200:
            raise CalibrationError(f"GET /metrics returned HTTP {code}")
        return body

    def stream(self, model: str, ids: list[int], max_tokens: int) -> dict[str, Any]:
        """Run one streamed completion and record its send time, chunk arrival times and usage."""
        body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        rec: dict[str, Any] = {"t_send": time.perf_counter(), "t_first": None, "t_last": None,
                               "t_chunks": [], "usage": None, "error": None}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                for raw in resp:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    now = time.perf_counter()
                    obj = json.loads(data)
                    if obj.get("choices"):
                        if rec["t_first"] is None:
                            rec["t_first"] = now
                        rec["t_last"] = now
                        rec["t_chunks"].append(now)
                    if obj.get("usage"):
                        rec["usage"] = obj["usage"]
        except urllib.error.HTTPError as exc:
            rec["error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            rec["error"] = repr(exc)
        return rec


def settled_metrics(engine: Engine, *, max_wait_s: float = 180.0) -> str:
    """Scrape /metrics once nothing is running or waiting and the counters have stopped moving.

    Stats reach /metrics a moment after the tokens they describe, so an early read under-counts.
    """
    t0, prev = time.monotonic(), None
    while True:
        text = engine.metrics()
        busy = ((metric(text, "vllm:num_requests_running") or 0)
                + (metric(text, "vllm:num_requests_waiting") or 0))
        now = _counters(text)
        if busy == 0 and now == prev:
            return text
        if time.monotonic() - t0 > max_wait_s:
            raise CalibrationError(f"the engine never settled within {max_wait_s:.0f} s")
        prev = now
        time.sleep(engine.poll_s)


def _delta(before: str, after: str, key: str) -> float:
    b, a = _counters(before)[key], _counters(after)[key]
    return (a or 0.0) - (b or 0.0)


@dataclass
class EngineInfo:
    base: str
    model: str
    version: str
    kv_cache_blocks: int
    block_size: int
    max_num_seqs: int
    max_model_len: int
    notes: list[str] = field(default_factory=list)


def _server_info(engine: Engine) -> dict[str, Any]:
    """GET /server_info (vLLM serves it with VLLM_SERVER_DEV_MODE=1). {} when it is not served."""
    for path in ("/server_info?config_format=json", "/server_info"):
        code, body = engine.request("GET", path, timeout=60)
        if code == 200:
            try:
                info = json.loads(body)
            except ValueError:
                return {}
            cfg = info.get("vllm_config", info)
            return cfg if isinstance(cfg, dict) else {}
    return {}


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def probe(engine: Engine, *, model: str | None = None, kv_blocks: int | None = None,
          block_size: int | None = None, max_num_seqs: int | None = None) -> EngineInfo:
    """Read what the fit needs to know about the engine; values passed in take precedence.

    The KV capacity comes from `kv_blocks`, else `vllm:cache_config_info` in /metrics, else
    `GET /server_info`, and is never estimated from memory arithmetic.
    """
    code, _ = engine.request("GET", "/health")
    if code != 200:
        raise CalibrationError(f"the engine at {engine.base} is not healthy (GET /health: {code})")
    text = engine.metrics()
    if metric(text, "vllm:prefix_cache_queries") is None:
        raise CalibrationError(
            "no vllm:prefix_cache_queries in /metrics, so prefix caching is off or this is not a "
            "vLLM V1 engine. Start it with --enable-prefix-caching.")
    notes: list[str] = []

    models: list[dict[str, Any]] = []
    code, body = engine.request("GET", "/v1/models")
    if code == 200:
        try:
            models = json.loads(body).get("data") or []
        except ValueError:
            models = []
    if model is None:
        if not models:
            raise CalibrationError("GET /v1/models listed no model; pass --model")
        model = str(models[0]["id"])
    listed = next((m for m in models if m.get("id") == model), {})

    code, body = engine.request("GET", "/version")
    version = "unknown"
    if code == 200:
        try:
            version = str(json.loads(body).get("version", "unknown"))
        except ValueError:
            pass
    if version == "unknown":
        notes.append("GET /version gave no version; the profile records 'unknown'")

    server = _server_info(engine)
    cache_cfg = server.get("cache_config") or {}
    sched_cfg = server.get("scheduler_config") or {}
    model_cfg = server.get("model_config") or {}

    info_line = next((ln for ln in text.splitlines()
                      if ln.startswith("vllm:cache_config_info{")), "")
    labels = dict(re.findall(r'(\w+)="([^"]*)"', info_line))

    candidates = (
        (kv_blocks, "--kv-blocks"),
        (_as_int(labels.get("num_gpu_blocks")), "vllm:cache_config_info in /metrics"),
        (_as_int(cache_cfg.get("num_gpu_blocks")), "GET /server_info"),
        (_as_int(cache_cfg.get("num_gpu_blocks_override")), "GET /server_info (the override)"),
    )
    kv_blocks, source = next(((v, src) for v, src in candidates if v), (None, ""))
    if kv_blocks is None:
        raise CalibrationError(
            "could not read the engine's KV capacity from /metrics or /server_info. Pass "
            "--kv-blocks N: the engine's log says 'GPU KV cache size: T tokens', and N = T / block "
            "size. Never estimate it from memory arithmetic.")
    notes.append(f"KV capacity: {kv_blocks:,} blocks, from {source}")

    if block_size is None:
        block_size = _as_int(labels.get("block_size")) or _as_int(cache_cfg.get("block_size"))
    if block_size is None:
        block_size = 16
        notes.append("block size not reported by the engine; assumed 16 (vLLM's default on CUDA)")
    if max_num_seqs is None:
        max_num_seqs = _as_int(sched_cfg.get("max_num_seqs"))
    if max_num_seqs is None:
        max_num_seqs = 256
        notes.append("max_num_seqs not reported by the engine; assumed 256 — pass --max-num-seqs")
    max_len = _as_int(listed.get("max_model_len")) or _as_int(model_cfg.get("max_model_len"))
    if max_len is None:
        max_len = 2048
        notes.append("max_model_len not reported; prompts kept under 2,048 tokens")
    return EngineInfo(engine.base, model, version, kv_blocks, block_size, max_num_seqs, max_len,
                      notes)


def ols(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Fit a least-squares line and return its slope, intercept, point count and R^2.

    Raises:
        CalibrationError: If there are fewer than two distinct x values.
    """
    if len(set(xs)) < 2:
        raise CalibrationError(f"a line needs at least two distinct points, got {sorted(set(xs))}")
    slope, intercept = statistics.linear_regression(xs, ys)
    mean_y = statistics.fmean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return {"slope": slope, "intercept": intercept, "n": len(xs),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0}


def _prompt(rng: random.Random, n: int, token_range: tuple[int, int]) -> list[int]:
    return [rng.randrange(*token_range) for _ in range(n)]


def _check(rec: dict[str, Any], prompt_tokens: int, max_tokens: int) -> str | None:
    """Return why a response cannot be used, or None if it can."""
    u = rec["usage"] or {}
    if rec["error"]:
        return rec["error"]
    if rec["t_first"] is None:
        return "no token was streamed"
    if u.get("prompt_tokens") != prompt_tokens:
        return f"the engine counted {u.get('prompt_tokens')} prompt tokens, we sent {prompt_tokens}"
    if u.get("completion_tokens") != max_tokens:
        return f"{u.get('completion_tokens')} tokens came back, {max_tokens} were asked for"
    return None


def measure_prefill(engine: Engine, info: EngineInfo, rng: random.Random, *,
                    lengths: tuple[int, ...], reps: int, token_range: tuple[int, int],
                    cached: tuple[int, ...] = (), new: tuple[int, ...] = (),
                    notes: list[str] | None = None) -> list[dict[str, float]]:
    """Measure TTFT from sequential prompts with one output token.

    Cold prompts of each length come first, then `new` tokens after each `cached` prefix, which a
    request of its own warms first. Agentic traffic is mostly a long cached prefix with few new
    tokens, which cold prompts never measure. The cached-token count is the engine's own report, or
    the warmed prefix when the engine gives none.

    Returns:
        One dict per request with "rep", "prompt_tokens", "uncached_tokens" and "ttft_s".
    """
    cap = info.max_model_len - 8
    lengths = tuple(sorted({min(n, cap) for n in lengths if n > 0}))
    prefixes = tuple(sorted({c - c % info.block_size for c in cached if c >= info.block_size}))
    for _ in range(2):  # warm-up: first requests after start-up pay one-off costs
        engine.stream(info.model, _prompt(rng, 32, token_range), 1)

    def timed(ids: list[int], assumed_cached: int, what: str, rep: int) -> dict[str, float]:
        rec = engine.stream(info.model, ids, 1)
        why = _check(rec, len(ids), 1)
        if why:
            raise CalibrationError(f"prefill request ({what}) failed: {why}")
        reported = ((rec["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if reported is None and assumed_cached and notes is not None and not notes:
            notes.append("the engine does not report per-request cached tokens "
                         "(--enable-prompt-tokens-details); each warmed prefix was assumed cached")
        hit = assumed_cached if reported is None else reported
        return {"rep": rep, "prompt_tokens": len(ids), "uncached_tokens": len(ids) - hit,
                "ttft_s": rec["t_first"] - rec["t_send"]}

    points = []
    for rep in range(reps):
        for n in lengths:
            points.append(timed(_prompt(rng, n, token_range), 0, f"{n} cold tokens", rep))
        for c in prefixes:
            for n in new:
                if n <= 0 or c + n > cap:
                    continue
                prefix = _prompt(rng, c, token_range)
                warm = engine.stream(info.model, prefix, 1)
                why = _check(warm, c, 1)
                if why:
                    raise CalibrationError(f"warming a {c}-token prefix failed: {why}")
                points.append(timed(prefix + _prompt(rng, n, token_range), c,
                                    f"{n} new tokens after {c} cached", rep))
    return points


def measure_decode(engine: Engine, info: EngineInfo, rng: random.Random, *,
                   batches: tuple[int, ...], contexts: tuple[int, ...], output_tokens: int,
                   token_range: tuple[int, int], skip: int = DECODE_SKIP_TOKENS,
                   reps: int = 1) -> list[dict[str, Any]]:
    """Measure the decode step time for each context length and batch size that fits the cache.

    A cell is b concurrent requests of L random tokens. A request's step is its mean gap from token
    `skip` on, and a cell's step the median over its requests; a cell that preempted is marked
    invalid. The whole set is measured `reps` times, one full pass after another, so a slow drift of
    the engine shows as a difference between repeats rather than hiding inside one cell.
    """
    rows = []
    budget = int(0.8 * info.kv_cache_blocks)  # headroom: a preempted cell times recomputation
    for rep in range(reps):
        for L in sorted({min(x, info.max_model_len - output_tokens - 1)
                         for x in contexts if x > 0}):
            per = -(-(L + output_tokens) // info.block_size)
            for b in sorted({x for x in batches if 1 <= x <= info.max_num_seqs}):
                if b * per > budget:
                    continue
                before = settled_metrics(engine)
                prompts = [_prompt(rng, L, token_range) for _ in range(b)]
                with ThreadPoolExecutor(max_workers=b) as pool:
                    recs = list(pool.map(
                        lambda ids: engine.stream(info.model, ids, output_tokens), prompts))
                preemptions = _delta(before, settled_metrics(engine), "preemptions")
                steps = sorted((r["t_chunks"][-1] - r["t_chunks"][skip])
                               / (len(r["t_chunks"]) - 1 - skip)
                               for r in recs if _check(r, L, output_tokens) is None
                               and len(r["t_chunks"]) == output_tokens)
                rows.append({
                    "rep": rep, "batch": b, "context": L,
                    "step_s": statistics.median(steps) if steps else None,
                    "requests_ok": len(steps), "preemptions": preemptions,
                    # A cell that preempted timed recomputation, not decode steps.
                    "valid": len(steps) == b and preemptions == 0,
                })
    return rows


def fit(prefill: list[dict[str, float]], decode: list[dict[str, Any]], *,
        output_tokens: int, skip: int = DECODE_SKIP_TOKENS) -> dict[str, Any]:
    """Fit the context-aware model of calib/fit.py: decode on the valid cells, then prefill.

    One fitted decode step is subtracted from each TTFT first, because the simulator emits the first
    token at the end of the step that prefills.
    """
    ok = [r for r in decode if r["valid"]]
    if len({(r["batch"], r["context"]) for r in ok}) < 3:
        raise CalibrationError(f"fewer than three usable decode cells ({len(ok)}): no decode fit")
    dec = fit_decode(ok, (skip + output_tokens) / 2.0)
    pre = fit_prefill([{"cached_tokens": p["prompt_tokens"] - p["uncached_tokens"],
                        "uncached_tokens": p["uncached_tokens"], "ttft_s": p["ttft_s"]}
                       for p in prefill], dec)
    notes = []
    if not dec["per_ctx_token_s"] and not pre["attn_s"]:
        notes.append("both context terms fitted to 0: on this engine cost did not grow with "
                     "context over the lengths measured")
    return {"prefill_base_s": pre["base_s"], "prefill_per_token_s": pre["per_token_s"],
            "prefill_attn_s": pre["attn_s"], "decode_base_s": dec["base_s"],
            "decode_per_seq_s": dec["per_seq_s"], "decode_per_ctx_token_s": dec["per_ctx_token_s"],
            "prefill_fit": pre, "decode_fit": dec, "notes": notes}


COEFFICIENTS = ("prefill_base_s", "prefill_per_token_s", "prefill_attn_s", "decode_base_s",
                "decode_per_seq_s", "decode_per_ctx_token_s")


def fit_replicates(prefill: list[dict[str, float]], decode: list[dict[str, Any]], *,
                   output_tokens: int, skip: int = DECODE_SKIP_TOKENS) -> list[dict[str, Any]]:
    """Fit each repeat's measurements alone, to show how much the coefficients move between runs.

    `kvsim replay` re-runs its sizing on each fit and shows the range of replica counts: a floor on
    the uncertainty, not a confidence interval. A repeat with too few usable decode cells is left
    out, and fewer than two fits give an empty list.
    """
    reps = sorted({p.get("rep", 0) for p in prefill} | {r.get("rep", 0) for r in decode})
    if len(reps) < 2:
        return []
    out = []
    for rep in reps:
        try:
            f = fit([p for p in prefill if p.get("rep", 0) == rep],
                    [r for r in decode if r.get("rep", 0) == rep],
                    output_tokens=output_tokens, skip=skip)
        except CalibrationError:
            continue
        out.append({"rep": rep, **{k: f[k] for k in COEFFICIENTS},
                    "decode_cells": f["decode_fit"]["points"],
                    "prefill_points": f["prefill_fit"]["points"]})
    return out if len(out) >= 2 else []


def check_workload(rng: random.Random, sessions: int, turns: int,
                   token_range: tuple[int, int]) -> list[tuple[str, list[int]]]:
    """Build sessions that share one system prompt, each re-sending its own growing history.

    No length is a multiple of 16, so the partial-block rule is exercised, and the shared prompt
    (150 tokens) ends mid-block at block sizes 16 and 32, so a wrong block size changes every
    session's first hit.
    """
    shared = _prompt(rng, 150, token_range)
    out = []
    for s in range(sessions):
        history = shared + _prompt(rng, 45, token_range)
        for _t in range(turns):
            history = history + _prompt(rng, 27, token_range)  # the user's turn
            out.append((f"s{s}", list(history)))
            history = history + _prompt(rng, 38, token_range)  # the assistant's reply, as text
    return out


def predict_check(work: list[tuple[str, list[int]]], profile: CostProfile):
    """Simulate the check workload on one replica with the block-pool model.

    Requests are 1,000 simulated seconds apart, far longer than any takes, so they run one at a
    time, as they are sent to the engine.
    """
    trace = [NormalizedRequest(f"c{i}", i * 1000.0, chain_hashes(ids, profile.block_size), len(ids),
                               1, sid) for i, (sid, ids) in enumerate(work)]
    return run(trace, policy="round-robin", n_replicas=1, profile=profile, cache_model="vllm",
               record_per_request=True)


def check_cache_model(engine: Engine, info: EngineInfo, profile: CostProfile,
                      rng: random.Random, *, sessions: int, turns: int,
                      token_range: tuple[int, int]) -> dict[str, Any]:
    """Compare the engine's prefix-cache counters over a sequential workload with the simulator's.

    The workload fits in the cache, so this checks units, block size, the partial-block rule and
    prefix matching on the engine, but not eviction under pressure.
    """
    work = check_workload(rng, sessions, turns, token_range)
    blocks = len({h for _, ids in work for h in chain_hashes(ids, info.block_size)}) + len(work)
    if blocks > info.kv_cache_blocks // 2:
        raise CalibrationError(
            f"the check needs ~{blocks:,} blocks, more than half this engine's cache "
            f"({info.kv_cache_blocks:,}); pass fewer --check-sessions")
    # A cold cache if the engine allows it (vLLM serves the reset with VLLM_SERVER_DEV_MODE=1).
    # Without one the check still holds: it fits in half the cache and uses fresh random prompts,
    # so older blocks are evicted before any of its own.
    code, body = engine.request("POST", "/reset_prefix_cache", timeout=60)
    try:
        reset = code == 200 and json.loads(body or "{}").get("success", True) is True
    except (ValueError, AttributeError):
        reset = False
    before = settled_metrics(engine)
    measured_cached = {}
    for i, (_sid, ids) in enumerate(work):
        rec = engine.stream(info.model, ids, 1)
        why = _check(rec, len(ids), 1)
        if why:
            raise CalibrationError(f"check request {i} failed: {why}")
        details = (rec["usage"] or {}).get("prompt_tokens_details") or {}
        measured_cached[f"c{i}"] = details.get("cached_tokens")
    after = settled_metrics(engine)
    queries, hits = _delta(before, after, "queries"), _delta(before, after, "hits")
    if queries <= 0:
        raise CalibrationError("the check moved no prefix-cache query counter")

    sim = predict_check(work, profile)
    measured, predicted = hits / queries, sim.hit_rate
    compared = {k: v for k, v in measured_cached.items() if v is not None}
    differ = sorted((k for k, v in compared.items() if sim.per_request[k]["cached_tokens"] != v),
                    key=lambda k: int(k[1:]))
    return {
        "requests": len(work),
        "sessions": sessions,
        "turns": turns,
        "prefix_cache_reset": bool(reset),
        "measured": {"queries": queries, "hits": hits, "hit_rate": measured},
        "predicted": {"queries": sim.cache_queries, "hits": sim.cache_hits,
                      "hit_rate": predicted},
        "relative_error": abs(measured - predicted) / measured if measured else None,
        "cached_tokens_compared": len(compared),
        "cached_tokens_equal": len(compared) - len(differ),
        "requests_that_differ": [
            {"request": k, "engine": compared[k], "model": sim.per_request[k]["cached_tokens"]}
            for k in differ[:10]
        ],
    }


def _local_gpu(base: str) -> str:
    """Return the GPU name from nvidia-smi when the engine runs on this machine, else ''."""
    host = urllib.parse.urlparse(base).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1") or not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip().splitlines()[0].strip() if out.strip() else ""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-") or "unknown"


def calibrate(engine: Engine, *, model: str | None = None, gpu: str | None = None,
              name: str | None = None, kv_blocks: int | None = None,
              block_size: int | None = None, max_num_seqs: int | None = None,
              prefill_lengths: tuple[int, ...] = DEFAULT_PREFILL_LENGTHS, reps: int = 3,
              prefill_cached: tuple[int, ...] = DEFAULT_PREFILL_CACHED,
              prefill_new: tuple[int, ...] = DEFAULT_PREFILL_NEW,
              decode_batches: tuple[int, ...] = DEFAULT_DECODE_BATCHES,
              decode_contexts: tuple[int, ...] = DEFAULT_DECODE_CONTEXTS,
              decode_tokens: int = 64, check_sessions: int = 16, check_turns: int = 3,
              max_error: float = 0.05, seed: int = 0,
              token_range: tuple[int, int] = DEFAULT_TOKEN_RANGE,
              log=print) -> tuple[dict[str, Any], bool]:
    """Probe the engine, fit the timing profile and check the cache model.

    The default prompt lengths reach 2,048 tokens and decode contexts 1,024; timings for longer
    traffic are extrapolated. The profile is marked calibrated only if the hit-rate error is within
    `max_error` and every per-request cached-token count the engine reports matches the model.

    Returns:
        The profile as a JSON-ready dict, and whether it is calibrated.
    """
    rng = random.Random(seed)
    info = probe(engine, model=model, kv_blocks=kv_blocks, block_size=block_size,
                 max_num_seqs=max_num_seqs)
    gpu = gpu or _local_gpu(engine.base) or "unknown"
    log(f"engine   {info.base} · vLLM {info.version} · {info.model} · GPU {gpu}")
    for note in info.notes:
        log(f"         {note}")

    prefill_notes: list[str] = []
    prefill = measure_prefill(engine, info, rng, lengths=prefill_lengths, reps=reps,
                              token_range=token_range, cached=prefill_cached, new=prefill_new,
                              notes=prefill_notes)
    decode = measure_decode(engine, info, rng, batches=decode_batches, contexts=decode_contexts,
                            output_tokens=decode_tokens, token_range=token_range, reps=reps)
    f = fit(prefill, decode, output_tokens=decode_tokens)
    replicates = fit_replicates(prefill, decode, output_tokens=decode_tokens)
    cells = [r for r in decode if r["valid"]]
    log(f"decode   {f['decode_base_s'] * 1e3:.1f} ms + {f['decode_per_seq_s'] * 1e3:.2f} ms × "
        f"batch + {f['decode_per_ctx_token_s'] * 1e6:.2f} ms per 1K context tokens  (R² "
        f"{f['decode_fit']['r2']:.2f}, {len(cells)} measurements of "
        f"{len({(r['batch'], r['context']) for r in cells})} cells: contexts "
        f"{' '.join(str(c) for c in sorted({r['context'] for r in cells}))})")
    log(f"prefill  {f['prefill_per_token_s'] * 1e3:.3f} ms per new token + "
        f"{f['prefill_attn_s'] * 1e6:.3f} ms per new token per 1K attended + "
        f"{f['prefill_base_s'] * 1e3:.1f} ms  (R² {f['prefill_fit']['r2']:.2f}, "
        f"{len(prefill)} requests, "
        f"{sum(1 for q in prefill if q['uncached_tokens'] < q['prompt_tokens'])} after a cached "
        "prefix)")
    if replicates:
        def span(key: str, scale: float, digits: int) -> str:
            vals = [r[key] * scale for r in replicates]
            return f"{min(vals):.{digits}f}–{max(vals):.{digits}f}"

        log(f"repeats  the same fit on each of the {len(replicates)} repeats alone: decode "
            f"{span('decode_base_s', 1e3, 1)} ms + {span('decode_per_seq_s', 1e3, 2)} × batch + "
            f"{span('decode_per_ctx_token_s', 1e6, 2)} per 1K context; prefill "
            f"{span('prefill_per_token_s', 1e3, 3)} per token + "
            f"{span('prefill_attn_s', 1e6, 3)} per 1K attended")
    f["notes"] = [*prefill_notes, *f["notes"]]
    for note in f["notes"]:
        log(f"         {note}")

    profile = CostProfile(
        name=name or f"{_slug(gpu)}-{_slug(info.model.split('/')[-1])}-vllm{_slug(info.version)}",
        prefill_base_s=f["prefill_base_s"],
        prefill_per_token_s=f["prefill_per_token_s"],
        decode_base_s=f["decode_base_s"],
        decode_per_seq_s=f["decode_per_seq_s"],
        prefill_attn_s=f["prefill_attn_s"],
        decode_per_ctx_token_s=f["decode_per_ctx_token_s"],
        kv_cache_blocks=info.kv_cache_blocks,
        block_size=info.block_size,
        max_num_seqs=info.max_num_seqs,
        gpu=gpu,
        model=info.model,
        engine_version=info.version,
        calibrated=False,
    )
    # The check has its own random stream, so its prompts depend only on the seed, not on how many
    # timing requests came first, and its expected hit rate can be computed in advance.
    check = check_cache_model(engine, info, profile, random.Random(f"check-{seed}"),
                              sessions=check_sessions, turns=check_turns, token_range=token_range)
    err = check["relative_error"]
    n_cmp, n_eq = check["cached_tokens_compared"], check["cached_tokens_equal"]
    # Both must hold. On a sequential workload that fits in the cache the model should be exact, so
    # one request whose cached tokens differ means a rule differs on this engine, even when the
    # aggregate error is small (a wrong block size moves it by only a few percent).
    ok = err is not None and err <= max_error and n_eq == n_cmp
    log(f"check    hit rate: engine {check['measured']['hit_rate']:.4f}, model "
        f"{check['predicted']['hit_rate']:.4f}, error {err:.2%} over {check['requests']} requests")
    if n_cmp:
        log(f"         per-request cached tokens: {n_eq} of {n_cmp} equal")
    else:
        log("         the engine does not report per-request cached tokens; start it with "
            "--enable-prompt-tokens-details to check every request, not only the total")

    out = asdict(profile)
    out["calibrated"] = ok
    out["calibration_error"] = None if err is None else round(err, 6)
    out = {
        "_what": (f"Fitted by `kvsim calibrate` against {info.base}. Timings are figures for "
                  f"{gpu} serving {info.model} on vLLM {info.version}; the capacity is what that "
                  "engine allocated."),
        "_source": {
            "tool": f"kvsim calibrate (kv-routing-sim {__version__})",
            "date_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "engine": asdict(info),
            "seed": seed,
            "max_error": max_error,
            "prefill_fit": f["prefill_fit"],
            "decode_fit": f["decode_fit"],
            "decode_cells": decode,
            "prefill_points": prefill,
            # The fit on each repeat alone; `kvsim replay` sizes on each and shows the range.
            "replicates": replicates,
            "check": check,
            "notes": f["notes"],
        },
        **out,
    }
    return out, ok
