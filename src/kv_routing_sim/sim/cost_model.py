"""The engine's time cost — and a hard refusal to pretend it has been measured.

⛔ THE DEFAULTS IN THIS FILE ARE NOT CALIBRATED. They are plausible shapes, not predictions.

This is the ONE place where the Phase-0 gate really bites. Everything else in this package is
algorithm — a trie, six policies, some statistics — and is correct or incorrect on its own terms.
These four numbers are a claim about how a real GPU behaves, and we have not yet run one. So the
profile carries a `calibrated` flag, every result carries the profile that produced it, and the
reporting layer refuses to present an uncalibrated run as a prediction (`eval/`).

`Mike-Ew/kv-cache-routing` is the cautionary case (docs/04-prior-art.md): they designed against
documentation, never started an engine, and shipped a README citing a metric that does not exist.
The difference between that repo and this one is not the design — it is whether the numbers were
ever checked. This flag is where that difference is enforced in code rather than in intention.

THE MODEL (ADR-001, iteration granularity):

    prefill(n, c) = prefill_base_s + prefill_per_token_s * n + prefill_attn_s * n * (c + n / 2)
    decode(b, S)  = decode_base_s  + decode_per_seq_s    * b + decode_per_ctx_token_s * S

n = uncached tokens computed, c = cached tokens before them, b = sequences in the step, S = the
sum of their context lengths (prompt + tokens generated so far).

The prefill form is not this project's invention: with L = c + n it is c_lin·n + c_attn·n·(L − n/2),
the cost model SMetric (arXiv 2607.08565, Eq. 1) takes from KunServe (EuroSys '26). C2 derived it
from the cost of attention without knowing either; what is this project's is the T4 fit and the
registered tests that it predicts traffic it never saw (docs/04-prior-art.md §2a).

Prefill is compute-bound and scales with the tokens actually computed — which is exactly the
quantity prefix caching reduces, and therefore exactly the quantity routing controls. Decode is
memory-bandwidth-bound and scales with how many sequences share the step.

⚠️ THE TWO CONTEXT TERMS (added 2026-09-28, after Phase 6): attention reads every earlier token, so
both costs also grow with context. Phase 6 found the context-free model optimistic by 2.7–4.0× per
decode step on ~1,700-token prompts, and C1 found prefill per token rising from 0.71 ms (≤ 1,024
tokens) to 1.12 ms (2,040). Both terms default to 0.0, which reproduces the old model exactly —
every profile fitted before C2 keeps its results to the bit. C2 (configs/c2_timing.yaml) fits them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from ..cache.capacity import SHAPES, kv_blocks_for

# The six coefficients of the timing model above. Everything else in a profile is capacity or
# provenance: a second calibration of the same engine changes these and nothing else.
TIMING_FIELDS = ("prefill_base_s", "prefill_per_token_s", "prefill_attn_s", "decode_base_s",
                 "decode_per_seq_s", "decode_per_ctx_token_s")


@dataclass(frozen=True, slots=True)
class CostProfile:
    """Engine timing + cache capacity for one (GPU, model, flags) combination.

    ⭐ ADR-010: a profile is the unit that does NOT transfer. A user running different hardware
    needs their own, which is what `kvsim calibrate` is for. The identity fields exist so a profile
    can never be silently reused outside the setup it was measured on.
    """

    name: str
    prefill_base_s: float
    prefill_per_token_s: float
    decode_base_s: float
    decode_per_seq_s: float
    kv_cache_blocks: int
    block_size: int = 16
    max_num_seqs: int = 256

    # ── Identity: what this profile was measured on. Empty means "nothing; it is a guess." ──
    gpu: str = ""
    model: str = ""
    engine_version: str = ""
    calibrated: bool = False
    calibration_error: float | None = None  # relative hit-rate error vs the live engine

    # ── The context terms (C2). 0.0 = the context-free model, exactly as before. ──
    prefill_attn_s: float = 0.0  # per computed token, per token it attends to
    decode_per_ctx_token_s: float = 0.0  # per context token summed over the batch

    def prefill_s(self, uncached_tokens: int, cached_tokens: int = 0) -> float:
        n = max(0, uncached_tokens)
        cost = self.prefill_base_s + self.prefill_per_token_s * n
        if self.prefill_attn_s:
            cost += self.prefill_attn_s * n * (max(0, cached_tokens) + n / 2.0)
        return cost

    def decode_s(self, batch_size: int, context_tokens: int = 0) -> float:
        cost = self.decode_base_s + self.decode_per_seq_s * max(0, batch_size)
        if self.decode_per_ctx_token_s:
            cost += self.decode_per_ctx_token_s * max(0, context_tokens)
        return cost

    @property
    def context_aware(self) -> bool:
        return bool(self.prefill_attn_s or self.decode_per_ctx_token_s)

    def provenance(self) -> dict[str, object]:
        """What every results file records about where its timing came from."""
        d = asdict(self)
        d["WARNING"] = (
            "UNCALIBRATED — timings are plausible shapes, not measurements. "
            "Policy RANKING may still be informative (H6); absolute latency and cost are not. "
            "Run `kvsim calibrate` against a live engine."
            if not self.calibrated
            else f"calibrated on {self.gpu} / {self.model} / {self.engine_version}"
        )
        return d


def load_profile(path: str | Path) -> CostProfile:
    """A profile from a JSON file — e.g. `configs/profiles/t4-qwen3-4b-vllm0.30.0.json`.

    Fails LOUDLY on a missing or unknown field (ADR-004): a mistyped key quietly falling back to a
    default is exactly how an uncalibrated number ends up published as a calibrated one. Keys that
    start with `_` are provenance for the reader, and are ignored here.
    """
    return _profile_from_json(Path(path).read_text(encoding="utf-8"), str(path))


def resolve_profile(spec: str | Path) -> CostProfile:
    """A profile from a file, or by the NAME of one shipped with the package — so the same command
    works from a clone and after `pip install` (`--profile t4-qwen3-4b-vllm0.30.0`).

    ⛔ No guessing between the two: an existing file wins, a bare shipped name is the only
    fallback, and anything else fails with the shipped names listed (ADR-004).
    """
    path = Path(spec)
    if path.is_file():
        return load_profile(path)
    from .. import data

    shipped = data.profiles()
    name = str(spec).removesuffix(".json")
    if name in shipped:
        return _profile_from_json(shipped[name].read_text(encoding="utf-8"), f"shipped {name}")
    raise FileNotFoundError(
        f"no profile file {str(spec)!r}, and no shipped profile of that name. Shipped: "
        f"{', '.join(sorted(shipped)) or 'none'}. Or measure your own: kvsim calibrate"
    )


def replicate_profiles(spec: str | Path) -> list[CostProfile]:
    """The profile's REPLICATES: the same profile with the timing coefficients fitted on each
    repeat of the calibration alone (`kvsim calibrate` writes them under `_source.replicates`).

    They exist to answer "how firm is this replica count?": sizing is re-run on each, and the
    spread is shown. Empty for a profile without them — the two shipped ones, and any profile
    written before 2026-10-05 — which is a statement that the spread is unknown, not that it is 0.
    """
    path = Path(spec)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        from .. import data

        shipped = data.profiles()
        name = str(spec).removesuffix(".json")
        if name not in shipped:
            return []
        text = shipped[name].read_text(encoding="utf-8")
    raw = json.loads(text)
    reps = ((raw.get("_source") or {}).get("replicates") if isinstance(raw, dict) else None) or []
    if not reps:
        return []
    base = _profile_from_json(text, str(spec))
    return [replace(base, name=f"{base.name}#repeat{r.get('rep', i)}",
                    **{k: float(r[k]) for k in TIMING_FIELDS})
            for i, r in enumerate(reps)]


def _profile_from_json(text: str, where: str) -> CostProfile:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: a profile must be a JSON object")
    data = {k: v for k, v in raw.items() if not k.startswith("_")}
    known = {f.name for f in fields(CostProfile)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{where}: unknown profile field(s) {unknown}; known: {sorted(known)}")
    try:
        return CostProfile(**data)
    except TypeError as exc:  # a required field is missing
        raise ValueError(f"{where}: {exc}") from None


# ⛔ A GUESS. Order-of-magnitude shapes for a Tesla T4 serving Qwen3-4B in fp16 — the hardware and
# model this project actually calibrates on (decided 2026-09-22; A100/H100 is not in the plan). Used
# so the simulator runs before Phase 3 replaces it with a measured profile. Not to be cited.
#
# The reasoning, so a reader can disagree: T4 fp16 tensor cores peak ~65 TFLOPS; a 4B model costs
# ~8 GFLOP per prefill token; at ~35% utilisation that is ~0.35 ms/token. Decode is memory-bound:
# ~7.5 GiB of weights over ~320 GB/s is ~25 ms per step at batch 1.
UNCALIBRATED_DEFAULT = CostProfile(
    name="uncalibrated-t4-qwen3-4b-shape",
    prefill_base_s=0.010,  # fixed per-batch overhead
    prefill_per_token_s=0.00035,  # ~0.35 ms/token (see above)
    decode_base_s=0.025,  # one decode step: read every weight once
    decode_per_seq_s=0.0005,  # marginal cost of one more sequence in the batch
    # DERIVED, not typed: qwen3-4b on one 15 GiB T4 at 0.90 util (~2,736 blocks). Computed from
    # cache/capacity.py so the two can never disagree. vLLM also reserves memory for CUDA graphs and
    # activations, so the real figure is LOWER — Phase 3 reads it from the engine's own log.
    kv_cache_blocks=kv_blocks_for(SHAPES["qwen3-4b"], 15.0),
    block_size=16,
    max_num_seqs=256,
    calibrated=False,
)
