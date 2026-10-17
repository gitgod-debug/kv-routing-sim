"""The engine's timing profile: prefill and decode step costs, and the KV cache capacity.

    prefill(n, c) = prefill_base_s + prefill_per_token_s * n + prefill_attn_s * n * (c + n / 2)
    decode(b, S)  = decode_base_s  + decode_per_seq_s    * b + decode_per_ctx_token_s * S

Here n is the uncached tokens computed, c the cached tokens before them, b the sequences in the step
and S the sum of their context lengths (prompt plus tokens generated so far). Prefill scales with
the tokens actually computed, which is what prefix caching reduces; the two context terms default to
0.0, the context-free model.
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
    """Engine timing and cache capacity for one combination of GPU, model and engine flags.

    A profile does not transfer to other hardware (ADR-010); `kvsim calibrate` measures a new one.
    The identity fields record the setup it was measured on, and results from a profile that is not
    `calibrated` are not presented as predictions.
    """

    name: str
    prefill_base_s: float
    prefill_per_token_s: float
    decode_base_s: float
    decode_per_seq_s: float
    kv_cache_blocks: int
    block_size: int = 16
    max_num_seqs: int = 256

    # Identity: what the profile was measured on; empty when the profile is a guess.
    gpu: str = ""
    model: str = ""
    engine_version: str = ""
    calibrated: bool = False
    calibration_error: float | None = None  # relative hit-rate error vs the live engine

    # The context terms; 0.0 gives the context-free model.
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
        """Return what a results file records about where its timing came from."""
        d = asdict(self)
        d["WARNING"] = (
            "UNCALIBRATED: timings are plausible shapes, not measurements. The ranking of "
            "policies may still be informative, though it changes with the hardware; absolute "
            "latency and cost are not. Run `kvsim calibrate` against a live engine."
            if not self.calibrated
            else f"calibrated on {self.gpu} / {self.model} / {self.engine_version}"
        )
        return d


def load_profile(path: str | Path) -> CostProfile:
    """Load a profile from a JSON file such as `configs/profiles/t4-qwen3-4b-vllm0.30.0.json`.

    A missing or unknown field raises (ADR-004), so a mistyped key cannot fall back to a default.
    Keys starting with `_` are provenance for the reader and are ignored.
    """
    return _profile_from_json(Path(path).read_text(encoding="utf-8"), str(path))


def resolve_profile(spec: str | Path) -> CostProfile:
    """Load a profile from a file, or by the name of one shipped with the package.

    An existing file wins; otherwise the name must match a shipped profile, and anything else raises
    FileNotFoundError listing the shipped names (ADR-004).
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
    """Return the profile refitted on each repeat of its calibration alone.

    `kvsim calibrate` writes these under `_source.replicates`; re-running the sizing on each shows
    how firm a replica count is. The list is empty for a profile without them, which means the
    spread is unknown, not zero.
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


# An uncalibrated guess: order-of-magnitude figures for a Tesla T4 serving Qwen3-4B in fp16, so the
# simulator runs without a measured profile. T4 fp16 tensor cores peak at ~65 TFLOPS and a 4B model
# costs ~8 GFLOP per prefill token, so ~35% utilisation gives ~0.35 ms/token. Decode is
# memory-bound: ~7.5 GiB of weights over ~320 GB/s is ~25 ms per step at batch 1.
UNCALIBRATED_DEFAULT = CostProfile(
    name="uncalibrated-t4-qwen3-4b-shape",
    prefill_base_s=0.010,  # fixed per-batch overhead
    prefill_per_token_s=0.00035,  # ~0.35 ms/token (see above)
    decode_base_s=0.025,  # one decode step: read every weight once
    decode_per_seq_s=0.0005,  # marginal cost of one more sequence in the batch
    # Computed from cache/capacity.py for qwen3-4b on one 15 GiB T4 at 0.90 utilisation (~2,736
    # blocks). vLLM also reserves memory for CUDA graphs and activations, so the real figure is
    # lower; a calibrated profile reads it from the engine.
    kv_cache_blocks=kv_blocks_for(SHAPES["qwen3-4b"], 15.0),
    block_size=16,
    max_num_seqs=256,
    calibrated=False,
)
