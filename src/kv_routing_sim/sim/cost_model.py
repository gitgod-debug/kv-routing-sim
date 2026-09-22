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

    prefill(uncached_tokens) = prefill_base_s + prefill_per_token_s * uncached_tokens
    decode(batch_size)       = decode_base_s  + decode_per_seq_s   * batch_size

Prefill is compute-bound and scales with the tokens actually computed — which is exactly the
quantity prefix caching reduces, and therefore exactly the quantity routing controls. Decode is
memory-bandwidth-bound and scales with how many sequences share the step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..cache.capacity import SHAPES, kv_blocks_for


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

    def prefill_s(self, uncached_tokens: int) -> float:
        return self.prefill_base_s + self.prefill_per_token_s * max(0, uncached_tokens)

    def decode_s(self, batch_size: int) -> float:
        return self.decode_base_s + self.decode_per_seq_s * max(0, batch_size)

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
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: a profile must be a JSON object")
    data = {k: v for k, v in raw.items() if not k.startswith("_")}
    known = {f.name for f in fields(CostProfile)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown profile field(s) {unknown}; known: {sorted(known)}")
    try:
        return CostProfile(**data)
    except TypeError as exc:  # a required field is missing
        raise ValueError(f"{path}: {exc}") from None


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
