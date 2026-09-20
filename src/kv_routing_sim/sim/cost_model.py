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

from dataclasses import asdict, dataclass


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


# ⛔ A GUESS. Order-of-magnitude shapes for an A100-class GPU serving an 8B model, used so the
# simulator runs at all before Phase 3. Replaced by a measured profile; not to be cited.
UNCALIBRATED_DEFAULT = CostProfile(
    name="uncalibrated-default",
    prefill_base_s=0.004,  # fixed per-batch overhead
    prefill_per_token_s=0.00002,  # ~20 us/token => ~50k tokens/s of prefill
    decode_base_s=0.006,  # a decode step with an empty batch
    decode_per_seq_s=0.0004,  # marginal cost of one more sequence in the batch
    kv_cache_blocks=28_672,  # from cache.capacity arithmetic: llama3-8b on 80 GiB
    block_size=16,
    max_num_seqs=256,
    calibrated=False,
)
