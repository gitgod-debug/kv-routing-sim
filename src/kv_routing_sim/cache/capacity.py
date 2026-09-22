"""How many KV blocks a replica can hold — and an honest admission that we have not measured it.

⛔ EVERY NUMBER IN THIS FILE IS UNCALIBRATED. It is derived from arithmetic, not from an engine.
The Phase-0/Phase-3 calibration replaces it with what vLLM actually reports, and until then anything
computed from it is a shape, not a prediction. The code refuses to call itself calibrated
(`CalibrationProfile.calibrated is False`), so a number cannot quietly acquire authority it has not
earned.

The arithmetic, so a reader can disagree with it:

    bytes per token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes
    bytes per block = bytes per token x block_size
    blocks          = (gpu_memory x utilization - weights) / bytes per block
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ModelShape:
    """Just enough of a model to size its KV cache."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16/bf16
    weight_bytes: int = 0

    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


def kv_blocks_for(
    model: ModelShape,
    gpu_memory_gib: float,
    *,
    block_size: int = 16,
    gpu_memory_utilization: float = 0.90,
) -> int:
    """Blocks that fit on one replica. ⚠️ Arithmetic, not a measurement — see the module docstring.

    Replaced by the calibration. Until then it is a shape, not a prediction.

    ⚠️ MEASURED 2026-09-22 (vLLM 0.30.0, Qwen3-0.6B, T4 @ 0.85): this OVERESTIMATES the engine
    by 7-12.5%, because vLLM also reserves memory for CUDA graphs and activations. Two launches with
    identical flags differed by 4.8% (94,496 vs 99,024 tokens). So never calibrate with this —
    read "GPU KV cache size: N tokens" from each run's own log.
    """
    usable = gpu_memory_gib * GIB * gpu_memory_utilization - model.weight_bytes
    if usable <= 0:
        raise ValueError(
            f"no room for a KV cache: {gpu_memory_gib} GiB at {gpu_memory_utilization} "
            f"utilization does not fit {model.weight_bytes / GIB:.1f} GiB of weights"
        )
    return max(1, int(usable // (model.bytes_per_token() * block_size)))


# Reference shapes, for making the default workload runnable. Architecture figures are public;
# ⚠️ none of these has been checked against a running engine on our hardware.
SHAPES = {
    # Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128, ~16 GiB of fp16 weights
    "llama3-8b": ModelShape("llama3-8b", 32, 8, 128, 2, 16 * GIB),
    # ✅ Read from each model's own config.json on 2026-09-21, not guessed:
    # Qwen3-0.6B — the Phase-0 spike model. Ungated, ~1.5 GB of weights.
    #   28 layers, 8 KV heads, head_dim 128 -> 112 KiB of KV per token.
    "qwen3-0.6b": ModelShape("qwen3-0.6b", 28, 8, 128, 2, 1_500_000_000),
    # Qwen3-8B — the Phase-3 calibration model. Ungated, unlike meta-llama/* (gated=manual).
    #   36 layers, 8 KV heads, head_dim 128 -> 144 KiB of KV per token, ~16 GB of bf16 weights.
    "qwen3-8b": ModelShape("qwen3-8b", 36, 8, 128, 2, 16 * GIB),
    # ⭐ Qwen3-4B — THE Phase-3 calibration model (decided 2026-09-22, Kaggle 2x T4).
    #   Qwen3-8B does not fit one T4 in fp16 (15.26 GiB of weights vs 13.5 GiB usable).
    #   4B's KV cache is architecturally IDENTICAL to 8B's — 36 layers, 8 KV heads, head_dim 128,
    #   144 KiB/token — so for calibrating a CACHE model it is a stand-in, not a downgrade.
    #   4.02B params -> 7.49 GiB of fp16 weights (T4 has no bf16).
    "qwen3-4b": ModelShape("qwen3-4b", 36, 8, 128, 2, 8_040_000_000),
}
