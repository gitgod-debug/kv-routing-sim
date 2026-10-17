"""Estimate how many KV blocks a replica can hold from a model's shape and a memory budget.

The estimate is arithmetic, not a measurement; a calibrated profile carries the engine's own figure
instead.

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
    """Return the number of KV blocks that fit on one replica, by the module's arithmetic.

    This overestimates what vLLM allocates (by 7-12.5% on a T4), because the engine also reserves
    memory for CUDA graphs and activations, and identical launches can differ by several percent.
    For a real figure, read "GPU KV cache size: N tokens" from each run's own log.
    """
    usable = gpu_memory_gib * GIB * gpu_memory_utilization - model.weight_bytes
    if usable <= 0:
        raise ValueError(
            f"no room for a KV cache: {gpu_memory_gib} GiB at {gpu_memory_utilization} "
            f"utilization does not fit {model.weight_bytes / GIB:.1f} GiB of weights"
        )
    return max(1, int(usable // (model.bytes_per_token() * block_size)))


# Reference shapes, so the default workload runs without a measured profile. Architecture figures
# are public; the Qwen3 ones are read from each model's config.json.
SHAPES = {
    # Llama-3-8B: 32 layers, 8 KV heads (GQA), head_dim 128, ~16 GiB of fp16 weights
    "llama3-8b": ModelShape("llama3-8b", 32, 8, 128, 2, 16 * GIB),
    # Qwen3-0.6B: 28 layers, 8 KV heads, head_dim 128 -> 112 KiB of KV per token, ~1.5 GB of
    # weights.
    "qwen3-0.6b": ModelShape("qwen3-0.6b", 28, 8, 128, 2, 1_500_000_000),
    # Qwen3-8B: 36 layers, 8 KV heads, head_dim 128 -> 144 KiB of KV per token, ~16 GB of bf16
    # weights.
    "qwen3-8b": ModelShape("qwen3-8b", 36, 8, 128, 2, 16 * GIB),
    # Qwen3-4B, the model of the shipped T4 profiles. Its KV cache has the same shape as
    # Qwen3-8B's (144 KiB per token), and its 4.02B params take 7.49 GiB in fp16 (a T4 has no
    # bf16), so it fits on one T4, where Qwen3-8B's 15.26 GiB exceed the 13.5 GiB usable.
    "qwen3-4b": ModelShape("qwen3-4b", 36, 8, 128, 2, 8_040_000_000),
}
