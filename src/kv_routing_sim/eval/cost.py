"""The cost model. Every constant comes from a committed, cited, dated config (ADR-006).

⛔ THIS MODULE REFUSES TO RUN ON AN UNFILLED CONFIG. `configs/cost_model.yaml` ships with every
value as `<TBD>`, and a missing constant raises instead of defaulting to zero. A cost model that
silently treats an unknown FX rate as 0 produces a confident, free-looking number — which is worse
than no number, because it looks like an answer.

    rupees per 1k successful requests
      = (replicas x price_per_hour_usd x usd_to_inr) / successful_requests_per_hour x 1000

The denominator is SUCCESSFUL requests — ones that met the SLO. A configuration that serves more
traffic by blowing p95 has not become cheaper; it has changed the product.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TBD = "<TBD>"


class UnfilledCostModel(RuntimeError):
    """Raised when a cost constant is still `<TBD>`. Deliberately fatal."""


@dataclass(frozen=True, slots=True)
class CostModel:
    usd_to_inr: float
    price_per_hour_usd: float
    sku: str
    ttft_p95_slo_ms: float
    provenance: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> CostModel:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        missing = _find_tbd(raw)
        if missing:
            raise UnfilledCostModel(
                f"{path}: {len(missing)} constant(s) still {TBD} -> {missing}. "
                "Every value needs a source and a date before any rupee figure is produced "
                "(ADR-006). No defaults are supplied on purpose."
            )
        return cls(
            usd_to_inr=float(raw["fx"]["usd_to_inr"]),
            price_per_hour_usd=float(raw["compute"]["price_per_hour_usd"]),
            sku=str(raw["compute"]["sku"]),
            ttft_p95_slo_ms=float(raw["slo"]["ttft_p95_ms"]),
            provenance=raw,
        )

    def inr_per_1k_requests(self, *, n_replicas: int, successful_requests_per_hour: float) -> float:
        if successful_requests_per_hour <= 0:
            return float("inf")  # served nothing; cost per unit of useful work is undefined
        inr_per_hour = n_replicas * self.price_per_hour_usd * self.usd_to_inr
        return inr_per_hour / successful_requests_per_hour * 1000.0


def _find_tbd(node: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_find_tbd(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_find_tbd(v, f"{path}[{i}]"))
    elif isinstance(node, str) and node.strip() == TBD:
        out.append(path)
    return out


def successful_requests(ttft_ms: list[float], slo_p95_ms: float) -> int:
    """Requests that met the SLO. The cost denominator."""
    return sum(1 for t in ttft_ms if t <= slo_p95_ms)
