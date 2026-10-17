"""The cost model, with every constant read from a cited config file (ADR-006).

    rupees per 1k successful requests
      = (replicas x price_per_hour_usd x usd_to_inr) / successful_requests_per_hour x 1000

A successful request is one within the SLO: its TTFT within `slo.ttft_p95_ms` and, when the optional
`slo.tpot_p95_ms` is set, its time per output token within that. Serving more traffic by missing the
SLO therefore does not lower the cost. A constant still marked `<TBD>` raises rather than defaulting
to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TBD = "<TBD>"


class UnfilledCostModel(RuntimeError):
    """Raised when a cost constant is still `<TBD>`."""


@dataclass(frozen=True, slots=True)
class CostModel:
    usd_to_inr: float
    price_per_hour_usd: float
    sku: str
    ttft_p95_slo_ms: float
    provenance: dict[str, Any]
    # The per-token half of the SLO, in ms; None leaves it out, so only TTFT is limited.
    tpot_p95_slo_ms: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> CostModel:
        """Load from a YAML file, or `"default"` for the cost model shipped with the package.

        Raises:
            UnfilledCostModel: If any constant is still `<TBD>`.
        """
        if str(path) == "default":
            from .. import data

            text = data.cost_model().read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
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
            tpot_p95_slo_ms=(None if raw["slo"].get("tpot_p95_ms") is None
                             else float(raw["slo"]["tpot_p95_ms"])),
        )

    @property
    def slo_label(self) -> str:
        """Name the SLO this cost model applies, for use in a sentence."""
        if self.tpot_p95_slo_ms is None:
            return "p95 TTFT SLO"
        return "SLO (p95 TTFT and p95 time per output token)"

    def meets_slo(self, p95_ttft_ms: float, p95_tpot_ms: float) -> bool:
        """Return whether p95 TTFT, and p95 TPOT when it is limited, are within the SLO.

        A run with no multi-token request has a NaN TPOT, which cannot miss the limit.
        """
        if not p95_ttft_ms <= self.ttft_p95_slo_ms:
            return False
        return self.tpot_p95_slo_ms is None or not p95_tpot_ms > self.tpot_p95_slo_ms

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


def successful_requests(ttft_ms: list[float], slo_p95_ms: float,
                        tpot_ms: list[float] | None = None,
                        tpot_slo_ms: float | None = None) -> int:
    """Count the requests that met the SLO, the denominator of the cost.

    With a per-token limit a request must meet both limits. `tpot_ms` is aligned with `ttft_ms`, and
    a NaN in it (a single-token request) cannot miss the limit.
    """
    if tpot_slo_ms is None or tpot_ms is None:
        return sum(1 for t in ttft_ms if t <= slo_p95_ms)
    return sum(1 for t, g in zip(ttft_ms, tpot_ms, strict=True)
               if t <= slo_p95_ms and not g > tpot_slo_ms)
