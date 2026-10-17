"""Population / Broadband-Serviceable-Location (BSL) demand weighting.

Phase 2 stub: weights station-grid results by population or BSL counts so that
burden can be expressed per-person rather than per-cell.  Data sources live in
``data/raw/`` (gitignored) with the download URL and date recorded in
``data/raw/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DemandWeight:
    """Per-cell demand weight and the source it was derived from."""

    weights: NDArray[np.float64]
    source: str

    def normalised(self) -> NDArray[np.float64]:
        total = float(np.sum(self.weights))
        if total <= 0.0:
            raise ValueError("weights sum to zero")
        return self.weights / total


def uniform_weights(n: int, source: str = "ASSUMED") -> DemandWeight:
    """Equal weighting over ``n`` cells (placeholder until population data lands)."""
    return DemandWeight(weights=np.full(n, 1.0 / n, dtype=np.float64), source=source)


def weighted_mean(values: NDArray[np.float64], demand: DemandWeight) -> float:
    """Demand-weighted mean of per-cell ``values``."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != demand.weights.shape:
        raise ValueError("values and demand weights must have the same shape")
    return float(np.sum(vals * demand.normalised()))
