"""Summary metrics: event rate, unresolvable rate, spectrum loss, burden concentration.

All functions are pure and return plain Python / NumPy values so they can be
written straight to ``results/`` and quoted verbatim in reports (AGENTS.md rule 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .diversity import DiversityOutcome, unresolvable_rate  # re-exported

__all__ = [
    "event_rate_per_cell_hour",
    "unresolvable_rate",
    "spectrum_loss_fraction",
    "gini_coefficient",
    "burden_concentration",
]


def event_rate_per_cell_hour(
    n_events: NDArray[np.float64] | float,
    exposure_time_s: NDArray[np.float64] | float,
) -> NDArray[np.float64] | float:
    """In-line events per cell per hour = n_events * 3600 / exposure_time_s."""
    ev = np.asarray(n_events, dtype=np.float64)
    exp = np.asarray(exposure_time_s, dtype=np.float64)
    if np.any(exp <= 0.0):
        raise ValueError("exposure_time_s must be positive")
    rate = ev * 3600.0 / exp
    if np.isscalar(n_events) and np.isscalar(exposure_time_s):
        return float(rate)
    return rate


def spectrum_loss_fraction(fraction_available: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """Fraction of spectrum lost to band-splitting = 1 - fraction_available."""
    frac = np.asarray(fraction_available, dtype=np.float64)
    loss = 1.0 - frac
    if np.isscalar(fraction_available):
        return float(loss)
    return loss


def gini_coefficient(values: NDArray[np.float64]) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, 1 = maximal).

    Used as the burden-concentration measure across the latitude grid.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if np.any(x < 0.0):
        raise ValueError("values must be non-negative")
    total = float(np.sum(x))
    if total == 0.0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def burden_concentration(
    per_cell_burden: NDArray[np.float64],
) -> dict[str, float]:
    """Concentration summary for a burden distribution across cells."""
    x = np.asarray(per_cell_burden, dtype=np.float64)
    total = float(np.sum(x))
    result = {"gini": gini_coefficient(x), "total": total}
    if total > 0.0:
        order = np.argsort(x)[::-1]
        top10 = max(1, int(np.ceil(0.10 * x.size)))
        result["top10pct_share"] = float(np.sum(x[order[:top10]]) / total)
    return result
