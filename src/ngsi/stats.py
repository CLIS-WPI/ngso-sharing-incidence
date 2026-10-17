"""Statistics over constellation phasings.

The unit of replication is an **independent constellation phasing** (AGENTS.md
rule 5): each phasing contributes one paired observation per cell, and tests are
run paired across cells/phasings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bootstrap_ci(
    samples: NDArray[np.float64],
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic``."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("samples must be non-empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = statistic(x[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def paired_permutation_test(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided paired permutation test on the mean of ``a - b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have the same shape")
    d = a - b
    observed = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.mean(signs * d, axis=1)
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return {"observed_mean_diff": observed, "p_value": p}


def wilcoxon_signed_rank(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test (``scipy.stats.wilcoxon``)."""
    from scipy.stats import wilcoxon

    stat, p = wilcoxon(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return {"statistic": float(stat), "p_value": float(p)}
