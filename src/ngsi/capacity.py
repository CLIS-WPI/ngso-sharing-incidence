"""Adaptive Coding and Modulation (ACM) throughput, ITU-R S.2131 style.

Stub for phase 2: maps a link C/(N+I) to a spectral efficiency via a set of ACM
modulation/coding modes and integrates over the visible time to give capacity.
No expected values are baked in; all mode tables come from configs/ or callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AcmMode:
    """One ACM mode: minimum C/(N+I) and its spectral efficiency (bits/s/Hz)."""

    min_cni_db: float
    spectral_efficiency_bps_hz: float


def spectral_efficiency(
    c_over_n_plus_i_db: NDArray[np.float64] | float,
    modes: list[AcmMode],
) -> NDArray[np.float64] | float:
    """Highest-efficiency ACM mode supportable at each C/(N+I) value (0 if none)."""
    ordered = sorted(modes, key=lambda m: m.min_cni_db)
    cni = np.asarray(c_over_n_plus_i_db, dtype=np.float64)
    eff = np.zeros_like(cni, dtype=np.float64)
    for mode in ordered:
        eff = np.where(cni >= mode.min_cni_db, mode.spectral_efficiency_bps_hz, eff)
    if np.isscalar(c_over_n_plus_i_db):
        return float(eff)
    return eff


def throughput_mbps(
    c_over_n_plus_i_db: NDArray[np.float64],
    bandwidth_mhz: float,
    modes: list[AcmMode],
) -> float:
    """Mean ACM throughput (Mbps) over a time series of C/(N+I) samples."""
    eff = np.asarray(spectral_efficiency(c_over_n_plus_i_db, modes), dtype=np.float64)
    return float(np.mean(eff) * bandwidth_mhz)
