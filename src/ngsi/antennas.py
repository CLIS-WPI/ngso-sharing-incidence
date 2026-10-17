"""Antenna radiation patterns.

Implementations follow ITU-R recommendations:

* :func:`s1528_gain`  - reference pattern for *space station* (satellite) antennas.
* :func:`s465_gain`   - reference pattern for *earth station* antennas.
* :func:`s1428_gain`  - NGSO earth-station reference pattern (S.1428 family).

Convention (see AGENTS.md rule 3): ``off_axis_deg`` is always the off-boresight
angle measured *at the antenna*, and each antenna's gain is counted exactly once.
Number of degrees, gains in dBi.
"""

from __future__ import annotations

import numpy as np

# --- ITU-R S.1528 reference pattern constants (revised pattern) -------------
_S1528_LS_DB = -6.75
_S1528_A = 2.58
_S1528_B = 6.32
_S1528_C = 14.6
_S1528_D = 26.6
_S1528_FLOOR_DBI = 0.0


def s1528_gain(
    off_axis_deg: float,
    peak_gain_dbi: float,
    beamwidth_deg: float,
    floor_dbi: float = _S1528_FLOOR_DBI,
) -> float:
    """ITU-R S.1528 reference pattern gain (dBi) of a satellite antenna.

    ``beamwidth_deg`` is the 3 dB half-power beamwidth (``psi_b``).  The pattern
    is exactly ``peak_gain_dbi`` at boresight and flattens to ``floor_dbi`` at
    large off-axis angles.
    """
    psi = abs(float(off_axis_deg))
    if psi == 0.0:
        return float(peak_gain_dbi)
    if beamwidth_deg <= 0.0:
        raise ValueError("beamwidth_deg must be positive")
    z = psi / beamwidth_deg

    if psi <= _S1528_A * beamwidth_deg:
        g = peak_gain_dbi - 3.0 * z**2
    elif psi <= _S1528_B * beamwidth_deg:
        g = peak_gain_dbi + _S1528_LS_DB
    elif psi <= _S1528_C * beamwidth_deg:
        g = peak_gain_dbi + _S1528_LS_DB - 20.0 * np.log10(z)
    elif psi <= _S1528_D * beamwidth_deg:
        g = peak_gain_dbi + _S1528_LS_DB - 20.0 * np.log10(z) - 25.0 * np.log10(z)
    else:
        g = floor_dbi
    return float(max(g, floor_dbi))


def s465_gain(
    off_axis_deg: float,
    peak_gain_dbi: float,
    diameter_m: float,
    freq_ghz: float,
) -> float:
    """ITU-R S.465 reference pattern gain (dBi) of an earth-station antenna.

    ``diameter_m`` is the antenna diameter and ``freq_ghz`` the operating
    frequency (used for ``D / lambda``).
    """
    psi = abs(float(off_axis_deg))
    if psi == 0.0:
        return float(peak_gain_dbi)
    if diameter_m <= 0.0 or freq_ghz <= 0.0:
        raise ValueError("diameter_m and freq_ghz must be positive")

    d_over_lambda = diameter_m * freq_ghz * 1e9 / 299792458.0
    g1 = 2.0 + 15.0 * np.log10(d_over_lambda)
    phi_m = (20.0 / d_over_lambda) * np.sqrt(max(peak_gain_dbi - g1, 0.0))

    if psi < phi_m:
        g = peak_gain_dbi - 2.5e-3 * (d_over_lambda * psi) ** 2
    elif psi < 1.0:
        g = g1
    elif psi < 48.0:
        g = 32.0 - 25.0 * np.log10(psi)
    else:
        g = -10.0
    return float(g)


def s1428_gain(
    off_axis_deg: float,
    peak_gain_dbi: float,
    diameter_m: float,
    freq_ghz: float,
) -> float:
    """ITU-R S.1428 earth-station reference pattern (NGSO sharing studies).

    TODO(phase-2): S.1428 specifies several patterns (``alpha``, ``beta`` ...)
    with optional sidelobe statistics.  For now this delegates to the S.465
    envelope, which S.1428 falls back to outside the main lobe.
    """
    return s465_gain(off_axis_deg, peak_gain_dbi, diameter_m, freq_ghz)
