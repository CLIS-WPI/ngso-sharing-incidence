"""Analytic sanity checks for the Gate-1 geometry (Section E).

These are *gates*: if the simulated sub-satellite latitude density or the mean
visible-satellite count disagrees with the analytic prediction by more than a
documented tolerance (away from the Walker singularity at ``|phi| = i``), the
experiments must stop rather than report event rates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def walker_sub_latitude_density(
    inc_deg: float, lat_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Walker sub-satellite latitude density ``cos(phi)/sqrt(sin^2 i - sin^2 phi)``.

    Proportional to the number of satellites per unit latitude; zero outside the
    latitude band ``|phi| >= inc_eff`` where ``inc_eff = min(i, 180 - i)`` (the
    caustic latitude, which is ``i`` prograde and ``180 - i`` retrograde).
    Singular at the caustic.
    """
    inc_eff = inc_deg if inc_deg <= 90.0 else 180.0 - inc_deg
    inc = np.deg2rad(inc_eff)
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    inside = np.abs(lat) < inc
    denom = np.sqrt(np.maximum(np.sin(inc) ** 2 - np.sin(lat) ** 2, 0.0))
    val = np.where(inside, np.cos(lat) / np.maximum(denom, 1e-300), 0.0)
    return val


def analytic_density_per_deg(
    inc_deg: float, lat_centers_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalised Walker density per degree of latitude (integrates to 1 over deg)."""
    centers = np.asarray(lat_centers_deg, dtype=np.float64)
    f = walker_sub_latitude_density(inc_deg, centers)
    area = np.trapezoid(f, centers)
    if area <= 0.0:
        return np.zeros_like(f)
    return f / area


def max_relative_deviation(
    sim_counts: NDArray[np.float64],
    bins_deg: NDArray[np.float64],
    inc_deg: float,
    exclude_deg: float = 7.5,
    min_density_frac: float = 0.02,
) -> float:
    """Max relative deviation between simulated and analytic latitude density.

    Both densities are renormalised **over the kept bins only**, so this is a pure
    *shape* comparison.  Bins within ``exclude_deg`` of the caustic
    ``|phi| = inc_eff`` are dropped (the continuum density is singular there and
    its per-bin value is ill-defined), as are bins with negligible analytic
    density (dominated by counting noise).
    """
    inc_eff = inc_deg if inc_deg <= 90.0 else 180.0 - inc_deg
    bins = np.asarray(bins_deg, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    total = sim_counts.sum()
    if total == 0:
        return float("inf")
    sim_density = sim_counts / (total * width)
    ana = analytic_density_per_deg(inc_deg, centers)
    peak = ana.max()
    keep = (np.abs(centers) < inc_eff - exclude_deg) & (ana > min_density_frac * peak) & (ana > 0)
    if not keep.any():
        return float("nan")
    ck = centers[keep]
    sim_k = sim_density[keep]
    ana_k = ana[keep]
    sim_k = sim_k / np.trapezoid(sim_k, ck)
    ana_k = ana_k / np.trapezoid(ana_k, ck)
    return float(np.max(np.abs(sim_k - ana_k) / ana_k))


def earth_central_angle_cap(altitude_km: float, min_elev_deg: float, radius_km: float) -> float:
    """Central angle (rad) of the visibility cap for a station at ``min_elev_deg``."""
    k = radius_km / (radius_km + altitude_km)
    return float(np.arccos(k * np.cos(np.deg2rad(min_elev_deg))) - np.deg2rad(min_elev_deg))


def analytic_visible_count(
    altitude_km: float,
    inc_deg: float,
    planes: int,
    sats_per_plane: int,
    lat0_deg: float,
    min_elev_deg: float,
    radius_km: float,
    n_phi: int = 20000,
) -> float:
    """Expected number of visible satellites of one shell at latitude ``lat0``.

    Averages the visibility-cap longitude fraction over a uniform argument of
    latitude ``u``; the sub-satellite latitude is ``asin(sin i sin u)``.  This is
    well conditioned through the caustic (unlike a latitude-space integral) and is
    valid for retrograde inclinations too.
    """
    n_sats = planes * sats_per_plane
    inc = np.deg2rad(inc_deg)
    phi0 = np.deg2rad(lat0_deg)
    psi = earth_central_angle_cap(altitude_km, min_elev_deg, radius_km)
    u = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    delta = np.arcsin(np.sin(inc) * np.sin(u))
    denom = np.cos(delta) * np.cos(phi0)
    cos_dlam = (np.cos(psi) - np.sin(delta) * np.sin(phi0)) / np.where(
        np.abs(denom) > 1e-12, denom, 1e-12
    )
    frac_lambda = np.arccos(np.clip(cos_dlam, -1.0, 1.0)) / np.pi
    return float(n_sats * np.mean(frac_lambda))
