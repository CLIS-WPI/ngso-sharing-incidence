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
    inclination band ``|phi| >= i``.  Singular at ``|phi| = i``.
    """
    inc = np.deg2rad(inc_deg)
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
    exclude_deg: float = 5.0,
    min_density_frac: float = 0.02,
) -> float:
    """Max relative deviation between simulated and analytic latitude density.

    Bins within ``exclude_deg`` of the singularity ``|phi| = i`` are ignored, as
    are bins whose analytic density is below ``min_density_frac`` of its peak
    (there the relative error is dominated by Monte-Carlo counting noise).
    """
    bins = np.asarray(bins_deg, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    total = sim_counts.sum()
    if total == 0:
        return float("inf")
    sim_density = sim_counts / (total * width)
    sim_density = sim_density / np.trapezoid(sim_density, centers)
    ana = analytic_density_per_deg(inc_deg, centers)
    ana = ana / np.trapezoid(ana, centers)
    peak = ana.max()
    keep = (np.abs(centers) < inc_deg - exclude_deg) & (ana > min_density_frac * peak) & (ana > 0)
    if not keep.any():
        return float("nan")
    return float(np.max(np.abs(sim_density[keep] - ana[keep]) / ana[keep]))


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
    n_phi: int = 4000,
) -> float:
    """Expected number of visible satellites of one shell at latitude ``lat0``.

    Integrates the (longitude-uniform) Walker latitude density over the spherical
    visibility cap: for each sub-satellite latitude ``phi`` the longitude fraction
    of the cap is ``dlambda / pi``.
    """
    n_sats = planes * sats_per_plane
    inc = np.deg2rad(inc_deg)
    phi0 = np.deg2rad(lat0_deg)
    psi = earth_central_angle_cap(altitude_km, min_elev_deg, radius_km)
    eps = 1e-9
    phi = np.linspace(-inc + eps, inc - eps, n_phi)
    dens = np.cos(phi) / np.sqrt(np.maximum(np.sin(inc) ** 2 - np.sin(phi) ** 2, 1e-300))
    dens = dens / np.trapezoid(dens, phi)
    cos_dlam = (np.cos(psi) - np.sin(phi) * np.sin(phi0)) / (
        np.cos(phi) * np.cos(phi0) + 1e-300
    )
    frac_lambda = np.arccos(np.clip(cos_dlam, -1.0, 1.0)) / np.pi
    return float(n_sats * np.trapezoid(dens * frac_lambda, phi))
