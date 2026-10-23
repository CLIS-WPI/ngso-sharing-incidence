"""Closed-form / semi-analytic model of the co-located in-line event fraction.

The simulator measures the fraction of station-time at which the *co-located*
interferer aggregate (the interferer's serving satellite, main beam, at the
victim station) drives ``Delta T/T`` above the 47 CFR 25.261 threshold.  This
module reconstructs that fraction from first principles, with **explicit and
deliberately simple approximations**, so the simulation can be checked rather
than merely trusted.

Model
-----
1. **Equivalent angle** ``theta*``: the off-boresight angle at which the victim
   terminal's ITU-R S.1428 gain reduces a co-located interferer to exactly the
   ``I/N = -12.2 dB`` (``Delta T/T = 6%``) threshold.  Under PFD-constant beam
   steering the received interference is range-independent
   (``I/N = PFD + 10 log10(lambda^2/4pi) + G_rx(theta) - N``), so ``theta*`` is a
   single number per (PFD, noise, aperture) and does not depend on the slant
   range.  It is obtained by inverting the S.1428 pattern numerically.

2. **Serving-satellite direction**: a constellation's serving satellite is the
   one whose sub-satellite point is nearest the station (highest elevation).
   Approximations:
   * the serving direction is the direction to that nearest sub-satellite point;
   * the visible satellites of each constellation are treated as a **Poisson
     field** uniform over the station's look-direction cap (half-angle
     ``alpha = 90 deg - min_elev``, set by the elevation mask and independent of
     altitude), so the nearest-to-zenith satellite's off-zenith radius ``r``
     follows the nearest-of-``N`` law
     ``f(r) = 2N r/alpha^2 (1 - r^2/alpha^2)^(N-1)`` with ``N`` the mean visible
     count from :func:`ngsi.sanity.analytic_visible_count`;
   * the two serving directions are statistically independent, with uniform
     independent azimuths.

3. **Event probability**: ``P_event(phi) = V_A(phi) V_B(phi) P_sep(theta*)`` where
   ``V = 1 - exp(-N)`` is the probability that a constellation has at least one
   visible satellite, and ``P_sep`` is the probability that two independent
   nearest-of-``N`` directions (one per constellation) subtend less than
   ``theta*`` at the station.  ``P_sep`` is evaluated by 2-D quadrature using the
   closed-form conditional probability over the azimuth difference.

The model is intentionally *not* tuned to the simulator; where it disagrees the
disagreement is reported and explained (see ``results/PHASE3D_REPORT.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .antennas import s1428_gain
from .constants import IN_THRESHOLD_DB
from .link_budget import noise_power_dbw_per_mhz, slant_range_km
from .sanity import (
    analytic_visible_count,
    earth_central_angle_cap,
    walker_sub_latitude_density,
)

LIGHT_SPEED_M_S = 299792458.0
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class ShellSpec:
    """One orbital shell (Walker-Delta) for the analytic visibility model."""

    altitude_km: float
    inclination_deg: float
    planes: int
    sats_per_plane: int


def visibility_stats(
    shells: "list[ShellSpec] | tuple[ShellSpec, ...]",
    lat_deg: float,
    min_elev_deg: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> tuple[float, float]:
    """Mean visible count ``N`` and look-direction cap half-angle ``alpha`` (deg).

    ``N`` sums the per-shell analytic visible counts.  ``alpha`` is the maximum
    zenith angle of a visible satellite, ``90 - min_elev``, the same for every
    shell (it is set by the elevation mask, not the altitude).  Shells of one
    operator therefore share a single cap; the union is approximated as a uniform
    Poisson field of ``N`` directions in that cap.
    """
    n_total = 0.0
    for s in shells:
        n_total += analytic_visible_count(
            s.altitude_km, s.inclination_deg, s.planes, s.sats_per_plane,
            lat_deg, min_elev_deg, earth_radius_km,
        )
    return n_total, 90.0 - float(min_elev_deg)




def effective_area_db(freq_ghz: float) -> float:
    """``10 log10(lambda^2 / 4 pi)`` (dB m^2): isotropic aperture at ``freq_ghz``."""
    lam = LIGHT_SPEED_M_S / (float(freq_ghz) * 1e9)
    return float(10.0 * np.log10(lam**2 / (4.0 * np.pi)))


def pfd_constant_dbw_per_m2_per_mhz(
    eirp_density_dbw_per_mhz: float,
    altitude_km: float,
    anchor_elevation_deg: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """PFD (dBW/m^2/MHz) held constant by power control to the service edge.

    ``PFD = EIRP_max - 10 log10(4 pi r_max^2)`` with ``r_max`` the slant range at
    ``anchor_elevation_deg`` (FCC-22-91A1 para. 42; the flat 25-90 deg limit in
    47 CFR 25.208(b)(2) is -116 dBW/m^2/MHz).
    """
    r_max = slant_range_km(altitude_km, anchor_elevation_deg, earth_radius_km)
    r_m = r_max * 1e3
    return float(eirp_density_dbw_per_mhz) - float(10.0 * np.log10(4.0 * np.pi * r_m**2))


def beam_pfd0_dbw_per_m2_per_mhz(beam, earth_radius_km: float = EARTH_RADIUS_KM) -> float:
    """PFD0 (dBW/m^2/MHz) of a PFD-constant beam, the same for every shell.

    ``EIRP_max - 10 log10(4 pi r_ref^2)`` with ``r_ref`` the slant range at
    ``beam.pfd_anchor_elevation_deg`` from ``beam.pfd_reference_altitude_km``.
    """
    if beam.pfd_anchor_elevation_deg is None or beam.pfd_reference_altitude_km is None:
        raise ValueError("PFD-constant beam needs pfd_anchor_elevation_deg and "
                         "pfd_reference_altitude_km")
    return pfd_constant_dbw_per_m2_per_mhz(
        beam.eirp_density_dbw_per_mhz, beam.pfd_reference_altitude_km,
        beam.pfd_anchor_elevation_deg, earth_radius_km)


def equivalent_angle_deg(
    rx_peak_gain_dbi: float,
    diameter_m: float,
    freq_ghz: float,
    pfd_dbw_per_m2_per_mhz: float,
    noise_dbw_per_mhz: float,
    i_over_n_db: float = IN_THRESHOLD_DB,
    lo_deg: float = 0.0,
    hi_deg: float = 180.0,
) -> float:
    """Off-axis angle (deg) at which the victim S.1428 gain meets the I/N threshold.

    Solves ``G_rx(theta) = i_over_n_db + noise - pfd - 10 log10(lambda^2/4 pi)``
    by bisection on the (non-increasing) S.1428 pattern.  Returns 0 when even the
    peak gain is below target, and ``hi_deg`` when the target is below the pattern
    floor (no resolvable angle).
    """
    target = (float(i_over_n_db) + float(noise_dbw_per_mhz)
              - float(pfd_dbw_per_m2_per_mhz) - effective_area_db(freq_ghz))

    def g(theta: float) -> float:
        return s1428_gain(theta, rx_peak_gain_dbi, diameter_m, freq_ghz)

    if g(lo_deg) <= target:
        return float(lo_deg)
    if g(hi_deg) >= target:
        return float(hi_deg)
    a, b = float(lo_deg), float(hi_deg)
    for _ in range(200):
        m = 0.5 * (a + b)
        if g(m) > target:
            a = m
        else:
            b = m
        if b - a < 1e-12:
            break
    return float(0.5 * (a + b))


def _nearest_offaxis_pdf(
    r_deg: NDArray[np.float64], alpha_deg: float, n: float
) -> NDArray[np.float64]:
    """Nearest-of-``n`` off-zenith radius density ``2 n r/alpha^2 (1-r^2/alpha^2)^(n-1)``."""
    if alpha_deg <= 0.0 or n <= 0.0:
        return np.zeros_like(r_deg)
    x = np.asarray(r_deg, dtype=np.float64) / alpha_deg
    inside = (x >= 0.0) & (x <= 1.0)
    val = np.where(inside, 2.0 * n * x / alpha_deg * (1.0 - np.clip(x, 0.0, 1.0) ** 2) ** (n - 1.0),
                   0.0)
    return val


def separation_probability(
    theta_star_deg: float,
    alpha_a_deg: float,
    n_a: float,
    alpha_b_deg: float,
    n_b: float,
    n_grid: int = 240,
) -> float:
    """P(|p_A - p_B| <= theta*) for independent nearest-of-``n`` directions.

    ``p_A`` and ``p_B`` are the off-zenith offsets of each constellation's serving
    (nearest-to-zenith) satellite, with independent uniform azimuths.  For given
    radii ``r_a, r_b`` the separation law of cosines gives a closed-form
    conditional probability ``arccos(q)/pi`` with
    ``q = (r_a^2 + r_b^2 - theta*^2) / (2 r_a r_b)``.
    """
    if theta_star_deg <= 0.0:
        return 0.0
    t2 = float(theta_star_deg) ** 2

    def _grid(alpha, n):
        r = (np.arange(n_grid) + 0.5) * (alpha / n_grid)
        w = _nearest_offaxis_pdf(r, alpha, n)
        w = w / w.sum() if w.sum() > 0 else w
        return r, w

    ra, wa = _grid(alpha_a_deg, n_a)
    rb, wb = _grid(alpha_b_deg, n_b)
    if wa.sum() == 0.0 or wb.sum() == 0.0:
        return 0.0
    RA, RB = np.meshgrid(ra, rb, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (RA**2 + RB**2 - t2) / (2.0 * RA * RB)
    cond = np.where(q <= -1.0, 1.0,
                    np.where(q >= 1.0, 0.0, np.arccos(np.clip(q, -1.0, 1.0)) / np.pi))
    W = wa[:, None] * wb[None, :]
    return float((W * cond).sum())


def ground_footprint_deg(
    shells: "list[ShellSpec] | tuple[ShellSpec, ...]",
    min_elev_deg: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """Largest ground central-angle footprint (deg) of the constellation's shells."""
    if not shells:
        return 0.0
    return float(max(
        np.rad2deg(earth_central_angle_cap(s.altitude_km, min_elev_deg, earth_radius_km))
        for s in shells
    ))


def serving_look_radius_pdf(
    shells: "list[ShellSpec] | tuple[ShellSpec, ...]",
    min_elev_deg: float,
    n: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
    n_grid: int = 240,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Density of the serving (highest-elevation) satellite's off-zenith radius.

    The sub-satellite points of the ``n`` visible satellites are treated as
    uniform over the ground footprint disc of radius ``psi_max`` (the validated
    Poisson-field approximation, but now in ground coordinates, where the
    sub-satellite points actually are), and the serving satellite is the
    nearest-to-zenith one.  Changing variables to the look angle ``r`` (deg)
    through the exact spherical elevation relation ``psi(r) = arccos(k cos eps) -
    eps`` with ``eps = 90 - r`` and ``k = R/(R+h)`` gives

        f_r(r) = f_psi(psi(r)) * |d psi/dr|,
        f_psi(psi) = 2 n psi/psi_max^2 (1 - psi^2/psi_max^2)^(n-1).

    The look-angle density is strongly skewed toward large ``r`` (low elevation),
    unlike a uniform disc in look space.  Multi-shell unions use the largest
    footprint (highest shell) and the total count ``n``.
    """
    alpha = 90.0 - float(min_elev_deg)
    if alpha <= 0.0 or n <= 0.0 or not shells:
        return np.zeros(n_grid), np.zeros(n_grid)
    psi_max = np.deg2rad(ground_footprint_deg(shells, min_elev_deg, earth_radius_km))
    alt = float(max(s.altitude_km for s in shells))
    k = earth_radius_km / (earth_radius_km + alt)
    r = (np.arange(n_grid) + 0.5) * (alpha / n_grid)      # deg
    eps = np.deg2rad(90.0 - r)                            # rad
    psi = np.arccos(k * np.cos(eps)) - eps                # rad
    if psi_max <= 0.0:
        return r, np.zeros(n_grid)
    x = np.clip(psi / psi_max, 0.0, 1.0)
    f_psi = 2.0 * float(n) * (psi / psi_max**2) * (1.0 - x**2) ** (float(n) - 1.0)
    dpsi_deps = k * np.sin(eps) / np.sqrt(1.0 - k**2 * np.cos(eps) ** 2) - 1.0
    w = f_psi * np.abs(dpsi_deps)                          # per rad; scale cancels
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    return r, (w / w.sum() if w.sum() > 0.0 else w)


def separation_probability_from_pdf(
    theta_star_deg: float,
    ra: NDArray[np.float64],
    wa: NDArray[np.float64],
    rb: NDArray[np.float64],
    wb: NDArray[np.float64],
) -> float:
    """P(|p_A - p_B| <= theta*) for independent directions with the given radius pdfs."""
    if theta_star_deg <= 0.0 or wa.sum() == 0.0 or wb.sum() == 0.0:
        return 0.0
    t2 = float(theta_star_deg) ** 2
    RA, RB = np.meshgrid(ra, rb, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (RA**2 + RB**2 - t2) / (2.0 * RA * RB)
    cond = np.where(q <= -1.0, 1.0,
                    np.where(q >= 1.0, 0.0, np.arccos(np.clip(q, -1.0, 1.0)) / np.pi))
    return float((wa[:, None] * wb[None, :] * cond).sum())


def co_located_event_fraction(
    lat_deg: float,
    theta_star_deg: float,
    shells_a: list[ShellSpec],
    shells_b: list[ShellSpec],
    min_elev_deg: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
    n_grid: int = 240,
) -> float:
    """Analytic co-located ``Delta T/T`` event fraction at one latitude.

    Uses the ground-space serving-radius density of
    :func:`serving_look_radius_pdf`; the two serving directions are independent
    (distinct constellations with independent phasing - confirmed against the
    simulation), so only the marginals enter.
    """
    n_a, _ = visibility_stats(shells_a, lat_deg, min_elev_deg, earth_radius_km)
    n_b, _ = visibility_stats(shells_b, lat_deg, min_elev_deg, earth_radius_km)
    if n_a <= 0.0 or n_b <= 0.0:
        return 0.0
    v_a = 1.0 - np.exp(-n_a)
    v_b = 1.0 - np.exp(-n_b)
    ra, wa = serving_look_radius_pdf(shells_a, min_elev_deg, n_a, earth_radius_km, n_grid)
    rb, wb = serving_look_radius_pdf(shells_b, min_elev_deg, n_b, earth_radius_km, n_grid)
    p_sep = separation_probability_from_pdf(theta_star_deg, ra, wa, rb, wb)
    return float(v_a * v_b * p_sep)


def event_fraction_vs_latitude(
    lats_deg: NDArray[np.float64],
    theta_star_deg: float,
    shells_a: list[ShellSpec],
    shells_b: list[ShellSpec],
    min_elev_deg: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
    n_grid: int = 240,
) -> NDArray[np.float64]:
    """Vectorised :func:`co_located_event_fraction` over latitudes."""
    return np.array([
        co_located_event_fraction(float(phi), theta_star_deg, shells_a, shells_b,
                                  min_elev_deg, earth_radius_km, n_grid)
        for phi in np.asarray(lats_deg, dtype=np.float64)
    ])


def theta_star_for_starlink(
    diameter_m: float,
    freq_ghz: float = 12.0,
    pfd_offset_db: float = 0.0,
    victim_noise_temp_k: float = 200.0,
    *,
    starlink_eirp_dbw_per_mhz: float = 16.0018,
    starlink_altitude_km: float = 550.0,
    anchor_elevation_deg: float = 25.0,
) -> float:
    """``theta*`` for the Starlink-interferer / OneWeb-victim binding direction.

    ``pfd_offset_db`` lowers the PFD-constant Starlink PFD by that many dB (0 = the
    47 CFR 25.208(b)-anchored value; -3/-6 dB are the sensitivity rows).
    """
    from .link_budget import terminal_peak_gain_dbi

    pfd = pfd_constant_dbw_per_m2_per_mhz(starlink_eirp_dbw_per_mhz, starlink_altitude_km,
                                          anchor_elevation_deg) + float(pfd_offset_db)
    peak = terminal_peak_gain_dbi(diameter_m, freq_ghz, 0.6)
    return equivalent_angle_deg(peak, diameter_m, freq_ghz, pfd,
                                noise_power_dbw_per_mhz(victim_noise_temp_k))


def walker_latitude_weight(lat_deg: float, inc_deg: float) -> float:
    """Validated Walker sub-satellite latitude density (shape weight), re-exported."""
    return float(walker_sub_latitude_density(inc_deg, np.array([lat_deg]))[0])
