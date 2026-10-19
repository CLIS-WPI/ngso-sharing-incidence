"""Link budget, I/N and the 47 CFR 25.261(d) Delta T/T criterion.

This module upgrades the geometric in-line screen to the regulatory power
criterion.  For a victim earth station at a site and time step:

* the **interferer** contribution uses its EIRP *density* toward the station,
  free-space path loss at the *shared-band centre frequency* (from
  ``configs/bands.yaml``), the victim receive-antenna gain at the off-axis angle
  between the victim serving satellite and the interfering satellite, and the
  victim system noise temperature;
* ``I/N = EIRP_density + G_rx(off-axis) - FSPL - 10 log10(k T_sys * 1 MHz)``;
* for a set of interferers the powers add in **linear** units, so
  ``Delta T/T = sum_i 10^(I/N_i / 10)``;
* the 25.261(d) short-term trigger is ``Delta T/T > 0.06`` (equivalently
  ``I/N > -12.2 dB``), the single constant in :mod:`ngsi.constants`.

All angles are at the antenna (AGENTS.md rule 3); each antenna gain is counted
exactly once.  The victim terminal pattern is ITU-R S.1428 (which delegates to the
S.465 envelope), selectable for sensitivity.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .antennas import s465_gain, s1428_gain

BOLTZMANN_W_PER_K_HZ: float = 1.380649e-23
LIGHT_SPEED_M_S: float = 299792458.0

PATTERN_FUNCS = {"S465": s465_gain, "S1428": s1428_gain}

# 47 CFR 25.218 states Ku-band (below 15 GHz) EIRP-density masks per 4 kHz,
# whereas this model stores every beam EIRP density per MHz.  A per-4-kHz EIRP
# becomes a per-MHz EIRP density by adding 10 log10(1000/4) = +23.9794 dB.
EIRP_DENSITY_4KHZ_TO_MHZ_DB: float = float(10.0 * np.log10(1000.0 / 4.0))


def eirp_density_dbw_per_mhz_from_dbw_per_4khz(eirp_dbw_per_4khz: float) -> float:
    """Convert a filed per-4-kHz EIRP (dBW) to a per-MHz EIRP density (dBW/MHz).

    Ku-band filings (``47 CFR 25.218``, and SpaceX's Ku technical attachments)
    tabulate maximum EIRP per 4 kHz, not per MHz.  This helper makes the unit
    conversion explicit and testable so a per-4-kHz value can never be silently
    read as a per-MHz density.
    """
    return float(eirp_dbw_per_4khz) + EIRP_DENSITY_4KHZ_TO_MHZ_DB


def terminal_peak_gain_dbi(diameter_m: float, freq_ghz: float, efficiency: float) -> float:
    """Peak gain (dBi) of a circular aperture: ``eta (pi D / lambda)^2``."""
    if diameter_m <= 0.0 or freq_ghz <= 0.0:
        raise ValueError("diameter_m and freq_ghz must be positive")
    if not 0.0 < efficiency < 1.0:
        raise ValueError("efficiency must be in (0, 1)")
    d_over_lambda = diameter_m * freq_ghz * 1e9 / LIGHT_SPEED_M_S
    return float(10.0 * np.log10(efficiency * (np.pi * d_over_lambda) ** 2))


def noise_power_dbw_per_mhz(system_noise_temp_k: float) -> float:
    """Receiver noise power in 1 MHz: ``10 log10(k T_sys * 1e6)`` dBW."""
    if system_noise_temp_k <= 0.0:
        raise ValueError("system_noise_temp_k must be positive")
    return float(10.0 * np.log10(BOLTZMANN_W_PER_K_HZ * system_noise_temp_k * 1e6))


def i_over_n_db(
    eirp_density_dbw_per_mhz: float,
    rx_gain_dbi: float,
    path_loss_db: float,
    system_noise_temp_k: float,
) -> float:
    """Single-entry I/N (dB) using a per-MHz EIRP density.

    The interferer bandwidth cancels because both the interference and the noise
    are referenced to the victim's 1 MHz resolution.
    """
    return float(
        eirp_density_dbw_per_mhz + rx_gain_dbi - path_loss_db
        - noise_power_dbw_per_mhz(system_noise_temp_k)
    )


def dtt_from_i_over_n_db(i_over_n_db_value):
    """``Delta T/T`` for a single entry: ``10^(I/N / 10)`` (scalar or array)."""
    return 10.0 ** (np.asarray(i_over_n_db_value, dtype=np.float64) / 10.0)


def aggregate_dtt_from_i_over_n_db(*i_over_n_db_values):
    """Aggregate ``Delta T/T`` for several entries (powers add in linear units)."""
    total = 0.0
    for v in i_over_n_db_values:
        total = total + dtt_from_i_over_n_db(v)
    return total


def cn_db(
    eirp_density_dbw_per_mhz: float,
    rx_gain_peak_dbi: float,
    path_loss_db: float,
    system_noise_temp_k: float,
) -> float:
    """Victim carrier-to-noise (dB) in a 1 MHz reference bandwidth."""
    return float(
        eirp_density_dbw_per_mhz + rx_gain_peak_dbi - path_loss_db
        - noise_power_dbw_per_mhz(system_noise_temp_k)
    )


def cn_over_n_plus_i_db(cn_db_value, dtt):
    """Carrier-to-(noise+interference) (dB): ``C/N - 10 log10(1 + Delta T/T)``."""
    return np.asarray(cn_db_value, dtype=np.float64) - 10.0 * np.log10(1.0 + np.asarray(dtt))


def shannon_throughput(spectral_snr_db, bandwidth_hz: float, margin_db: float):
    """Shannon rate (bit/s) with a linear implementation margin on the SNR.

    ``R = B log2(1 + SNR / margin_linear)``.  The margin is a documented
    ``ASSUMED`` allowance for finite coding/modulation.
    """
    snr = 10.0 ** (np.asarray(spectral_snr_db, dtype=np.float64) / 10.0)
    margin = 10.0 ** (float(margin_db) / 10.0)
    return bandwidth_hz * np.log2(1.0 + snr / margin)


# --- batched (torch) helpers --------------------------------------------------
def terminal_peak_gain_dbi_batch(diameter_m: float, freq_ghz: float, efficiency: float,
                                 device: str = "cuda") -> torch.Tensor:
    d_over_lambda = diameter_m * freq_ghz * 1e9 / LIGHT_SPEED_M_S
    return torch.tensor(
        10.0 * np.log10(efficiency * (np.pi * d_over_lambda) ** 2),
        dtype=torch.float64, device=device,
    )


def pattern_gain_batch(
    off_axis_deg: torch.Tensor,
    peak_gain_dbi: float,
    diameter_m: float,
    freq_ghz: float,
    kind: str = "S1428",
) -> torch.Tensor:
    """Batched earth-station pattern gain (dBi) at the S.465/S.1428 envelope."""
    if kind not in ("S465", "S1428"):
        raise ValueError(f"unknown pattern {kind!r}")
    psi = torch.abs(off_axis_deg)
    d_over_lambda = diameter_m * freq_ghz * 1e9 / LIGHT_SPEED_M_S
    g1 = 2.0 + 15.0 * torch.log10(torch.tensor(d_over_lambda, dtype=torch.float64,
                                               device=off_axis_deg.device))
    phi_m = (20.0 / d_over_lambda) * float(max(peak_gain_dbi - g1.item(), 0.0)) ** 0.5
    small = psi < phi_m
    mid = (psi >= phi_m) & (psi < 1.0)
    far = (psi >= 1.0) & (psi < 48.0)
    g = torch.where(
        small,
        peak_gain_dbi - 2.5e-3 * (d_over_lambda * psi) ** 2,
        torch.where(
            mid,
            torch.full_like(psi, g1.item()),
            torch.where(
                far,
                32.0 - 25.0 * torch.log10(torch.clamp(psi, min=1e-9)),
                torch.full_like(psi, -10.0),
            ),
        ),
    )
    return g


# --- ITU-R S.1528 satellite-pattern batch gain ------------------------------
S1528_LS_DB: float = -6.75
S1528_A: float = 2.58
S1528_B: float = 6.32
S1528_C: float = 14.6
S1528_D: float = 26.6
S1528_FLOOR_DBI: float = 0.0


def s1528_gain_batch(
    off_axis_deg: torch.Tensor,
    peak_gain_dbi: float,
    beamwidth_deg: float,
    floor_dbi: float = S1528_FLOOR_DBI,
) -> torch.Tensor:
    """Batched ITU-R S.1528 (revised) satellite-antenna gain (dBi).

    Matches :func:`ngsi.antennas.s1528_gain` element-wise.  ``off_axis_deg`` is the
    angle measured *at the satellite* between the beam's pointing direction and the
    victim; ``beamwidth_deg`` is the 3 dB half-power beamwidth ``psi_b``.
    """
    if beamwidth_deg <= 0.0:
        raise ValueError("beamwidth_deg must be positive")
    psi = torch.abs(off_axis_deg)
    z = psi / float(beamwidth_deg)
    zc = torch.clamp(z, min=1e-12)
    g1 = peak_gain_dbi - 3.0 * z**2
    g2 = torch.full_like(psi, peak_gain_dbi + S1528_LS_DB)
    g3 = peak_gain_dbi + S1528_LS_DB - 20.0 * torch.log10(zc)
    g4 = g3 - 25.0 * torch.log10(zc)
    g = torch.where(
        psi <= S1528_A * beamwidth_deg,
        g1,
        torch.where(
            psi <= S1528_B * beamwidth_deg,
            g2,
            torch.where(
                psi <= S1528_C * beamwidth_deg,
                g3,
                torch.where(psi <= S1528_D * beamwidth_deg, g4, torch.full_like(psi, floor_dbi)),
            ),
        ),
    )
    return torch.clamp(g, min=floor_dbi)


# --- PFD-constant beam steering (FCC-22-91A1 para. 42) ----------------------
def slant_range_km(
    altitude_km: float,
    elevation_deg: float,
    earth_radius_km: float = 6371.0,
) -> float:
    """Slant range (km) to a satellite at ``altitude_km`` seen at ``elevation_deg``.

    Spherical earth: ``r = -Re sin e + sqrt(Re^2 sin^2 e + h^2 + 2 Re h)``.  This is
    the geometric maximum range of a link held at the system's minimum service
    elevation, i.e. the ``r_max`` PFD-constant anchoring range.
    """
    h = float(altitude_km)
    e = math.radians(float(elevation_deg))
    re = float(earth_radius_km)
    return float(-re * math.sin(e) + math.sqrt(re * re * math.sin(e) ** 2 + h * h + 2.0 * re * h))


def slant_range_km_batch(
    altitude_km: torch.Tensor,
    elevation_deg: float,
    earth_radius_km: float = 6371.0,
) -> torch.Tensor:
    """Batched :func:`slant_range_km` over a tensor of altitudes."""
    e = math.radians(float(elevation_deg))
    re = float(earth_radius_km)
    sin_e = math.sin(e)
    h = altitude_km
    return -re * sin_e + torch.sqrt(re * re * sin_e**2 + h * h + 2.0 * re * h)


def pfd_spreading_db(r_km) -> torch.Tensor:
    """``10 log10(4 pi r^2)`` (dB) with ``r`` in km: the EIRP-defined PFD spreading.

    A PFD (dBW/m^2) equals an EIRP density (dBW) minus this term, so PFD-constant
    beam steering sets ``EIRP_density(r) = PFD_const + pfd_spreading_db(r)``.
    """
    r_m = torch.as_tensor(r_km, dtype=torch.float64) * 1e3
    return 10.0 * torch.log10(4.0 * np.pi * r_m**2)


def pfd_constant_from_eirp(eirp_max_density_dbw_per_mhz: float, r_max_km: float) -> float:
    """PFD (dBW/m^2/MHz) held constant by power control at the service edge.

    ``PFD_const = EIRP_max - 10 log10(4 pi r_max^2)`` with ``r_max`` the slant range
    at the system's minimum service elevation (FCC-22-91A1 para. 42).
    """
    r_m = float(r_max_km) * 1e3
    return float(eirp_max_density_dbw_per_mhz) - float(10.0 * np.log10(4.0 * np.pi * r_m**2))


def pfd_eirp_density_dbw_per_mhz(pfd_const_dbw_per_m2_per_mhz: float, r_km) -> torch.Tensor:
    """EIRP density (dBW/MHz) toward a point at range ``r`` under PFD-constant steering."""
    return float(pfd_const_dbw_per_m2_per_mhz) + pfd_spreading_db(r_km)
