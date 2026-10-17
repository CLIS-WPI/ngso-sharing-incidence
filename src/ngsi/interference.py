"""Interference power, I/N and Delta T/T computation.

References: ITU-R SM.337 (I/N), ITU-R S.741 / S.1323 (Delta T/T as an increase in
equivalent noise temperature of the victim receiver).
"""

from __future__ import annotations

import numpy as np

BOLTZMANN_W_PER_K_HZ: float = 1.380649e-23
LIGHT_SPEED_M_S: float = 299792458.0


def free_space_path_loss_db(freq_ghz: float, distance_km: float) -> float:
    """Free-space path loss (dB) between isotropic antennas."""
    if freq_ghz <= 0.0 or distance_km <= 0.0:
        raise ValueError("freq_ghz and distance_km must be positive")
    d_m = distance_km * 1e3
    lam_m = LIGHT_SPEED_M_S / (freq_ghz * 1e9)
    return float(20.0 * np.log10(4.0 * np.pi * d_m / lam_m))


def received_power_dbw(
    tx_eirp_density_dbw_per_mhz: float,
    bandwidth_mhz: float,
    rx_gain_dbi: float,
    path_loss_db: float,
) -> float:
    """Received power (dBW) from an EIRP *density*, a receive gain and a path loss."""
    return float(
        tx_eirp_density_dbw_per_mhz + 10.0 * np.log10(bandwidth_mhz) + rx_gain_dbi - path_loss_db
    )


def i_over_n_db(interference_w: float, noise_w: float) -> float:
    """I/N ratio (dB) for a single-entry interference into a receiver."""
    if interference_w <= 0.0 or noise_w <= 0.0:
        raise ValueError("interference_w and noise_w must be positive")
    return float(10.0 * np.log10(interference_w / noise_w))


def delta_t_over_t(interference_w: float, system_noise_temp_k: float, bandwidth_hz: float) -> float:
    """Delta T/T: fractional increase in receiver noise temperature from interference.

    ``Delta T / T = I / (k * T_sys * B)``.  The 47 CFR 25.261(d) trigger is
    ``Delta T/T > 0.06`` (see :mod:`ngsi.events`).
    """
    if interference_w < 0.0:
        raise ValueError("interference_w must be non-negative")
    if system_noise_temp_k <= 0.0 or bandwidth_hz <= 0.0:
        raise ValueError("system_noise_temp_k and bandwidth_hz must be positive")
    noise_w = BOLTZMANN_W_PER_K_HZ * system_noise_temp_k * bandwidth_hz
    return float(interference_w / noise_w)
