"""Propagation: J2 secular rates and SGP4 validation (AGENTS.md rule 6).

The SGP4 test builds a *TLE-equivalent* for three circular satellites with
``Satrec.sgp4init`` (the same initialiser ``twoline2rv`` uses), propagates both
models for one day, and compares ECEF positions.  Our Gate-1 model is J2
*secular* only (no J2 short-period terms, no drag), so a residual of order the J2
short-period amplitude is expected; the tolerance below is documented from that
physics, not tuned to the run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngsi import frames
from ngsi.constellation import WalkerConstellation
from ngsi.propagation import J2, _secular_rates_j2, propagate_ecef

# WGS-72 constants SGP4 uses internally, so the analytic orbit matches SGP4.
WGS72_MU = 398600.8
WGS72_RE = 6378.135
EPOCH_JD = 2460676.5  # 2025-01-01T00:00:00Z
EPOCH_DAYS_1949 = EPOCH_JD - 2433281.5
# Observed residual is the J2 short-period amplitude (~16 km over 1 day for these
# 6900 km orbits): our model keeps J2 secular terms only, and SGP4 additionally
# carries J2 short-period terms, drag and the Kozai/Brouwer mean-motion
# conversion. The bound is set from that physics, not tuned to a run.
MAX_ERROR_KM = 25.0


def test_j2_raan_regresses_for_prograde() -> None:
    a = torch.tensor([6900.0], dtype=torch.float64)
    inc = torch.tensor([np.deg2rad(53.0)], dtype=torch.float64)
    n = torch.tensor([np.sqrt(WGS72_MU / 6900.0**3)], dtype=torch.float64)
    raan_dot, _u_dot = _secular_rates_j2(a, inc, n, WGS72_RE)
    assert float(raan_dot) < 0.0  # prograde orbit -> westward node regression


def test_j2_argument_of_latitude_coefficient() -> None:
    # u_dot - n must equal (3/4) J2 n (Re/a)^2 (5 cos^2 i - 1).
    a = 7000.0
    inc_deg = 53.0
    n = np.sqrt(WGS72_MU / a**3)
    a_t = torch.tensor([a], dtype=torch.float64)
    inc_t = torch.tensor([np.deg2rad(inc_deg)], dtype=torch.float64)
    n_t = torch.tensor([n], dtype=torch.float64)
    _raan_dot, u_dot = _secular_rates_j2(a_t, inc_t, n_t, WGS72_RE)
    factor = J2 * n * (WGS72_RE / a) ** 2
    expected = n + 0.75 * factor * (5.0 * np.cos(np.deg2rad(inc_deg)) ** 2 - 1.0)
    assert float(u_dot) == pytest.approx(expected, rel=1e-12)


def _analytic_elements(inc_deg: float, raan_deg: float, m_deg: float) -> WalkerConstellation:
    a = 6900.0
    n_rad_s = np.sqrt(WGS72_MU / a**3)
    return WalkerConstellation(
        semi_major_axis_km=np.array([a]),
        eccentricity=np.array([0.0]),
        inclination_rad=np.array([np.deg2rad(inc_deg)]),
        raan_rad=np.array([np.deg2rad(raan_deg)]),
        arg_perigee_rad=np.array([0.0]),
        mean_anomaly_rad=np.array([np.deg2rad(m_deg)]),
        mean_motion_rad_s=np.array([n_rad_s]),
        plane_index=np.array([0]),
        slot_index=np.array([0]),
    )


def test_propagation_matches_sgp4_over_one_day() -> None:
    from sgp4.api import Satrec, WGS72

    sats = [
        _analytic_elements(53.0, 0.0, 0.0),
        _analytic_elements(70.0, 120.0, 90.0),
        _analytic_elements(97.6, 240.0, 200.0),
    ]

    t = np.arange(0.0, 86400.0 + 1.0, 600.0)
    jd = EPOCH_JD + t / 86400.0
    gmst = frames.gmst_rad_series(jd)

    max_err = 0.0
    for elements, inc_deg, raan_deg, m_deg in zip(sats, (53.0, 70.0, 97.6), (0.0, 120.0, 240.0), (0.0, 90.0, 200.0)):
        n_rad_min = elements.mean_motion_rad_s[0] * 60.0
        sat = Satrec()
        sat.sgp4init(
            WGS72,
            "i",
            1,
            EPOCH_DAYS_1949,
            0.0,  # bstar
            0.0,  # ndot
            0.0,  # nddot
            elements.eccentricity[0],
            elements.arg_perigee_rad[0],
            np.deg2rad(inc_deg),
            np.deg2rad(m_deg),
            n_rad_min,
            np.deg2rad(raan_deg),
        )
        sgp4_ecef = np.empty((len(t), 3), dtype=np.float64)
        for i, (j, g) in enumerate(zip(jd, gmst)):
            err, r_eci, _v = sat.sgp4(float(j), 0.0)
            assert err == 0
            sgp4_ecef[i] = frames.eci_to_ecef(np.asarray(r_eci, dtype=np.float64), g)

        ours = np.asarray(propagate_ecef(elements, t, EPOCH_JD, WGS72_RE, device="cpu"))[:, 0, :]
        err_km = float(np.max(np.linalg.norm(ours - sgp4_ecef, axis=-1)))
        max_err = max(max_err, err_km)

    # Documented residual budget; the actual value is reported by the failure.
    assert max_err < MAX_ERROR_KM, f"max SGP4-ECEF error {max_err:.3f} km exceeds {MAX_ERROR_KM} km"


def test_gpu_and_cpu_paths_agree() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    elements = _analytic_elements(53.0, 10.0, 20.0)
    t = np.arange(0.0, 3600.0, 60.0)
    cpu = np.asarray(propagate_ecef(elements, t, EPOCH_JD, 6371.0, device="cpu"))
    gpu = propagate_ecef(elements, t, EPOCH_JD, 6371.0, device="cuda").cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-6)
