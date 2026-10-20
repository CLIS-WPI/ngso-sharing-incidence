"""Orbital propagation, batched on the GPU with a NumPy CPU fallback.

Two propagation paths are provided:

* :func:`propagate_two_body` - analytic Keplerian two-body propagation, batched
  over satellites and times.  Uses ``torch`` when available and falls back to
  NumPy.
* :func:`propagate_sgp4` - SGP4 from a TLE, for validation against the analytic
  model.  SGP4 is inherently per-satellite and runs through the ``sgp4`` package.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constellation import MU_EARTH_KM3_S2, WalkerConstellation

try:  # torch is always present in the project image; import defensively anyway.
    import torch
except ImportError:  # pragma: no cover - exercised only outside the container
    torch = None  # type: ignore[assignment]

Device = Literal["cpu", "cuda", "auto"]


def _resolve_device(device: Device):
    if torch is None:
        return None
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _solve_kepler(mean_anomaly, eccentricity, iters: int = 12):
    """Solve Kepler's equation E - e sin E = M by Newton iteration (batched)."""
    ecc_anom = mean_anomaly.clone()
    for _ in range(iters):
        f = ecc_anom - eccentricity * torch.sin(ecc_anom) - mean_anomaly
        fp = 1.0 - eccentricity * torch.cos(ecc_anom)
        ecc_anom = ecc_anom - f / fp
    return ecc_anom


def propagate_two_body(
    elements: WalkerConstellation,
    times_s: NDArray[np.float64],
    device: Device = "auto",
) -> NDArray[np.float64]:
    """Return ECI positions (km) with shape ``(n_times, n_sats, 3)``.

    Satellites are propagated as circular/elliptical Keplerian orbits from their
    mean anomaly at ``t = 0``; Earth oblateness (J2) and drag are ignored.  This
    is the fast bulk path used for event screening.
    """
    dev = _resolve_device(device)
    t = np.asarray(times_s, dtype=np.float64)

    if dev is None:  # NumPy fallback
        return _two_body_numpy(elements, t)

    a = torch.as_tensor(elements.semi_major_axis_km, dtype=torch.float64, device=dev)
    e = torch.as_tensor(elements.eccentricity, dtype=torch.float64, device=dev)
    inc = torch.as_tensor(elements.inclination_rad, dtype=torch.float64, device=dev)
    raan = torch.as_tensor(elements.raan_rad, dtype=torch.float64, device=dev)
    argp = torch.as_tensor(elements.arg_perigee_rad, dtype=torch.float64, device=dev)
    m0 = torch.as_tensor(elements.mean_anomaly_rad, dtype=torch.float64, device=dev)
    n = torch.as_tensor(elements.mean_motion_rad_s, dtype=torch.float64, device=dev)

    tt = torch.as_tensor(t, dtype=torch.float64, device=dev).reshape(-1, 1)
    m = m0.reshape(1, -1) + n.reshape(1, -1) * tt
    ecc_anom = _solve_kepler(m, e.reshape(1, -1))

    cos_e, sin_e = torch.cos(ecc_anom), torch.sin(ecc_anom)
    a2 = a.reshape(1, -1)
    x_orb = a2 * (cos_e - e.reshape(1, -1))
    y_orb = a2 * torch.sqrt(torch.clamp(1.0 - e.reshape(1, -1) ** 2, min=0.0)) * sin_e
    z_orb = torch.zeros_like(x_orb)

    # Rotation Rz(raan) Rx(inc) Rz(argp) applied to the orbital-plane vector.
    cr, sr = torch.cos(raan).reshape(1, -1), torch.sin(raan).reshape(1, -1)
    ci, si = torch.cos(inc).reshape(1, -1), torch.sin(inc).reshape(1, -1)
    cw, sw = torch.cos(argp).reshape(1, -1), torch.sin(argp).reshape(1, -1)

    # Combined rotation matrix components (per satellite row).
    r11 = cr * cw - sr * sw * ci
    r12 = -cr * sw - sr * cw * ci
    r21 = sr * cw + cr * sw * ci
    r22 = -sr * sw + cr * cw * ci
    r31 = sw * si
    r32 = cw * si

    x = r11 * x_orb + r12 * y_orb
    y = r21 * x_orb + r22 * y_orb
    z = r31 * x_orb + r32 * y_orb
    out = torch.stack([x, y, z], dim=-1)
    return out.detach().cpu().numpy()


def _two_body_numpy(
    elements: WalkerConstellation, times_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """NumPy fallback for :func:`propagate_two_body` (same math, no torch)."""
    a = elements.semi_major_axis_km[None, :]
    e = elements.eccentricity[None, :]
    inc = elements.inclination_rad[None, :]
    raan = elements.raan_rad[None, :]
    argp = elements.arg_perigee_rad[None, :]
    m0 = elements.mean_anomaly_rad[None, :]
    n = elements.mean_motion_rad_s[None, :]

    m = m0 + n * times_s[:, None]
    ecc = m.copy()
    for _ in range(12):
        ecc = ecc - (ecc - e * np.sin(ecc) - m) / (1.0 - e * np.cos(ecc))

    x_orb = a * (np.cos(ecc) - e)
    y_orb = a * np.sqrt(np.clip(1.0 - e**2, 0.0, None)) * np.sin(ecc)
    z_orb = np.zeros_like(x_orb)

    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    r11 = cr * cw - sr * sw * ci
    r12 = -cr * sw - sr * cw * ci
    r21 = sr * cw + cr * sw * ci
    r22 = -sr * sw + cr * cw * ci
    r31 = sw * si
    r32 = cw * si

    return np.stack(
        [
            r11 * x_orb + r12 * y_orb,
            r21 * x_orb + r22 * y_orb,
            r31 * x_orb + r32 * y_orb,
        ],
        axis=-1,
    )


def propagate_sgp4(
    tle_line1: str, tle_line2: str, jd: NDArray[np.float64], fr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Propagate one TLE with SGP4 and return ECI positions (km, shape ``(n, 3)``)."""
    from sgp4.api import Satrec

    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    out = np.empty((len(jd), 3), dtype=np.float64)
    for i, (j, f) in enumerate(zip(jd, fr)):
        err, r, _v = sat.sgp4(float(j), float(f))
        if err != 0:
            raise RuntimeError(f"SGP4 error {err} at jd={j}, fr={f}")
        out[i] = r
    return out


# --- J2 secular propagation, fully vectorised in torch -----------------------
#
# Gate-1 production model: circular orbits (eccentricity ignored) with the J2
# secular rates for the right ascension of the ascending node (RAAN) and the
# argument of latitude.  Positions are computed in ECI, then rotated to ECEF with
# GMST for a fixed epoch (from configs/constellations.yaml).
J2: float = 1.08262668e-3


def _secular_rates_j2(a, inc, n, earth_radius_km):
    """J2 secular rates (rad/s) for circular orbits: ``(raan_dot, u_dot)``.

    ``raan_dot`` is the standard nodal regression
    ``-(3/2) J2 n (Re/a)^2 cos i``.  ``u_dot`` is the argument-of-latitude rate
    ``n [1 + (3/4) J2 (Re/a)^2 (5 cos^2 i - 1)]``; the ``(5 cos^2 i - 1)``
    coefficient was verified against an independent DOP853 two-body+J2
    integration and against SGP4 at 14 inclinations (tests/test_propagation.py).
    """
    factor = J2 * n * (earth_radius_km / a) ** 2
    cos_i = torch.cos(inc)
    raan_dot = -1.5 * factor * cos_i
    u_dot = n + 0.75 * factor * (5.0 * cos_i**2 - 1.0)
    return raan_dot, u_dot


def propagate_ecef(
    elements: WalkerConstellation,
    times_s: NDArray[np.float64] | object,
    epoch_jd: float,
    earth_radius_km: float,
    device: Device = "auto",
):
    """Return ECEF positions ``(n_times, n_sats, 3)`` in km.

    On the GPU path the returned object is a ``torch.Tensor`` (float64) that stays
    on the device; the NumPy path returns an ``ndarray``.  Circular orbits with J2
    secular RAAN / argument-of-latitude drift, GMST from ``epoch_jd``.
    """
    from .frames import gmst_rad_series

    dev = _resolve_device(device)
    t = np.asarray(times_s, dtype=np.float64)

    if dev is not None:
        a = torch.as_tensor(elements.semi_major_axis_km, dtype=torch.float64, device=dev)
        inc = torch.as_tensor(elements.inclination_rad, dtype=torch.float64, device=dev)
        n = torch.as_tensor(elements.mean_motion_rad_s, dtype=torch.float64, device=dev)
        raan0 = torch.as_tensor(elements.raan_rad, dtype=torch.float64, device=dev)
        m0 = torch.as_tensor(elements.mean_anomaly_rad, dtype=torch.float64, device=dev)
        argp0 = torch.as_tensor(elements.arg_perigee_rad, dtype=torch.float64, device=dev)

        raan_dot, u_dot = _secular_rates_j2(a, inc, n, earth_radius_km)
        tt = torch.as_tensor(t, dtype=torch.float64, device=dev).reshape(-1, 1)
        raan = raan0.reshape(1, -1) + raan_dot.reshape(1, -1) * tt
        u = (argp0 + m0).reshape(1, -1) + u_dot.reshape(1, -1) * tt

        ci, si = torch.cos(inc), torch.sin(inc)
        cr, sr = torch.cos(raan), torch.sin(raan)
        cu, su = torch.cos(u), torch.sin(u)
        a2 = a.reshape(1, -1)
        x = a2 * (cr * cu - sr * su * ci)
        y = a2 * (sr * cu + cr * su * ci)
        z = a2 * (su * si)
        eci = torch.stack([x, y, z], dim=-1)

        gmst = torch.as_tensor(
            gmst_rad_series(epoch_jd + t / 86400.0), dtype=torch.float64, device=dev
        ).reshape(-1, 1)
        c, s = torch.cos(gmst), torch.sin(gmst)
        # r_ECEF = Rz(-gmst) r_ECI  (see ngsi.frames.eci_to_ecef).
        ecef = torch.stack(
            [c * eci[..., 0] + s * eci[..., 1], -s * eci[..., 0] + c * eci[..., 1], eci[..., 2]],
            dim=-1,
        )
        # Phases and trigonometry above stay float64; the heavy per-station
        # geometry downstream follows the configured compute dtype.
        from .precision import compute_dtype

        return ecef.to(compute_dtype())

    return _ecef_numpy(elements, t, epoch_jd, earth_radius_km)


def _ecef_numpy(
    elements: WalkerConstellation, t: NDArray[np.float64], epoch_jd: float, earth_radius_km: float
) -> NDArray[np.float64]:
    """NumPy fallback for :func:`propagate_ecef` (same math as the torch path)."""
    from .frames import gmst_rad_series

    a = elements.semi_major_axis_km[None, :]
    inc = elements.inclination_rad[None, :]
    n = elements.mean_motion_rad_s[None, :]
    raan0 = elements.raan_rad[None, :]
    u0 = (elements.arg_perigee_rad + elements.mean_anomaly_rad)[None, :]

    factor = J2 * n * (earth_radius_km / a) ** 2
    cos_i = np.cos(inc)
    raan_dot = -1.5 * factor * cos_i
    u_dot = n + 0.75 * factor * (5.0 * cos_i**2 - 1.0)

    raan = raan0 + raan_dot * t[:, None]
    u = u0 + u_dot * t[:, None]
    ci, si = np.cos(inc), np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)
    cu, su = np.cos(u), np.sin(u)
    x = a * (cr * cu - sr * su * ci)
    y = a * (sr * cu + cr * su * ci)
    z = a * (su * si)
    eci = np.stack([x, y, z], axis=-1)

    gmst = np.asarray(gmst_rad_series(epoch_jd + t / 86400.0))[:, None]
    c, s = np.cos(gmst), np.sin(gmst)
    return np.stack(
        [c * eci[..., 0] + s * eci[..., 1], -s * eci[..., 0] + c * eci[..., 1], eci[..., 2]],
        axis=-1,
    )


__all__ = [
    "propagate_two_body",
    "propagate_sgp4",
    "propagate_ecef",
    "MU_EARTH_KM3_S2",
    "J2",
]
