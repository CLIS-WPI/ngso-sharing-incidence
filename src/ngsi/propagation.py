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


__all__ = ["propagate_two_body", "propagate_sgp4", "MU_EARTH_KM3_S2"]
