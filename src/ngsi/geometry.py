"""Elevation, azimuth, slant range, angular separation and visibility.

Convention: all off-boresight / separation angles returned here are *angles at
the antenna* measured at the earth station or satellite, never ground distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .frames import ecef_to_enu, enu_basis

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LookAngles:
    """Topocentric look angles of a target from an earth station."""

    elevation_deg: float
    azimuth_deg: float
    slant_range_km: float


def look_angles(
    station_ecef: NDArray[np.float64],
    station_lat_deg: float,
    station_lon_deg: float,
    target_ecef: NDArray[np.float64],
) -> LookAngles:
    """Return elevation, azimuth and slant range of ``target_ecef`` from a station."""
    e, n, u = ecef_to_enu(target_ecef, station_ecef, station_lat_deg, station_lon_deg)
    rng = float(np.sqrt(e * e + n * n + u * u))
    elev = float(np.rad2deg(np.arcsin(np.clip(u / rng, -1.0, 1.0))))
    az = float(np.rad2deg(np.arctan2(e, n)) % 360.0)
    return LookAngles(elevation_deg=elev, azimuth_deg=az, slant_range_km=rng)


def elevation_deg(
    station_ecef: NDArray[np.float64],
    station_lat_deg: float,
    station_lon_deg: float,
    target_ecef: NDArray[np.float64],
) -> float:
    """Convenience wrapper returning only the elevation in degrees."""
    return look_angles(station_ecef, station_lat_deg, station_lon_deg, target_ecef).elevation_deg


def angular_separation_deg(
    target_a_ecef: NDArray[np.float64],
    target_b_ecef: NDArray[np.float64],
    origin_ecef: NDArray[np.float64],
) -> float:
    """Angle (deg) between two targets as subtended at ``origin_ecef`` (e.g. a station).

    Uses ``atan2(|a x b|, a.b)`` rather than ``acos`` so it stays accurate for the
    small separations (~0.1-5 deg) that in-line events depend on.
    """
    a = np.asarray(target_a_ecef, dtype=np.float64) - np.asarray(origin_ecef, dtype=np.float64)
    b = np.asarray(target_b_ecef, dtype=np.float64) - np.asarray(origin_ecef, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def is_visible(elevation: float, min_elevation_deg: float) -> bool:
    """True when a target at ``elevation`` (deg) is above the station's mask."""
    return elevation >= min_elevation_deg


# --- Batched (torch) geometry for screening ----------------------------------
def station_ecef_spherical(
    lat_deg, lon_deg, radius_km: float, device: str = "cpu"
):
    """Spherical-earth station ECEF positions ``[G, 3]`` (torch, float64)."""
    lat = torch.deg2rad(torch.as_tensor(lat_deg, dtype=torch.float64, device=device))
    lon = torch.deg2rad(torch.as_tensor(lon_deg, dtype=torch.float64, device=device))
    x = radius_km * torch.cos(lat) * torch.cos(lon)
    y = radius_km * torch.cos(lat) * torch.sin(lon)
    z = radius_km * torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def unit_look_vectors(sat_ecef, station_ecef):
    """Unit look vectors ``[T, G, S, 3]`` from each station to each satellite."""
    # Follow the satellites' dtype (ngsi.precision) instead of promoting to float64.
    rel = sat_ecef.unsqueeze(1) - station_ecef.to(sat_ecef.dtype).unsqueeze(0).unsqueeze(2)
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
    return rel / torch.clamp(dist, min=1e-12), dist.squeeze(-1)


def elevation_deg_batch(unit_look, up_ecef):
    """Elevation (deg) ``[T, G, S]`` of unit look vectors relative to local up ``[G, 3]``."""
    sin_elev = (unit_look * up_ecef.to(unit_look.dtype).unsqueeze(0).unsqueeze(2)).sum(-1)
    return torch.rad2deg(torch.asin(torch.clamp(sin_elev, -1.0, 1.0)))


def angular_separation_batch(u, v):
    """Element-wise angle (deg) between two unit-vector tensors with last dim 3.

    ``atan2(|u x v|, u.v)``; stable for tiny angles, unlike ``acos``.
    """
    cross = torch.cross(u, v, dim=-1)
    return torch.rad2deg(torch.atan2(torch.linalg.norm(cross, dim=-1), (u * v).sum(-1)))
