"""Elevation, azimuth, slant range, angular separation and visibility.

Convention: all off-boresight / separation angles returned here are *angles at
the antenna* measured at the earth station or satellite, never ground distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .frames import ecef_to_enu, enu_basis


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
    """Angle (deg) between two targets as subtended at ``origin_ecef`` (e.g. a station)."""
    a = np.asarray(target_a_ecef, dtype=np.float64) - np.asarray(origin_ecef, dtype=np.float64)
    b = np.asarray(target_b_ecef, dtype=np.float64) - np.asarray(origin_ecef, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosang)))


def is_visible(elevation: float, min_elevation_deg: float) -> bool:
    """True when a target at ``elevation`` (deg) is above the station's mask."""
    return elevation >= min_elevation_deg
