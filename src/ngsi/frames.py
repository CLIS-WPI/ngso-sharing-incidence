"""WGS-84 / ECI / ECEF / geodetic / ENU frame conversions.

All positions are in kilometres, all angles in degrees unless the name says
``_rad``.  Reference: WGS-84 ellipsoid parameters (NIMA TR8350.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- WGS-84 ellipsoid -------------------------------------------------------
WGS84_A_KM: float = 6378.137
WGS84_F: float = 1.0 / 298.257223563
WGS84_B_KM: float = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)

# Earth rotation rate (rad/s), ITU-R / IERS nominal value.
OMEGA_EARTH_RAD_S: float = 7.2921150e-5


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, height_km: float = 0.0
) -> NDArray[np.float64]:
    """Convert geodetic latitude/longitude/height to ECEF (km)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + height_km) * cos_lat * np.cos(lon)
    y = (n + height_km) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height_km) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(
    r_ecef: NDArray[np.float64], tol: float = 1e-10, max_iter: int = 20
) -> tuple[float, float, float]:
    """Convert ECEF (km) to (lat_deg, lon_deg, height_km) via fixed-point iteration."""
    x, y, z = (float(v) for v in r_ecef)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    height = 0.0
    for _ in range(max_iter):
        sin_lat = np.sin(lat)
        n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        height = p / np.cos(lat) - n
        lat_new = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(height)


def enu_basis(lat_deg: float, lon_deg: float) -> NDArray[np.float64]:
    """Return the 3x3 matrix whose rows are the local East, North, Up unit vectors
    expressed in ECEF."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.vstack([east, north, up])


def ecef_to_enu(
    r_ecef: NDArray[np.float64],
    station_ecef: NDArray[np.float64],
    lat_deg: float,
    lon_deg: float,
) -> NDArray[np.float64]:
    """Return (east, north, up) components (km) of ``r_ecef`` relative to a station."""
    return enu_basis(lat_deg, lon_deg) @ (np.asarray(r_ecef) - np.asarray(station_ecef))


def rotation_z(theta_rad: float) -> NDArray[np.float64]:
    """Right-handed active rotation about +Z by ``theta_rad``."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gmst_rad(jd_ut1: float) -> float:
    """Greenwich mean sidereal time (rad) from a Julian date (IAU 1982, ~1e-4 rad)."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - 2451545.0)
        + 0.000387933 * t**2
        - t**3 / 38710000.0
    )
    return float(np.deg2rad(np.mod(gmst_deg, 360.0)))


def eci_to_ecef(r_eci: NDArray[np.float64], gmst: float) -> NDArray[np.float64]:
    """Rotate an ECI position (km) into the ECEF frame at sidereal time ``gmst``."""
    return rotation_z(gmst) @ np.asarray(r_eci, dtype=np.float64)
