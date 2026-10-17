"""Walker-Delta constellation element generation.

A Walker-Delta pattern is written ``i: T/P/F`` where ``T`` is the total number
of satellites, ``P`` the number of orbital planes, and ``F`` the phasing factor
(``0 .. P-1``).  Satellites are numbered ``k = 0 .. T-1`` with ``k = j*P + p``
for plane ``p`` and slot ``j``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import Shell

MU_EARTH_KM3_S2: float = 398600.4418
EARTH_RADIUS_KM: float = 6378.137


@dataclass(frozen=True)
class WalkerConstellation:
    """Keplerian elements (angles in radians, distances in km) for every satellite."""

    semi_major_axis_km: NDArray[np.float64]
    eccentricity: NDArray[np.float64]
    inclination_rad: NDArray[np.float64]
    raan_rad: NDArray[np.float64]
    arg_perigee_rad: NDArray[np.float64]
    mean_anomaly_rad: NDArray[np.float64]
    mean_motion_rad_s: NDArray[np.float64]
    plane_index: NDArray[np.int64]
    slot_index: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.semi_major_axis_km.size)


def mean_motion_rad_s(semi_major_axis_km: float) -> float:
    """Keplerian mean motion n = sqrt(mu / a^3) [rad/s]."""
    return float(np.sqrt(MU_EARTH_KM3_S2 / semi_major_axis_km**3))


def walker_delta(
    altitude_km: float,
    inclination_deg: float,
    planes: int,
    sats_per_plane: int,
    phasing_f: int,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> WalkerConstellation:
    """Generate a Walker-Delta i:T/P/F constellation.

    ``F`` is the relative phasing between adjacent planes: satellite ``j`` in
    plane ``p`` has argument of latitude ``360/sats_per_plane * j + 360*F/T * p``.
    ``earth_radius_km`` sets the orbital radius ``a = earth_radius_km + altitude``
    (the Gate-1 spherical-earth choice is 6371 km).
    """
    if planes <= 0 or sats_per_plane <= 0:
        raise ValueError("planes and sats_per_plane must be positive")
    if not 0 <= phasing_f < planes:
        raise ValueError("phasing_f must satisfy 0 <= F < planes")

    total = planes * sats_per_plane
    inc = np.deg2rad(inclination_deg)
    a = earth_radius_km + altitude_km
    n = mean_motion_rad_s(a)

    plane_index = np.repeat(np.arange(planes), sats_per_plane)
    slot_index = np.tile(np.arange(sats_per_plane), planes)

    raan = np.deg2rad(360.0 / planes * plane_index)
    arg_lat = np.deg2rad(
        360.0 / sats_per_plane * slot_index + 360.0 * phasing_f / total * plane_index
    )

    zeros = np.zeros(total, dtype=np.float64)
    return WalkerConstellation(
        semi_major_axis_km=np.full(total, a, dtype=np.float64),
        eccentricity=zeros.copy(),
        inclination_rad=np.full(total, inc, dtype=np.float64),
        raan_rad=raan,
        arg_perigee_rad=zeros.copy(),
        # Circular orbit: mean anomaly == argument of latitude.
        mean_anomaly_rad=arg_lat,
        mean_motion_rad_s=np.full(total, n, dtype=np.float64),
        plane_index=plane_index,
        slot_index=slot_index,
    )


def from_shell(shell: Shell, earth_radius_km: float = EARTH_RADIUS_KM) -> WalkerConstellation:
    """Build a :class:`WalkerConstellation` from a config :class:`~ngsi.config.Shell`."""
    return walker_delta(
        altitude_km=shell.altitude_km,
        inclination_deg=shell.inclination_deg,
        planes=shell.planes,
        sats_per_plane=shell.sats_per_plane,
        phasing_f=shell.phasing_f,
        earth_radius_km=earth_radius_km,
    )
