"""Geometry: elevation/range hand checks (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np

from ngsi.frames import WGS84_A_KM, geodetic_to_ecef
from ngsi.geometry import angular_separation_deg, is_visible, look_angles


def test_elevation_at_nadir_is_90() -> None:
    station = geodetic_to_ecef(0.0, 0.0, 0.0)
    height_km = 550.0
    sat = np.array([WGS84_A_KM + height_km, 0.0, 0.0])
    la = look_angles(station, 0.0, 0.0, sat)
    assert abs(la.elevation_deg - 90.0) < 1e-9
    assert abs(la.slant_range_km - height_km) < 1e-9


def test_azimuth_east_is_90() -> None:
    # At (lat=0, lon=0) the ECEF +Y axis is local East, so an eastward offset
    # from the station must give azimuth 90 deg.
    station = geodetic_to_ecef(0.0, 0.0, 0.0)
    sat = station + np.array([0.0, 1000.0, 0.0])
    la = look_angles(station, 0.0, 0.0, sat)
    assert abs(la.azimuth_deg - 90.0) < 1e-6
    assert abs(la.elevation_deg) < 1e-9


def test_angular_separation_same_direction_is_zero() -> None:
    station = np.array([1.0, 2.0, 3.0])
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([20.0, 40.0, 60.0])
    assert abs(angular_separation_deg(a, b, station)) < 1e-9


def test_angular_separation_orthogonal_is_90() -> None:
    station = np.zeros(3)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(angular_separation_deg(a, b, station) - 90.0) < 1e-9


def test_visibility_mask() -> None:
    assert is_visible(30.0, 25.0)
    assert not is_visible(10.0, 25.0)


def test_angular_separation_small_angles_atan2() -> None:
    # Two unit vectors at a known small separation; atan2 form must recover it.
    st = np.array([7000.0, 0.0, 0.0])
    for theta_deg in (0.1, 1.0, 5.0):
        th = np.deg2rad(theta_deg)
        a = st + np.array([0.0, 0.0, 100.0])
        b = st + 100.0 * np.array([0.0, np.sin(th), np.cos(th)])
        assert abs(angular_separation_deg(a, b, st) - theta_deg) < 1e-6


def test_angular_separation_batch_matches_known_angle() -> None:
    import torch

    from ngsi.geometry import angular_separation_batch

    th = np.deg2rad(2.0)
    u = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    v = torch.tensor([[np.cos(th), np.sin(th), 0.0]], dtype=torch.float64)
    np.testing.assert_allclose(float(angular_separation_batch(u, v)[0]), 2.0, atol=1e-9)
