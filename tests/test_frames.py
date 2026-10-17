"""Frame conversions: hand-checkable points (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np

from ngsi.frames import (
    WGS84_A_KM,
    ecef_to_geodetic,
    eci_to_ecef,
    enu_basis,
    geodetic_to_ecef,
    gmst_rad,
)


def test_geodetic_to_ecef_equator_prime_meridian() -> None:
    r = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(r, [WGS84_A_KM, 0.0, 0.0], atol=1e-9)


def test_geodetic_to_ecef_north_pole() -> None:
    r = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert np.allclose(r[:2], [0.0, 0.0], atol=1e-6)
    assert r[2] > 0.0


def test_geodetic_ecef_round_trip() -> None:
    for lat, lon, h in [(0.0, 0.0, 0.0), (45.0, 120.0, 0.5), (-30.0, 250.0, 2.0)]:
        r = geodetic_to_ecef(lat, lon, h)
        lat2, lon2, h2 = ecef_to_geodetic(r)
        assert abs(lat2 - lat) < 1e-8
        assert abs(((lon2 - lon + 180.0) % 360.0) - 180.0) < 1e-8
        assert abs(h2 - h) < 1e-6


def test_enu_basis_is_orthonormal() -> None:
    m = enu_basis(37.0, -122.0)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(m), 1.0)


def test_eci_to_ecef_identity_at_zero_gmst() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    assert np.allclose(eci_to_ecef(r, 0.0), r)


def test_eci_to_ecef_quarter_turn() -> None:
    # At GMST = 90 deg the Greenwich meridian is at ECI RA = 90 deg, so a point on
    # the ECI x-axis (RA = 0) sits at ECEF longitude -90 deg: [0, -7000, 0].
    r = np.array([7000.0, 0.0, 0.0])
    out = eci_to_ecef(r, np.pi / 2.0)
    assert np.allclose(out, [0.0, -7000.0, 0.0], atol=1e-6)


def test_gmst_is_bounded() -> None:
    g = gmst_rad(2451545.0)  # J2000.0
    assert 0.0 <= g < 2.0 * np.pi
