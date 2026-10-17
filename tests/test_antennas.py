"""Antenna patterns: boresight and floor hand checks (AGENTS.md rule 6)."""

from __future__ import annotations

import pytest

from ngsi.antennas import s1428_gain, s1528_gain, s465_gain


def test_s1528_boresight_equals_peak_gain() -> None:
    assert s1528_gain(0.0, 34.0, 2.0) == pytest.approx(34.0)


def test_s1528_floor_at_large_angle() -> None:
    assert s1528_gain(89.0, 34.0, 2.0) == pytest.approx(0.0)


def test_s1528_never_below_floor() -> None:
    for ang in range(0, 90, 3):
        assert s1528_gain(float(ang), 34.0, 2.0) >= 0.0


def test_s1528_first_sidelobe_region_is_minus_6_75() -> None:
    # Just outside a*psi_b and inside b*psi_b the pattern is flat at Gmax - 6.75.
    assert s1528_gain(2.58 * 2.0 + 0.05, 34.0, 2.0) == pytest.approx(34.0 - 6.75)


def test_s465_boresight_equals_peak_gain() -> None:
    assert s465_gain(0.0, 45.0, 3.0, 20.0) == pytest.approx(45.0)


def test_s465_far_sidelobe_floor() -> None:
    assert s465_gain(90.0, 45.0, 3.0, 20.0) == pytest.approx(-10.0)


def test_s1428_delegates_to_s465() -> None:
    assert s1428_gain(5.0, 45.0, 3.0, 20.0) == pytest.approx(s465_gain(5.0, 45.0, 3.0, 20.0))
