"""Hand-checkable values for the analytic sanity functions (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np
import pytest

from ngsi.sanity import (
    analytic_density_per_deg,
    analytic_visible_count,
    earth_central_angle_cap,
    max_relative_deviation,
    walker_sub_latitude_density,
)


def test_polar_density_is_uniform() -> None:
    lats = np.linspace(-89.0, 89.0, 51)
    dens = walker_sub_latitude_density(90.0, lats)
    np.testing.assert_allclose(dens, 1.0, rtol=1e-12)
    per_deg = analytic_density_per_deg(90.0, lats)
    np.testing.assert_allclose(per_deg, 1.0 / 178.0, rtol=1e-12)


def test_density_at_equator_matches_formula() -> None:
    # inc=53 deg: cos(0)/sqrt(sin^2 53) = 1/sin(53) = 1.2520...
    val = walker_sub_latitude_density(53.0, np.array([0.0]))[0]
    assert val == pytest.approx(1.0 / np.sin(np.deg2rad(53.0)), rel=1e-12)


def test_density_zero_outside_inclination_band() -> None:
    dens = walker_sub_latitude_density(53.0, np.array([60.0, -60.0, 90.0]))
    np.testing.assert_array_equal(dens, 0.0)


def test_uniform_sim_has_zero_deviation_for_polar() -> None:
    bins = np.arange(-90.0, 90.1, 5.0)
    counts = np.full(len(bins) - 1, 1000.0)
    assert max_relative_deviation(counts, bins, 90.0) == pytest.approx(0.0, abs=1e-12)


def test_visibility_cap_nadir() -> None:
    # Looking straight down (min elevation with a finite cap radius).
    psi = earth_central_angle_cap(550.0, 0.0, 6371.0)
    assert 0.0 < psi < np.pi / 2
    k = 6371.0 / (6371.0 + 550.0)
    assert psi == pytest.approx(np.arccos(k), rel=1e-12)


def test_visible_count_monotonic_in_elevation() -> None:
    lo = analytic_visible_count(550.0, 53.0, 72, 22, 45.0, 35.0, 6371.0)
    hi = analytic_visible_count(550.0, 53.0, 72, 22, 45.0, 20.0, 6371.0)
    assert hi > lo > 0.0
