"""Hand-checkable values for the analytic co-located event-fraction model."""

from __future__ import annotations

import numpy as np
import pytest

from ngsi.analytic import (
    ShellSpec,
    effective_area_db,
    equivalent_angle_deg,
    event_fraction_vs_latitude,
    pfd_constant_dbw_per_m2_per_mhz,
    separation_probability,
    theta_star_for_starlink,
)
from ngsi.antennas import s1428_gain
from ngsi.constants import IN_THRESHOLD_DB
from ngsi.link_budget import noise_power_dbw_per_mhz, terminal_peak_gain_dbi

FREQ = 11.7
DIAM = 0.45


def test_effective_area_hand_check() -> None:
    # lambda = 0.0256410 m -> 10 log10(lambda^2/4pi) = -42.81 dB.
    assert effective_area_db(FREQ) == pytest.approx(-42.81, abs=0.01)


def test_pfd_matches_25_208b() -> None:
    # EIRP_max +15.68 dBW/MHz over r_max(550 km, 25 deg) -> ~-116.3 dBW/m^2/MHz,
    # within 0.4 dB of the 47 CFR 25.208(b)(2) flat 25-90 deg limit (-116).
    pfd = pfd_constant_dbw_per_m2_per_mhz(15.68, 550.0, 25.0)
    assert pfd == pytest.approx(-116.3, abs=0.05)
    assert abs(pfd - (-116.0)) < 0.4


def test_equivalent_angle_round_trip() -> None:
    peak = terminal_peak_gain_dbi(DIAM, FREQ, 0.6)
    pfd = -116.3
    noise = noise_power_dbw_per_mhz(200.0)
    t = equivalent_angle_deg(peak, DIAM, FREQ, pfd, noise)
    target = IN_THRESHOLD_DB + noise - pfd - effective_area_db(FREQ)
    # By construction the returned angle reproduces the threshold gain.
    assert s1428_gain(t, peak, DIAM, FREQ) == pytest.approx(target, abs=0.05)
    # In the universal sidelobe envelope 32 - 25 log10(theta): theta ~ 17 deg.
    assert t == pytest.approx(10.0 ** ((32.0 - target) / 25.0), rel=0.05)
    assert 10.0 < t < 30.0


def test_equivalent_angle_aperture_independence() -> None:
    # The S.1428 envelope is aperture-independent, so theta* is the same for all
    # three headline apertures as long as the threshold sits in the envelope.
    pfd = -116.3
    noise = noise_power_dbw_per_mhz(200.0)
    ts = [equivalent_angle_deg(terminal_peak_gain_dbi(d, FREQ, 0.6), d, FREQ, pfd, noise)
          for d in (0.45, 0.6, 0.75)]
    assert max(ts) - min(ts) < 0.05


def test_equivalent_angle_bounds() -> None:
    peak = terminal_peak_gain_dbi(DIAM, FREQ, 0.6)
    noise = noise_power_dbw_per_mhz(200.0)
    # Very strong PFD: the threshold is met at every off-axis angle -> theta* = 180.
    assert equivalent_angle_deg(peak, DIAM, FREQ, -40.0, noise) == 180.0
    # Very weak PFD: even boresight cannot reach the threshold -> theta* = 0.
    assert equivalent_angle_deg(peak, DIAM, FREQ, -300.0, noise) == 0.0


def test_separation_probability_limits_and_monotonicity() -> None:
    alpha, n = 65.0, 40.0
    assert separation_probability(0.0, alpha, n, alpha, n) == 0.0
    # theta >= 2 alpha covers the whole disc pair.
    assert separation_probability(2.0 * alpha + 1.0, alpha, n, alpha, n) == pytest.approx(1.0)
    vals = [separation_probability(t, alpha, n, alpha, n) for t in (2.0, 5.0, 10.0, 20.0)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_separation_probability_matches_monte_carlo() -> None:
    alpha, n, theta = 10.0, 5.0, 5.0
    p = separation_probability(theta, alpha, n, alpha, n, n_grid=400)
    rng = np.random.default_rng(0)
    m = 400_000

    def _sample(k):
        x = np.sqrt(1.0 - (1.0 - rng.random(k)) ** (1.0 / n)) * alpha
        az = rng.random(k) * 2.0 * np.pi
        return x * np.cos(az), x * np.sin(az)

    xa, ya = _sample(m)
    xb, yb = _sample(m)
    sep = np.hypot(xa - xb, ya - yb)
    mc = float((sep <= theta).mean())
    assert p == pytest.approx(mc, abs=0.01)


def test_co_located_fraction_bounds() -> None:
    oneweb = [ShellSpec(1200.0, 87.9, 18, 36)]
    starlink = [ShellSpec(550.0, 53.0, 72, 22)]
    vals = event_fraction_vs_latitude(np.array([0.0, 25.0, 50.0, 65.0]), 17.0,
                                      oneweb, starlink, 25.0)
    assert np.all((vals >= 0.0) & (vals <= 1.0))
    assert vals.max() > 0.0


def test_theta_star_pfd_offset_shrinks_angle() -> None:
    t0 = theta_star_for_starlink(DIAM)
    t3 = theta_star_for_starlink(DIAM, pfd_offset_db=-3.0)
    t6 = theta_star_for_starlink(DIAM, pfd_offset_db=-6.0)
    # A weaker interferer needs a higher victim gain to trigger -> smaller angle.
    assert t0 > t3 > t6 > 0.0
