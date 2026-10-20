"""Handoff (satellite-diversity) policies of the Delta T/T screener (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngsi.config import Beam, Constellation, Shell
from ngsi.dtt_screen import (
    LEGACY_POLICY,
    HandoffPolicy,
    rate_limited_choice,
    screen_pair_dtt,
)
from ngsi.geometry import station_ecef_spherical
from ngsi.link_budget import terminal_peak_gain_dbi
from ngsi.systems import build_system

EPOCH_JD = 2460676.5
R_EARTH = 6371.0
FREQ = 11.7
DIAMETER = 0.45
EFF = 0.6
MARGIN = 3.0
VICTIM_BEAM = Beam(30.0, 3.0, 10.6, 12.0, 250.0, 350.0, "ASSUMED")
INTERF_BEAM = Beam(44.0, 1.1, 16.0, 12.0, 250.0, 362.9, "ASSUMED")


# --- rate_limited_choice: hand-checkable sequences -------------------------

def _one(ids_row, n_t):
    """[T, 1, K] ids that do not change with time."""
    return np.tile(np.asarray(ids_row, dtype=np.int64)[None, None, :], (n_t, 1, 1))


def test_rate_limit_blocks_switch_until_interval_elapsed() -> None:
    # Satellites 10 (rank 0) and 20 (rank 1); the memoryless policy wants
    # 10, 20, 10, 20 at t = 0, 10, 20, 30 s.  With a 25 s limit the terminal
    # acquires 10 at t=0, cannot leave it at t=10 (10 s < 25 s), has no reason
    # to at t=20, and switches to 20 at t=30 (30 s >= 25 s).
    ids = _one([10, 20], 4)
    valid = np.ones((4, 1, 2), dtype=bool)
    desired = np.array([[0], [1], [0], [1]])
    r, sw = rate_limited_choice(ids, valid, desired, np.array([0.0, 10.0, 20.0, 30.0]), 25.0, {})
    assert r[:, 0].tolist() == [0, 0, 0, 1]
    assert sw[:, 0].tolist() == [False, False, False, True]


def test_rate_limit_forced_switch_when_satellite_lost() -> None:
    # The current satellite (10) drops out of the candidate set at t=10 s; the
    # terminal must switch to 20 immediately even though the limit is 1000 s.
    ids = _one([10, 20], 2)
    valid = np.array([[[True, True]], [[False, True]]])
    desired = np.array([[0], [1]])
    r, sw = rate_limited_choice(ids, valid, desired, np.array([0.0, 10.0]), 1000.0, {})
    assert r[:, 0].tolist() == [0, 1]
    assert sw[:, 0].tolist() == [False, True]


def test_rate_limit_state_carries_across_calls() -> None:
    # Same as the first test, split into two calls (two time chunks).
    ids = _one([10, 20], 2)
    valid = np.ones((2, 1, 2), dtype=bool)
    state: dict = {}
    r1, _ = rate_limited_choice(ids, valid, np.array([[0], [1]]), np.array([0.0, 10.0]), 25.0,
                                state)
    r2, sw2 = rate_limited_choice(ids, valid, np.array([[0], [1]]), np.array([20.0, 30.0]), 25.0,
                                  state)
    assert r1[:, 0].tolist() + r2[:, 0].tolist() == [0, 0, 0, 1]
    assert sw2[:, 0].tolist() == [False, True]


def test_rate_limit_tracks_satellite_identity_not_rank() -> None:
    # Ranks reorder (satellite 20 becomes the highest-elevation one); a terminal
    # on 10 that may not switch keeps satellite 10, now at rank 1.
    ids = np.array([[[10, 20]], [[20, 10]]], dtype=np.int64)
    valid = np.ones((2, 1, 2), dtype=bool)
    r, sw = rate_limited_choice(ids, valid, np.array([[0], [0]]), np.array([0.0, 10.0]), 60.0, {})
    assert r[:, 0].tolist() == [0, 1]
    assert not sw.any()


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        HandoffPolicy("x", link_cn="nope")
    with pytest.raises(ValueError):
        HandoffPolicy("x", link_cn="serving", k=2)
    with pytest.raises(ValueError):
        HandoffPolicy("x", k=0)
    with pytest.raises(ValueError):
        HandoffPolicy("x", link_cn="serving", keep_serving_if_better=True)
    p = HandoffPolicy.from_dict({"name": "k2", "k": 2})
    assert p.link_cn == "candidate" and p.k == 2 and p.min_interval_s is None


# --- screener-level consistency ----------------------------------------------

def _systems():
    a = Constellation("A", "opA", [Shell("a1", 1200.0, 87.9, 12, 20, 1, VICTIM_BEAM, "ASSUMED")])
    b = Constellation("B", "opB", [Shell("b1", 550.0, 53.0, 24, 22, 1, INTERF_BEAM, "ASSUMED")])
    rng = np.random.default_rng(3)
    return (build_system(a, rng, earth_radius_km=R_EARTH),
            build_system(b, rng, earth_radius_km=R_EARTH))


@pytest.fixture(scope="module")
def screened():
    a, b = _systems()
    stations = np.asarray(station_ecef_spherical(np.array([0.0, 30.0, 50.0]),
                                                 np.array([0.0, 0.0, 0.0]), R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 7200.0, 10.0)
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)
    pols = (LEGACY_POLICY, HandoffPolicy("full"), HandoffPolicy("k1", k=1),
            HandoffPolicy("keep", keep_serving_if_better=True),
            HandoffPolicy("n_dt", min_interval_s=10.0), HandoffPolicy("n600", min_interval_s=600.0))
    return screen_pair_dtt(
        a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
        VICTIM_BEAM, INTERF_BEAM, FREQ, peak, DIAMETER, MARGIN, min_elevation_deg=25.0,
        earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD, device="cpu", chunk=64,
        handoff_policies=pols,
    )


def test_legacy_policy_reproduces_top_level(screened) -> None:
    pol = screened["policies"]["legacy"]
    for vi in range(3):
        np.testing.assert_allclose(pol["r_div_mean"][vi], screened["r_div_mean"][vi], rtol=1e-12)
        np.testing.assert_allclose(pol["unavail_incr_div_frac"][vi],
                                   screened["unavail_incr_div_frac"][vi], atol=1e-15)
        np.testing.assert_allclose(pol["unres_frac"][vi], screened["unres_frac"][vi], atol=1e-15)


def test_events_occur_in_fixture(screened) -> None:
    assert screened["event_frac"][0].max() > 0.0


def test_k1_means_no_diversity(screened) -> None:
    # K = 1 leaves only the serving satellite: the rate equals the no-diversity
    # rate and every event is unresolvable.
    pol = screened["policies"]["k1"]
    for vi in range(3):
        np.testing.assert_allclose(pol["r_div_mean"][vi], screened["r_int_mean"][vi], rtol=1e-12)
        np.testing.assert_allclose(pol["unres_frac"][vi], screened["event_frac"][vi], atol=1e-15)
        assert (pol["diverted_frac"][vi] == 0.0).all()


def test_primary_rule_switches_on_every_resolvable_event(screened) -> None:
    # Event-triggered rule: the terminal is on an alternate exactly when the
    # serving link is in event and some alternate resolves it.
    pol = screened["policies"]["full"]
    for vi in range(3):
        np.testing.assert_allclose(
            pol["diverted_frac"][vi], screened["event_frac"][vi] - pol["unres_frac"][vi],
            atol=1e-12)


def test_keep_serving_variant_never_worse_than_staying(screened) -> None:
    # With the serving satellite in the choice set the chosen link's rate is at
    # least the no-diversity rate, and at least the strict rule's rate.
    pol = screened["policies"]["keep"]
    for vi in range(3):
        assert (pol["r_div_mean"][vi] >= screened["r_int_mean"][vi] * (1 - 1e-12)).all()
        assert (pol["r_div_mean"][vi]
                >= screened["policies"]["full"]["r_div_mean"][vi] * (1 - 1e-12)).all()


def test_rate_limit_equal_to_time_step_is_memoryless(screened) -> None:
    full, n_dt = screened["policies"]["full"], screened["policies"]["n_dt"]
    for vi in range(3):
        np.testing.assert_allclose(n_dt["r_div_mean"][vi], full["r_div_mean"][vi], rtol=1e-12)
    # Its interference-free baseline follows the highest-elevation satellite: R_0.
    np.testing.assert_allclose(n_dt["r_base_mean"], screened["r0_mean"], rtol=1e-12)


def test_longer_rate_limit_lowers_baseline_and_switch_rate(screened) -> None:
    n600, n_dt = screened["policies"]["n600"], screened["policies"]["n_dt"]
    assert (n600["r_base_mean"] <= screened["r0_mean"] * (1 + 1e-12)).all()
    assert (n600["switches_per_hour"][0] <= n_dt["switches_per_hour"][0] + 1e-12).all()


def test_event_in_histogram_counts_events(screened) -> None:
    # Every realistic-variant event sample lands in exactly one I/N bin, and the
    # aggregate I/N of an event is above the 6% threshold (-12.2 dB).
    from ngsi.dtt_screen import IN_HIST_EDGES_DB, hist_quantile

    n_events = screened["event_frac"][0].sum() * screened["n_samples"]
    for key in ("serving", "aggregate"):
        assert screened["event_in_hist"][key].sum() == pytest.approx(n_events)
    agg = screened["event_in_hist"]["aggregate"]
    assert agg[: int(np.searchsorted(IN_HIST_EDGES_DB, -12.22))].sum() == 0
    # Serving beam alone never exceeds the aggregate it is part of.
    assert hist_quantile(screened["event_in_hist"]["serving"], 0.5) <= hist_quantile(agg, 0.5)


def test_hist_quantile_hand_check() -> None:
    from ngsi.dtt_screen import hist_quantile

    edges = np.array([0.0, 1.0, 2.0, 3.0])
    counts = np.array([0, 1, 2, 1, 0])  # 1 in (0,1], 2 in (1,2], 1 in (2,3]
    assert hist_quantile(counts, 0.5, edges) == 2.0
    assert hist_quantile(counts, 0.1, edges) == 1.0
    assert np.isnan(hist_quantile(np.zeros(5), 0.5, edges))
