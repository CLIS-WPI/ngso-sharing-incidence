"""The victim's own EIRP must not move the Delta T/T screen.

The 25.261(c) trigger depends on the *interferer's* power, the victim's receive
gain and noise, and the geometry -- never on the victim's own transmit EIRP.  This
test pins that separation: changing the victim downlink EIRP density leaves the
event/unresolvable fractions bit-identical while moving the victim's baseline C/N
and availability, which is exactly what makes the baseline calibration a clean,
independent change.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ngsi.config import Beam, Constellation, Shell
from ngsi.dtt_screen import screen_pair_dtt
from ngsi.geometry import station_ecef_spherical
from ngsi.link_budget import terminal_peak_gain_dbi
from ngsi.systems import build_system

EPOCH_JD = 2460676.5
R_EARTH = 6371.0
FREQ = 11.7
DIAMETER = 0.6
EFF = 0.6
MARGIN = 3.0
VICTIM_BEAM = Beam(35.0, 1.1, -8.3, 12.0, 240.0, 150.0, "ASSUMED")
INTERF_BEAM = Beam(30.0, 3.0, 0.2, 12.0, 125.0, 200.0, "ASSUMED")


def _systems(victim_beam: Beam):
    def shell(name, beam):
        return Shell(name, 550.0, 53.0, 2, 3, 1, beam, "ASSUMED")

    a = Constellation("A", "opA", [shell("a1", victim_beam)])
    b = Constellation("B", "opB", [shell("b1", INTERF_BEAM)])
    return build_system(a, earth_radius_km=R_EARTH), build_system(b, earth_radius_km=R_EARTH)


def _screen(victim_beam: Beam) -> dict:
    a, b = _systems(victim_beam)
    latg, long = np.meshgrid([0.0, 40.0, 60.0], [0.0, 150.0], indexing="ij")
    stations = np.asarray(station_ecef_spherical(latg.ravel(), long.ravel(), R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 6.0 * 3600.0, 60.0)
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)
    return screen_pair_dtt(
        a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
        victim_beam, INTERF_BEAM, FREQ, peak, DIAMETER, MARGIN,
        min_elevation_deg=20.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
        device="cpu", chunk=32,
    )


def test_events_invariant_to_victim_eirp() -> None:
    low = _screen(VICTIM_BEAM)
    high = _screen(replace(VICTIM_BEAM, eirp_density_dbw_per_mhz=-8.3 + 20.2))

    # The screen itself must be identical: only the victim's own link budget moves.
    np.testing.assert_array_equal(low["event_frac"], high["event_frac"])
    np.testing.assert_array_equal(low["unres_frac"], high["unres_frac"])
    assert low["event_frac"].sum() > 0.0

    # The victim's baseline C/N shifts by exactly the EIRP increment, and its
    # baseline availability is at least as high (a stronger victim link closes more).
    delta = high["cn0_median"] - low["cn0_median"]
    np.testing.assert_allclose(delta[np.isfinite(delta)], 20.2, atol=1e-9)
    assert np.all(high["avail0_frac"] >= low["avail0_frac"] - 1e-12)


def test_screen_reports_cn0_distribution_and_availability() -> None:
    res = _screen(VICTIM_BEAM)
    g = res["event_frac"].shape[1]
    for key in ("cn0_mean", "cn0_median", "cn0_p10", "avail0_frac"):
        assert res[key].shape == (g,)
    assert np.all((res["avail0_frac"] >= 0.0) & (res["avail0_frac"] <= 1.0))
