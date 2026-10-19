"""Delta T/T screener vs a brute-force NumPy reference (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngsi.antennas import s1428_gain
from ngsi.config import Beam, Constellation, Shell
from ngsi.constants import DTT_THRESHOLD
from ngsi.dtt_screen import screen_pair_dtt
from ngsi.geometry import angular_separation_deg, station_ecef_spherical
from ngsi.link_budget import (
    noise_power_dbw_per_mhz,
    pattern_gain_batch,
    shannon_throughput,
    terminal_peak_gain_dbi,
)
from ngsi.propagation import propagate_ecef
from ngsi.systems import build_system

EPOCH_JD = 2460676.5
R_EARTH = 6371.0
FREQ = 11.7
DIAMETER = 0.6
EFF = 0.6
MARGIN = 3.0
VICTIM_BEAM = Beam(35.0, 1.1, -8.3, 12.0, 240.0, 150.0, "ASSUMED")
INTERF_BEAM = Beam(30.0, 3.0, -20.0, 12.0, 125.0, 200.0, "ASSUMED")


def _shell(name, alt, inc, planes, spp, f, beam):
    return Shell(name, alt, inc, planes, spp, f, beam, "ASSUMED")


def _systems():
    a = Constellation("A", "opA", [_shell("a1", 550.0, 53.0, 3, 4, 1, VICTIM_BEAM)])
    b = Constellation("B", "opB", [_shell("b1", 550.0, 53.0, 3, 4, 1, INTERF_BEAM)])
    return build_system(a, earth_radius_km=R_EARTH), build_system(b, earth_radius_km=R_EARTH)


def _fspl(dist_km):
    lam = 299792458.0 / (FREQ * 1e9)
    return 20.0 * np.log10(4.0 * np.pi * (dist_km * 1e3) / lam)


def _in_db(off_axis, dist, eirp_density):
    g = s1428_gain(off_axis, PEAK, DIAMETER, FREQ)
    return eirp_density + g - _fspl(dist) - noise_power_dbw_per_mhz(VICTIM_BEAM.system_noise_temp_k)


def _reference(a, b, stations, up, times, min_elev):
    g = len(stations)
    event = np.zeros((2, g))
    unres = np.zeros((2, g))
    served = np.zeros(g)
    r0_sum = np.zeros(g)
    pa = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))
    pb = np.asarray(propagate_ecef(b.elements, times, EPOCH_JD, R_EARTH, "cpu"))
    for gi in range(g):
        ra = pa - stations[gi]
        rb = pb - stations[gi]
        da = np.linalg.norm(ra, axis=-1)
        db = np.linalg.norm(rb, axis=-1)
        ua = ra / da[..., None]
        ub = rb / db[..., None]
        elev_a = np.rad2deg(np.arcsin(np.clip(ua @ up[gi], -1, 1)))
        elev_b = np.rad2deg(np.arcsin(np.clip(ub @ up[gi], -1, 1)))
        vis_a = elev_a >= min_elev
        vis_b = elev_b >= min_elev
        for t in range(len(times)):
            av = np.flatnonzero(vis_a[t])
            if av.size == 0:
                continue
            served[gi] += 1.0
            sa = int(av[np.argmax(elev_a[t, av])])
            cn0 = VICTIM_BEAM.eirp_density_dbw_per_mhz + PEAK - _fspl(da[t, sa]) \
                - noise_power_dbw_per_mhz(VICTIM_BEAM.system_noise_temp_k)
            r0_sum[gi] += shannon_throughput(cn0, VICTIM_BEAM.bandwidth_mhz * 1e6, MARGIN)
            bv = np.flatnonzero(vis_b[t])
            sb = int(bv[np.argmax(elev_b[t, bv])]) if bv.size else None
            for vi in (0, 1):
                interf = bv if vi == 0 else (np.array([sb]) if sb is not None else np.array([], int))
                if interf.size == 0:
                    continue
                dtt = sum(
                    10.0 ** (_in_db(angular_separation_deg(pa[t, sa], pb[t, k], stations[gi]),
                                    db[t, k], INTERF_BEAM.eirp_density_dbw_per_mhz) / 10.0)
                    for k in interf
                )
                if dtt <= DTT_THRESHOLD:
                    continue
                resolvable = False
                for ca in av:
                    if ca == sa:
                        continue
                    dtt_c = sum(
                        10.0 ** (_in_db(angular_separation_deg(pa[t, ca], pb[t, k], stations[gi]),
                                        db[t, k], INTERF_BEAM.eirp_density_dbw_per_mhz) / 10.0)
                        for k in interf
                    )
                    if dtt_c <= DTT_THRESHOLD:
                        resolvable = True
                        break
                event[vi, gi] += 1.0
                if not resolvable:
                    unres[vi, gi] += 1.0
    n = len(times)
    return event / n, unres / n, served / n, r0_sum / np.maximum(served, 1.0)


def test_screen_pair_dtt_matches_reference() -> None:
    a, b = _systems()
    lats = [0.0, 30.0, 50.0]
    lons = [0.0, 120.0, 240.0]
    latg, long = np.meshgrid(lats, lons, indexing="ij")
    stations = np.asarray(station_ecef_spherical(latg.ravel(), long.ravel(), R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 86400.0, 3600.0)

    global PEAK
    PEAK = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)

    res = screen_pair_dtt(
        a, b, a.n_sats, b.n_sats,
        torch.as_tensor(stations), torch.as_tensor(up), times,
        VICTIM_BEAM, INTERF_BEAM, FREQ, PEAK, DIAMETER, MARGIN,
        min_elevation_deg=25.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
        device="cpu", chunk=32,
    )
    ref_event, ref_unres, ref_served, ref_r0 = _reference(a, b, stations, up, times, 25.0)

    np.testing.assert_allclose(res["event_frac"], ref_event, atol=1e-12)
    np.testing.assert_allclose(res["unres_frac"], ref_unres, atol=1e-12)
    np.testing.assert_allclose(res["served_frac"], ref_served, atol=1e-12)
    np.testing.assert_allclose(res["r0_mean"], ref_r0, rtol=1e-9)
    assert ref_event.sum() > 0.0


def test_pfd_constant_uses_altitude_instead_of_range() -> None:
    """PFD-constant co-located DTT must use FSPL(altitude), not FSPL(slant range).

    The victim and interferer share identical orbits, so the serving victim and
    interferer satellites coincide: the co-located off-axis angle is 0 and the
    receive gain is the peak.  The co-located Delta T/T is then
    ``EIRP + G_peak - FSPL - noise`` with FSPL at the actual slant range
    (constant-EIRP) or at the satellite altitude (PFD-constant).
    """
    a, b = _systems()
    stations = np.asarray(station_ecef_spherical(np.array([45.0]), np.array([0.0]),
                                                 R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.array([0.0])
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)

    def _run(pfd: bool) -> dict:
        return screen_pair_dtt(
            a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
            VICTIM_BEAM, INTERF_BEAM, FREQ, peak, DIAMETER, MARGIN,
            min_elevation_deg=-90.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
            device="cpu", chunk=32, pfd_constant=pfd,
        )

    res_e = _run(False)
    res_p = _run(True)

    pa = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))[0]
    rel = pa - stations[0]
    rng = np.linalg.norm(rel, axis=-1)
    k = int(np.argmax((rel / rng[:, None]) @ up[0]))  # highest-elevation satellite
    noise = noise_power_dbw_per_mhz(VICTIM_BEAM.system_noise_temp_k)
    g0 = float(pattern_gain_batch(torch.zeros(1, dtype=torch.float64), peak, DIAMETER, FREQ,
                                  kind="S1428")[0])
    eirp = INTERF_BEAM.eirp_density_dbw_per_mhz
    exp_e = 10.0 ** ((eirp + g0 - _fspl(rng[k]) - noise) / 10.0)
    exp_p = 10.0 ** ((eirp + g0 - _fspl(np.linalg.norm(pa[k]) - R_EARTH) - noise) / 10.0)
    assert res_e["dtt_colo_mean"][0] == pytest.approx(exp_e, rel=1e-9)
    assert res_p["dtt_colo_mean"][0] == pytest.approx(exp_p, rel=1e-9)
    assert exp_p > exp_e  # altitude < slant range everywhere except nadir
