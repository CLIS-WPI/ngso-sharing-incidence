"""Delta T/T screener vs a brute-force NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ngsi.antennas import s1428_gain, s1528_gain
from ngsi.config import Beam, Constellation, Shell
from ngsi.constants import DTT_THRESHOLD
from ngsi.dtt_screen import VARIANT_NAMES, screen_pair_dtt
from ngsi.geometry import angular_separation_deg, station_ecef_spherical
from ngsi.link_budget import (
    noise_power_dbw_per_mhz,
    shannon_throughput,
    slant_range_km,
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
NOISE = noise_power_dbw_per_mhz(150.0)
VICTIM_BEAM = Beam(35.0, 1.1, -8.3, 12.0, 240.0, 150.0, "ASSUMED")
INTERF_BEAM = Beam(30.0, 3.0, -20.0, 12.0, 125.0, 200.0, "ASSUMED")
INTERF_BEAM_PFD = Beam(30.0, 3.0, -20.0, 12.0, 125.0, 200.0, "ASSUMED",
                       beam_steering="pfd_constant", pfd_anchor_elevation_deg=25.0,
                       pfd_reference_altitude_km=550.0)
ZERO = np.zeros(3)


def _shell(name, alt, inc, planes, spp, f, beam):
    return Shell(name, alt, inc, planes, spp, f, beam, "ASSUMED")


def _systems():
    a = Constellation("A", "opA", [_shell("a1", 550.0, 53.0, 3, 4, 1, VICTIM_BEAM)])
    b = Constellation("B", "opB", [_shell("b1", 550.0, 53.0, 3, 4, 1, INTERF_BEAM)])
    return build_system(a, earth_radius_km=R_EARTH), build_system(b, earth_radius_km=R_EARTH)


def _fspl(dist_km):
    lam = 299792458.0 / (FREQ * 1e9)
    return 20.0 * np.log10(4.0 * np.pi * (dist_km * 1e3) / lam)


def _lin(sep_from_boresight, fspl_db, eirp_density, peak):
    """Linear I/N for one interferer (S.1428 victim RX, given TX EIRP density)."""
    g = s1428_gain(sep_from_boresight, peak, DIAMETER, FREQ)
    return 10.0 ** ((eirp_density + g - fspl_db - NOISE) / 10.0)


def _reference(a, b, stations, up, times, min_elev, peak, interf_beam):
    g = len(stations)
    event = np.zeros((3, g))
    unres = np.zeros((3, g))
    served = np.zeros(g)
    r0_sum = np.zeros(g)
    e_max = interf_beam.eirp_density_dbw_per_mhz
    e_peak = interf_beam.peak_gain_dbi
    e_bw = interf_beam.beamwidth_deg
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
            cn0 = VICTIM_BEAM.eirp_density_dbw_per_mhz + peak - _fspl(da[t, sa]) - NOISE
            r0_sum[gi] += shannon_throughput(cn0, VICTIM_BEAM.bandwidth_mhz * 1e6, MARGIN)
            bv = np.flatnonzero(vis_b[t])
            sb = int(bv[np.argmax(elev_b[t, bv])]) if bv.size else None

            dtt_colo = (_lin(angular_separation_deg(pa[t, sa], pb[t, sb], stations[gi]),
                             _fspl(db[t, sb]), e_max, peak) if sb is not None else 0.0)
            dtt_upper = sum(_lin(angular_separation_deg(pa[t, sa], pb[t, k], stations[gi]),
                                 _fspl(db[t, k]), e_max, peak) for k in bv)
            dtt_side = 0.0
            for k in bv:
                if k == sb:
                    continue
                nadir = -pb[t, k] / np.linalg.norm(pb[t, k])
                theta = angular_separation_deg(nadir, -ub[t, k], ZERO)
                e_oth = e_max + (s1528_gain(theta, e_peak, e_bw) - e_peak)
                dtt_side += _lin(angular_separation_deg(pa[t, sa], pb[t, k], stations[gi]),
                                 _fspl(db[t, k]), e_oth, peak)
            dtts = {0: dtt_colo + dtt_side, 1: dtt_colo, 2: dtt_upper}
            has = {0: bv.size > 0, 1: sb is not None, 2: bv.size > 0}
            for vi in range(3):
                if not has[vi] or dtts[vi] <= DTT_THRESHOLD:
                    continue
                resolvable = False
                for ca in av:
                    if ca == sa:
                        continue
                    cand_c = _lin(
                        angular_separation_deg(pa[t, ca], pb[t, sb], stations[gi]),
                        _fspl(db[t, sb]), e_max, peak) if sb is not None else 0.0
                    cand_s = 0.0
                    cand_u = 0.0
                    for k in bv:
                        sep = angular_separation_deg(pa[t, ca], pb[t, k], stations[gi])
                        cand_u += _lin(sep, _fspl(db[t, k]), e_max, peak)
                        if k != sb:
                            nadir = -pb[t, k] / np.linalg.norm(pb[t, k])
                            theta = angular_separation_deg(nadir, -ub[t, k], ZERO)
                            e_oth = e_max + (s1528_gain(theta, e_peak, e_bw) - e_peak)
                            cand_s += _lin(sep, _fspl(db[t, k]), e_oth, peak)
                    cand = {0: cand_c + cand_s, 1: cand_c, 2: cand_u}[vi]
                    if cand <= DTT_THRESHOLD:
                        resolvable = True
                        break
                event[vi, gi] += 1.0
                if not resolvable:
                    unres[vi, gi] += 1.0
    n = len(times)
    return event / n, unres / n, served / n, r0_sum / np.maximum(served, 1.0)


def test_variant_names_order() -> None:
    assert VARIANT_NAMES == ("realistic", "co_located", "upper_bound")


def test_screen_pair_dtt_matches_reference() -> None:
    a, b = _systems()
    lats = [0.0, 30.0, 50.0]
    lons = [0.0, 120.0, 240.0]
    latg, long = np.meshgrid(lats, lons, indexing="ij")
    stations = np.asarray(station_ecef_spherical(latg.ravel(), long.ravel(), R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 86400.0, 3600.0)
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)

    res = screen_pair_dtt(
        a, b, a.n_sats, b.n_sats,
        torch.as_tensor(stations), torch.as_tensor(up), times,
        VICTIM_BEAM, INTERF_BEAM, FREQ, peak, DIAMETER, MARGIN,
        min_elevation_deg=25.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
        device="cpu", chunk=32,
    )
    ref_event, ref_unres, ref_served, ref_r0 = _reference(
        a, b, stations, up, times, 25.0, peak, INTERF_BEAM)

    np.testing.assert_allclose(res["event_frac"], ref_event, atol=1e-12)
    np.testing.assert_allclose(res["unres_frac"], ref_unres, atol=1e-12)
    np.testing.assert_allclose(res["served_frac"], ref_served, atol=1e-12)
    np.testing.assert_allclose(res["r0_mean"], ref_r0, rtol=1e-9)
    assert ref_event.sum() > 0.0
    # Realistic (main beam + S.1528 sidelobes) <= upper bound, per station/variant.
    assert (ref_event[0] <= ref_event[2] + 1e-12).all()


def test_pfd_constant_anchors_at_r_max() -> None:
    """PFD-constant co-located DTT must use FSPL(r_max), not FSPL(altitude).

    The victim and interferer share identical orbits, so the serving victim and
    interferer satellites coincide: the co-located off-axis angle is 0 and the
    receive gain is the peak.  Under PFD-constant steering the EIRP topology
    cannot exceed its value at the minimum-elevation service edge, so the
    co-located Delta T/T is ``EIRP + G_peak - max(FSPL(r), FSPL(r_max)) - noise``.
    """
    a, b = _systems()
    stations = np.asarray(station_ecef_spherical(np.array([45.0]), np.array([0.0]),
                                                 R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.array([0.0])
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)

    def _run(interf_beam, steering):
        return screen_pair_dtt(
            a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
            VICTIM_BEAM, interf_beam, FREQ, peak, DIAMETER, MARGIN,
            min_elevation_deg=-90.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
            device="cpu", chunk=32, steering=steering,
        )

    res_e = _run(INTERF_BEAM, None)
    res_p = _run(INTERF_BEAM_PFD, None)

    pa = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))[0]
    rel = pa - stations[0]
    rng = np.linalg.norm(rel, axis=-1)
    k = int(np.argmax((rel / rng[:, None]) @ up[0]))  # highest-elevation satellite
    g0 = float(s1428_gain(0.0, peak, DIAMETER, FREQ))
    eirp = INTERF_BEAM.eirp_density_dbw_per_mhz
    r_max = slant_range_km(550.0, 25.0, R_EARTH)
    exp_e = 10.0 ** ((eirp + g0 - _fspl(rng[k]) - NOISE) / 10.0)
    exp_p = 10.0 ** ((eirp + g0 - max(_fspl(rng[k]), _fspl(r_max)) - NOISE) / 10.0)
    assert res_e["dtt_colo_mean"][0] == pytest.approx(exp_e, rel=1e-9)
    assert res_p["dtt_colo_mean"][0] == pytest.approx(exp_p, rel=1e-9)
    # The EIRP ceiling reduces interference for any satellite above the edge.
    assert exp_p <= exp_e
    # Legacy (wrong) anchor used altitude, giving a *larger* clamp than r_max.
    exp_alt = 10.0 ** ((eirp + g0 - _fspl(550.0) - NOISE) / 10.0)
    assert exp_alt > exp_e


def test_per_system_min_elevation() -> None:
    """A victim-only elevation override changes only the victim's visibility.

    The served fraction must equal the brute-force fraction of time some victim
    satellite is at or above the victim's own minimum elevation, and overriding
    both sides with the common value must reproduce the default exactly.
    """
    a, b = _systems()
    stations = np.asarray(station_ecef_spherical(np.array([0.0, 45.0]), np.array([0.0, 90.0]),
                                                 R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 86400.0, 600.0)
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)

    def _run(**kw):
        return screen_pair_dtt(
            a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
            VICTIM_BEAM, INTERF_BEAM, FREQ, peak, DIAMETER, MARGIN,
            min_elevation_deg=25.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
            device="cpu", chunk=32, **kw,
        )

    res55 = _run(victim_min_elevation_deg=55.0)
    pa = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))  # [T,N,3]
    rel = pa[:, None, :, :] - stations[None, :, None, :]
    sin_el = (rel / np.linalg.norm(rel, axis=-1, keepdims=True) * up[None, :, None, :]).sum(-1)
    ref_served = (np.rad2deg(np.arcsin(sin_el)).max(-1) >= 55.0).mean(0)
    np.testing.assert_allclose(res55["served_frac"], ref_served, atol=1e-12)
    assert (res55["served_frac"] <= _run()["served_frac"] + 1e-12).all()

    base, same = _run(), _run(victim_min_elevation_deg=25.0, interferer_min_elevation_deg=25.0)
    np.testing.assert_array_equal(base["event_frac"], same["event_frac"])


def test_pfd_constant_uses_reference_altitude_not_shell() -> None:
    """The PFD0 of a PFD-constant beam does not depend on the interferer's altitude.

    Victim and interferer share a 500 km shell (co-located serving satellites), the
    beam's reference altitude is 550 km: the EIRP clamp is FSPL(r_ref) with r_ref the
    25 deg slant range from 550 km, not from the 500 km shell.
    """
    a = Constellation("A", "opA", [_shell("a1", 500.0, 53.0, 3, 4, 1, VICTIM_BEAM)])
    b = Constellation("B", "opB", [_shell("b1", 500.0, 53.0, 3, 4, 1, INTERF_BEAM_PFD)])
    a, b = build_system(a, earth_radius_km=R_EARTH), build_system(b, earth_radius_km=R_EARTH)
    times = np.array([0.0])
    # Station at a satellite's sub-satellite point, so the serving range (~500 km) is
    # below both anchor ranges and the EIRP clamp is active.
    p0 = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))[0][0]
    stations = (p0 / np.linalg.norm(p0) * R_EARTH)[None, :]
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    peak = terminal_peak_gain_dbi(DIAMETER, FREQ, EFF)
    res = screen_pair_dtt(
        a, b, a.n_sats, b.n_sats, torch.as_tensor(stations), torch.as_tensor(up), times,
        VICTIM_BEAM, INTERF_BEAM_PFD, FREQ, peak, DIAMETER, MARGIN,
        min_elevation_deg=-90.0, earth_radius_km=R_EARTH, epoch_jd=EPOCH_JD,
        device="cpu", chunk=32)
    pa = np.asarray(propagate_ecef(a.elements, times, EPOCH_JD, R_EARTH, "cpu"))[0]
    rel = pa - stations[0]
    rng = np.linalg.norm(rel, axis=-1)
    k = int(np.argmax((rel / rng[:, None]) @ up[0]))
    g0 = float(s1428_gain(0.0, peak, DIAMETER, FREQ))
    eirp = INTERF_BEAM_PFD.eirp_density_dbw_per_mhz
    r_ref = slant_range_km(550.0, 25.0, R_EARTH)
    r_own = slant_range_km(500.0, 25.0, R_EARTH)
    exp = 10.0 ** ((eirp + g0 - max(_fspl(rng[k]), _fspl(r_ref)) - NOISE) / 10.0)
    old = 10.0 ** ((eirp + g0 - max(_fspl(rng[k]), _fspl(r_own)) - NOISE) / 10.0)
    assert res["dtt_colo_mean"][0] == pytest.approx(exp, rel=1e-9)
    assert exp < old  # the per-shell anchor overstated the PFD of a lower shell
