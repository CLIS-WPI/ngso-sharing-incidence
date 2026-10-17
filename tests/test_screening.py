"""Screening engine vs a brute-force NumPy reference (AGENTS.md rule 6)."""

from __future__ import annotations

import numpy as np
import torch

from ngsi.config import Beam, Constellation, Shell
from ngsi.geometry import angular_separation_deg, station_ecef_spherical
from ngsi.propagation import propagate_ecef
from ngsi.screening import ScreenConfig, screen_pair, system_visibility
from ngsi.systems import build_system

EPOCH_JD = 2460676.5
R_EARTH = 6371.0


def _shell(name: str, alt: float, inc: float, planes: int, spp: int, f: int) -> Shell:
    return Shell(
        name=name,
        altitude_km=alt,
        inclination_deg=inc,
        planes=planes,
        sats_per_plane=spp,
        phasing_f=f,
        beam=Beam(30.0, 3.0, -18.0, 12.0, 240.0, 150.0, "ASSUMED"),
        source="ASSUMED",
    )


def _systems():
    a = Constellation("A", "opA", [_shell("a1", 550.0, 53.0, 3, 4, 1)])
    b = Constellation("B", "opB", [_shell("b1", 600.0, 53.0, 3, 4, 1)])
    return build_system(a, earth_radius_km=R_EARTH), build_system(b, earth_radius_km=R_EARTH)


def _reference(a, b, stations, up, times, cfg):
    g = len(stations)
    thetas = np.asarray(cfg.thetas_deg, dtype=np.float64)
    event = np.zeros((len(thetas), 2, g))
    unres = np.zeros((len(thetas), 2, g))
    pa = np.asarray(propagate_ecef(a.elements, times, cfg.epoch_jd, cfg.earth_radius_km, "cpu"))
    pb = np.asarray(propagate_ecef(b.elements, times, cfg.epoch_jd, cfg.earth_radius_km, "cpu"))
    for gi in range(g):
        ra = pa - stations[gi]
        rb = pb - stations[gi]
        ua = ra / np.linalg.norm(ra, axis=-1, keepdims=True)
        ub = rb / np.linalg.norm(rb, axis=-1, keepdims=True)
        elev_a = np.rad2deg(np.arcsin(np.clip(ua @ up[gi], -1, 1)))
        elev_b = np.rad2deg(np.arcsin(np.clip(ub @ up[gi], -1, 1)))
        vis_a = elev_a >= cfg.min_elevation_deg
        vis_b = elev_b >= cfg.min_elevation_deg
        for t in range(len(times)):
            if not vis_a[t].any():
                continue
            sa = int(np.argmax(np.where(vis_a[t], elev_a[t], -np.inf)))
            vis_bidx = np.flatnonzero(vis_b[t])
            visible_a = np.flatnonzero(vis_a[t])
            sb = int(np.argmax(np.where(vis_b[t], elev_b[t], -np.inf))) if vis_bidx.size else None
            for vi in (0, 1):
                interferers = vis_bidx if vi == 0 else (np.array([sb]) if sb is not None else np.array([], int))
                if interferers.size == 0:
                    continue
                for ti, th in enumerate(thetas):
                    sep_serving = [angular_separation_deg(pa[t, sa], pb[t, k], stations[gi]) for k in interferers]
                    fired = min(sep_serving) < th
                    if not fired:
                        continue
                    resolvable = False
                    for ca in visible_a:
                        if ca == sa:
                            continue
                        mins = min(
                            angular_separation_deg(pa[t, ca], pb[t, k], stations[gi]) for k in interferers
                        )
                        if mins >= th:
                            resolvable = True
                            break
                    event[ti, vi, gi] += 1.0
                    if not resolvable:
                        unres[ti, vi, gi] += 1.0
    event /= len(times)
    unres /= len(times)
    return event, unres


def test_screen_pair_matches_reference() -> None:
    a, b = _systems()
    lats = [0.0, 30.0, 55.0]
    lons = [0.0, 120.0, 240.0]
    latg, long = np.meshgrid(lats, lons, indexing="ij")
    lat_flat, lon_flat = latg.ravel(), long.ravel()
    stations = np.asarray(station_ecef_spherical(lat_flat, lon_flat, R_EARTH, "cpu"))
    up = stations / np.linalg.norm(stations, axis=-1, keepdims=True)
    times = np.arange(0.0, 86400.0, 300.0)
    cfg = ScreenConfig(
        min_elevation_deg=25.0,
        earth_radius_km=R_EARTH,
        epoch_jd=EPOCH_JD,
        thetas_deg=(5.0, 30.0),
        chunk=128,
        device="cpu",
    )

    st = system_visibility(a, torch.as_tensor(stations), torch.as_tensor(up), times, cfg, np.arange(-90, 91, 5))
    sb = system_visibility(b, torch.as_tensor(stations), torch.as_tensor(up), times, cfg, np.arange(-90, 91, 5))
    res = screen_pair(
        a, b, st["max_visible"], sb["max_visible"],
        torch.as_tensor(stations), torch.as_tensor(up), times, cfg,
    )
    ref_event, ref_unres = _reference(a, b, stations, up, times, cfg)

    np.testing.assert_allclose(res["event_frac"], ref_event, atol=1e-12)
    np.testing.assert_allclose(res["unres_frac"], ref_unres, atol=1e-12)
    # The scenario should actually exercise events/diversity, not just zeros.
    assert ref_event.sum() > 0
    assert res["covisible_frac"].shape == (len(stations),)
