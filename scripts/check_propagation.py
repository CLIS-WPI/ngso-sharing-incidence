"""Validity of the J2 secular propagator over the 8-day as-deployed window (Phase 4b).

E5/E6 advance every TLE to a common epoch with the circular J2 secular model
(``ngsi.tle.advance_to_epoch`` / ``ngsi.propagation.propagate_ecef``) and replicate
over consecutive 24 h windows.  This script compares that model with SGP4 run on
the same TLEs:

A. A random sample of operational satellites (default 50 Gen2 Starlink, 20 OneWeb):
   ECEF position error and station look-angle error at day 0, 1, 4 and 8 after the
   simulation epoch (a ``--window-s`` window at each, ``--step-s`` sampling).
B. Every operational satellite of the E5 headline set (Gen2 Starlink + OneWeb),
   each of the ``n_start_times`` days: per-station mean visible counts and a
   geometric in-line proxy (a visible Starlink satellite within ``screen_theta_deg``
   of the highest-elevation OneWeb satellite), under both propagators, so a
   systematic drift with day can be seen in either.

SGP4 positions are TEME, rotated to ECEF with GMST only (no polar motion, no
equation of the equinoxes), the same rotation the J2 model uses.

Usage (in the container):
    python scripts/check_propagation.py [--config configs/experiments/e5_as_deployed.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sgp4.api import SatrecArray, Satrec

from ngsi.config import load_experiment, load_simulation, load_stations
from ngsi.frames import enu_basis, geodetic_to_ecef, gmst_rad_series
from ngsi.io import new_run_dir
from ngsi.propagation import propagate_ecef
from ngsi.tle import (
    classify_generation,
    common_epoch_jd,
    elements_from_records,
    fetch_group,
    filter_operational,
    parse_tle,
)

REPO = Path(__file__).resolve().parents[1]
SYSTEMS = ("STARLINK", "ONEWEB")


def operational_records(exp, radius: float) -> dict[str, list]:
    """Operational records per system, Starlink reduced to the configured generation."""
    p = exp.params
    out = {}
    for n in exp.constellations:
        recs = parse_tle(fetch_group(str(p["tle_groups"][n]), REPO / str(p["tle_cache_dir"])))
        band = p["operational_altitude_km"].get(n, [0.0, 1e9])
        kept, _, _ = filter_operational(recs, radius, float(band[0]), float(band[1]))
        gen = p.get("starlink_generation")
        if n == "STARLINK" and gen not in (None, "all"):
            lab = classify_generation(kept, radius, list(p["starlink_generation_rule"]))
            kept = [r for r, g in zip(kept, lab) if g == gen]
        out[n] = kept
    return out


def ecef_j2(recs: list, t: np.ndarray, epoch: float, radius: float) -> np.ndarray:
    """(n_t, n_sat, 3) ECEF km from the production J2 model."""
    el = elements_from_records(recs, epoch, radius)
    r = propagate_ecef(el, t, epoch, radius, device="cpu")
    return np.asarray(r.double().numpy() if hasattr(r, "numpy") else r, dtype=np.float64)


def ecef_sgp4(recs: list, t: np.ndarray, epoch: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_t, n_sat, 3) ECEF km from SGP4 and the (n_sat, n_t) SGP4 error codes."""
    sats = SatrecArray([Satrec.twoline2rv(r.line1, r.line2) for r in recs])
    jd = epoch + t / 86400.0
    jd_i = np.floor(jd)
    err, r, _v = sats.sgp4(jd_i, jd - jd_i)
    r = np.transpose(r, (1, 0, 2))  # (n_t, n_sat, 3) TEME
    g = gmst_rad_series(jd)[:, None]
    c, s = np.cos(g), np.sin(g)
    ecef = np.stack([c * r[..., 0] + s * r[..., 1], -s * r[..., 0] + c * r[..., 1], r[..., 2]],
                    axis=-1)
    return ecef, err


def stations(grid_cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = np.meshgrid(grid_cfg.latitudes_deg, grid_cfg.longitudes_deg, indexing="ij")
    lat, lon = lat.ravel().astype(float), lon.ravel().astype(float)
    pos = np.stack([geodetic_to_ecef(a, b) for a, b in zip(lat, lon)])
    up = np.stack([enu_basis(a, b)[2] for a, b in zip(lat, lon)])
    return lat, pos, up


def look(sat: np.ndarray, pos: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit look vectors (n_t, n_st, n_sat, 3) and elevations (deg)."""
    d = sat[:, None, :, :] - pos[None, :, None, :]
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    el = np.rad2deg(np.arcsin(np.clip(np.einsum("tsnk,sk->tsn", u, up), -1, 1)))
    return u, el


def sample_errors(recs: dict, epoch: float, radius: float, days, window: float, step: float,
                  grid, min_el: float, n_pick: dict, rng) -> list[dict]:
    lat, pos, up = grid
    rows = []
    for n, k in n_pick.items():
        idx = rng.choice(len(recs[n]), size=min(k, len(recs[n])), replace=False)
        sub = [recs[n][i] for i in sorted(idx)]
        for d in days:
            t = d * 86400.0 + np.arange(0.0, window, step)
            rj = ecef_j2(sub, t, epoch, radius)
            rs, err = ecef_sgp4(sub, t, epoch)
            ok = (err == 0).T  # (n_t, n_sat)
            dr = np.linalg.norm(rj - rs, axis=-1)[ok]
            rad = (np.linalg.norm(rj, axis=-1) - np.linalg.norm(rs, axis=-1))[ok]
            uj, ej = look(rj, pos, up)
            us, es = look(rs, pos, up)
            vis = (es >= min_el) & ok[:, None, :]
            ang = np.rad2deg(np.arccos(np.clip(np.sum(uj * us, axis=-1), -1, 1)))[vis]
            rows.append({
                "system": n, "day": d, "n_sats": len(sub), "n_samples": int(ok.sum()),
                "n_sgp4_errors": int((err != 0).sum()),
                "pos_err_median_km": float(np.median(dr)), "pos_err_p90_km": float(np.percentile(dr, 90)),
                "pos_err_max_km": float(dr.max()),
                "radial_err_median_km": float(np.median(np.abs(rad))),
                "look_err_n": int(vis.sum()),
                "look_err_median_deg": float(np.median(ang)) if ang.size else np.nan,
                "look_err_p90_deg": float(np.percentile(ang, 90)) if ang.size else np.nan,
                "look_err_max_deg": float(ang.max()) if ang.size else np.nan,
                "elev_err_median_deg": float(np.median(np.abs(ej - es)[vis])) if ang.size else np.nan,
            })
            print(f"[prop] {n} day {d}: |dr| median {rows[-1]['pos_err_median_km']:.1f} km, "
                  f"look median {rows[-1]['look_err_median_deg']:.3f} deg", flush=True)
    return rows


def daily_geometry(recs: dict, epoch: float, radius: float, n_days: int, step: float,
                   chunk_s: float, grid, min_el: float, theta: float) -> list[dict]:
    """Per day x station x propagator: mean visible counts and the in-line proxy."""
    lat, pos, up = grid
    cos_th = np.cos(np.deg2rad(theta))
    rows = []
    for d in range(n_days):
        acc = {p: {"vis_s": 0.0, "vis_o": 0.0, "served": 0.0, "ev": 0.0} for p in ("j2", "sgp4")}
        acc = {p: {k: np.zeros(len(lat)) for k in v} for p, v in acc.items()}
        for c0 in np.arange(0.0, 86400.0, chunk_s):
            t = d * 86400.0 + c0 + np.arange(0.0, min(chunk_s, 86400.0 - c0), step)
            for prop in ("j2", "sgp4"):
                look_ = {}
                for n in SYSTEMS:
                    if prop == "j2":
                        r = ecef_j2(recs[n], t, epoch, radius)
                        ok = np.ones(r.shape[:2], bool)
                    else:
                        r, err = ecef_sgp4(recs[n], t, epoch)
                        ok = (err == 0).T
                    u, el = look(r, pos, up)
                    look_[n] = (u, (el >= min_el) & ok[:, None, :], el)
                us, vs, _ = look_["STARLINK"]
                uo, vo, eo = look_["ONEWEB"]
                a = acc[prop]
                a["vis_s"] += vs.sum(-1).sum(0)
                a["vis_o"] += vo.sum(-1).sum(0)
                served = vo.any(-1)
                best = np.where(vo, eo, -np.inf).argmax(-1)  # (n_t, n_st)
                serv = np.take_along_axis(uo, best[..., None, None], axis=2)[:, :, 0, :]
                cosang = np.einsum("tsnk,tsk->tsn", us, serv)
                ev = ((cosang >= cos_th) & vs).any(-1) & served
                a["served"] += served.sum(0)
                a["ev"] += ev.sum(0)
        n_t = len(np.arange(0.0, 86400.0, step))
        for prop, a in acc.items():
            for i in range(len(lat)):
                rows.append({"day": d, "propagator": prop, "station": i, "latitude_deg": lat[i],
                             "vis_starlink_mean": a["vis_s"][i] / n_t,
                             "vis_oneweb_mean": a["vis_o"][i] / n_t,
                             "served_frac": a["served"][i] / n_t,
                             "inline_frac": (a["ev"][i] / a["served"][i]
                                             if a["served"][i] else np.nan)})
        print(f"[prop] daily geometry day {d} done", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/experiments/e5_as_deployed.yaml")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--n-starlink", type=int, default=50)
    ap.add_argument("--n-oneweb", type=int, default=20)
    ap.add_argument("--days", default="0,1,4,8")
    ap.add_argument("--window-s", type=float, default=3600.0)
    ap.add_argument("--step-s", type=float, default=30.0)
    ap.add_argument("--daily-step-s", type=float, default=60.0)
    ap.add_argument("--n-days", type=int, default=None,
                    help="days for part B (default: the config's n_start_times)")
    args = ap.parse_args()

    exp_path = REPO / args.config
    sim = load_simulation(REPO / "configs" / "constellations.yaml")
    exp = load_experiment(exp_path)
    grid_cfg = load_stations(REPO / "configs" / exp.stations)
    radius = sim.earth.radius_km
    p = exp.params
    theta = float(yaml.safe_load((REPO / "configs/experiments/e1_event_rate.yaml").read_text())
                  ["params"]["screen_theta_deg"])
    recs = operational_records(exp, radius)
    epoch = common_epoch_jd([r for v in recs.values() for r in v])
    grid = stations(grid_cfg)
    min_el = float(grid_cfg.min_elevation_deg)
    run = new_run_dir(args.results_root, "e12_propagation_check", exp.seed,
                      config_paths=[exp_path, REPO / "configs" / "constellations.yaml",
                                    REPO / "configs" / exp.stations], repo_root=REPO)
    rng = np.random.default_rng(exp.seed)
    days = [int(x) for x in args.days.split(",")]
    a = sample_errors(recs, epoch, radius, days, args.window_s, args.step_s, grid, min_el,
                      {"STARLINK": args.n_starlink, "ONEWEB": args.n_oneweb}, rng)
    pd.DataFrame(a).to_csv(run / "summary_sample_errors.csv", index=False)
    n_days = int(args.n_days if args.n_days is not None else p.get("n_start_times", 8))
    b = pd.DataFrame(daily_geometry(recs, epoch, radius, n_days, args.daily_step_s, 1800.0,
                                    grid, min_el, theta))
    b.to_parquet(run / "raw_daily_geometry.parquet", index=False)
    s = (b.groupby(["day", "propagator"])[["vis_starlink_mean", "vis_oneweb_mean", "served_frac",
                                           "inline_frac"]].mean().reset_index())
    s.to_csv(run / "summary_daily_geometry.csv", index=False)
    (run / "status.json").write_text(json.dumps({
        "experiment": "e12_propagation_check", "status": "ok", "config": args.config,
        "starlink_generation": p.get("starlink_generation"), "common_epoch_jd": epoch,
        "n_sats": {n: len(v) for n, v in recs.items()}, "days": days, "n_days": n_days,
        "window_s": args.window_s, "step_s": args.step_s, "daily_step_s": args.daily_step_s,
        "min_elevation_deg": min_el, "theta_deg": theta,
        "start_offset_s": float(p["start_offset_s"]),
    }, indent=2))
    print(f"[e12] run complete: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
