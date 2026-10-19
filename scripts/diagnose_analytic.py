"""Diagnose the analytic co-located model against the simulated serving directions.

For one constellation phasing over 24 h, directly measures, at every station
latitude, the off-zenith radius and azimuth of each constellation's
highest-elevation (serving) satellite and the angle between them.  From the same
E4 geometry it then evaluates, at theta* = theta_star_for_starlink(0.45 m):

* ``sim_cond``   - empirical P(sep < theta* | both served)  (the simulation truth);
* ``indep_cond`` - P(sep < theta*) using the **empirical marginal** radius
                  distributions but independent, uniform azimuths (the controlled
                  substitution of the simulated marginals into the analytic
                  separation integral; tests the independence assumption);
* ``model_cond`` - the current independent nearest-of-N uniform-disc prediction.

If ``indep_cond`` tracks ``sim_cond`` the independence assumption is sound and the
residual error comes from the marginal (uniform-disc nearest-of-N) model.  This
script does not tune anything.

Run:  python scripts/diagnose_analytic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ngsi.analytic import (
    ShellSpec,
    _nearest_offaxis_pdf,
    event_fraction_vs_latitude,
    separation_probability,
    separation_probability_from_pdf,
    serving_look_radius_pdf,
    theta_star_for_starlink,
    visibility_stats,
)
from ngsi.geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from ngsi.io import new_run_dir
from ngsi.propagation import propagate_ecef
from ngsi.runner import (
    REPO_ROOT,
    _load_inputs,
    _phase_systems,
    device_str,
    epoch_jd,
)

RESULTS = REPO_ROOT / "results"
THETA = theta_star_for_starlink(0.45)
MIN_EL = 25.0
N_BINS = 90


def _empirical_marginal(r: np.ndarray, alpha: float):
    edges = np.linspace(0.0, alpha, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(np.clip(r, 0.0, alpha - 1e-9), bins=edges)
    w = hist.astype(np.float64)
    w /= w.sum()
    return centers, w


def _sep_prob_from_marginals(ra, wa, rb, wb, theta: float) -> float:
    t2 = theta**2
    RA, RB = np.meshgrid(ra, rb, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (RA**2 + RB**2 - t2) / (2.0 * RA * RB)
    cond = np.where(q <= -1.0, 1.0,
                    np.where(q >= 1.0, 0.0, np.arccos(np.clip(q, -1.0, 1.0)) / np.pi))
    return float((wa[:, None] * wb[None, :] * cond).sum())


def main() -> int:
    exp_path = REPO_ROOT / "configs" / "experiments" / "e4_dtt.yaml"
    sim, exp, grid_cfg, grid, _ = _load_inputs(exp_path, REPO_ROOT)
    device = device_str()
    radius = sim.earth.radius_km
    jd = epoch_jd(sim)
    times = np.arange(0.0, 86400.0, 10.0)
    rng = np.random.default_rng(exp.seed)
    systems = _phase_systems(sim, exp.constellations, rng, radius)

    victim, interf = "ONEWEB", "STARLINK"
    shells_a = [ShellSpec(s.altitude_km, s.inclination_deg, s.planes, s.sats_per_plane)
                for s in sim.systems[victim].shells]
    shells_b = [ShellSpec(s.altitude_km, s.inclination_deg, s.planes, s.sats_per_plane)
                for s in sim.systems[interf].shells]

    # East/North unit vectors from the local up vector (schematic ENU).
    up = grid.up.cpu().numpy()
    zhat = np.array([0.0, 0.0, 1.0])
    east = np.cross(zhat[None, :], up)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(up, east)
    east_t = torch.as_tensor(east, dtype=torch.float64, device=device)
    north_t = torch.as_tensor(north, dtype=torch.float64, device=device)

    rows, lat_rows = [], []
    lats = np.asarray(grid_cfg.latitudes_deg, dtype=np.float64)
    seen: set[float] = set()
    for gi, lat in enumerate(grid.latitudes):
        key = round(float(lat), 6)
        if key in seen:
            continue
        seen.add(key)
        st = grid.ecef[gi:gi + 1]
        u = grid.up[gi:gi + 1]
        ra, rb = [], []
        for name, out_r in ((victim, ra), (interf, rb)):
            p = propagate_ecef(systems[name].elements, times, jd, radius, device)
            uu, _ = unit_look_vectors(p, st)          # [T,1,K,3]
            el = elevation_deg_batch(uu, u)[:, 0, :]  # [T,K]
            vis = el >= MIN_EL
            el_masked = el.masked_fill(~vis, float("-inf"))
            best = el_masked.argmax(-1)               # [T]
            has = el_masked.max(-1).values >= MIN_EL  # [T]
            ub = torch.gather(uu[:, 0], 1,
                              best.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # [T,3]
            r = (90.0 - el_masked.max(-1).values)     # [T]
            out_r.extend(r.cpu().numpy().tolist())
            if name == victim:
                ua_s, has_a = ub, has
            else:
                ub_s, has_b = ub, has
        both = (has_a & has_b).cpu().numpy()
        sep = angular_separation_batch(ua_s, ub_s).cpu().numpy()  # [T]
        sep_ok = both & np.isfinite(sep)
        ra_arr = np.array(ra)[sep_ok]
        rb_arr = np.array(rb)[sep_ok]
        sim_cond = float((sep[sep_ok] < THETA).mean()) if sep_ok.any() else float("nan")
        alpha = 90.0 - MIN_EL
        ca, wa = _empirical_marginal(ra_arr, alpha)
        cb, wb = _empirical_marginal(rb_arr, alpha)
        indep_cond = _sep_prob_from_marginals(ca, wa, cb, wb, THETA)
        n_a, _ = visibility_stats(shells_a, float(lat), MIN_EL, radius)
        n_b, _ = visibility_stats(shells_b, float(lat), MIN_EL, radius)
        model_cond = separation_probability(THETA, alpha, n_a, alpha, n_b)
        ra_p, wa_p = serving_look_radius_pdf(shells_a, MIN_EL, n_a, radius, N_BINS)
        rb_p, wb_p = serving_look_radius_pdf(shells_b, MIN_EL, n_b, radius, N_BINS)
        new_cond = separation_probability_from_pdf(THETA, ra_p, wa_p, rb_p, wb_p)
        v_a, v_b = 1.0 - np.exp(-n_a), 1.0 - np.exp(-n_b)
        frac_model = event_fraction_vs_latitude(
            np.array([float(lat)]), THETA, shells_a, shells_b, MIN_EL,
            earth_radius_km=radius, n_grid=240)[0]
        lat_rows.append({
            "latitude_deg": float(lat), "theta_star_deg": THETA,
            "n_samples_both": int(sep_ok.sum()),
            "r_a_mean_deg": float(np.mean(ra_arr)), "r_b_mean_deg": float(np.mean(rb_arr)),
            "r_a_median_deg": float(np.median(ra_arr)), "r_b_median_deg": float(np.median(rb_arr)),
            "corr_r": float(np.corrcoef(ra_arr, rb_arr)[0, 1]) if sep_ok.sum() > 2 else float("nan"),
            "sim_cond": sim_cond, "indep_cond": indep_cond,             "model_cond": model_cond,
            "new_cond": new_cond,
            "r_a_new_model_mean_deg": float((ra_p * wa_p).sum()),
            "r_b_new_model_mean_deg": float((rb_p * wb_p).sum()),
            "old_event_frac": float(v_a * v_b * model_cond),
            "new_event_frac": float(v_a * v_b * new_cond),
            "event_frac_vs_lat": float(frac_model),
            "sim_event_frac": float((both & (sep < THETA)).mean()),
        })
        rows.extend([{"latitude_deg": float(lat), "r_a_deg": float(a), "r_b_deg": float(b)}
                     for a, b in zip(ra_arr, rb_arr)])

    df = pd.DataFrame(lat_rows)
    run_dir = new_run_dir(RESULTS, "e8_analytic_diagnosis", exp.seed,
                          config_paths=[exp_path, REPO_ROOT / "configs" / "constellations.yaml"],
                          repo_root=REPO_ROOT)
    df.to_csv(run_dir / "summary_diagnosis.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "raw_serving_radii.csv", index=False)
    summary = {
        "theta_star_deg": THETA,
        "max_abs_err_old_cond": float((df["model_cond"] - df["sim_cond"]).abs().max()),
        "max_abs_err_new_cond": float((df["new_cond"] - df["sim_cond"]).abs().max()),
        "max_abs_err_indep_cond": float((df["indep_cond"] - df["sim_cond"]).abs().max()),
        "old_above_sim_everywhere": bool((df["old_event_frac"] > df["sim_event_frac"]).all()),
        "new_above_sim_everywhere": bool((df["new_event_frac"] > df["sim_event_frac"]).all()),
        "old_event_minus_sim_min": float((df["old_event_frac"] - df["sim_event_frac"]).min()),
        "old_event_minus_sim_max": float((df["old_event_frac"] - df["sim_event_frac"]).max()),
        "new_event_minus_sim_min": float((df["new_event_frac"] - df["sim_event_frac"]).min()),
        "new_event_minus_sim_max": float((df["new_event_frac"] - df["sim_event_frac"]).max()),
        "new_event_frac_vs_analytic_max": float(
            (df["new_event_frac"] - df["event_frac_vs_lat"]).abs().max()),
        "mean_r_a_empirical_deg": float(df["r_a_mean_deg"].mean()),
        "mean_r_b_empirical_deg": float(df["r_b_mean_deg"].mean()),
        "mean_r_a_new_model_deg": float(df["r_a_new_model_mean_deg"].mean()),
        "mean_r_b_new_model_deg": float(df["r_b_new_model_mean_deg"].mean()),
        "r_a_model_mean_deg": None,
        "results": df.to_dict(orient="records"),
    }
    # model mean nearest-of-N radius, for comparison with the empirical mean
    alpha = 90.0 - MIN_EL
    means = []
    for _, row in df.iterrows():
        n_a, _ = visibility_stats(shells_a, float(row["latitude_deg"]), MIN_EL, radius)
        edges = np.linspace(0.0, alpha, N_BINS + 1)
        c = 0.5 * (edges[:-1] + edges[1:])
        w = _nearest_offaxis_pdf(c, alpha, n_a)
        w /= w.sum()
        means.append(float((c * w).sum()))
    summary["r_a_model_mean_deg"] = float(np.mean(means))
    (run_dir / "diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[e8] wrote {run_dir}")
    print(df.to_string(index=False))
    print(json.dumps({k: summary[k] for k in (
        "max_abs_err_old_cond", "max_abs_err_new_cond", "max_abs_err_indep_cond",
        "old_above_sim_everywhere", "new_above_sim_everywhere",
        "new_event_minus_sim_min", "new_event_minus_sim_max",
        "mean_r_a_empirical_deg", "mean_r_b_empirical_deg", "r_a_model_mean_deg")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
