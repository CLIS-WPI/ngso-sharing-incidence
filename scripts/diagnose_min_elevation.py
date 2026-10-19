"""Diagnose the E6 ``oneweb_min_elev_55`` case (as-deployed CelesTrak geometry).

Uses the E6 configuration and TLE snapshot (the geometry of that case) and, per
station and start time, measures:

* the number of OneWeb satellites at or above the 55 deg mask (mean, p10) and the
  fraction of time with none (the victim is then unserved, which counts as
  unavailable in the baseline);
* the OneWeb-victim / Starlink-interferer 25.261(d) metrics (realistic variant,
  headline aperture) under the common 25 deg mask and the 55 deg victim mask,
  computed exactly as in E6 (``screen_pair_dtt``);
* the same 25 deg metrics with the victim bandwidth replaced by 250 MHz
  (del Portillo Table 3), to check that the results do not depend on the
  channel bandwidth (I/N and C/N are per-MHz densities; degradation is a rate
  ratio, so B cancels).

It does not tune anything.  Run:  python scripts/diagnose_min_elevation.py
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from ngsi.bands import load_bands, overlap
from ngsi.config import load_link_budget
from ngsi.dtt_screen import screen_pair_dtt
from ngsi.geometry import elevation_deg_batch, unit_look_vectors
from ngsi.io import new_run_dir
from ngsi.link_budget import terminal_peak_gain_dbi
from ngsi.propagation import propagate_ecef
from ngsi.runner import REPO_ROOT, _as_deployed_inputs, _load_inputs, device_str
from ngsi.screening import ScreenConfig, system_visibility

RESULTS = REPO_ROOT / "results"
VICTIM, INTERF = "ONEWEB", "STARLINK"
HEAD_DIAM = 0.45
CASE = "oneweb_min_elev_55"
BW_CHECK_MHZ = 250.0  # del Portillo et al., Table 3 (user downlink channel)


def _with_beam(m, **over):
    beam = replace(m.shells[0].beam, source="diagnosis override", **over)
    return replace(m, shells=tuple(replace(sh, beam=beam) for sh in m.shells))


def _metrics(d: dict) -> dict:
    r0 = d["r0_mean"]
    ok = r0 > 0.0
    safe = np.where(ok, r0, 1.0)
    return {
        "served_frac": d["served_frac"],
        "avail0_frac": d["avail0_frac"],
        "event_frac": d["event_frac"][0],
        "degr_nodiv": np.where(ok, 1.0 - d["r_int_mean"][0] / safe, np.nan),
        "degr_div": np.where(ok, 1.0 - d["r_div_mean"][0] / safe, np.nan),
        "unavail_nodiv": d["unavail_incr_nodiv_frac"][0],
        "unavail_div": d["unavail_incr_div_frac"][0],
    }


def main() -> int:
    exp_path = REPO_ROOT / "configs" / "experiments" / "e6_eirp_sweep.yaml"
    sim, exp, grid_cfg, grid, _ = _load_inputs(exp_path, REPO_ROOT)
    bands = load_bands(REPO_ROOT / "configs" / "bands.yaml")
    lb = load_link_budget(REPO_ROOT / "configs" / "link_budget.yaml")
    p = exp.params
    case = next(c for c in p["cases"] if c["name"] == CASE)
    mask = float(case["min_elevation_deg"][VICTIM])
    eirp = {k: float(v) for k, v in case["eirp_dbw_per_mhz"].items()}
    n_st, dt = int(p["n_start_times"]), float(p["time_step_s"])
    dur, offset, chunk = float(p["duration_s"]), float(p["start_offset_s"]), int(p["chunk"])
    min_el = float(grid_cfg.min_elevation_deg)
    device = device_str()
    radius = sim.earth.radius_km

    systems0, tle_meta, _ = _as_deployed_inputs(exp, sim, REPO_ROOT, radius)
    epoch = float(tle_meta["common_epoch_jd"])
    sysd = {n: _with_beam(systems0[n], eirp_density_dbw_per_mhz=eirp[n])
            for n in (VICTIM, INTERF)}
    sys_bw = dict(sysd, **{VICTIM: _with_beam(sysd[VICTIM], bandwidth_mhz=BW_CHECK_MHZ)})
    freq = float(overlap(bands[VICTIM], bands[INTERF], "user_downlink").center_ghz)
    rx_peak = terminal_peak_gain_dbi(HEAD_DIAM, freq, lb.aperture_efficiency)
    cfg = ScreenConfig(min_elevation_deg=min_el, earth_radius_km=radius, epoch_jd=epoch,
                       thetas_deg=(5.0,), chunk=chunk, device=device)
    lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)

    rows = []
    for k in range(n_st):
        times = k * offset + np.arange(0.0, dur, dt)
        # Count OneWeb satellites at/above the mask, per station and time step.
        counts = []
        for s in range(0, len(times), chunk):
            pa = propagate_ecef(sysd[VICTIM].elements, times[s:s + chunk], epoch, radius, device)
            ua, _ = unit_look_vectors(pa, grid.ecef)
            counts.append((elevation_deg_batch(ua, grid.up) >= mask).sum(-1).cpu().numpy())
        cnt = np.concatenate(counts, axis=0)  # [T, G]
        vis = {n: system_visibility(sysd[n], grid.ecef, grid.up, times, cfg, lat_bins)
               for n in (VICTIM, INTERF)}

        def _run(systems, victim_mask):
            return _metrics(screen_pair_dtt(
                systems[VICTIM], systems[INTERF], vis[VICTIM]["max_visible"],
                vis[INTERF]["max_visible"], grid.ecef, grid.up, times,
                systems[VICTIM].shells[0].beam, systems[INTERF].shells[0].beam, freq,
                rx_peak, HEAD_DIAM, lb.implementation_margin_db, min_el, radius, epoch,
                device=device, chunk=chunk, pattern=lb.pattern,
                victim_min_elevation_deg=victim_mask))

        m25, m55, mbw = _run(sysd, None), _run(sysd, mask), _run(sys_bw, None)
        for g in range(len(grid.latitudes)):
            row = {"start": k, "latitude_deg": float(grid.latitudes[g]),
                   "longitude_deg": float(grid.longitudes[g]),
                   "n_above_mask_mean": float(cnt[:, g].mean()),
                   "n_above_mask_p10": float(np.percentile(cnt[:, g], 10.0)),
                   "frac_none_above_mask": float((cnt[:, g] == 0).mean())}
            for tag, m in (("m25", m25), ("m55", m55), ("bw250", mbw)):
                for key, v in m.items():
                    row[f"{key}_{tag}"] = float(v[g])
            rows.append(row)

    df = pd.DataFrame(rows)
    for tag in ("m25", "m55"):
        df[f"qual_{tag}"] = df[f"avail0_frac_{tag}"] >= 0.99
    lat = df.groupby("latitude_deg").agg(
        n_above_mask_mean=("n_above_mask_mean", "mean"),
        n_above_mask_p10=("n_above_mask_p10", "min"),
        frac_none_above_mask=("frac_none_above_mask", "mean"),
        avail0_min_m55=("avail0_frac_m55", "min"),
        n_qual_m55=("qual_m55", "sum"),
        n_cells=("qual_m55", "size"),
    ).reset_index()

    def _worst(tag: str, col: str):
        q = df[df[f"qual_{tag}"]]
        r = q.loc[q.groupby("start")[f"{col}_{tag}"].idxmax()]
        return r[["start", "latitude_deg", "longitude_deg", f"{col}_{tag}",
                  "frac_none_above_mask"]].to_dict(orient="records")

    qual55 = df[df["qual_m55"]]
    bw_cols = ("event_frac", "degr_nodiv", "degr_div", "unavail_nodiv", "unavail_div",
               "avail0_frac")
    summary = {
        "geometry": "as-deployed CelesTrak snapshot (E6 TLEs)",
        "case": CASE, "mask_deg": mask, "n_start_times": n_st,
        "n_stations": int(len(grid.latitudes)),
        "qualifying_latitudes_m55": sorted(qual55["latitude_deg"].unique().tolist()),
        "n_qual_stations_m55_per_start": df.groupby("start")["qual_m55"].sum().tolist(),
        "qualifying_frac_none_above_mask_max": float(qual55["frac_none_above_mask"].max()),
        "worst_degr_nodiv_m25": _worst("m25", "degr_nodiv"),
        "worst_degr_nodiv_m55": _worst("m55", "degr_nodiv"),
        "worst_unavail_div_m55": _worst("m55", "unavail_div"),
        "bandwidth_check_mhz": BW_CHECK_MHZ,
        "bandwidth_max_abs_diff": {c: float((df[f"{c}_bw250"] - df[f"{c}_m25"]).abs().max())
                                   for c in bw_cols},
        "latitude_table": lat.to_dict(orient="records"),
    }
    run_dir = new_run_dir(RESULTS, "e9_min_elevation_diagnosis", exp.seed,
                          config_paths=[exp_path, REPO_ROOT / "configs" / "constellations.yaml",
                                        REPO_ROOT / "configs" / "link_budget.yaml",
                                        REPO_ROOT / "configs" / exp.stations],
                          repo_root=REPO_ROOT)
    lat.to_csv(run_dir / "summary_min_elevation_latitude.csv", index=False)
    df.to_csv(run_dir / "summary_min_elevation_stations.csv", index=False)
    (run_dir / "diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[e9] wrote {run_dir}")
    print(lat.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "latitude_table"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
