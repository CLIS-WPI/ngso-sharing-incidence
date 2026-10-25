"""Geometric in-line screen at several separation thresholds, Gen2 as deployed (E15).

Runs only the geometric (theta) screen of E5 (``ngsi.runner.run_e5``): same config,
TLE snapshot, Starlink generation, stations, start times, step and duration, and
the same aggregation (``summary_event_rate.csv``), for every ``--thetas`` value.
Before writing the run it checks that every theta the headline Gen2 E5 run also
screened reproduces that run's ``event_frac_mean`` exactly (fails otherwise).

Usage (in the container):
    python scripts/theta_screen_sweep.py [--thetas 1,2,5,10,13.88,15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402

from ngsi.bands import co_frequency_pairs, load_bands  # noqa: E402
from ngsi.io import new_run_dir  # noqa: E402
from ngsi.runner import (_as_deployed_inputs, _load_inputs, _starlink_generation,  # noqa: E402
                         device_str, summarise)
from ngsi.screening import ScreenConfig, screen_pair, system_visibility  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments" / "e5_as_deployed.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", type=Path, default=REPO / "results")
    ap.add_argument("--thetas", default="1,2,5,10,13.88,15")
    args = ap.parse_args()
    thetas = tuple(float(t) for t in args.thetas.split(","))

    mn.RESULTS = args.results_root.resolve()
    head = mn.headline_run("e5_as_deployed", "gen2")
    if head is None:
        raise SystemExit("no Gen2 E5 headline run to check against")

    # As in run_e5 (geometric screen only).
    sim, exp, grid_cfg, grid, _ = _load_inputs(EXP, REPO)
    bands = load_bands(REPO / "configs" / "bands.yaml")
    pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    p = exp.params
    gen = _starlink_generation(p, None)
    n_st = int(p.get("n_start_times", 8))
    dt = float(p.get("time_step_s", 10.0))
    dur = float(p.get("duration_s", 86400.0))
    offset = float(p.get("start_offset_s", 18000.0))
    radius = sim.earth.radius_km
    lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)
    systems, tle_meta, _ = _as_deployed_inputs(exp, sim, REPO, radius, gen)
    cfg = ScreenConfig(min_elevation_deg=float(grid_cfg.min_elevation_deg),
                       earth_radius_km=radius, epoch_jd=float(tle_meta["common_epoch_jd"]),
                       thetas_deg=thetas, chunk=int(p.get("chunk", 64)), device=device_str())

    rows = []
    for k in range(n_st):
        times = k * offset + np.arange(0.0, dur, dt)
        vis = {n: system_visibility(systems[n], grid.ecef, grid.up, times, cfg, lat_bins)
               for n in exp.constellations}
        for a, b in pairs:
            res = screen_pair(systems[a], systems[b], vis[a]["max_visible"],
                              vis[b]["max_visible"], grid.ecef, grid.up, times, cfg)
            for ti, th in enumerate(thetas):
                for vi in (0, 1):
                    for g in range(grid.ecef.shape[0]):
                        rows.append({"phasing": k, "system_a": a, "system_b": b, "variant": vi,
                                     "theta_deg": th, "latitude_deg": float(grid.latitudes[g]),
                                     "event_frac": float(res["event_frac"][ti, vi, g])})
    s = summarise(pd.DataFrame(rows), ("event_frac",),
                  ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])

    # Reproduction check against the headline run for the thetas it has.
    keys = ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"]
    ref = pd.read_csv(head / "summary_event_rate.csv")[keys + ["event_frac_mean"]]
    m = ref.merge(s[keys + ["event_frac_mean"]], on=keys, suffixes=("_ref", ""))
    common = sorted(set(ref["theta_deg"]) & set(thetas))
    if m.empty or len(m) != len(ref[ref["theta_deg"].isin(common)]):
        raise SystemExit("reproduction check: rows do not line up with the headline run")
    dev = float((m["event_frac_mean"] - m["event_frac_mean_ref"]).abs().max())
    if dev > 1e-12:
        raise SystemExit(f"reproduction check failed: max |diff| = {dev:g}")

    out = new_run_dir(args.results_root, "e15_theta_screen", int(exp.seed),
                      config_paths=[EXP], repo_root=REPO)
    s.to_csv(out / "summary_theta_screen.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "experiment": "e15_theta_screen", "status": "ok", "starlink_generation": gen,
        "thetas_deg": list(thetas), "n_start_times": n_st, "time_step_s": dt,
        "duration_s": dur, "start_offset_s": offset,
        "checked_against": head.name, "checked_thetas_deg": common,
        "checked_rows": int(len(m)), "checked_max_abs_diff": dev}, indent=2))
    print(f"[e15] reproduces {head.name} at theta {common} ({len(m)} rows, max diff {dev:g})")
    print(f"[e15] run complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
