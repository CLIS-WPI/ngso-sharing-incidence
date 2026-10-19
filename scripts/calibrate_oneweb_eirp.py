"""Calibrate the modelled OneWeb user-downlink EIRP density to a stated C/N target.

The OneWeb user-downlink EIRP density in ``configs/constellations.yaml`` is a
physical parameter with no unambiguous public value (candidate filings disagree by
more than 10 dB and mix per-beam, per-MHz and PFD quantities).  Rather than import
an unreliable absolute value, we calibrate OneWeb *in the model's own scale* so
that its user link closes realistically, and record the calibrated value in the
config with the rule that produced it.

Rule: choose the OneWeb downlink EIRP density (dBW/MHz) at which the modelled
time-median clear-sky C/N of the 0.6 m reference terminal at 45 deg latitude in
the shared Ku user-downlink band is ``TARGET_MEDIAN_DB`` (target band 4-6 dB).
That places the OneWeb link at positive margin above the 0 dB availability
reference and comparable to the peer Starlink link in the same model.

The victim C/N is linear in the victim EIRP density (``C/N = EIRP + const``), so
the script measures the offset once and solves for the target.

Run:  python scripts/calibrate_oneweb_eirp.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ngsi.bands import load_bands, overlap
from ngsi.config import load_link_budget, load_simulation, load_stations
from ngsi.dtt_screen import screen_pair_dtt
from ngsi.link_budget import terminal_peak_gain_dbi
from ngsi.runner import _phase_systems, build_grid, device_str, epoch_jd
from ngsi.screening import ScreenConfig, system_visibility

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MEDIAN_DB = 5.0
TARGET_BAND_DB = (4.0, 6.0)
APERTURE_M = 0.6
LATITUDE_DEG = 45.0


def main() -> int:
    sim = load_simulation(REPO_ROOT / "configs" / "constellations.yaml")
    grid_cfg = load_stations(REPO_ROOT / "configs" / "stations.yaml")
    lb = load_link_budget(REPO_ROOT / "configs" / "link_budget.yaml")
    bands = load_bands(REPO_ROOT / "configs" / "bands.yaml")
    freq = float(overlap(bands["ONEWEB"], bands["STARLINK"], "user_downlink").center_ghz)
    device = device_str()
    grid = build_grid(grid_cfg, sim.earth.radius_km, device)
    jd = epoch_jd(sim)
    times = np.arange(0.0, 86400.0, 10.0)
    lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)
    cfg = ScreenConfig(min_elevation_deg=float(grid_cfg.min_elevation_deg),
                       earth_radius_km=sim.earth.radius_km, epoch_jd=jd,
                       thetas_deg=(2.0,), chunk=64, device=device)

    rng = np.random.default_rng(0)
    systems = _phase_systems(sim, ["STARLINK", "ONEWEB"], rng, sim.earth.radius_km)
    rx_peak = terminal_peak_gain_dbi(APERTURE_M, freq, lb.aperture_efficiency)
    lats = np.asarray(grid.latitudes, dtype=np.float64)
    sel = np.isclose(lats, LATITUDE_DEG)

    def _run(eirp: float) -> dict:
        beam = replace(systems["ONEWEB"].shells[0].beam, eirp_density_dbw_per_mhz=eirp)
        ow = replace(systems["ONEWEB"],
                     shells=tuple(replace(sh, beam=beam) for sh in systems["ONEWEB"].shells))
        vis = {n: system_visibility(m, grid.ecef, grid.up, times, cfg, lat_bins)
               for n, m in (("ONEWEB", ow), ("STARLINK", systems["STARLINK"]))}
        return screen_pair_dtt(
            ow, systems["STARLINK"], vis["ONEWEB"]["max_visible"],
            vis["STARLINK"]["max_visible"], grid.ecef, grid.up, times,
            beam, systems["STARLINK"].shells[0].beam, freq, rx_peak, APERTURE_M,
            lb.implementation_margin_db, float(grid_cfg.min_elevation_deg),
            sim.earth.radius_km, jd, device=device, chunk=64, pattern=lb.pattern)

    eirp_cfg = float(systems["ONEWEB"].shells[0].beam.eirp_density_dbw_per_mhz)
    res_cfg = _run(eirp_cfg)
    cn0_at_cfg = float(np.nanmedian(res_cfg["cn0_median"][sel]))
    offset = cn0_at_cfg - eirp_cfg
    calibrated = TARGET_MEDIAN_DB - offset
    res_cal = _run(calibrated)
    cn0_at_cal = float(np.nanmedian(res_cal["cn0_median"][sel]))
    avail_at_cal = float(np.mean(res_cal["avail0_frac"][sel]))

    print(f"band centre           : {freq} GHz")
    print(f"aperture              : {APERTURE_M} m (peak {rx_peak:.2f} dBi)")
    print(f"config OneWeb EIRP    : {eirp_cfg:.2f} dBW/MHz -> median C/N {cn0_at_cfg:.2f} dB")
    print(f"C/N offset            : {offset:.3f} dB")
    print(f"target band           : {TARGET_BAND_DB} dB (target {TARGET_MEDIAN_DB} dB)")
    print(f"CALIBRATED OneWeb EIRP: {calibrated:.3f} dBW/MHz "
          f"-> median C/N {cn0_at_cal:.2f} dB, avail {avail_at_cal * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
