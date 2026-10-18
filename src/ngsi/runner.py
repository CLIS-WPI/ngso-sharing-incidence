"""Experiment orchestration: phasings, screening, aggregation with CI, results.

Heavy logic for E1 (event rate vs latitude) and E2 (satellite diversity and the
25.261 burden) lives here; ``experiments/*.py`` are thin entrypoints.

Replication unit (AGENTS.md rule 5) is an independent constellation *phasing*:
each phasing draws one random RAAN offset and mean-anomaly offset per shell.  All
confidence intervals are computed **across phasings** (t-interval, n-1 df).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bands import LINK_TYPES, co_frequency_pairs, load_bands, overlap
from .capacity_diversity import capacity_limited_unresolvable, regional_grid
from .config import (
    Constellation,
    ExperimentConfig,
    StationGrid,
    load_experiment,
    load_link_budget,
    load_sanity_stations,
    load_simulation,
    load_stations,
)
from .link_budget import terminal_peak_gain_dbi
from .frames import utc_to_jd
from .geometry import station_ecef_spherical
from .io import new_run_dir
from .rules import BAND_SPLIT_FRACTION
from .sanity import analytic_density_per_deg, analytic_visible_count, max_relative_deviation
from .dtt_screen import screen_pair_dtt
from .screening import ScreenConfig, screen_pair, system_visibility
from .systems import build_as_deployed_system, build_system
from .tle import celestrak_url, common_epoch_jd, fetch_group, parse_tle

REPO_ROOT = Path(__file__).resolve().parents[2]

_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
        10: 2.262, 11: 2.228, 13: 2.179, 16: 2.131, 21: 2.086, 31: 2.042}


def t95(n: int) -> float:
    """Two-sided 95% t critical value for ``n`` samples (normal fallback)."""
    if n < 2:
        return float("nan")
    if n in _T95:
        return _T95[n]
    for k in sorted(_T95):
        if n <= k:
            return _T95[k]
    return 1.96


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Grid:
    """Station grid as torch tensors plus per-station latitude/longitude."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    ecef: torch.Tensor
    up: torch.Tensor


def build_grid(grid: StationGrid, radius_km: float, device: str) -> Grid:
    lat, lon = np.meshgrid(grid.latitudes_deg, grid.longitudes_deg, indexing="ij")
    lat_flat, lon_flat = lat.ravel(), lon.ravel()
    ecef = station_ecef_spherical(lat_flat, lon_flat, radius_km, device)
    up = ecef / torch.linalg.norm(ecef, dim=-1, keepdim=True)
    return Grid(lat_flat, lon_flat, ecef, up)


def epoch_jd(sim) -> float:
    dt = datetime.fromisoformat(sim.epoch_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    return utc_to_jd(dt)


def ordered_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in names for b in names if a != b]


def _phase_systems(sim, names: list[str], rng: np.random.Generator, radius_km: float) -> dict:
    return {n: build_system(sim.systems[n], rng, earth_radius_km=radius_km) for n in names}


def screen_phasings(
    sim,
    names: list[str],
    grid: Grid,
    times: np.ndarray,
    device: str,
    seed: int,
    n_phasings: int,
    thetas_deg: tuple[float, ...],
    min_elevation_deg: float,
    chunk: int,
    collect_series: bool = False,
    pairs: list[tuple[str, str]] | None = None,
    pair_meta: dict[tuple[str, str], dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Screen every ordered pair over ``n_phasings`` independent phasings.

    ``pairs`` defaults to every ordered pair of ``names``; pass the co-frequency
    subset to screen only pairs that share a band.  ``pair_meta`` maps each pair
    to ``{"link_type": str, "shared_freq_ghz": float | None}`` recorded on every
    event row.

    Returns ``(pair_events, vis_pool, sanity_material)`` where ``pair_events`` is a
    tidy table of per-cell fractions and ``vis_pool`` holds per-system visibility
    statistics (pool mean, per-shell means, sub-satellite latitude histogram).
    """
    radius = sim.earth.radius_km
    jd = epoch_jd(sim)
    thetas = tuple(float(t) for t in thetas_deg)
    if pairs is None:
        pairs = ordered_pairs(names)
    pair_meta = pair_meta or {}
    lats = np.asarray(sorted(set(grid.latitudes)), dtype=np.float64)
    nlon = int((grid.latitudes == grid.latitudes[0]).sum())
    event_rows: list[dict] = []
    pool_rows: list[dict] = []
    series_rows: list[dict] = []
    sub_lat = {n: {"hist": None, "incs": None} for n in names}
    shell_specs = {n: None for n in names}

    for p in range(n_phasings):
        rng = np.random.default_rng(seed + 137 * p)
        systems = _phase_systems(sim, names, rng, radius)
        cfg = ScreenConfig(
            min_elevation_deg=min_elevation_deg,
            earth_radius_km=radius,
            epoch_jd=jd,
            thetas_deg=thetas,
            chunk=chunk,
            device=device,
        )
        lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)
        vis = {
            n: system_visibility(systems[n], grid.ecef, grid.up, times, cfg, lat_bins)
            for n in names
        }
        for n in names:
            if p == 0:
                sub_lat[n]["hist"] = vis[n]["shell_lat_hist"]
                sub_lat[n]["incs"] = [s.inclination_deg for s in systems[n].shells]
                shell_specs[n] = [
                    {"name": s.name, "altitude_km": s.altitude_km, "inclination_deg": s.inclination_deg,
                     "planes": s.planes, "sats_per_plane": s.sats_per_plane}
                    for s in systems[n].shells
                ]
            pool_mean = (vis[n]["pool_sum"] / vis[n]["n_samples"]).reshape(len(lats), nlon).mean(1)
            served = vis[n]["served_frac"].reshape(len(lats), nlon).mean(1)
            pool_hist = vis[n]["pool_hist"].reshape(len(lats), nlon, -1).sum(1)  # [nlat, B]
            pool_p10 = np.array([_weighted_percentile(pool_hist[li], 10.0) for li in range(len(lats))])
            for li, la in enumerate(lats):
                pool_rows.append(
                    {"phasing": p, "system": n, "mode": "pool", "shell_or_all": "all",
                     "latitude_deg": float(la), "pool_mean": float(pool_mean[li]),
                     "shell_visible_mean": np.nan, "served_frac": float(served[li]),
                     "pool_p10": float(pool_p10[li])}
                )
            shell_mean = vis[n]["shell_vis_sum"] / vis[n]["n_samples"]
            for si, sh in enumerate(systems[n].shells):
                shm = shell_mean[si].reshape(len(lats), nlon).mean(1)
                for li, la in enumerate(lats):
                    pool_rows.append(
                        {"phasing": p, "system": n, "mode": "shell", "shell_or_all": sh.name,
                         "latitude_deg": float(la), "pool_mean": np.nan,
                         "shell_visible_mean": float(shm[li]), "served_frac": np.nan,
                         "pool_p10": np.nan}
                    )
        for a, b in pairs:
            res = screen_pair(
                systems[a], systems[b], vis[a]["max_visible"], vis[b]["max_visible"],
                grid.ecef, grid.up, times, cfg, collect_series=collect_series,
            )
            meta = pair_meta.get((a, b), {})
            link_type = meta.get("link_type")
            shared_freq = meta.get("shared_freq_ghz")
            svc_a = vis[a]["served_frac"]
            svc_b = vis[b]["served_frac"]
            for ti, th in enumerate(thetas):
                for vi in (0, 1):
                    ev, un = res["event_frac"][ti, vi], res["unres_frac"][ti, vi]
                    for g in range(grid.ecef.shape[0]):
                        event_rows.append(
                            {"phasing": p, "system_a": a, "system_b": b, "variant": vi,
                             "theta_deg": th, "latitude_deg": float(grid.latitudes[g]),
                             "longitude_deg": float(grid.longitudes[g]),
                             "link_type": link_type, "shared_freq_ghz": shared_freq,
                             "event_frac": float(ev[g]), "unres_frac": float(un[g]),
                             "covisible_frac": float(res["covisible_frac"][g]),
                             "service_a_frac": float(svc_a[g]), "service_b_frac": float(svc_b[g]),
                             "n_samples": int(res["n_samples"])}
                        )
            if collect_series:
                series_rows.extend(_series_durations(res, grid, thetas, a, b, p, times))
    sanity = {
        "sub_lat": sub_lat,
        "shell_specs": shell_specs,
        "lat_bins": np.arange(-90.0, 90.0 + 1e-9, 5.0).tolist(),
        "n_phasings": n_phasings,
    }
    return pd.DataFrame(event_rows), pd.DataFrame(pool_rows), pd.DataFrame(series_rows), sanity


def _series_durations(res, grid, thetas, a, b, p, times) -> list[dict]:
    """Run lengths (s) of event / unresolvable series per cell for duration analysis."""
    if "event_series" not in res or res["event_series"] is None:
        return []
    step = float(times[1] - times[0]) if len(times) > 1 else float(times[0])
    rows = []
    ev = res["event_series"]
    un = res["unres_series"]
    for ti, th in enumerate(thetas):
        for vi in (0, 1):
            for g in range(ev.shape[2]):
                for label, series in (("event", ev[ti, vi, g]), ("unres", un[ti, vi, g])):
                    lengths = _run_lengths(series) * step
                    for L in lengths:
                        rows.append(
                            {
                                "phasing": p,
                                "system_a": a,
                                "system_b": b,
                                "variant": vi,
                                "theta_deg": th,
                                "latitude_deg": float(grid.latitudes[g]),
                                "longitude_deg": float(grid.longitudes[g]),
                                "kind": label,
                                "duration_s": float(L),
                            }
                        )
    return rows


def _weighted_percentile(hist: np.ndarray, q: float) -> float:
    """Percentile of a non-negative integer count distribution given as a histogram."""
    h = np.asarray(hist, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return float("nan")
    cdf = np.cumsum(h) / total
    idx = int(np.searchsorted(cdf, q / 100.0, side="left"))
    return float(min(idx, h.size - 1))


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths (in samples) of consecutive-True runs in a 1-D boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return np.array([], dtype=np.float64)
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return (edges[1::2] - edges[0::2]).astype(np.float64)


def summarise(df: pd.DataFrame, value_cols: tuple[str, ...], group_cols: list[str]) -> pd.DataFrame:
    """Mean over longitudes per phasing, then mean +/- 95% t-CI over phasings."""
    per_phase = (
        df.groupby(group_cols + ["phasing"])[list(value_cols)].mean().reset_index()
    )
    records = []
    for key, sub in per_phase.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        n = len(sub)
        tcrit = t95(n)
        for col in value_cols:
            vals = sub[col].to_numpy(dtype=np.float64)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            half = tcrit * sd / np.sqrt(n) if n > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95"] = float(half)
            row[f"{col}_sd"] = sd
        row["n_phasings"] = n
        records.append(row)
    return pd.DataFrame(records)


def event_rate_per_hour(event_frac: float, time_step_s: float, duration_s: float) -> float:
    """Events per cell per hour from the event time-fraction (sampling-limited)."""
    return float(event_frac) * 3600.0 / float(time_step_s)


def run_sanity(
    sim,
    exp: ExperimentConfig,
    grid: Grid,
    sanity_grid: Grid | None,
    time_step_s: float,
    duration_s: float,
    min_elevation_deg: float,
    device: str,
    n_phasings: int = 2,
    chunk: int = 128,
) -> dict:
    """Section E gate: latitude density, visible-count, symmetry, runtime/memory."""
    radius = sim.earth.radius_km
    jd = epoch_jd(sim)
    times = np.arange(0.0, duration_s, time_step_s)
    lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)
    names = exp.constellations
    checks = {"density": [], "visible": [], "symmetry": [], "runtime": {}, "sub_lat": {}}
    thetas = (float(exp.params.get("inline_threshold_deg", 1.0)),)
    cfg = ScreenConfig(min_elevation_deg, radius, jd, thetas, chunk, device)

    rng = np.random.default_rng(exp.seed)
    systems = _phase_systems(sim, names, rng, radius)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for n in names:
        vis = system_visibility(systems[n], grid.ecef, grid.up, times, cfg, lat_bins)
        incs = [s.inclination_deg for s in systems[n].shells]
        checks["sub_lat"][n] = {"hist": vis["shell_lat_hist"].tolist(), "incs": incs}
        for si, sh in enumerate(systems[n].shells):
            dev = max_relative_deviation(vis["shell_lat_hist"][si], lat_bins, sh.inclination_deg)
            checks["density"].append(
                {"system": n, "shell": sh.name, "max_rel_dev": float(dev)}
            )
        shell_mean = vis["shell_vis_sum"] / vis["n_samples"]
        nlon = int((grid.latitudes == grid.latitudes[0]).sum())
        nlat = int(grid.ecef.shape[0] // nlon)
        lats_u = np.asarray(sorted(set(grid.latitudes)), dtype=np.float64)
        for si, sh in enumerate(systems[n].shells):
            sim_curve = shell_mean[si].reshape(nlat, nlon).mean(1)
            ana_curve = np.array([
                analytic_visible_count(sh.altitude_km, sh.inclination_deg, sh.planes,
                                       sh.sats_per_plane, float(la), min_elevation_deg, radius)
                for la in lats_u
            ])
            peak = float(ana_curve.max())
            mask = ana_curve >= 0.05 * peak if peak > 0 else np.zeros_like(ana_curve, dtype=bool)
            rel = (
                float(np.max(np.abs(sim_curve[mask] - ana_curve[mask]) / ana_curve[mask]))
                if mask.any() else float("nan")
            )
            checks["visible"].append(
                {"system": n, "shell": sh.name, "sim_peak": float(sim_curve.max()),
                 "analytic_peak": peak, "rel_dev": rel}
            )
    checks["runtime"]["peak_mem_bytes"] = (
        torch.cuda.max_memory_allocated() if device == "cuda" else 0
    )

    if sanity_grid is not None:
        for n in names:
            vis_s = system_visibility(systems[n], sanity_grid.ecef, sanity_grid.up, times, cfg, lat_bins)
            nlon_s = int((sanity_grid.latitudes == sanity_grid.latitudes[0]).sum())
            nlat_s = int(sanity_grid.ecef.shape[0] // nlon_s)
            total_vis = vis_s["shell_vis_sum"].sum(0) / vis_s["n_samples"]
            sm = total_vis.reshape(nlat_s, nlon_s).mean(1)
            checks["symmetry"].append(
                {
                    "system": n,
                    "max_abs_diff_pos_neg": float(np.max(np.abs(sm - sm[::-1]))),
                    "mean_visible": float(sm.mean()),
                }
            )
    checks["n_phasings"] = n_phasings
    checks["time_step_s"] = time_step_s
    checks["duration_s"] = duration_s
    checks["min_elevation_deg"] = min_elevation_deg
    checks["lat_bins"] = lat_bins.tolist()
    return checks


# --- result writing helpers -------------------------------------------------

def _write_summary(run_dir: Path, name: str, df: pd.DataFrame) -> Path:
    path = run_dir / f"summary_{name}.csv"
    df.to_csv(path, index=False)
    return path


def _write_json(run_dir: Path, name: str, obj) -> Path:
    import json

    path = run_dir / name
    path.write_text(
        json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)),
        encoding="utf-8",
    )
    return path


def _sanity_ok(checks: dict) -> tuple[bool, list[str]]:
    reasons = []
    for row in checks["density"]:
        if not np.isfinite(row["max_rel_dev"]) or row["max_rel_dev"] > 0.05:
            reasons.append(f"density {row['system']}: max_rel_dev={row['max_rel_dev']:.3f} > 0.05")
    for row in checks["visible"]:
        if not np.isfinite(row["rel_dev"]) or abs(row["rel_dev"]) > 0.20:
            reasons.append(
                f"visible {row['system']}/{row['shell']}: rel_dev={row['rel_dev']:.3f} > 0.20"
            )
    for row in checks["symmetry"]:
        if row["max_abs_diff_pos_neg"] > 1.0:
            reasons.append(
                f"symmetry {row['system']}: |pos-neg|={row['max_abs_diff_pos_neg']:.3f} > 1.0"
            )
    return (len(reasons) == 0, reasons)


def _load_inputs(exp_path: Path, repo_root: Path = REPO_ROOT):
    sim = load_simulation(repo_root / "configs" / "constellations.yaml")
    exp = load_experiment(exp_path)
    grid_cfg = load_stations(repo_root / "configs" / exp.stations)
    try:
        sanity_cfg = load_sanity_stations(repo_root / "configs" / exp.stations)
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"sanity grid unavailable: {exc!r}")
        sanity_cfg = None
    device = device_str()
    grid = build_grid(grid_cfg, sim.earth.radius_km, device)
    sanity_grid = build_grid(sanity_cfg, sim.earth.radius_km, device) if sanity_cfg else None
    return sim, exp, grid_cfg, grid, sanity_grid


def _sub_latitude_table(sanity: dict, sim) -> pd.DataFrame:
    bins = np.asarray(sanity["lat_bins"], dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    rows = []
    for n, info in sanity["sub_lat"].items():
        hists = np.asarray(info["hist"], dtype=np.float64)
        if hists.ndim == 1:
            hists = hists[None, :]
        for hist, inc in zip(hists, info["incs"]):
            total = hist.sum()
            sim_density = hist / (total * width) if total else hist
            ana = analytic_density_per_deg(inc, centers)
            for c, s_d, a_d in zip(centers, sim_density, ana):
                rows.append({"system": n, "inclination_deg": float(inc), "lat_center_deg": float(c),
                             "sim_density_per_deg": float(s_d), "analytic_density_per_deg": float(a_d)})
    return pd.DataFrame(rows)


def _band_overlap_table(names: list[str], bands) -> pd.DataFrame:
    """Every ordered pair x link type with overlap, shared bandwidth and centre."""
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            for lt in LINK_TYPES:
                ov = overlap(bands[a], bands[b], lt)
                rows.append(
                    {"system_a": a, "system_b": b, "link_type": lt,
                     "overlaps": bool(ov.overlaps),
                     "shared_bandwidth_ghz": ov.total_bandwidth_ghz,
                     "center_ghz": ov.center_ghz,
                     "segments_ghz": ";".join(f"{lo}-{hi}" for lo, hi in ov.segments_ghz)}
                )
    return pd.DataFrame(rows)


def _service_pool_summary(pool: pd.DataFrame) -> pd.DataFrame:
    """Per victim system and latitude: service availability and diversity pool."""
    p = pool[pool["mode"] == "pool"]
    return summarise(p, ("pool_mean", "served_frac", "pool_p10"), ["system", "latitude_deg"])


def run_e1(
    exp_path: Path,
    results_root: Path,
    repo_root: Path = REPO_ROOT,
    n_phasings: int | None = None,
    time_step_s: float | None = None,
    duration_s: float | None = None,
) -> Path:
    """Run E1 (event rate vs latitude). Returns the run directory."""
    sim, exp, grid_cfg, grid, sanity_grid = _load_inputs(exp_path, repo_root)
    bands = load_bands(repo_root / "configs" / "bands.yaml")
    udl_pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    pair_meta = {
        (a, b): {"link_type": "user_downlink",
                 "shared_freq_ghz": overlap(bands[a], bands[b], "user_downlink").center_ghz}
        for (a, b) in udl_pairs
    }
    p = exp.params
    n_ph = int(n_phasings if n_phasings is not None else p.get("n_phasings", 8))
    dt = float(time_step_s if time_step_s is not None else p.get("time_step_s", 10.0))
    dur = float(duration_s if duration_s is not None else p.get("duration_s", 86400.0))
    thetas = (1.0, 2.0, 5.0)
    min_el = float(grid_cfg.min_elevation_deg)
    chunk = int(p.get("chunk", 64))

    run_dir = new_run_dir(
        results_root, exp.name, exp.seed,
        config_paths=[exp_path, repo_root / "configs" / "constellations.yaml",
                      repo_root / "configs" / "bands.yaml",
                      repo_root / "configs" / exp.stations],
        repo_root=repo_root,
    )
    _band_overlap_table(exp.constellations, bands).to_csv(
        run_dir / "summary_band_overlap.csv", index=False
    )
    device = device_str()
    checks = run_sanity(sim, exp, grid, sanity_grid, dt, dur, min_el, device, chunk=chunk)
    _write_json(run_dir, "sanity.json", checks)
    _sub_latitude_table(checks, sim).to_csv(run_dir / "summary_sub_latitude_density.csv", index=False)
    ok, reasons = _sanity_ok(checks)
    if not ok:
        _write_json(run_dir, "status.json", {"experiment": exp.name, "status": "sanity_failed",
                                             "reasons": reasons})
        raise SystemExit(f"[e1] sanity gate FAILED: {reasons}")

    times = np.arange(0.0, dur, dt)
    events, pool, _series, sanity_material = screen_phasings(
        sim, exp.constellations, grid, times, device, exp.seed, n_ph, thetas, min_el, chunk,
        pairs=udl_pairs, pair_meta=pair_meta,
    )
    events.to_parquet(run_dir / "raw_pair_events.parquet", index=False)
    pool.to_parquet(run_dir / "raw_pool.parquet", index=False)

    ev = summarise(events, ("event_frac", "unres_frac", "service_a_frac", "service_b_frac"),
                   ["system_a", "system_b", "link_type", "shared_freq_ghz",
                    "variant", "theta_deg", "latitude_deg"])
    ev["event_rate_per_hour"] = ev["event_frac_mean"] * 3600.0 / dt
    ev.to_csv(run_dir / "summary_event_rate.csv", index=False)
    _service_pool_summary(pool).to_csv(run_dir / "summary_service_pool.csv", index=False)

    _write_json(run_dir, "status.json",
                {"experiment": exp.name, "status": "ok", "n_phasings": n_ph,
                 "time_step_s": dt, "duration_s": dur, "thetas_deg": list(thetas),
                 "min_elevation_deg": min_el, "device": device,
                 "user_downlink_pairs": [list(q) for q in udl_pairs]})
    return run_dir


def _starlink_f_variant(sim, f_value: int):
    """Copy of the simulation with the Starlink ~53 deg shells given phasing ``F``."""
    from dataclasses import replace as _replace

    def is53(sh):
        return abs(sh.inclination_deg - 53.0) < 1.0

    new_systems = {}
    for name, con in sim.systems.items():
        if name == "STARLINK":
            shells = [_replace(s, phasing_f=int(f_value)) if is53(s) else s for s in con.shells]
            new_systems[name] = _replace(con, shells=shells)
        else:
            new_systems[name] = con
    return _replace(sim, systems=new_systems)


def _oneweb_shell_variant(sim, planes: int, sats_per_plane: int):
    """Copy of the simulation with the OneWeb shell re-sized (ASSUMED geometry)."""
    from dataclasses import replace as _replace

    new_systems = {}
    for name, con in sim.systems.items():
        if name == "ONEWEB":
            shells = [_replace(s, planes=int(planes), sats_per_plane=int(sats_per_plane))
                      for s in con.shells]
            new_systems[name] = _replace(con, shells=shells)
        else:
            new_systems[name] = con
    return _replace(sim, systems=new_systems)


def _duration_summary(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["system_a", "system_b", "variant", "theta_deg", "kind",
                                     "latitude_deg", "count", "mean_s", "median_s", "p90_s",
                                     "mean_ci95_s"])
    keys = ["system_a", "system_b", "variant", "theta_deg", "kind", "latitude_deg"]
    pooled = series.groupby(keys)["duration_s"].agg(
        count="count", mean_s="mean", median_s="median", p90_s=lambda x: float(np.percentile(x, 90))
    ).reset_index()
    per_phase = series.groupby(keys + ["phasing"])["duration_s"].mean().reset_index()
    ci = summarise(per_phase, ("duration_s",), keys)
    ci = ci.rename(columns={"duration_s_mean": "mean_perphase_s", "duration_s_ci95": "mean_ci95_s"})[
        keys + ["mean_ci95_s"]
    ]
    return pooled.merge(ci, on=keys, how="left")


def run_e2(
    exp_path: Path,
    results_root: Path,
    repo_root: Path = REPO_ROOT,
    n_phasings: int | None = None,
    time_step_s: float | None = None,
    duration_s: float | None = None,
    duration_step_s: float = 1.0,
    duration_window_s: float = 7200.0,
) -> Path:
    """Run E2 (diversity, durations, min-elevation sweep, F sensitivity, 25.261 loss)."""
    sim, exp, grid_cfg, grid, sanity_grid = _load_inputs(exp_path, repo_root)
    bands = load_bands(repo_root / "configs" / "bands.yaml")
    udl_pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    pair_meta = {
        (a, b): {"link_type": "user_downlink",
                 "shared_freq_ghz": overlap(bands[a], bands[b], "user_downlink").center_ghz}
        for (a, b) in udl_pairs
    }
    p = exp.params
    n_ph = int(n_phasings if n_phasings is not None else p.get("n_phasings", 8))
    dt = float(time_step_s if time_step_s is not None else p.get("time_step_s", 10.0))
    dur = float(duration_s if duration_s is not None else p.get("duration_s", 86400.0))
    thetas = (1.0, 2.0, 5.0)
    min_el = float(grid_cfg.min_elevation_deg)
    chunk = int(p.get("chunk", 64))
    device = device_str()

    run_dir = new_run_dir(
        results_root, exp.name, exp.seed,
        config_paths=[exp_path, repo_root / "configs" / "constellations.yaml",
                      repo_root / "configs" / "bands.yaml",
                      repo_root / "configs" / exp.stations],
        repo_root=repo_root,
    )
    _band_overlap_table(exp.constellations, bands).to_csv(
        run_dir / "summary_band_overlap.csv", index=False
    )
    checks = run_sanity(sim, exp, grid, sanity_grid, dt, dur, min_el, device, chunk=chunk)
    _write_json(run_dir, "sanity.json", checks)
    ok, reasons = _sanity_ok(checks)
    if not ok:
        _write_json(run_dir, "status.json", {"experiment": exp.name, "status": "sanity_failed",
                                             "reasons": reasons})
        raise SystemExit(f"[e2] sanity gate FAILED: {reasons}")

    times = np.arange(0.0, dur, dt)
    events, pool, _s, sanity_material = screen_phasings(
        sim, exp.constellations, grid, times, device, exp.seed, n_ph, thetas, min_el, chunk,
        pairs=udl_pairs, pair_meta=pair_meta,
    )
    events.to_parquet(run_dir / "raw_pair_events.parquet", index=False)
    pool.to_parquet(run_dir / "raw_pool.parquet", index=False)
    _sub_latitude_table(sanity_material, sim).to_csv(
        run_dir / "summary_sub_latitude_density.csv", index=False
    )

    # Diversity: unresolvable fraction (of time and conditional on an event).  The
    # 25.261(c) band-split loss is half the time the band is split: with diversity
    # only unresolvable events split it, without diversity every event does.
    un = summarise(events, ("event_frac", "unres_frac", "service_a_frac", "service_b_frac"),
                   ["system_a", "system_b", "link_type", "shared_freq_ghz",
                    "variant", "theta_deg", "latitude_deg"])
    un["unres_conditional"] = np.where(
        un["event_frac_mean"] > 0, un["unres_frac_mean"] / un["event_frac_mean"], np.nan
    )
    un["band_split_loss_with_diversity"] = BAND_SPLIT_FRACTION * un["unres_frac_mean"]
    un["band_split_loss_no_diversity"] = BAND_SPLIT_FRACTION * un["event_frac_mean"]
    # Condition on the victim being in service (events already require it, so the
    # ratio is exact: event_frac / service_a_frac = P(event | victim served)).
    serv_a = un["service_a_frac_mean"].to_numpy()
    un["event_when_served"] = np.where(serv_a > 0, un["event_frac_mean"] / serv_a, np.nan)
    un["unres_when_served"] = np.where(serv_a > 0, un["unres_frac_mean"] / serv_a, np.nan)
    pool_stats = _service_pool_summary(pool).rename(columns={"system": "system_a"})
    un = un.merge(
        pool_stats[["system_a", "latitude_deg", "pool_mean_mean", "pool_p10_mean",
                    "served_frac_mean"]],
        on=["system_a", "latitude_deg"], how="left",
    )
    un.to_csv(run_dir / "summary_unresolvable.csv", index=False)

    # Pool / visible summary and analytic check.
    pool_s = summarise(pool.drop(columns=["mode"]), ("pool_mean", "shell_visible_mean"),
                       ["system", "shell_or_all", "latitude_deg"])
    pool_s.to_csv(run_dir / "summary_pool.csv", index=False)
    _service_pool_summary(pool).to_csv(run_dir / "summary_service_pool.csv", index=False)
    rows = []
    for n, specs in sanity_material["shell_specs"].items():
        for spec in specs:
            for la in sorted(set(grid.latitudes)):
                ana = analytic_visible_count(spec["altitude_km"], spec["inclination_deg"],
                                             spec["planes"], spec["sats_per_plane"], float(la),
                                             min_el, sim.earth.radius_km)
                sim_v = pool[(pool["system"] == n) & (pool["shell_or_all"] == spec["name"]) &
                             (pool["latitude_deg"] == la)]["shell_visible_mean"].mean()
                rows.append({"system": n, "shell": spec["name"], "latitude_deg": float(la),
                             "sim_mean": float(sim_v), "analytic": float(ana)})
    pd.DataFrame(rows).to_csv(run_dir / "summary_shell_visible_analytic.csv", index=False)

    # Min-elevation sweep (20 and 35 deg) re-using the same phasing seeds.
    sweep_frames = []
    for me in (20.0, 35.0):
        ev_me, _, _s2, _m = screen_phasings(
            sim, exp.constellations, grid, times, device, exp.seed, n_ph, thetas, me, chunk,
            pairs=udl_pairs, pair_meta=pair_meta,
        )
        s = summarise(ev_me, ("event_frac", "unres_frac"),
                      ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
        s["min_elevation_deg"] = me
        sweep_frames.append(s)
    pd.concat(sweep_frames).to_csv(run_dir / "summary_min_elevation.csv", index=False)

    # Durations at fine (1 s) resolution over a short window.
    dt_times = np.arange(0.0, duration_window_s, duration_step_s)
    _e, _p, series, _m = screen_phasings(
        sim, exp.constellations, grid, dt_times, device, exp.seed, n_ph, thetas, min_el, chunk,
        collect_series=True, pairs=udl_pairs, pair_meta=pair_meta,
    )
    if not series.empty:
        series.to_parquet(run_dir / "raw_durations.parquet", index=False)
        _duration_summary(series).to_csv(run_dir / "summary_durations.csv", index=False)

    # Walker F sensitivity on the Starlink 53 deg shells only.
    sens_frames = []
    base = summarise(events, ("event_frac",),
                     ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
    base["f"] = 1
    sens_frames.append(base)
    for f_val in (0, 36):
        sim_f = _starlink_f_variant(sim, f_val)
        ev_f, _, _s3, _m = screen_phasings(
            sim_f, exp.constellations, grid, times, device, exp.seed, n_ph, thetas, min_el, chunk,
            pairs=udl_pairs, pair_meta=pair_meta,
        )
        s = summarise(ev_f, ("event_frac",),
                      ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
        s["f"] = f_val
        sens_frames.append(s)
    pd.concat(sens_frames).to_csv(run_dir / "summary_sensitivity_f.csv", index=False)

    # OneWeb shell-size sensitivity: shipped 18x36 (FCC 17-77) vs 12x49 (ASSUMED).
    ow_frames = []
    base_ow = summarise(events, ("event_frac", "unres_frac"),
                        ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
    base_ow["oneweb_config"] = "18x36"
    ow_frames.append(base_ow)
    sim_ow = _oneweb_shell_variant(sim, 12, 49)
    ev_ow, _, _s4, _m = screen_phasings(
        sim_ow, exp.constellations, grid, times, device, exp.seed, n_ph, thetas, min_el, chunk,
        pairs=udl_pairs, pair_meta=pair_meta,
    )
    s_ow = summarise(ev_ow, ("event_frac", "unres_frac"),
                     ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
    s_ow["oneweb_config"] = "12x49"
    ow_frames.append(s_ow)
    pd.concat(ow_frames).to_csv(run_dir / "summary_sensitivity_oneweb.csv", index=False)

    _write_json(run_dir, "status.json",
                {"experiment": exp.name, "status": "ok", "n_phasings": n_ph, "time_step_s": dt,
                 "duration_s": dur, "duration_step_s": duration_step_s,
                 "duration_window_s": duration_window_s, "thetas_deg": list(thetas),
                 "min_elevation_deg": min_el, "device": device,
                 "user_downlink_pairs": [list(q) for q in udl_pairs]})
    return run_dir


def run_e3(
    exp_path: Path,
    results_root: Path,
    repo_root: Path = REPO_ROOT,
    n_phasings: int | None = None,
    duration_s: float | None = None,
) -> Path:
    """Run E3 (capacity-limited satellite diversity over regional cell grids)."""
    sim, exp, grid_cfg, grid, sanity_grid = _load_inputs(exp_path, repo_root)
    bands = load_bands(repo_root / "configs" / "bands.yaml")
    lb = load_link_budget(repo_root / "configs" / "link_budget.yaml")
    udl_pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    p = exp.params
    n_ph = int(n_phasings if n_phasings is not None else p.get("n_phasings", 8))
    dt = float(p.get("time_step_s", 60.0))
    dur = float(duration_s if duration_s is not None else p.get("duration_s", 86400.0))
    min_el = float(grid_cfg.min_elevation_deg)
    chunk = int(p.get("chunk", 8))
    region_size = float(p.get("region_size_km", 1000.0))
    spacing = float(p.get("cell_spacing_km", 20.0))
    lat_centers = [float(x) for x in p.get("lat_center_deg", [0, 10, 20, 30, 40, 50, 60])]
    meridians = [float(x) for x in p.get("meridians_deg", [0, 90, 180, 270])]
    n_beams = [int(x) for x in p.get("n_beam_sweep", [8, 16, 32])]
    n_rank = int(p.get("n_rank", 6))
    device = device_str()
    radius = sim.earth.radius_km
    jd = epoch_jd(sim)

    run_dir = new_run_dir(
        results_root, exp.name, exp.seed,
        config_paths=[exp_path, repo_root / "configs" / "constellations.yaml",
                      repo_root / "configs" / "bands.yaml",
                      repo_root / "configs" / "link_budget.yaml",
                      repo_root / "configs" / exp.stations],
        repo_root=repo_root,
    )
    _band_overlap_table(exp.constellations, bands).to_csv(
        run_dir / "summary_band_overlap.csv", index=False
    )
    checks = run_sanity(sim, exp, grid, sanity_grid, dt, dur, min_el, device, chunk=64)
    _write_json(run_dir, "sanity.json", checks)
    ok, reasons = _sanity_ok(checks)
    if not ok:
        _write_json(run_dir, "status.json", {"experiment": exp.name, "status": "sanity_failed",
                                             "reasons": reasons})
        raise SystemExit(f"[e3] sanity gate FAILED: {reasons}")

    times = np.arange(0.0, dur, dt)
    # Four meridians per latitude, concatenated into one cell grid (Phase 3a).
    cells = {}
    for la in lat_centers:
        ecefs, ups = [], []
        for lon in meridians:
            ce, cu = regional_grid(la, lon, region_size, spacing, radius)
            ecefs.append(ce)
            ups.append(cu)
        ecef = np.concatenate(ecefs, axis=0)
        up = np.concatenate(ups, axis=0)
        cells[la] = (torch.as_tensor(ecef, dtype=torch.float64, device=device),
                     torch.as_tensor(up, dtype=torch.float64, device=device))
    rows = []
    for pi in range(n_ph):
        rng = np.random.default_rng(exp.seed + 137 * pi)
        systems = _phase_systems(sim, exp.constellations, rng, radius)
        for a, b in udl_pairs:
            meta = {"link_type": "user_downlink",
                    "shared_freq_ghz": overlap(bands[a], bands[b], "user_downlink").center_ghz}
            freq = float(meta["shared_freq_ghz"])
            rx_peak = terminal_peak_gain_dbi(lb.diameter_m, freq, lb.aperture_efficiency)
            victim_beam = systems[a].shells[0].beam
            interf_beam = systems[b].shells[0].beam
            for la in lat_centers:
                ce, cu = cells[la]
                for nb in n_beams:
                    r = capacity_limited_unresolvable(
                        systems[a], systems[b], ce, cu, times, jd, radius, nb,
                        victim_beam=victim_beam, interf_beam=interf_beam, freq_ghz=freq,
                        rx_peak_gain_dbi=rx_peak, rx_diameter_m=lb.diameter_m,
                        min_elevation_deg=min_el, device=device, chunk=chunk, n_rank=n_rank,
                        pattern=lb.pattern,
                    )
                    rows.append(
                        {"phasing": pi, "system_a": a, "system_b": b,
                         "link_type": meta["link_type"], "shared_freq_ghz": meta["shared_freq_ghz"],
                         "n_beam": nb, "latitude_deg": float(la), **r}
                    )
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "raw_capacity_diversity.parquet", index=False)
    summ = summarise(
        df, ("event_cell_frac", "unres_cell_frac", "served_frac",
             "event_per_served", "unres_per_served"),
        ["system_a", "system_b", "link_type", "shared_freq_ghz", "n_beam", "latitude_deg"],
    )
    summ.to_csv(run_dir / "summary_capacity_diversity.csv", index=False)

    _write_json(run_dir, "status.json",
                {"experiment": exp.name, "status": "ok", "n_phasings": n_ph, "time_step_s": dt,
                 "duration_s": dur, "min_elevation_deg": min_el, "device": device,
                 "region_size_km": region_size, "cell_spacing_km": spacing,
                 "lat_center_deg": lat_centers, "meridians_deg": meridians,
                 "n_beam_sweep": n_beams, "n_rank": n_rank, "dtt_threshold": 0.06,
                 "user_terminal_diameter_m": lb.diameter_m,
                 "user_downlink_pairs": [list(q) for q in udl_pairs]})
    return run_dir


def run_e4(
    exp_path: Path,
    results_root: Path,
    repo_root: Path = REPO_ROOT,
    n_phasings: int | None = None,
    time_step_s: float | None = None,
    duration_s: float | None = None,
) -> Path:
    """Run E4 (47 CFR 25.261(d) Delta T/T screening + throughput degradation).

    The geometric theta proxy stays in E1/E2 as the secondary comparison; this
    experiment uses the regulatory power criterion for every co-frequency
    user-downlink pair and sweeps the ASSUMED terminal aperture.
    """
    sim, exp, grid_cfg, grid, sanity_grid = _load_inputs(exp_path, repo_root)
    bands = load_bands(repo_root / "configs" / "bands.yaml")
    lb = load_link_budget(repo_root / "configs" / "link_budget.yaml")
    udl_pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    p = exp.params
    n_ph = int(n_phasings if n_phasings is not None else p.get("n_phasings", 8))
    dt = float(time_step_s if time_step_s is not None else p.get("time_step_s", 10.0))
    dur = float(duration_s if duration_s is not None else p.get("duration_s", 86400.0))
    min_el = float(grid_cfg.min_elevation_deg)
    chunk = int(p.get("chunk", 64))
    device = device_str()
    radius = sim.earth.radius_km
    jd = epoch_jd(sim)

    run_dir = new_run_dir(
        results_root, exp.name, exp.seed,
        config_paths=[exp_path, repo_root / "configs" / "constellations.yaml",
                      repo_root / "configs" / "bands.yaml",
                      repo_root / "configs" / "link_budget.yaml",
                      repo_root / "configs" / exp.stations],
        repo_root=repo_root,
    )
    _band_overlap_table(exp.constellations, bands).to_csv(
        run_dir / "summary_band_overlap.csv", index=False
    )
    checks = run_sanity(sim, exp, grid, sanity_grid, dt, dur, min_el, device, chunk=chunk)
    _write_json(run_dir, "sanity.json", checks)
    ok, reasons = _sanity_ok(checks)
    if not ok:
        _write_json(run_dir, "status.json", {"experiment": exp.name, "status": "sanity_failed",
                                             "reasons": reasons})
        raise SystemExit(f"[e4] sanity gate FAILED: {reasons}")

    times = np.arange(0.0, dur, dt)
    cfg = ScreenConfig(min_elevation_deg=min_el, earth_radius_km=radius, epoch_jd=jd,
                       thetas_deg=(2.0,), chunk=chunk, device=device)
    rows = []
    for pi in range(n_ph):
        rng = np.random.default_rng(exp.seed + 137 * pi)
        systems = _phase_systems(sim, exp.constellations, rng, radius)
        vis = {n: system_visibility(systems[n], grid.ecef, grid.up, times, cfg,
                                    np.arange(-90.0, 90.0 + 1e-9, 5.0))
               for n in exp.constellations}
        for a, b in udl_pairs:
            ov = overlap(bands[a], bands[b], "user_downlink")
            freq = float(ov.center_ghz)
            victim_beam = systems[a].shells[0].beam
            interf_beam = systems[b].shells[0].beam
            for diam in lb.diameter_sweep_m:
                rx_peak = terminal_peak_gain_dbi(diam, freq, lb.aperture_efficiency)
                res = screen_pair_dtt(
                    systems[a], systems[b], vis[a]["max_visible"], vis[b]["max_visible"],
                    grid.ecef, grid.up, times, victim_beam, interf_beam, freq, rx_peak, diam,
                    lb.implementation_margin_db, min_el, radius, jd, device=device, chunk=chunk,
                    pattern=lb.pattern,
                )
                for vi in (0, 1):
                    unavail_int = (res["unavail_nodiv_worst_frac"] if vi == 0
                                   else res["unavail_nodiv_colo_frac"])
                    for gi in range(grid.ecef.shape[0]):
                        r0 = res["r0_mean"][gi]
                        r1i = res["r_int_mean"][vi][gi]
                        r1d = res["r_div_mean"][vi][gi]
                        rows.append(
                            {"phasing": pi, "system_a": a, "system_b": b,
                             "link_type": "user_downlink", "shared_freq_ghz": freq,
                             "diameter_m": float(diam), "rx_peak_gain_dbi": rx_peak,
                             "variant": vi, "latitude_deg": float(grid.latitudes[gi]),
                             "longitude_deg": float(grid.longitudes[gi]),
                             "event_frac": float(res["event_frac"][vi, gi]),
                             "unres_frac": float(res["unres_frac"][vi, gi]),
                             "served_frac": float(res["served_frac"][gi]),
                             "cn0_mean": float(res["cn0_mean"][gi]),
                             "dtt_mean": float(res["dtt_colo_mean"][gi] if vi == 1
                                               else res["dtt_worst_mean"][gi]),
                             "r0_mean": float(r0), "r_int_mean": float(r1i),
                             "r_div_mean": float(r1d),
                             "degradation_nodiv": 1.0 - float(r1i) / float(r0) if r0 else np.nan,
                             "degradation_div": 1.0 - float(r1d) / float(r0) if r0 else np.nan,
                             "unavail0_frac": float(res["unavail0_frac"][gi]),
                             "unavail_int_frac": float(unavail_int[gi])}
                        )
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "raw_dtt.parquet", index=False)
    keys = ["system_a", "system_b", "link_type", "shared_freq_ghz", "diameter_m", "variant",
            "latitude_deg"]
    summ = summarise(
        df, ("event_frac", "unres_frac", "served_frac", "cn0_mean", "dtt_mean",
             "degradation_nodiv", "degradation_div", "unavail0_frac", "unavail_int_frac"),
        keys,
    )
    summ.to_csv(run_dir / "summary_dtt.csv", index=False)

    thr = summ[["system_a", "system_b", "diameter_m", "variant", "latitude_deg",
                "degradation_nodiv_mean", "degradation_nodiv_ci95",
                "degradation_div_mean", "degradation_div_ci95",
                "unavail0_frac_mean", "unavail_int_frac_mean"]]
    thr.to_csv(run_dir / "summary_throughput.csv", index=False)

    _write_json(run_dir, "status.json",
                {"experiment": exp.name, "status": "ok", "n_phasings": n_ph, "time_step_s": dt,
                 "duration_s": dur, "min_elevation_deg": min_el, "device": device,
                 "dtt_threshold": 0.06, "diameter_sweep_m": lb.diameter_sweep_m,
                 "pattern": lb.pattern, "implementation_margin_db": lb.implementation_margin_db,
                 "user_downlink_pairs": [list(q) for q in udl_pairs]})
    return run_dir


def run_e5(
    exp_path: Path,
    results_root: Path,
    repo_root: Path = REPO_ROOT,
    n_start_times: int | None = None,
    time_step_s: float | None = None,
    duration_s: float | None = None,
) -> Path:
    """Run E5 (Phase 3b: as-deployed CelesTrak TLE scenario).

    Replaces the modelled Walker pattern + random phasing with a real snapshot:
    every satellite's TLE mean elements are advanced to a common epoch and the
    analytic J2 propagator is run from N independent *start times* (AGENTS.md
    rule 5).  Event-rate (geometric theta) and Delta T/T (power) screens are
    produced on the same station grid as E1/E4.
    """
    from datetime import datetime, timezone

    sim, exp, grid_cfg, grid, sanity_grid = _load_inputs(exp_path, repo_root)
    bands = load_bands(repo_root / "configs" / "bands.yaml")
    lb = load_link_budget(repo_root / "configs" / "link_budget.yaml")
    udl_pairs = co_frequency_pairs(exp.constellations, bands, "user_downlink")
    p = exp.params
    n_st = int(n_start_times if n_start_times is not None else p.get("n_start_times", 8))
    dt = float(time_step_s if time_step_s is not None else p.get("time_step_s", 10.0))
    dur = float(duration_s if duration_s is not None else p.get("duration_s", 86400.0))
    offset = float(p.get("start_offset_s", 18000.0))
    thetas = tuple(float(t) for t in p.get("thetas_deg", [1.0, 2.0, 5.0]))
    chunk = int(p.get("chunk", 64))
    groups = dict(p.get("tle_groups", {}))
    cache_dir = repo_root / str(p.get("tle_cache_dir", "data/raw/tle"))
    min_el = float(grid_cfg.min_elevation_deg)
    device = device_str()
    radius = sim.earth.radius_km
    lat_bins = np.arange(-90.0, 90.0 + 1e-9, 5.0)

    run_dir = new_run_dir(
        results_root, exp.name, exp.seed,
        config_paths=[exp_path, repo_root / "configs" / "constellations.yaml",
                      repo_root / "configs" / "bands.yaml",
                      repo_root / "configs" / "link_budget.yaml",
                      repo_root / "configs" / exp.stations],
        repo_root=repo_root,
    )
    _band_overlap_table(exp.constellations, bands).to_csv(
        run_dir / "summary_band_overlap.csv", index=False
    )

    fetched = str(datetime.now(timezone.utc).isoformat())
    records: dict[str, list] = {}
    tle_meta: dict = {"fetched_utc": fetched, "cache_dir": str(cache_dir), "groups": {}}
    for n in exp.constellations:
        grp = str(groups.get(n, n.lower()))
        text = fetch_group(grp, cache_dir)
        recs = parse_tle(text)
        records[n] = recs
        tle_meta["groups"][n] = {
            "group": grp, "url": celestrak_url(grp), "n_sats": len(recs),
            "epoch_jd_min": float(min(r.epoch_jd for r in recs)),
            "epoch_jd_max": float(max(r.epoch_jd for r in recs)),
        }
    epoch = float(max(common_epoch_jd(records[n]) for n in exp.constellations))
    tle_meta["common_epoch_jd"] = epoch
    _write_json(run_dir, "tle_meta.json", tle_meta)

    systems = {
        n: build_as_deployed_system(sim.systems[n], records[n], radius)
        for n in exp.constellations
    }
    shell_rows = []
    for n, model in systems.items():
        for si, sh in enumerate(model.shells):
            shell_rows.append(
                {"system": n, "shell": sh.name, "n_sats": int((model.shell_index == si).sum()),
                 "altitude_km": sh.altitude_km, "inclination_deg": sh.inclination_deg,
                 "planes": sh.planes, "sats_per_plane": sh.sats_per_plane,
                 "source": sh.source}
            )
    pd.DataFrame(shell_rows).to_csv(run_dir / "summary_as_deployed_shells.csv", index=False)

    cfg = ScreenConfig(min_elevation_deg=min_el, earth_radius_km=radius, epoch_jd=epoch,
                       thetas_deg=thetas, chunk=chunk, device=device)
    event_rows: list[dict] = []
    dtt_rows: list[dict] = []
    for k in range(n_st):
        times = k * offset + np.arange(0.0, dur, dt)
        vis = {n: system_visibility(systems[n], grid.ecef, grid.up, times, cfg, lat_bins)
               for n in exp.constellations}
        for a, b in udl_pairs:
            freq = float(overlap(bands[a], bands[b], "user_downlink").center_ghz)
            res = screen_pair(systems[a], systems[b], vis[a]["max_visible"],
                              vis[b]["max_visible"], grid.ecef, grid.up, times, cfg)
            for ti, th in enumerate(thetas):
                for vi in (0, 1):
                    for g in range(grid.ecef.shape[0]):
                        event_rows.append(
                            {"phasing": k, "system_a": a, "system_b": b, "variant": vi,
                             "theta_deg": th, "latitude_deg": float(grid.latitudes[g]),
                             "longitude_deg": float(grid.longitudes[g]),
                             "event_frac": float(res["event_frac"][ti, vi, g]),
                             "unres_frac": float(res["unres_frac"][ti, vi, g]),
                             "covisible_frac": float(res["covisible_frac"][g]),
                             "service_a_frac": float(vis[a]["served_frac"][g]),
                             "service_b_frac": float(vis[b]["served_frac"][g])}
                        )
            victim_beam = systems[a].shells[0].beam
            interf_beam = systems[b].shells[0].beam
            for diam in lb.diameter_sweep_m:
                rx_peak = terminal_peak_gain_dbi(diam, freq, lb.aperture_efficiency)
                dres = screen_pair_dtt(
                    systems[a], systems[b], vis[a]["max_visible"], vis[b]["max_visible"],
                    grid.ecef, grid.up, times, victim_beam, interf_beam, freq, rx_peak, diam,
                    lb.implementation_margin_db, min_el, radius, epoch, device=device,
                    chunk=chunk, pattern=lb.pattern,
                )
                for vi in (0, 1):
                    for g in range(grid.ecef.shape[0]):
                        r0 = dres["r0_mean"][g]
                        r1i = dres["r_int_mean"][vi][g]
                        r1d = dres["r_div_mean"][vi][g]
                        dtt_rows.append(
                            {"phasing": k, "system_a": a, "system_b": b,
                             "shared_freq_ghz": freq, "diameter_m": float(diam),
                             "variant": vi, "latitude_deg": float(grid.latitudes[g]),
                             "longitude_deg": float(grid.longitudes[g]),
                             "event_frac": float(dres["event_frac"][vi, g]),
                             "unres_frac": float(dres["unres_frac"][vi, g]),
                             "cn0_mean": float(dres["cn0_mean"][g]),
                             "dtt_mean": float(dres["dtt_worst_mean"][g] if vi == 0
                                               else dres["dtt_colo_mean"][g]),
                             "degradation_nodiv": 1.0 - float(r1i) / float(r0) if r0 else np.nan,
                             "degradation_div": 1.0 - float(r1d) / float(r0) if r0 else np.nan}
                        )

    ev = pd.DataFrame(event_rows)
    ev.to_parquet(run_dir / "raw_as_deployed_events.parquet", index=False)
    ev_s = summarise(ev, ("event_frac", "unres_frac", "service_a_frac", "service_b_frac"),
                     ["system_a", "system_b", "variant", "theta_deg", "latitude_deg"])
    ev_s["event_rate_per_hour"] = ev_s["event_frac_mean"] * 3600.0 / dt
    ev_s.to_csv(run_dir / "summary_event_rate.csv", index=False)

    dtt = pd.DataFrame(dtt_rows)
    dtt.to_parquet(run_dir / "raw_as_deployed_dtt.parquet", index=False)
    dtt_s = summarise(
        dtt, ("event_frac", "unres_frac", "cn0_mean", "dtt_mean",
              "degradation_nodiv", "degradation_div"),
        ["system_a", "system_b", "diameter_m", "variant", "latitude_deg"],
    )
    dtt_s.to_csv(run_dir / "summary_dtt.csv", index=False)

    _write_json(run_dir, "status.json",
                {"experiment": exp.name, "status": "ok", "n_start_times": n_st,
                 "start_offset_s": offset, "time_step_s": dt, "duration_s": dur,
                 "thetas_deg": list(thetas), "min_elevation_deg": min_el,
                 "device": device, "common_epoch_jd": epoch,
                 "n_sats": {n: len(records[n]) for n in exp.constellations},
                 "dtt_threshold": 0.06, "diameter_sweep_m": lb.diameter_sweep_m,
                 "tle_groups": {n: str(groups.get(n, n.lower())) for n in exp.constellations},
                 "user_downlink_pairs": [list(q) for q in udl_pairs]})
    return run_dir
