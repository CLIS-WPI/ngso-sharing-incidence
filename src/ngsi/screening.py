"""Vectorised in-line event screening and satellite-diversity resolution.

The heavy work happens on the GPU with torch in float64.  Time is processed in
chunks so peak memory stays bounded; within a chunk every reduction is a tensor
op (``topk``, ``min``, ``sum``) -- there are **no Python loops over satellites or
time steps** (only over the handful of shells, thetas and interferer variants).

For an ordered pair of *different* systems ``(A victim, B interferer)`` and each
station/time sample:

* A's **serving** satellite is the highest-elevation visible A satellite.
* The interferer set is either
  (i) *worst case* -- every visible B satellite, or
  (ii) *co-located* -- B's serving satellite at the same site.
* An event fires when the angle at the station between the A serving satellite and
  some interferer is below ``theta``.
* The event is **resolvable** when some *other* visible A satellite is at least
  ``theta`` away from every interferer (same variant); otherwise it is
  **unresolvable**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from .propagation import propagate_ecef
from .systems import SystemModel

VARIANT_NAMES = ("worst_case", "co_located")


@dataclass
class ScreenConfig:
    """Runtime knobs for :func:`screen_pair`."""

    min_elevation_deg: float
    earth_radius_km: float
    epoch_jd: float
    thetas_deg: tuple[float, ...] = (1.0, 2.0, 5.0)
    chunk: int = 64
    device: str = "cuda"


def _chunk_ranges(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _sat_lat_deg(pos: torch.Tensor) -> torch.Tensor:
    """Geocentric sub-satellite latitude (deg) from ECEF positions ``[..., 3]``."""
    r = torch.linalg.norm(pos, dim=-1)
    return torch.rad2deg(torch.asin(torch.clamp(pos[..., 2] / r, -1.0, 1.0)))


def visibility_stats(
    a: SystemModel,
    b: SystemModel,
    station_ecef: torch.Tensor,
    up: torch.Tensor,
    times: np.ndarray,
    config: ScreenConfig,
    lat_bins: np.ndarray,
) -> dict:
    """Visibility-only pass: max visible counts, per-shell means, pool stats, lat hist.

    Returns a dict with ``max_vis_a``/``max_vis_b`` (ints, used to size the exact
    ``topk`` in :func:`screen_pair`), ``pool_sum``/``pool_sqsum``/``n_samples`` per
    station, ``shell_vis_sum_a``/``shell_vis_sum_b`` of shape ``[n_shells, G]``, and
    sub-satellite latitude histograms for both systems.
    """
    dev = config.device
    g = station_ecef.shape[0]
    t_total = len(times)
    max_a = max_b = 1
    pool_sum = torch.zeros(g, dtype=torch.float64, device=dev)
    pool_sqsum = torch.zeros(g, dtype=torch.float64, device=dev)
    shell_a = torch.zeros(a.n_shells, g, dtype=torch.float64, device=dev)
    shell_b = torch.zeros(b.n_shells, g, dtype=torch.float64, device=dev)
    hist_a = np.zeros(len(lat_bins) - 1, dtype=np.int64)
    hist_b = np.zeros(len(lat_bins) - 1, dtype=np.int64)
    idx_a = torch.as_tensor(a.shell_index, device=dev)
    idx_b = torch.as_tensor(b.shell_index, device=dev)

    for s, e in _chunk_ranges(t_total, config.chunk):
        tc = times[s:e]
        pa = propagate_ecef(a.elements, tc, config.epoch_jd, config.earth_radius_km, dev)
        pb = propagate_ecef(b.elements, tc, config.epoch_jd, config.earth_radius_km, dev)
        ua, _ = unit_look_vectors(pa, station_ecef)
        ub, _ = unit_look_vectors(pb, station_ecef)
        vis_a = elevation_deg_batch(ua, up) >= config.min_elevation_deg
        vis_b = elevation_deg_batch(ub, up) >= config.min_elevation_deg
        count_a = vis_a.sum(-1)  # [Tc, G]
        count_b = vis_b.sum(-1)
        max_a = max(max_a, int(count_a.max()))
        max_b = max(max_b, int(count_b.max()))
        pool_sum += count_a.sum(0)
        pool_sqsum += (count_a.to(torch.float64) ** 2).sum(0)
        for i in range(a.n_shells):
            shell_a[i] += vis_a[:, :, idx_a == i].sum(dim=(0, 2))
        for i in range(b.n_shells):
            shell_b[i] += vis_b[:, :, idx_b == i].sum(dim=(0, 2))
        hist_a += np.histogram(_sat_lat_deg(pa).reshape(-1).cpu().numpy(), bins=lat_bins)[0]
        hist_b += np.histogram(_sat_lat_deg(pb).reshape(-1).cpu().numpy(), bins=lat_bins)[0]

    return {
        "max_vis_a": max_a,
        "max_vis_b": max_b,
        "pool_sum": pool_sum.cpu().numpy(),
        "pool_sqsum": pool_sqsum.cpu().numpy(),
        "n_samples": t_total,
        "shell_vis_sum_a": shell_a.cpu().numpy(),
        "shell_vis_sum_b": shell_b.cpu().numpy(),
        "sub_lat_hist_a": hist_a,
        "sub_lat_hist_b": hist_b,
    }


def screen_pair(
    a: SystemModel,
    b: SystemModel,
    k_a: int,
    k_b: int,
    station_ecef: torch.Tensor,
    up: torch.Tensor,
    times: np.ndarray,
    config: ScreenConfig,
    collect_series: bool = False,
) -> dict:
    """Screen one ordered system pair.

    Returns ``event_frac`` and ``unres_frac`` of shape ``[n_theta, 2, G]`` (fraction
    of time samples), ``n_samples`` per station, and -- when ``collect_series`` --
    ``event_series``/``unres_series`` of shape ``[n_theta, 2, G, T]`` for run-length
    (duration) analysis.
    """
    dev = config.device
    g = station_ecef.shape[0]
    t_total = len(times)
    n_theta = len(config.thetas_deg)
    thetas = torch.tensor(config.thetas_deg, dtype=torch.float64, device=dev).reshape(-1, 1, 1)
    k_a = min(int(k_a), a.n_sats)
    k_b = min(int(k_b), b.n_sats)
    eye_a = torch.arange(k_a, device=dev)

    event_count = torch.zeros(n_theta, 2, g, dtype=torch.float64, device=dev)
    unres_count = torch.zeros(n_theta, 2, g, dtype=torch.float64, device=dev)
    if collect_series:
        event_series = torch.zeros(n_theta, 2, g, t_total, dtype=torch.bool, device=dev)
        unres_series = torch.zeros(n_theta, 2, g, t_total, dtype=torch.bool, device=dev)
    else:
        event_series = unres_series = None

    for s, e in _chunk_ranges(t_total, config.chunk):
        tc = times[s:e]
        pa = propagate_ecef(a.elements, tc, config.epoch_jd, config.earth_radius_km, dev)
        pb = propagate_ecef(b.elements, tc, config.epoch_jd, config.earth_radius_km, dev)
        ua, _ = unit_look_vectors(pa, station_ecef)  # [Tc,G,Sa,3]
        ub, _ = unit_look_vectors(pb, station_ecef)  # [Tc,G,Sb,3]
        elev_a = elevation_deg_batch(ua, up)
        elev_b = elevation_deg_batch(ub, up)
        vis_a = elev_a >= config.min_elevation_deg
        vis_b = elev_b >= config.min_elevation_deg

        va, ia = torch.topk(
            elev_a.masked_fill(~vis_a, float("-inf")), k_a, dim=-1
        )  # [Tc,G,K]
        vb, ib = torch.topk(elev_b.masked_fill(~vis_b, float("-inf")), k_b, dim=-1)
        vis_ka = va >= config.min_elevation_deg  # [Tc,G,K]
        vis_kb = vb >= config.min_elevation_deg

        ua_k = torch.gather(ua, 2, ia.unsqueeze(-1).expand(-1, -1, -1, 3))  # [Tc,G,K,3]
        ub_k = torch.gather(ub, 2, ib.unsqueeze(-1).expand(-1, -1, -1, 3))

        serving_a = ua_k[:, :, 0, :]  # [Tc,G,3]
        serving_b = ub_k[:, :, 0, :]
        has_serving_a = vis_ka[:, :, 0]  # [Tc,G]
        any_vis_b = vis_kb.any(-1)
        has_serving_b = vis_kb[:, :, 0]

        # Variant 0 (worst case): min separation from A serving to every visible B.
        sep0 = angular_separation_batch(serving_a.unsqueeze(2), ub_k)  # [Tc,G,K_b]
        min_sep0 = sep0.masked_fill(~vis_kb, float("inf")).min(-1).values  # [Tc,G]
        # Variant 1 (co-located): separation from A serving to B serving.
        sep1 = angular_separation_batch(serving_a, serving_b)  # [Tc,G]

        # Diversity: per A candidate, min separation over the interferer set.
        sep_all = angular_separation_batch(ua_k.unsqueeze(3), ub_k.unsqueeze(2))  # [Tc,G,Ka,Kb]
        min_a0 = sep_all.masked_fill(~vis_kb.unsqueeze(2), float("inf")).min(-1).values  # [Tc,G,Ka]
        min_a1 = angular_separation_batch(ua_k, serving_b.unsqueeze(2))  # [Tc,G,Ka]
        cand = vis_ka & (eye_a.reshape(1, 1, -1) != 0)  # exclude serving, require visible

        min_seps = {0: min_a0, 1: min_a1}
        for vi in (0, 1):
            sep_serving = (min_sep0, sep1)[vi]
            event = (sep_serving < thetas) & has_serving_a.unsqueeze(0)
            event = event & (any_vis_b.unsqueeze(0) if vi == 0 else has_serving_b.unsqueeze(0))
            clear = min_seps[vi] >= thetas.reshape(-1, 1, 1, 1)
            resolvable = (clear & cand).any(-1)  # [n_theta,Tc,G]
            event_ = event  # [n_theta,Tc,G]
            unres = event_ & ~resolvable
            event_count[:, vi, :] += event_.sum(1).to(torch.float64)
            unres_count[:, vi, :] += unres.sum(1).to(torch.float64)
            if collect_series:
                event_series[:, vi, :, s:e] = event_.permute(0, 2, 1)
                unres_series[:, vi, :, s:e] = unres.permute(0, 2, 1)

    return {
        "event_frac": (event_count / t_total).cpu().numpy(),
        "unres_frac": (unres_count / t_total).cpu().numpy(),
        "n_samples": t_total,
        "event_series": event_series.cpu().numpy() if collect_series else None,
        "unres_series": unres_series.cpu().numpy() if collect_series else None,
        "peak_mem_bytes": torch.cuda.max_memory_allocated() if dev == "cuda" else 0,
    }
