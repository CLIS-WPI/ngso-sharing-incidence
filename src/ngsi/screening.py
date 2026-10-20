"""Vectorised in-line event screening and satellite-diversity resolution.

The heavy work happens on the GPU with torch in float64 (or float32, see ngsi.precision).  Time is processed in
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


def system_visibility(
    system: SystemModel,
    station_ecef: torch.Tensor,
    up: torch.Tensor,
    times: np.ndarray,
    config: ScreenConfig,
    lat_bins: np.ndarray,
) -> dict:
    """Visibility-only pass for one system (independent of any pair).

    Returns ``max_visible`` (int, used to size the exact ``topk`` in
    :func:`screen_pair`), ``shell_vis_sum`` of shape ``[n_shells, G]``, per-station
    ``pool_sum``/``pool_sqsum`` over time, ``n_samples`` and the sub-satellite
    latitude histogram.
    """
    dev = config.device
    g = station_ecef.shape[0]
    t_total = len(times)
    max_vis = 1
    pool_bins = 128
    pool_sum = torch.zeros(g, dtype=torch.float64, device=dev)
    pool_sqsum = torch.zeros(g, dtype=torch.float64, device=dev)
    served_count = torch.zeros(g, dtype=torch.float64, device=dev)
    pool_hist = torch.zeros(g, pool_bins, dtype=torch.float64, device=dev)
    shell_vis = torch.zeros(system.n_shells, g, dtype=torch.float64, device=dev)
    shell_hist = np.zeros((system.n_shells, len(lat_bins) - 1), dtype=np.int64)
    hist = np.zeros(len(lat_bins) - 1, dtype=np.int64)
    idx = torch.as_tensor(system.shell_index, device=dev)

    for s, e in _chunk_ranges(t_total, config.chunk):
        tc = times[s:e]
        pos = propagate_ecef(system.elements, tc, config.epoch_jd, config.earth_radius_km, dev)
        look, _ = unit_look_vectors(pos, station_ecef)
        vis = elevation_deg_batch(look, up) >= config.min_elevation_deg
        count = vis.sum(-1)  # [Tc, G]
        max_vis = max(max_vis, int(count.max()))
        pool_sum += count.sum(0)
        pool_sqsum += (count.to(torch.float64) ** 2).sum(0)
        served_count += (count > 0).sum(0).to(torch.float64)
        pool_hist += torch.nn.functional.one_hot(
            count.to(torch.long).clamp(max=pool_bins - 1), pool_bins
        ).sum(0).to(torch.float64)
        sat_lat = _sat_lat_deg(pos)  # [Tc, S]
        for i in range(system.n_shells):
            shell_vis[i] += vis[:, :, idx == i].sum(dim=(0, 2))
            shell_hist[i] += np.histogram(sat_lat[:, idx == i].reshape(-1).cpu().numpy(), bins=lat_bins)[0]
        hist += np.histogram(sat_lat.reshape(-1).cpu().numpy(), bins=lat_bins)[0]

    return {
        "max_visible": max_vis,
        "pool_sum": pool_sum.cpu().numpy(),
        "pool_sqsum": pool_sqsum.cpu().numpy(),
        "served_frac": (served_count / t_total).cpu().numpy(),
        "pool_hist": pool_hist.cpu().numpy(),
        "n_samples": t_total,
        "shell_vis_sum": shell_vis.cpu().numpy(),
        "sub_lat_hist": hist,
        "shell_lat_hist": shell_hist,
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
    covisible_count = torch.zeros(g, dtype=torch.float64, device=dev)
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
        covisible_count += (vis_a.any(-1) & vis_b.any(-1)).sum(0).to(torch.float64)

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
        "covisible_frac": (covisible_count / t_total).cpu().numpy(),
        "event_series": event_series.cpu().numpy() if collect_series else None,
        "unres_series": unres_series.cpu().numpy() if collect_series else None,
        "peak_mem_bytes": torch.cuda.max_memory_allocated() if dev == "cuda" else 0,
    }
