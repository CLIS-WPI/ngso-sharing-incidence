"""Capacity-limited satellite diversity over a regional cell grid.

The E1/E2 diversity model gives the victim station an *unlimited* pool of
candidate satellites: an in-line event is resolvable whenever some other visible
victim satellite is at least ``theta`` away from the interferer.  Real victim
satellites carry a finite number of spot beams, so a satellite can serve at most
``n_beam`` cell centres at once.

Model (documented approximation; every quantity below is computed, none is
hard-coded):

* A *region* is a ``size_km`` x ``size_km`` square of cell centres on a
  hex-offset lattice of nearest-neighbour spacing ``spacing_km``, laid out on the
  local tangent plane at (``latitude_deg``, ``longitude_deg``) and mapped to ECEF
  through small-angle offsets on a sphere of radius ``earth_radius_km``.  The
  local vertical at each cell is its radial direction.
* Each cell's *requested* victim satellite is its highest-elevation visible
  victim satellite.  Each victim satellite grants at most ``n_beam`` requests,
  keeping the highest-elevation requests (greedy by elevation).  A cell whose
  request is not granted is *unserved*.
* An **event** is a served cell whose requested victim satellite is within
  ``theta`` of the interferer's serving satellite at the same cell (co-located
  variant).
* The event is **resolvable** when some other visible victim satellite is at
  least ``theta`` from the interferer *and* that satellite still has a free beam
  (spare capacity after the primary grants).  Otherwise it is **unresolvable**:
  the cell loses service for the duration of the in-line geometry.

Rescue does not sequentially deplete spare beams (a cell that rescues does not
reduce another cell's spare) -- an optimistic bias noted in the report.
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from .propagation import propagate_ecef
from .systems import SystemModel


def regional_grid(
    latitude_deg: float,
    longitude_deg: float,
    size_km: float,
    spacing_km: float,
    earth_radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres ``[G, 3]`` (ECEF, km) and local up ``[G, 3]`` on a hex lattice."""
    half = size_km / 2.0
    n = int(np.floor(size_km / spacing_km))
    east, north = [], []
    ys = -half + np.arange(n) * spacing_km
    for j, y in enumerate(ys):
        xs = -half + np.arange(n) * spacing_km + (0.5 * spacing_km if j % 2 else 0.0)
        east.append(xs)
        north.append(np.full(n, y))
    east = np.concatenate(east)
    north = np.concatenate(north)
    r = earth_radius_km
    lat0 = np.deg2rad(latitude_deg)
    lat = lat0 + north / r
    lon = np.deg2rad(longitude_deg) + east / (r * np.cos(lat0))
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    ecef = np.stack([x, y, z], axis=1)
    up = ecef / np.linalg.norm(ecef, axis=1, keepdims=True)
    return ecef, up


def _nth_max_per_group(values: torch.Tensor, groups: torch.Tensor, n: int,
                       n_groups: int) -> torch.Tensor:
    """``n``-th largest ``values`` per group (1-based); ``-inf`` if fewer than ``n``."""
    t = values.shape[0]
    dev, dt = values.device, values.dtype
    work = values.clone()
    out = torch.full((t, n_groups), float("-inf"), dtype=dt, device=dev)
    for _ in range(n):
        out = torch.full((t, n_groups), float("-inf"), dtype=dt, device=dev)
        out.scatter_reduce_(1, groups, work, reduce="amax", include_self=True)
        picked = out.gather(1, groups)
        work = torch.where(work >= picked, torch.full_like(work, float("-inf")), work)
    return out


def served_under_capacity(
    elev_best: torch.Tensor,
    sat_best: torch.Tensor,
    visible: torch.Tensor,
    n_beam: int,
    n_sats: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy-by-elevation beam grant per cell.

    ``elev_best``/``sat_best``/``visible`` are ``[T, G]``.  Returns ``served``
    ``[T, G]``, ``granted`` ``[T, S]`` and ``spare`` ``[T, S]``.
    """
    neg_inf = torch.full_like(elev_best, float("-inf"))
    vals = torch.where(visible, elev_best, neg_inf)
    sat = sat_best.clamp(min=0)
    thresh = _nth_max_per_group(vals, sat, n_beam, n_sats)  # [T,S]
    thr_cell = thresh.gather(1, sat)  # [T,G]
    served = visible & (thr_cell > float("-inf")) & (elev_best >= thr_cell)
    granted = torch.zeros_like(thresh)
    granted.scatter_add_(1, sat, served.to(granted.dtype))
    spare = (float(n_beam) - granted).clamp(min=0.0)
    return served, granted, spare


def capacity_limited_unresolvable(
    victim: SystemModel,
    interferer: SystemModel,
    cells_ecef: torch.Tensor,
    cells_up: torch.Tensor,
    times: np.ndarray,
    epoch_jd: float,
    earth_radius_km: float,
    n_beam: int,
    theta_deg: float = 2.0,
    min_elevation_deg: float = 25.0,
    device: str = "cuda",
    chunk: int = 16,
    n_rank: int = 4,
) -> dict:
    """Cell-time fractions of event / served / unresolvable under beam capacity."""
    g = cells_ecef.shape[0]
    t_total = len(times)
    k = min(int(n_rank), victim.n_sats)
    theta = torch.tensor(theta_deg, dtype=torch.float64, device=device)
    eye = torch.arange(k, device=device)

    event_tot = 0.0
    unres_tot = 0.0
    served_tot = 0.0
    for s in range(0, t_total, chunk):
        e = min(s + chunk, t_total)
        tc = times[s:e]
        pa = propagate_ecef(victim.elements, tc, epoch_jd, earth_radius_km, device)
        pb = propagate_ecef(interferer.elements, tc, epoch_jd, earth_radius_km, device)
        ua, _ = unit_look_vectors(pa, cells_ecef)  # [Tc,G,Sa,3]
        ub, _ = unit_look_vectors(pb, cells_ecef)  # [Tc,G,Sb,3]
        elev_a = elevation_deg_batch(ua, cells_up)
        vis_a = elev_a >= min_elevation_deg
        va, ia = torch.topk(elev_a.masked_fill(~vis_a, float("-inf")), k, dim=-1)
        vis_ka = va >= min_elevation_deg

        elev_b = elevation_deg_batch(ub, cells_up)
        vis_b = elev_b >= min_elevation_deg
        vb, ib = torch.topk(elev_b.masked_fill(~vis_b, float("-inf")), 1, dim=-1)
        has_b = vb[..., 0] >= min_elevation_deg
        ub0 = torch.gather(ub, 2, ib.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)  # [Tc,G,3]

        elev_best = va[..., 0]
        sat_best = ia[..., 0]
        visible = vis_ka[..., 0]
        served, _granted, spare = served_under_capacity(
            elev_best, sat_best, visible, n_beam, victim.n_sats
        )

        ua_k = torch.gather(ua, 2, ia.unsqueeze(-1).expand(-1, -1, -1, 3))  # [Tc,G,k,3]
        serving_a = ua_k[..., 0, :]
        sep0 = angular_separation_batch(serving_a, ub0)  # [Tc,G]
        event = served & has_b & (sep0 < theta)

        seps = angular_separation_batch(ua_k, ub0.unsqueeze(2))  # [Tc,G,k]
        spare_r = spare.gather(1, ia.reshape(e - s, -1)).reshape(e - s, g, k)
        cand = vis_ka & (eye.reshape(1, 1, -1) != 0)
        rescuable = (cand & (seps >= theta) & (spare_r > 0)).any(-1)
        unres = event & ~rescuable

        event_tot += float(event.sum())
        unres_tot += float(unres.sum())
        served_tot += float(served.sum())

    n = t_total * g
    return {
        "unres_cell_frac": unres_tot / n,
        "event_cell_frac": event_tot / n,
        "served_frac": served_tot / n,
        "n_cell_samples": int(n),
    }
