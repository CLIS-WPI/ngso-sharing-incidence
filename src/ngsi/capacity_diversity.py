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
* An **event** is a served cell whose aggregate ``Delta T/T`` from the
  interferer's serving satellite exceeds the 47 CFR 25.261(d) threshold
  (co-located variant: the interferer serves the same cell).
* The event is **resolvable** when some other visible victim satellite keeps
  ``Delta T/T`` below the threshold *and* still has a free beam after the primary
  grants.  Rescue **sequentially depletes** spare beams: a satellite that rescues
  one cell has one fewer free beam for the next.  Otherwise the event is
  **unresolvable**: the cell loses service for the duration of the in-line
  geometry.

The returned fractions exist on two denominators, both reported so the two
effects cannot be confused (a larger ``n_beam`` serves more cells, so an absolute
cell-time fraction can rise even as the conditional risk falls):

* ``*_cell_frac`` uses every cell-time sample (numerator/denominator tilted by
  service coverage);
* ``*_per_served`` uses only served cell-time samples (conditional on a cell
  being served at all).
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import DTT_THRESHOLD
from .geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from .link_budget import noise_power_dbw_per_mhz, pattern_gain_batch
from .propagation import propagate_ecef
from .systems import SystemModel

LIGHT_SPEED_M_S = 299792458.0


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


def _fspl_db(dist_km: torch.Tensor, freq_ghz: float) -> torch.Tensor:
    lam = LIGHT_SPEED_M_S / (freq_ghz * 1e9)
    return 20.0 * torch.log10(4.0 * np.pi * (dist_km * 1e3) / lam)


def rescue_with_depletion(
    event: torch.Tensor,
    dtt_cand: torch.Tensor,
    vis_ka: torch.Tensor,
    ia: torch.Tensor,
    va: torch.Tensor,
    spare: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Greedily rescue event cells, depleting one spare beam per rescue.

    ``event`` ``[T,G]``; ``dtt_cand``/``vis_ka``/``va`` are ``[T,G,k]`` (candidate
    ``Delta T/T``, visibility, elevation by rank); ``ia`` ``[T,G,k]`` candidate
    satellite index; ``spare`` ``[T,S]``.  Within each time step the event cells
    are processed in descending serving elevation; each takes its best candidate
    satellite (highest elevation) that keeps ``Delta T/T`` at or below the
    threshold **and still has a free beam**, consuming one spare beam.  Events are
    sparse, so the exact greedy runs on CPU.  Returns the rescued mask ``[T,G]``.
    """
    t, g = event.shape
    k = dtt_cand.shape[-1]
    ev = event.detach().cpu().numpy()
    ok = ((dtt_cand <= threshold) & vis_ka).detach().cpu().numpy()
    ian = ia.detach().cpu().numpy()
    van = np.where(np.isfinite(va.detach().cpu().numpy()), va.detach().cpu().numpy(), -np.inf)
    sp = spare.detach().cpu().numpy().astype(np.float64, copy=True)
    rescued = np.zeros((t, g), dtype=bool)
    for ti in range(t):
        cells = np.flatnonzero(ev[ti])
        if cells.size == 0:
            continue
        for c in cells[np.argsort(-van[ti, cells, 0])]:
            best_r, best_e = -1, -np.inf
            for r in range(1, k):
                if not ok[ti, c, r]:
                    continue
                s = int(ian[ti, c, r])
                if sp[ti, s] <= 0:
                    continue
                if van[ti, c, r] > best_e:
                    best_e, best_r = van[ti, c, r], r
            if best_r >= 0:
                sp[ti, int(ian[ti, c, best_r])] -= 1.0
                rescued[ti, c] = True
    return torch.from_numpy(rescued).to(event.device)


def capacity_limited_unresolvable(
    victim: SystemModel,
    interferer: SystemModel,
    cells_ecef: torch.Tensor,
    cells_up: torch.Tensor,
    times: np.ndarray,
    epoch_jd: float,
    earth_radius_km: float,
    n_beam: int,
    victim_beam,
    interf_beam,
    freq_ghz: float,
    rx_peak_gain_dbi: float,
    rx_diameter_m: float,
    min_elevation_deg: float = 25.0,
    device: str = "cuda",
    chunk: int = 16,
    n_rank: int = 4,
    pattern: str = "S1428",
) -> dict:
    """Cell-time event / served / unresolvable fractions under beam capacity.

    Events use the 25.261(d) ``Delta T/T`` criterion (co-located variant).  Rescue
    depletes spare beams sequentially.  See the module docstring for the two
    denominators.
    """
    g = cells_ecef.shape[0]
    t_total = len(times)
    k = min(int(n_rank), victim.n_sats)
    n_sats = victim.n_sats
    noise_v = noise_power_dbw_per_mhz(victim_beam.system_noise_temp_k)

    event_tot = torch.zeros((), dtype=torch.float64, device=device)
    unres_tot = torch.zeros((), dtype=torch.float64, device=device)
    served_tot = torch.zeros((), dtype=torch.float64, device=device)

    def _dtt(off_axis, dist):
        g_rx = pattern_gain_batch(off_axis, rx_peak_gain_dbi, rx_diameter_m, freq_ghz, pattern)
        i_over_n = interf_beam.eirp_density_dbw_per_mhz + g_rx - _fspl_db(dist, freq_ghz) - noise_v
        return 10.0 ** (i_over_n / 10.0)

    for s in range(0, t_total, chunk):
        e = min(s + chunk, t_total)
        tc = times[s:e]
        pa = propagate_ecef(victim.elements, tc, epoch_jd, earth_radius_km, device)
        pb = propagate_ecef(interferer.elements, tc, epoch_jd, earth_radius_km, device)
        ua, da = unit_look_vectors(pa, cells_ecef)  # [Tc,G,Sa,3], [Tc,G,Sa]
        ub, db = unit_look_vectors(pb, cells_ecef)
        elev_a = elevation_deg_batch(ua, cells_up)
        vis_a = elev_a >= min_elevation_deg
        va, ia = torch.topk(elev_a.masked_fill(~vis_a, float("-inf")), k, dim=-1)
        vis_ka = va >= min_elevation_deg

        elev_b = elevation_deg_batch(ub, cells_up)
        vis_b = elev_b >= min_elevation_deg
        vb, ib = torch.topk(elev_b.masked_fill(~vis_b, float("-inf")), 1, dim=-1)
        has_b = vb[..., 0] >= min_elevation_deg
        ub0 = torch.gather(ub, 2, ib.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)  # [Tc,G,3]
        dist_b0 = torch.gather(db, 2, ib)[..., 0]  # [Tc,G]

        elev_best = va[..., 0]
        sat_best = ia[..., 0]
        visible = vis_ka[..., 0]
        served, _granted, spare = served_under_capacity(
            elev_best, sat_best, visible, n_beam, n_sats
        )

        ua_k = torch.gather(ua, 2, ia.unsqueeze(-1).expand(-1, -1, -1, 3))  # [Tc,G,k,3]
        dist_a_k = torch.gather(da, 2, ia)  # [Tc,G,k]
        serving_a = ua_k[..., 0, :]
        sep0 = angular_separation_batch(serving_a, ub0)
        event = served & has_b & (_dtt(sep0, dist_b0) > DTT_THRESHOLD)

        # Rescue: try candidate ranks 1..k-1 in order; each rescuing satellite
        # accepts at most one cell per round and has one fewer spare thereafter.
        seps = angular_separation_batch(ua_k, ub0.unsqueeze(2))  # [Tc,G,k]
        dtt_cand = _dtt(seps, dist_a_k)  # [Tc,G,k]
        rescued = rescue_with_depletion(event, dtt_cand, vis_ka, ia, va, spare, DTT_THRESHOLD)
        unres = event & ~rescued

        event_tot += event.sum().to(torch.float64)
        unres_tot += unres.sum().to(torch.float64)
        served_tot += served.sum().to(torch.float64)

    n = t_total * g
    ev = float(event_tot)
    un = float(unres_tot)
    sv = float(served_tot)
    return {
        "event_cell_frac": ev / n,
        "unres_cell_frac": un / n,
        "served_frac": sv / n,
        "event_per_served": ev / sv if sv else float("nan"),
        "unres_per_served": un / sv if sv else float("nan"),
        "unres_no_div_per_served": ev / sv if sv else float("nan"),
        "n_cell_samples": int(n),
        "n_served_samples": int(sv),
    }
