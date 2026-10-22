"""Batched Delta T/T screening of a co-frequency pair (47 CFR 25.261(c)/(d)).

For an ordered pair (A victim, B interferer) that is co-frequency in the user
downlink, and each station/time sample:

* A's serving satellite is the highest-elevation visible A satellite; its own
  carrier sets the baseline C/N.
* Three interferer aggregates are evaluated per sample (``VARIANT_NAMES``):

  0. ``realistic`` - B's serving satellite points its main beam at the site
     (a co-located B terminal) at B's EIRP density; every *other* visible B
     satellite contributes through the ITU-R S.1528 space-station sidelobe, with
     its beam-pointing direction approximated as **nadir** (stated assumption).
     Individual I/N add in linear units (a two-gain product: S.1528 TX sidelobe
     and S.1428 victim RX).
  1. ``co_located`` - only B's serving satellite at its peak (main-beam) EIRP.
  2. ``upper_bound`` - every visible B satellite at B's peak EIRP density toward
     the site (the all-at-peak bound; no sidelobe reduction).

* An **event** is aggregate ``Delta T/T > 0.06`` (``I/N > -12.2 dB``).
* The event is **resolvable** when some *other* visible A satellite keeps
  aggregate ``Delta T/T <= 0.06`` (diversity); the link actually used after a
  handoff contributes its own (<= 6%) aggregate.

Handoff (satellite diversity) is evaluated under one or more
:class:`HandoffPolicy` objects; the definition is the same for all of them:

* **Candidate set** - the visible A satellites (elevation >= the victim mask)
  among the ``k_a`` highest-elevation ones, optionally limited to the ``k``
  highest-elevation satellites (rank 0, the serving satellite, included).  An
  alternate (rank >= 1) is admissible only if its own aggregate
  ``Delta T/T <= 0.06`` (it resolves the event).
* **Trigger** - the terminal stays on the highest-elevation satellite (the
  baseline policy) and considers a handoff only while that link is in event.
* **Selection rule** - ``link_cn="candidate"`` (full link budget): every
  candidate's C/N is computed from its own slant range,
  ``C/N_k = EIRP_A + G_rx,peak - FSPL(r_k) - N`` (the terminal re-points, so the
  victim receive gain is the peak gain toward the new satellite), the
  interference toward it is the aggregate at the off-axis angles of the
  re-pointed antenna, and the terminal switches to the admissible
  (non-triggered) alternate with the highest ``C/(N+I)``.  If no alternate is
  admissible (unresolvable) it stays on the serving satellite with its degraded
  SINR.  ``keep_serving_if_better=True`` (sensitivity) also keeps the serving
  satellite when its degraded ``C/(N+I)`` beats every admissible alternate.
  ``link_cn="serving"`` (legacy, retained for attribution) takes the admissible
  alternate with the lowest ``Delta T/T`` and reuses the *serving* satellite's
  clear-sky C/N for it.
* **Rate limit** - ``min_interval_s`` makes the choice stateful: the terminal
  keeps its current satellite until at least ``min_interval_s`` has passed since
  its last switch, unless that satellite leaves the candidate set (a forced
  switch).  Every change of satellite counts, including the geometric handover
  to a new highest-elevation satellite.
* **Baseline** - degradation is ``1 - R_used / R_0``, with ``R_0`` the
  interference-free rate of the highest-elevation satellite (the rule-free
  reference link).  A rate-limited policy also reports its own interference-free
  baseline (the same rate limit, no interferer), so the part of the loss caused
  by the rate limit itself can be separated from the interference.

Interferer beam steering follows ``interf_beam.beam_steering`` (or the ``steering``
override).  Under ``pfd_constant`` the interferer raises EIRP with slant range so
the ground PFD stays at its value at the minimum-elevation service edge
(``r_max``): the effective path loss is ``max(FSPL(r), FSPL(r_max))``, so the
received interference stops growing once the satellite is at its EIRP ceiling
(FCC-22-91A1 para. 42).  ``constant_eirp`` holds the EIRP density fixed at its
maximum.  The victim's own carrier is always constant-EIRP, so the baseline C/N
distribution and the 25.261(d) availability denominators do not depend on the
interferer model.

The same geometry drives the 25.261(d) secondary results: the victim throughput
with and without interference and with and without diversity, and the absolute
unavailability change at C/N = 0 dB.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from .constants import DTT_THRESHOLD
from .geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from .link_budget import (
    noise_power_dbw_per_mhz,
    pattern_gain_batch,
    s1528_gain_batch,
    slant_range_km_batch,
)
from .propagation import propagate_ecef
from .systems import SystemModel

LIGHT_SPEED_M_S = 299792458.0
VARIANT_NAMES = ("realistic", "co_located", "upper_bound")
_STEERING = ("constant_eirp", "pfd_constant")
_LINK_CN = ("candidate", "serving")
# I/N histogram bin edges (dB) for the event I/N statistics (0.01 dB bins).
IN_HIST_EDGES_DB = np.round(np.arange(-30.0, 50.0 + 1e-9, 0.01), 2)


def hist_quantile(counts: np.ndarray, q: float, edges: np.ndarray = IN_HIST_EDGES_DB) -> float:
    """``q``-quantile (0-1) from ``event_in_hist`` counts (upper bin edge; NaN if empty)."""
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum()
    if tot <= 0:
        return float("nan")
    i = int(np.searchsorted(np.cumsum(c), q * tot))
    return float(edges[min(max(i, 0), len(edges) - 1)])


@dataclass(frozen=True)
class HandoffPolicy:
    """One satellite-diversity (handoff) policy; see the module docstring."""

    name: str
    link_cn: str = "candidate"
    k: int | None = None
    min_interval_s: float | None = None
    keep_serving_if_better: bool = False

    def __post_init__(self) -> None:
        if self.link_cn not in _LINK_CN:
            raise ValueError(f"link_cn must be one of {_LINK_CN!r}")
        if self.k is not None and int(self.k) < 1:
            raise ValueError("k must be >= 1")
        if self.link_cn == "serving" and (self.k is not None or self.min_interval_s is not None
                                          or self.keep_serving_if_better):
            raise ValueError("the legacy 'serving' policy takes no k / min_interval_s / "
                             "keep_serving_if_better")

    @classmethod
    def from_dict(cls, d: dict) -> "HandoffPolicy":
        return cls(name=str(d["name"]), link_cn=str(d.get("link_cn", "candidate")),
                   k=None if d.get("k") is None else int(d["k"]),
                   min_interval_s=(None if d.get("min_interval_s") is None
                                   else float(d["min_interval_s"])),
                   keep_serving_if_better=bool(d.get("keep_serving_if_better", False)))


LEGACY_POLICY = HandoffPolicy("legacy", link_cn="serving")


def rate_limited_choice(
    ids: np.ndarray,
    valid: np.ndarray,
    desired: np.ndarray,
    t_s: np.ndarray,
    min_interval_s: float,
    state: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateful satellite choice under a minimum time between switches.

    ``ids``/``valid`` are ``[T, S, K]`` (satellite id and candidate validity per
    rank, ``S`` independent terminals), ``desired`` is ``[T, S]`` (the rank the
    memoryless policy would use), ``t_s`` is ``[T]``.  ``state`` carries
    ``cur`` (current satellite id, -1 = none) and ``last`` (time of the last
    switch) across calls.  Returns ``(rank_used [T, S], switched [T, S])``; rank
    -1 means no candidate.  A terminal keeps its satellite while it is still a
    valid candidate and fewer than ``min_interval_s`` seconds have passed since
    its last switch; it is forced to switch when its satellite drops out.
    """
    n_t, n_s, _ = ids.shape
    cur = state.setdefault("cur", np.full(n_s, -1, dtype=np.int64))
    last = state.setdefault("last", np.full(n_s, -np.inf))
    rank_used = np.full((n_t, n_s), -1, dtype=np.int64)
    switched = np.zeros((n_t, n_s), dtype=bool)
    cols = np.arange(n_s)
    for t in range(n_t):
        match = (ids[t] == cur[:, None]) & valid[t]
        found = match.any(-1)
        cur_rank = np.where(found, match.argmax(-1), -1)
        d = desired[t]
        has_d = d >= 0
        d_id = np.where(has_d, ids[t][cols, np.clip(d, 0, None)], -1)
        want = has_d & (d_id != cur)
        allowed = (t_s[t] - last) >= min_interval_s
        switch = want & (~found | allowed)
        r = np.where(switch, d, np.where(found, cur_rank, -1))
        new_cur = np.where(r >= 0, ids[t][cols, np.clip(r, 0, None)], -1)
        sw = switch & (cur >= 0)
        switched[t] = sw
        last = np.where(switch, t_s[t], last)
        cur = new_cur
        rank_used[t] = r
    state["cur"], state["last"] = cur, last
    return rank_used, switched


def count_switches(ids_used: np.ndarray, state: dict, key: str = "prev_id") -> np.ndarray:
    """Satellite changes per terminal over ``ids_used`` ``[T, S]`` (-1 = not served).

    A switch is a step whose satellite differs from the previous step's while both
    steps are served; ``state[key]`` carries the last step's ids across calls.
    """
    ids_used = np.asarray(ids_used, dtype=np.int64)
    prev = state.get(key)
    if prev is None:
        prev = np.full(ids_used.shape[1], -1, dtype=np.int64)
    seq = np.concatenate([prev[None, :], ids_used], axis=0)
    a, b = seq[:-1], seq[1:]
    n = ((a >= 0) & (b >= 0) & (a != b)).sum(0)
    state[key] = ids_used[-1].copy() if len(ids_used) else prev
    return n.astype(np.float64)


def _fspl_db(dist_km: torch.Tensor, freq_ghz: float) -> torch.Tensor:
    lam = LIGHT_SPEED_M_S / (freq_ghz * 1e9)
    return 20.0 * torch.log10(4.0 * np.pi * (dist_km * 1e3) / lam)


def _dbw_to_linear(db):
    return 10.0 ** (db / 10.0)


def _cni_from_dtt(cn0_db, dtt):
    return cn0_db - 10.0 * torch.log10(1.0 + dtt)


def _shannon_rate(snr_db, bandwidth_hz: float, margin_db: float):
    snr = 10.0 ** (snr_db / 10.0)
    margin = 10.0 ** (float(margin_db) / 10.0)
    return bandwidth_hz * torch.log2(1.0 + snr / margin)


def _shell_altitudes(interferer: SystemModel, device: str) -> torch.Tensor:
    """Design altitude (km) of every interferer satellite, from its shell."""
    alts = np.asarray(
        [interferer.shells[int(si)].altitude_km for si in np.asarray(interferer.shell_index)],
        dtype=np.float64,
    )
    from .precision import compute_dtype

    return torch.as_tensor(alts, dtype=compute_dtype(), device=device)


def _evaluate_policies(policies, acc, states, tc, dt_s, k_a, eye_a, cn0_k, cn0, has_a,
                       vis_ka, ia, dtts, cand_dtt, has_b_var, bw_hz, margin_db) -> None:
    """Accumulate each handoff policy's rate / availability over one time chunk.

    ``cn0_k`` [Tc,G,Ka] is the clear-sky C/N of every ranked victim satellite
    (own slant range, peak receive gain); ``cand_dtt[vi]`` the aggregate
    Delta T/T toward each of them with the antenna re-pointed.
    """
    rank = eye_a.reshape(1, 1, -1)
    served = has_a
    ldtt, sinr_k, event = {}, {}, {}
    for vi in range(3):
        d = torch.where(has_b_var[vi].unsqueeze(-1), cand_dtt[vi], torch.zeros_like(cand_dtt[vi]))
        d = torch.cat([dtts[vi].unsqueeze(-1), d[..., 1:]], dim=-1)  # exact serving aggregate
        ldtt[vi] = d
        sinr_k[vi] = _cni_from_dtt(cn0_k, d)
        event[vi] = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
    tc_np = np.asarray(tc, dtype=np.float64)
    for p in policies:
        a = acc[p.name]
        k = k_a if p.k is None else min(int(p.k), k_a)
        in_k = vis_ka & (rank < k)
        for vi in range(3):
            resolv = in_k & (rank >= 1) & (ldtt[vi] <= DTT_THRESHOLD)
            any_res = resolv.any(-1)
            unres = event[vi] & ~any_res
            if p.link_cn == "serving":
                big = torch.full_like(ldtt[vi], float("inf"))
                best_v, best_i = torch.where(resolv, ldtt[vi], big).min(-1)
                use_alt = event[vi] & any_res
                cni = _cni_from_dtt(cn0, torch.where(use_alt, best_v, dtts[vi]))
                diverted = use_alt
                used_rank = torch.where(use_alt, best_i, torch.zeros_like(best_i))
            else:
                # Event-triggered: during an event switch to the best admissible
                # alternate (optionally also keeping the serving satellite when it
                # is better); unresolvable events stay on the serving satellite.
                allowed = resolv | (rank == 0) if p.keep_serving_if_better else resolv
                best_rank = sinr_k[vi].masked_fill(~allowed, float("-inf")).argmax(-1)
                switch = event[vi] & (any_res if not p.keep_serving_if_better else True)
                desired = torch.where(switch, best_rank, torch.zeros_like(best_rank))
                if p.min_interval_s is None:
                    cni = torch.gather(sinr_k[vi], -1, desired.unsqueeze(-1)).squeeze(-1)
                    diverted = served & (desired > 0)
                    used_rank = desired
                else:
                    desired_np = torch.where(served, desired, torch.full_like(desired, -1)).cpu().numpy()
                    r_used, sw = rate_limited_choice(
                        ia.cpu().numpy(), in_k.cpu().numpy(), desired_np, tc_np,
                        float(p.min_interval_s), states[p.name][vi])
                    r_t = torch.as_tensor(np.clip(r_used, 0, None), device=cn0.device)
                    cni = torch.gather(sinr_k[vi], -1, r_t.unsqueeze(-1)).squeeze(-1)
                    cni = torch.where(torch.as_tensor(r_used >= 0, device=cn0.device), cni,
                                      torch.full_like(cni, float("-inf")))
                    diverted = served & (r_t > 0)
                    a["sw"][vi] += sw.sum(0)
                    used_rank = torch.where(torch.as_tensor(r_used >= 0, device=cn0.device),
                                            r_t, torch.full_like(r_t, -1))
            # Satellite actually used (-1 = none) -> switch count, every policy.
            ur = used_rank.clamp(min=0).unsqueeze(-1)
            uid = torch.gather(ia, -1, ur).squeeze(-1)
            uid = torch.where(served & (used_rank >= 0), uid, torch.full_like(uid, -1))
            a["sw_all"][vi] += count_switches(uid.cpu().numpy(), states[p.name][vi], "prev_id")
            rate = torch.where(served, _shannon_rate(cni, bw_hz, margin_db), torch.zeros_like(cni))
            a["r"][vi] += rate.sum(0).cpu().numpy()
            a["unav"][vi] += ((cn0 >= 0.0) & (cni < 0.0) & served).sum(0).cpu().numpy()
            a["unres"][vi] += unres.sum(0).cpu().numpy()
            a["div"][vi] += diverted.sum(0).cpu().numpy()
        # Switches of the unconstrained interference-free baseline (highest elevation).
        bid = torch.where(served, ia[..., 0], torch.full_like(ia[..., 0], -1))
        a["sw_base0"] += count_switches(bid.cpu().numpy(), states[p.name][3], "prev_base")
        if p.min_interval_s is not None:
            # Interference-free baseline under the same rate limit: the terminal
            # wants the highest-elevation satellite and switches as allowed.
            desired0 = torch.where(served, torch.zeros_like(ia[..., 0]),
                                   torch.full_like(ia[..., 0], -1)).cpu().numpy()
            r0_used, _ = rate_limited_choice(ia.cpu().numpy(), in_k.cpu().numpy(), desired0,
                                             tc_np, float(p.min_interval_s), states[p.name][3])
            r0_t = torch.as_tensor(np.clip(r0_used, 0, None), device=cn0.device)
            c0 = torch.gather(cn0_k, -1, r0_t.unsqueeze(-1)).squeeze(-1)
            rate0 = torch.where(served, _shannon_rate(c0, bw_hz, margin_db), torch.zeros_like(c0))
            a["r_base"] += rate0.sum(0).cpu().numpy()
            a["unav0_base"] += ((c0 < 0.0) & served).sum(0).cpu().numpy()


def screen_pair_dtt(
    victim: SystemModel,
    interferer: SystemModel,
    k_a: int,
    k_b: int,
    station_ecef: torch.Tensor,
    up: torch.Tensor,
    times: np.ndarray,
    victim_beam,
    interf_beam,
    freq_ghz: float,
    rx_peak_gain_dbi: float,
    rx_diameter_m: float,
    margin_db: float,
    min_elevation_deg: float,
    earth_radius_km: float,
    epoch_jd: float,
    device: str = "cuda",
    chunk: int = 64,
    pattern: str = "S1428",
    steering: str | None = None,
    diversity_fspl: str = "interferer",
    victim_min_elevation_deg: float | None = None,
    interferer_min_elevation_deg: float | None = None,
    handoff_policies: tuple[HandoffPolicy, ...] = (LEGACY_POLICY,),
) -> dict:
    """Screen one ordered pair under the Delta T/T criterion.

    Returns per-station arrays of shape ``[3, G]`` for ``event_frac``/``unres_frac``
    (variants 0 realistic, 1 co-located, 2 upper bound), the victim C/N
    distribution and baseline availability, and the per-variant throughput /
    unavailability accumulators for the 25.261(d) secondary result.

    ``steering`` selects the interferer beam-steering model; ``None`` (default)
    uses ``interf_beam.beam_steering``.  ``"pfd_constant"`` requires
    ``interf_beam.pfd_anchor_elevation_deg``.  ``diversity_fspl="interferer"``
    (correct) evaluates the interference at the interferer's slant range;
    ``"victim"`` reproduces the earlier, incorrect reuse of the candidate victim
    range and is retained only for attribution of that fix.
    ``victim_min_elevation_deg`` / ``interferer_min_elevation_deg`` override
    ``min_elevation_deg`` for that system's visibility (default: the common value).

    ``handoff_policies`` lists the diversity policies to evaluate on the same
    geometry; their results are under ``"policies"`` keyed by name.  The
    top-level ``r_div_mean`` / ``unavail_incr_div_frac`` keep the legacy
    definition (``LEGACY_POLICY``) for backward compatibility.
    """
    min_el_a = min_elevation_deg if victim_min_elevation_deg is None else float(victim_min_elevation_deg)
    min_el_b = (min_elevation_deg if interferer_min_elevation_deg is None
                else float(interferer_min_elevation_deg))
    if diversity_fspl not in ("interferer", "victim"):
        raise ValueError("diversity_fspl must be 'interferer' or 'victim'")
    mode = interf_beam.beam_steering if steering is None else str(steering)
    if mode not in _STEERING:
        raise ValueError(f"steering must be one of {_STEERING!r}")
    pfd = mode == "pfd_constant"
    anchor_elev = interf_beam.pfd_anchor_elevation_deg
    if pfd and anchor_elev is None:
        raise ValueError("pfd_constant steering requires interf_beam.pfd_anchor_elevation_deg")
    ref_alt = interf_beam.pfd_reference_altitude_km
    if pfd and ref_alt is None:
        raise ValueError("pfd_constant steering requires interf_beam.pfd_reference_altitude_km")

    g = station_ecef.shape[0]
    t_total = len(times)
    k_a = min(int(k_a), victim.n_sats)
    k_b = min(int(k_b), interferer.n_sats)
    eye_a = torch.arange(k_a, device=device)
    eye_b = torch.arange(k_b, device=device)
    noise_v = noise_power_dbw_per_mhz(victim_beam.system_noise_temp_k)
    bw_hz = float(victim_beam.bandwidth_mhz) * 1e6
    eirp_max = float(interf_beam.eirp_density_dbw_per_mhz)
    interf_peak = float(interf_beam.peak_gain_dbi)
    interf_bw = float(interf_beam.beamwidth_deg)

    # PFD-constant: one r_ref for every satellite, so the surface PFD is
    # PFD0 = EIRP_max - 10 log10(4 pi r_ref^2) whatever the satellite's altitude.
    alt_all = _shell_altitudes(interferer, device)  # [Nb]
    r_max_all = (
        slant_range_km_batch(torch.full_like(alt_all, float(ref_alt)), float(anchor_elev),
                             earth_radius_km) if pfd else None
    )

    event_count = torch.zeros(3, g, dtype=torch.float64, device=device)
    unres_count = torch.zeros(3, g, dtype=torch.float64, device=device)
    served_count = torch.zeros(g, dtype=torch.float64, device=device)
    r0_sum = torch.zeros(g, dtype=torch.float64, device=device)
    r_int = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    r_div = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    unavail0 = torch.zeros(g, dtype=torch.float64, device=device)
    unavail_nodiv = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    unavail_incr_nodiv = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    unavail_incr_div = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    cn0_sum = torch.zeros(g, dtype=torch.float64, device=device)
    dtt_sum = {vi: torch.zeros(g, dtype=torch.float64, device=device) for vi in range(3)}
    cn0_blocks: list[np.ndarray] = []
    in_edges_t = torch.as_tensor(IN_HIST_EDGES_DB, dtype=torch.float64, device=device)
    in_hist = {k: torch.zeros(len(IN_HIST_EDGES_DB) + 1, dtype=torch.float64, device=device)
               for k in ("serving", "aggregate")}
    names = [p.name for p in handoff_policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate handoff policy names: {names}")
    dt_s = float(times[1] - times[0]) if t_total > 1 else 0.0
    pol_acc = {
        p.name: {key: {vi: np.zeros(g) for vi in range(3)}
                 for key in ("r", "unav", "unres", "div", "sw", "sw_all")}
        | {"r_base": np.zeros(g), "unav0_base": np.zeros(g), "sw_base0": np.zeros(g)}
        for p in handoff_policies
    }
    pol_state = {p.name: [dict() for _ in range(4)] for p in handoff_policies}

    def _in_db(off_axis, fspl_db, eirp_density):
        g_rx = pattern_gain_batch(off_axis, rx_peak_gain_dbi, rx_diameter_m, freq_ghz, pattern)
        return eirp_density + g_rx - fspl_db - noise_v

    def _best_candidate(dtt_cand, cand_mask):
        """Min aggregate DTT among *resolvable* A candidates (else +inf)."""
        big = torch.full_like(dtt_cand, float("inf"))
        masked = torch.where(cand_mask & (dtt_cand <= DTT_THRESHOLD), dtt_cand, big)
        best = masked.min(-1).values
        return best, torch.isfinite(best)

    for s in range(0, t_total, chunk):
        e = min(s + chunk, t_total)
        tc = times[s:e]
        pa = propagate_ecef(victim.elements, tc, epoch_jd, earth_radius_km, device)
        pb = propagate_ecef(interferer.elements, tc, epoch_jd, earth_radius_km, device)
        ua, da = unit_look_vectors(pa, station_ecef)  # [Tc,G,Ka,3], [Tc,G,Ka]
        ub, db = unit_look_vectors(pb, station_ecef)
        elev_a = elevation_deg_batch(ua, up)
        elev_b = elevation_deg_batch(ub, up)
        vis_a = elev_a >= min_el_a
        vis_b = elev_b >= min_el_b

        va, ia = torch.topk(elev_a.masked_fill(~vis_a, float("-inf")), k_a, dim=-1)
        vb, ib = torch.topk(elev_b.masked_fill(~vis_b, float("-inf")), k_b, dim=-1)
        vis_ka = va >= min_el_a
        vis_kb = vb >= min_el_b
        ua_k = torch.gather(ua, 2, ia.unsqueeze(-1).expand(-1, -1, -1, 3))
        ub_k = torch.gather(ub, 2, ib.unsqueeze(-1).expand(-1, -1, -1, 3))
        da_k = torch.gather(da, 2, ia)
        db_k = torch.gather(db, 2, ib)

        serving_a = ua_k[:, :, 0, :]
        serving_b = ub_k[:, :, 0, :]
        dist_a = da_k[:, :, 0]
        dist_b = db_k[:, :, 0]
        has_a = vis_ka[:, :, 0]
        has_b = vis_kb[:, :, 0]

        # Victim own carrier: serving satellite at boresight (peak gain).
        cn0 = victim_beam.eirp_density_dbw_per_mhz + rx_peak_gain_dbi - _fspl_db(dist_a, freq_ghz) - noise_v

        fspl_all_b = _fspl_db(db_k, freq_ghz)  # [Tc,G,Kb]
        fspl_dist_b = _fspl_db(dist_b, freq_ghz)  # [Tc,G]
        if pfd:
            # Effective path loss: the interferer cannot exceed its EIRP ceiling, so
            # the received interference stops falling once the slant range is below
            # r_ref (the reference-altitude anchor range, the same for every shell).
            r_max_k = torch.gather(
                r_max_all.reshape(1, 1, -1).expand(tc.shape[0], g, -1), 2, ib
            )
            fspl_serve = torch.maximum(fspl_dist_b, _fspl_db(r_max_k[:, :, 0], freq_ghz))
        else:
            fspl_serve = fspl_dist_b

        # --- serving interferer (B serving satellite, main beam) ---
        sep_c = angular_separation_batch(serving_a, serving_b)
        dtt_colo = _dbw_to_linear(_in_db(sep_c, fspl_serve, eirp_max))

        # --- upper bound: every visible B satellite at peak EIRP, actual range ---
        sep_w = angular_separation_batch(serving_a.unsqueeze(2), ub_k)  # [Tc,G,Kb]
        dtt_upper = (
            _dbw_to_linear(_in_db(sep_w, fspl_all_b, eirp_max)) * vis_kb.to(pb.dtype)
        ).sum(-1)

        # --- realistic: main beam + S.1528 sidelobe of every OTHER visible B sat ---
        # Beam-pointing directions of the non-serving B satellites are approximated
        # as nadir (stated assumption); theta_tx is the angle at that satellite
        # between its pointing direction and the victim station.
        pb_unit = -pb / torch.clamp(torch.linalg.norm(pb, dim=-1, keepdim=True), min=1e-12)
        nadir_k = torch.gather(
            pb_unit.unsqueeze(1).expand(-1, g, -1, -1), 2,
            ib.unsqueeze(-1).expand(-1, -1, -1, 3),
        )  # [Tc,G,Kb,3]
        theta_tx = angular_separation_batch(nadir_k, -ub_k)  # [Tc,G,Kb]
        g_tx = s1528_gain_batch(theta_tx, interf_peak, interf_bw)  # [Tc,G,Kb]
        eirp_other = eirp_max + (g_tx - interf_peak)  # [Tc,G,Kb]
        other_mask = vis_kb & (eye_b.reshape(1, 1, -1) != 0)
        dtt_side = (
            _dbw_to_linear(_in_db(sep_w, fspl_all_b, eirp_other))
            * other_mask.to(pb.dtype)
        ).sum(-1)
        dtt_realistic = dtt_colo + dtt_side

        dtts = {
            0: torch.where(vis_kb.any(-1), dtt_realistic, torch.zeros_like(dtt_realistic)),
            1: torch.where(has_b, dtt_colo, torch.zeros_like(dtt_colo)),
            2: dtt_upper,
        }
        has_b_var = {0: vis_kb.any(-1), 1: has_b, 2: vis_kb.any(-1)}
        for vi in range(3):
            event = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
            event_count[vi] += event.sum(0).to(torch.float64)
        # I/N histograms over realistic-variant events (all stations pooled): the
        # serving interferer beam alone and the realistic aggregate.
        ev0 = has_a & has_b_var[0] & (dtts[0] > DTT_THRESHOLD)
        for key, lin in (("serving", dtt_colo), ("aggregate", dtts[0])):
            in_db = 10.0 * torch.log10(torch.clamp(lin[ev0], min=1e-30))
            b = torch.bucketize(in_db.to(torch.float64), in_edges_t).clamp_(0, len(IN_HIST_EDGES_DB))
            in_hist[key] += torch.bincount(b, minlength=len(IN_HIST_EDGES_DB) + 1).to(torch.float64)

        # --- diversity: alternate visible A satellites (link actually used) ---
        cand = vis_ka & (eye_a.reshape(1, 1, -1) != 0)
        if diversity_fspl == "victim":
            fspl_div_serve = _fspl_db(da_k, freq_ghz)  # [Tc,G,Ka]
            fspl_div_all = fspl_div_serve.unsqueeze(-1)  # [Tc,G,Ka,1]
        else:
            fspl_div_serve = fspl_serve.unsqueeze(2)  # [Tc,G,1]
            fspl_div_all = fspl_all_b.unsqueeze(2)  # [Tc,G,1,Kb]
        sep_ca = angular_separation_batch(ua_k, serving_b.unsqueeze(2))  # [Tc,G,Ka]
        dtt_ca = _dbw_to_linear(_in_db(sep_ca, fspl_div_serve, eirp_max))  # [Tc,G,Ka]
        sep_wa = angular_separation_batch(ua_k.unsqueeze(3), ub_k.unsqueeze(2))  # [Tc,G,Ka,Kb]
        side_lin = _dbw_to_linear(_in_db(sep_wa, fspl_div_all, eirp_other.unsqueeze(2)))
        peak_lin = _dbw_to_linear(_in_db(sep_wa, fspl_div_all, eirp_max))
        vis_kb_f = vis_kb.unsqueeze(2).to(pb.dtype)
        # Upper bound uses *all* visible B satellites at peak EIRP (no sidelobe
        # reduction), matching dtt_upper; the realistic variant adds the S.1528
        # sidelobe only of the non-serving B satellites.
        dtt_ua = (peak_lin * vis_kb_f).sum(-1)  # [Tc,G,Ka]
        dtt_ra = dtt_ca + (side_lin * other_mask.unsqueeze(2).to(pb.dtype)).sum(-1)

        cand_dtt = {0: dtt_ra, 1: dtt_ca, 2: dtt_ua}
        for vi in range(3):
            best, resolvable = _best_candidate(cand_dtt[vi], cand)
            event = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
            unres = event & ~resolvable
            unres_count[vi] += unres.sum(0).to(torch.float64)

            dtt_used = dtts[vi]
            dtt_div = torch.where(event & resolvable, best, dtt_used)
            cni_nodiv = _cni_from_dtt(cn0, dtt_used)
            cni_div = _cni_from_dtt(cn0, dtt_div)
            rate_nodiv = _shannon_rate(cni_nodiv, bw_hz, margin_db)
            rate_div = _shannon_rate(cni_div, bw_hz, margin_db)
            served = has_a
            r_int[vi] += (rate_nodiv * served).sum(0)
            r_div[vi] += (rate_div * served).sum(0)
            unavail_nodiv[vi] += ((cni_nodiv < 0.0) & event).sum(0).to(torch.float64)
            unavail_incr_nodiv[vi] += ((cn0 >= 0.0) & (cni_nodiv < 0.0) & served).sum(0).to(torch.float64)
            unavail_incr_div[vi] += ((cn0 >= 0.0) & (cni_div < 0.0) & served).sum(0).to(torch.float64)
            dtt_sum[vi] += (dtts[vi] * served).sum(0)

        _evaluate_policies(
            handoff_policies, pol_acc, pol_state, times[s:e], dt_s, k_a, eye_a,
            victim_beam.eirp_density_dbw_per_mhz + rx_peak_gain_dbi - _fspl_db(da_k, freq_ghz)
            - noise_v,
            cn0, has_a, vis_ka, ia, dtts, cand_dtt, has_b_var, bw_hz, margin_db,
        )

        served = has_a
        served_count += served.sum(0).to(torch.float64)
        r0 = _shannon_rate(cn0, bw_hz, margin_db)
        r0_sum += (r0 * served).sum(0)
        unavail0 += ((cn0 < 0.0) & served).sum(0).to(torch.float64)
        cn0_sum += (cn0 * served).sum(0)
        cn0_blocks.append(
            torch.where(served, cn0, torch.full_like(cn0, float("nan"))).cpu().numpy()
        )

    n = float(t_total)
    served_np = served_count.clamp(min=1.0).cpu().numpy()
    cn0_all =np.concatenate(cn0_blocks, axis=0) if cn0_blocks else np.empty((0, g))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cn0_median = np.nanmedian(cn0_all, axis=0) if t_total else np.full(g, np.nan)
        cn0_p10 = np.nanpercentile(cn0_all, 10.0, axis=0) if t_total else np.full(g, np.nan)
    avail0_frac = (cn0_all >= 0.0).mean(axis=0) if t_total else np.zeros(g)
    return {
        "event_frac": (event_count / n).cpu().numpy(),
        "unres_frac": (unres_count / n).cpu().numpy(),
        "served_frac": (served_count / n).cpu().numpy(),
        "r0_mean": (r0_sum / served_count.clamp(min=1.0)).cpu().numpy(),
        "r_int_mean": {vi: (r_int[vi] / served_count.clamp(min=1.0)).cpu().numpy() for vi in range(3)},
        "r_div_mean": {vi: (r_div[vi] / served_count.clamp(min=1.0)).cpu().numpy() for vi in range(3)},
        "cn0_mean": (cn0_sum / served_count.clamp(min=1.0)).cpu().numpy(),
        "cn0_median": cn0_median,
        "cn0_p10": cn0_p10,
        "avail0_frac": avail0_frac,
        "dtt_realistic_mean": (dtt_sum[0] / served_count.clamp(min=1.0)).cpu().numpy(),
        "dtt_colo_mean": (dtt_sum[1] / served_count.clamp(min=1.0)).cpu().numpy(),
        "dtt_upper_mean": (dtt_sum[2] / served_count.clamp(min=1.0)).cpu().numpy(),
        "unavail_nodiv_frac": {vi: (unavail_nodiv[vi] / n).cpu().numpy() for vi in range(3)},
        "unavail_incr_nodiv_frac": {vi: (unavail_incr_nodiv[vi] / n).cpu().numpy() for vi in range(3)},
        "unavail_incr_div_frac": {vi: (unavail_incr_div[vi] / n).cpu().numpy() for vi in range(3)},
        "unavail0_frac": (unavail0 / n).cpu().numpy(),
        "n_samples": t_total,
        # Counts of realistic-variant event samples per I/N bin (dB); bin i holds
        # IN_HIST_EDGES_DB[i-1] < I/N <= IN_HIST_EDGES_DB[i] (0 / last = under / over).
        "event_in_hist": {k: v.cpu().numpy() for k, v in in_hist.items()},
        "policies": {
            p.name: {
                "r_div_mean": {vi: a["r"][vi] / served_np for vi in range(3)},
                "unavail_incr_div_frac": {vi: a["unav"][vi] / n for vi in range(3)},
                "unres_frac": {vi: a["unres"][vi] / n for vi in range(3)},
                "diverted_frac": {vi: a["div"][vi] / n for vi in range(3)},
                "switches_per_hour": {
                    vi: (a["sw"][vi] / (n * dt_s / 3600.0) if p.min_interval_s is not None
                         and dt_s > 0 else np.full(g, np.nan)) for vi in range(3)},
                "switches_all_per_hour": {
                    vi: (a["sw_all"][vi] / (n * dt_s / 3600.0) if dt_s > 0
                         else np.full(g, np.nan)) for vi in range(3)},
                "base_switches_per_hour": (a["sw_base0"] / (n * dt_s / 3600.0) if dt_s > 0
                                           else np.full(g, np.nan)),
                "r_base_mean": (a["r_base"] / served_np if p.min_interval_s is not None
                                else (r0_sum / served_count.clamp(min=1.0)).cpu().numpy()),
                "unavail0_base_frac": (a["unav0_base"] / n if p.min_interval_s is not None
                                       else (unavail0 / n).cpu().numpy()),
            }
            for p in handoff_policies
            for a in (pol_acc[p.name],)
        },
    }
