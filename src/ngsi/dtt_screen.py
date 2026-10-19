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
    return torch.as_tensor(alts, dtype=torch.float64, device=device)


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

    alt_all = _shell_altitudes(interferer, device)  # [Nb]
    r_max_all = (
        slant_range_km_batch(alt_all, float(anchor_elev), earth_radius_km) if pfd else None
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
            # r_max.  r_max from the *serving* satellite's shell.
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
            _dbw_to_linear(_in_db(sep_w, fspl_all_b, eirp_max)) * vis_kb.to(torch.float64)
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
            * other_mask.to(torch.float64)
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
        vis_kb_f = vis_kb.unsqueeze(2).to(torch.float64)
        # Upper bound uses *all* visible B satellites at peak EIRP (no sidelobe
        # reduction), matching dtt_upper; the realistic variant adds the S.1528
        # sidelobe only of the non-serving B satellites.
        dtt_ua = (peak_lin * vis_kb_f).sum(-1)  # [Tc,G,Ka]
        dtt_ra = dtt_ca + (side_lin * other_mask.unsqueeze(2).to(torch.float64)).sum(-1)

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
    cn0_all = np.concatenate(cn0_blocks, axis=0) if cn0_blocks else np.empty((0, g))
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
    }
