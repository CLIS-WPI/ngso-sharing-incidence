"""Batched Delta T/T screening of a co-frequency pair (47 CFR 25.261(d)).

For an ordered pair (A victim, B interferer) that is co-frequency in the user
downlink, and each station/time sample:

* A's serving satellite is the highest-elevation visible A satellite.
* **Co-located variant**: the interferer is B's serving satellite at the same
  site (both systems serve a terminal there), at B's peak EIRP density; the
  off-axis angle is that between the two serving satellites.
* **Worst-case variant (upper bound)**: every visible B satellite radiates at B's
  peak EIRP density toward the site; the individual I/N add in linear units.
* An **event** is aggregate ``Delta T/T > 0.06`` (``I/N > -12.2 dB``).
* The event is **resolvable** when some *other* visible A satellite keeps
  aggregate ``Delta T/T <= 0.06`` (diversity); otherwise it is unresolvable.

The same geometry drives the 25.261(d) secondary result: the victim's own C/N,
its Shannon throughput with and without interference, and the unavailability
increase at C/N = 0 dB.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .constants import DTT_THRESHOLD
from .geometry import angular_separation_batch, elevation_deg_batch, unit_look_vectors
from .link_budget import noise_power_dbw_per_mhz, pattern_gain_batch
from .propagation import propagate_ecef
from .systems import SystemModel

LIGHT_SPEED_M_S = 299792458.0
VARIANT_NAMES = ("worst_case", "co_located")


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
    pfd_constant: bool = False,
    diversity_fspl: str = "interferer",
) -> dict:
    """Screen one ordered pair under the Delta T/T criterion.

    Returns per-station ``event_frac``/``unres_frac`` of shape ``[2, G]`` (variants
    0 worst case, 1 co-located), the victim C/N distribution (``cn0_mean``,
    ``cn0_median``, ``cn0_p10``) and its baseline availability at C/N = 0 dB
    (``avail0_frac``), and accumulator sums for the throughput secondary result
    (all means are ``sum / n_samples``).

    ``pfd_constant=False`` (default) holds the interferer EIRP density fixed
    (constant-EIRP).  ``pfd_constant=True`` models the SpaceX/SpaceX-style
    beam-steering power control: the interferer raises its EIRP with slant range
    so the PFD on the ground stays at its nadir value.  The interferer
    contribution then uses ``FSPL(altitude)`` instead of ``FSPL(actual range)``
    and is range-independent.  The victim's own carrier is left constant-EIRP so
    the baseline C/N distribution (and the availability denominators used by
    25.261(d)) is identical between the two models.

    ``diversity_fspl="interferer"`` (correct) evaluates the interference in the
    diversity test at the interferer's slant range.  ``"victim"`` reproduces the
    earlier behaviour that (incorrectly) reused the candidate victim satellite's
    range, retained only for attribution of that fix.
    """
    if diversity_fspl not in ("interferer", "victim"):
        raise ValueError("diversity_fspl must be 'interferer' or 'victim'")
    g = station_ecef.shape[0]
    t_total = len(times)
    k_a = min(int(k_a), victim.n_sats)
    k_b = min(int(k_b), interferer.n_sats)
    eye_a = torch.arange(k_a, device=device)
    noise_v = noise_power_dbw_per_mhz(victim_beam.system_noise_temp_k)
    bw_hz = float(victim_beam.bandwidth_mhz) * 1e6

    event_count = torch.zeros(2, g, dtype=torch.float64, device=device)
    unres_count = torch.zeros(2, g, dtype=torch.float64, device=device)
    served_count = torch.zeros(g, dtype=torch.float64, device=device)
    r0_sum = torch.zeros(g, dtype=torch.float64, device=device)
    r_int = {0: torch.zeros(g, dtype=torch.float64, device=device),
             1: torch.zeros(g, dtype=torch.float64, device=device)}
    r_div = {0: torch.zeros(g, dtype=torch.float64, device=device),
             1: torch.zeros(g, dtype=torch.float64, device=device)}
    unavail0 = torch.zeros(g, dtype=torch.float64, device=device)
    unavail_int = {0: torch.zeros(g, dtype=torch.float64, device=device),
                   1: torch.zeros(g, dtype=torch.float64, device=device)}
    unavail_incr = {0: torch.zeros(g, dtype=torch.float64, device=device),
                    1: torch.zeros(g, dtype=torch.float64, device=device)}
    cn0_sum = torch.zeros(g, dtype=torch.float64, device=device)
    dtt_colo_sum = torch.zeros(g, dtype=torch.float64, device=device)
    dtt_worst_sum = torch.zeros(g, dtype=torch.float64, device=device)
    cn0_blocks: list[np.ndarray] = []

    for s in range(0, t_total, chunk):
        e = min(s + chunk, t_total)
        tc = times[s:e]
        pa = propagate_ecef(victim.elements, tc, epoch_jd, earth_radius_km, device)
        pb = propagate_ecef(interferer.elements, tc, epoch_jd, earth_radius_km, device)
        ua, da = unit_look_vectors(pa, station_ecef)  # [Tc,G,Ka,3], [Tc,G,Ka]
        ub, db = unit_look_vectors(pb, station_ecef)
        elev_a = elevation_deg_batch(ua, up)
        elev_b = elevation_deg_batch(ub, up)
        vis_a = elev_a >= min_elevation_deg
        vis_b = elev_b >= min_elevation_deg

        va, ia = torch.topk(elev_a.masked_fill(~vis_a, float("-inf")), k_a, dim=-1)
        vb, ib = torch.topk(elev_b.masked_fill(~vis_b, float("-inf")), k_b, dim=-1)
        vis_ka = va >= min_elevation_deg
        vis_kb = vb >= min_elevation_deg
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

        # Interferer path loss.  With PFD-constant beam steering the satellite
        # raises EIRP with slant range so the PFD on the ground stays at its
        # nadir value; the received interference is then range-independent and
        # uses FSPL(altitude) in place of FSPL(actual range).
        if pfd_constant:
            alt_b = torch.linalg.norm(pb, dim=-1) - earth_radius_km  # [Tc, Nb] km
            alt_b_k = torch.gather(alt_b.unsqueeze(1).expand(-1, g, -1), 2, ib)
            fspl_serving_b = _fspl_db(alt_b_k[:, :, 0], freq_ghz)
            fspl_all_b = _fspl_db(alt_b_k, freq_ghz)
        else:
            fspl_serving_b = _fspl_db(dist_b, freq_ghz)
            fspl_all_b = _fspl_db(db_k, freq_ghz)

        def _in_db(off_axis, fspl_db, eirp_density):
            g_rx = pattern_gain_batch(off_axis, rx_peak_gain_dbi, rx_diameter_m, freq_ghz, pattern)
            return eirp_density + g_rx - fspl_db - noise_v

        # --- co-located: single interferer (B serving satellite) ---
        sep_c = angular_separation_batch(serving_a, serving_b)
        dtt_c = _dbw_to_linear(_in_db(sep_c, fspl_serving_b, interf_beam.eirp_density_dbw_per_mhz))
        # --- worst case: aggregate over every visible B satellite ---
        sep_w = angular_separation_batch(serving_a.unsqueeze(2), ub_k)  # [Tc,G,Kb]
        dtt_w = (
            _dbw_to_linear(_in_db(sep_w, fspl_all_b, interf_beam.eirp_density_dbw_per_mhz))
            * vis_kb.to(torch.float64)
        ).sum(-1)

        dtts = {0: dtt_w, 1: dtt_c}
        has_b_var = {0: vis_kb.any(-1), 1: has_b}
        for vi in (0, 1):
            event = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
            event_count[vi] += event.sum(0).to(torch.float64)

        # Diversity: alternate visible A satellites that keep DTT <= threshold.
        # The interfering signal travels from the interferer, so the correct path
        # loss is the station->interferer range.  ``"victim"`` reproduces the
        # earlier (incorrect) reuse of the candidate victim range for attribution.
        cand = vis_ka & (eye_a.reshape(1, 1, -1) != 0)
        if diversity_fspl == "victim":
            fspl_div = _fspl_db(da_k, freq_ghz)  # [Tc,G,Ka]
            fspl_div_serving = fspl_div
            fspl_div_all = fspl_div.unsqueeze(-1)  # [Tc,G,Ka,1]
        else:
            fspl_div_serving = fspl_serving_b.unsqueeze(2)  # [Tc,G,1]
            fspl_div_all = fspl_all_b.unsqueeze(2)  # [Tc,G,1,Kb]
        sep_ca = angular_separation_batch(ua_k, serving_b.unsqueeze(2))  # [Tc,G,Ka]
        dtt_ca = _dbw_to_linear(_in_db(sep_ca, fspl_div_serving, interf_beam.eirp_density_dbw_per_mhz))
        sep_wa = angular_separation_batch(ua_k.unsqueeze(3), ub_k.unsqueeze(2))  # [Tc,G,Ka,Kb]
        dtt_wa = (
            _dbw_to_linear(_in_db(sep_wa, fspl_div_all, interf_beam.eirp_density_dbw_per_mhz))
            * vis_kb.reshape(vis_kb.shape[0], vis_kb.shape[1], 1, -1)
        ).sum(-1)
        resolvable = {
            0: (cand & (dtt_wa <= DTT_THRESHOLD)).any(-1),
            1: (cand & (dtt_ca <= DTT_THRESHOLD)).any(-1),
        }
        for vi in (0, 1):
            event = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
            unres = event & ~resolvable[vi]
            unres_count[vi] += unres.sum(0).to(torch.float64)

        # --- throughput secondary result ---
        served = has_a
        served_count += served.sum(0).to(torch.float64)
        r0 = _shannon_rate(cn0, bw_hz, margin_db)
        for vi in (0, 1):
            event = has_a & has_b_var[vi] & (dtts[vi] > DTT_THRESHOLD)
            unres = event & ~resolvable[vi]
            cni = _cni_from_dtt(cn0, dtts[vi])
            r1 = _shannon_rate(cni, bw_hz, margin_db)
            r_int[vi] += (torch.where(event, r1, r0) * served).sum(0)
            r_div[vi] += (torch.where(unres, r1, r0) * served).sum(0)
            unavail_int[vi] += ((cni < 0.0) & event).sum(0).to(torch.float64)
            # Absolute availability change = A0 - A_int = P(C/N0 >= 0 and C/Nint < 0):
            # only samples that were available at baseline count as a *new* outage.
            unavail_incr[vi] += ((cn0 >= 0.0) & (cni < 0.0) & served).sum(0).to(torch.float64)
        r0_sum += (r0 * served).sum(0)
        unavail0 += ((cn0 < 0.0) & served).sum(0).to(torch.float64)
        cn0_sum += (cn0 * served).sum(0)
        dtt_colo_sum += (dtt_c * served).sum(0)
        dtt_worst_sum += (dtt_w * served).sum(0)
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
        "r_int_mean": {vi: (r_int[vi] / served_count.clamp(min=1.0)).cpu().numpy() for vi in (0, 1)},
        "r_div_mean": {vi: (r_div[vi] / served_count.clamp(min=1.0)).cpu().numpy() for vi in (0, 1)},
        "cn0_mean": (cn0_sum / served_count.clamp(min=1.0)).cpu().numpy(),
        "cn0_median": cn0_median,
        "cn0_p10": cn0_p10,
        "avail0_frac": avail0_frac,
        "dtt_colo_mean": (dtt_colo_sum / served_count.clamp(min=1.0)).cpu().numpy(),
        "dtt_worst_mean": (dtt_worst_sum / served_count.clamp(min=1.0)).cpu().numpy(),
        "unavail_nodiv_colo_frac": (unavail_int[1] / n).cpu().numpy(),
        "unavail_nodiv_worst_frac": (unavail_int[0] / n).cpu().numpy(),
        "unavail_incr_colo_frac": (unavail_incr[1] / n).cpu().numpy(),
        "unavail_incr_worst_frac": (unavail_incr[0] / n).cpu().numpy(),
        "unavail0_frac": (unavail0 / n).cpu().numpy(),
        "n_samples": t_total,
    }
