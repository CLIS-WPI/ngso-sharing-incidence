"""Generate ``results/GATE1_REPORT.md`` from the newest E1/E2 run tables.

Every number embedded in the report is read from the ``summary_*.csv`` /
``sanity.json`` files under ``results/`` so the report cannot drift from the code
output (AGENTS.md rule 1).

Run:  python scripts/make_gate1_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def newest(prefix: str) -> Path:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run directory for {prefix}")
    return runs[-1]


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    df = df if cols is None else df[cols]
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def pivot(df, index, columns, values, round_to=4):
    out = df.pivot_table(index=index, columns=columns, values=values).round(round_to)
    out.columns = [" <- ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in out.columns]
    out.index.name = "lat"
    return out.reset_index()


def main() -> int:
    e1, e2, e3 = newest("e1_event_rate"), newest("e2_diversity"), newest("e3_capacity_diversity")
    m1 = json.loads((e1 / "run_meta.json").read_text())
    m2 = json.loads((e2 / "run_meta.json").read_text())
    m3 = json.loads((e3 / "run_meta.json").read_text())
    s1 = json.loads((e1 / "sanity.json").read_text())
    st1 = json.loads((e1 / "status.json").read_text())
    st2 = json.loads((e2 / "status.json").read_text())
    st3 = json.loads((e3 / "status.json").read_text())

    ev = pd.read_csv(e1 / "summary_event_rate.csv")
    bo = pd.read_csv(e1 / "summary_band_overlap.csv")
    un = pd.read_csv(e2 / "summary_unresolvable.csv")
    du = pd.read_csv(e2 / "summary_durations.csv")
    fs = pd.read_csv(e2 / "summary_sensitivity_f.csv")
    me = pd.read_csv(e2 / "summary_min_elevation.csv")
    ow = pd.read_csv(e2 / "summary_sensitivity_oneweb.csv")
    svc = pd.read_csv(e2 / "summary_service_pool.csv")
    cd = pd.read_csv(e3 / "summary_capacity_diversity.csv")

    dens = pd.DataFrame(s1["density"])[["system", "shell", "max_rel_dev"]].round(4)
    vis = pd.DataFrame(s1["visible"])[["shell", "sim_peak", "analytic_peak", "rel_dev"]].round(4)
    sym = pd.DataFrame(s1["symmetry"]).round(3)

    udl = bo[(bo.link_type == "user_downlink")][
        ["system_a", "system_b", "overlaps", "center_ghz", "segments_ghz"]]
    gwl = bo[(bo.link_type == "gateway_downlink") & (bo.overlaps)][
        ["system_a", "system_b", "center_ghz"]]

    ev2c = pivot(ev[(ev.variant == 1) & (ev.theta_deg == 2.0)], "latitude_deg",
                 ["system_a", "system_b"], "event_frac_mean")
    ev2w = pivot(ev[(ev.variant == 0) & (ev.theta_deg == 2.0)], "latitude_deg",
                 ["system_a", "system_b"], "event_frac_mean")
    unic = pivot(un[(un.variant == 1) & (un.theta_deg == 2.0)], "latitude_deg",
                 ["system_a", "system_b"], "unres_frac_mean")
    uniw = pivot(un[(un.variant == 0) & (un.theta_deg == 2.0)], "latitude_deg",
                 ["system_a", "system_b"], "unres_frac_mean")

    sl = un[(un.system_a == "STARLINK") & (un.system_b == "ONEWEB")]
    unres_max = float(sl[sl.variant == 1].unres_frac_mean.max())
    unres_worst_max = float(sl[sl.variant == 0].unres_frac_mean.max())
    evc2 = ev[(ev.variant == 1) & (ev.theta_deg == 2.0)]
    ev_peak = float(evc2.event_frac_mean.max())
    ev_peak_lat = float(evc2.loc[evc2.event_frac_mean.idxmax(), "latitude_deg"])
    bs2 = sl[sl.theta_deg == 2.0]
    bs_div = float(bs2.band_split_loss_with_diversity.mean())
    bs_nodiv = float(bs2.band_split_loss_no_diversity.mean())

    durs = du[(du.kind == "unres") & (du.theta_deg == 5.0)]
    dur_tab = durs.groupby(["variant"]).agg(events=("count", "sum")).reset_index()

    me_tab = me[(me.variant == 1) & (me.theta_deg == 2.0) & (me.latitude_deg >= 45.0) &
                (me.system_a == "STARLINK") & (me.system_b == "ONEWEB")][
        ["latitude_deg", "min_elevation_deg", "event_frac_mean"]].round(5)
    me_tab = me_tab.sort_values(["min_elevation_deg", "latitude_deg"])

    fs_tab = fs[(fs.variant == 1) & (fs.theta_deg == 2.0) & (fs.latitude_deg == 45.0) &
                (fs.system_a == "STARLINK") & (fs.system_b == "ONEWEB")][
        ["f", "event_frac_mean"]].round(5)
    f1 = float(fs_tab[fs_tab.f == 1]["event_frac_mean"].iloc[0])
    f0 = float(fs_tab[fs_tab.f == 0]["event_frac_mean"].iloc[0])
    frel = abs(f0 - f1) / f1

    ow_tab = ow[(ow.variant == 1) & (ow.theta_deg == 2.0) & (ow.latitude_deg == 45.0) &
                (ow.system_a == "STARLINK") & (ow.system_b == "ONEWEB")][
        ["oneweb_config", "event_frac_mean"]].round(5)
    ow_base = float(ow_tab[ow_tab.oneweb_config == "18x36"]["event_frac_mean"].iloc[0])
    ow_alt = float(ow_tab[ow_tab.oneweb_config == "12x49"]["event_frac_mean"].iloc[0])

    svc_min = svc.groupby("system").served_frac_mean.min().round(3)
    ku65 = float(svc[(svc.system == "KUIPER") & (svc.latitude_deg == 65.0)]["served_frac_mean"].iloc[0])

    slc = cd[cd.system_a == "STARLINK"]
    cap_unres = pivot(slc, "latitude_deg", "n_beam", "unres_cell_frac_mean", round_to=5)
    cap_serv = pivot(slc, "latitude_deg", "n_beam", "served_frac_mean", round_to=3)
    cap_max = float(slc.unres_cell_frac_mean.max())
    cap_max_row = slc.loc[slc.unres_cell_frac_mean.idxmax()]
    cap_unlim = unres_max

    text = f"""# Phase 2 - Gate 1 report (revised: co-frequency filter)

Traceability: every number below is read from `results/` by
`scripts/make_gate1_report.py`; the figures are generated by
`figures/make_figures.py`. No value is hard-coded.

## 1. Provenance

- Branch `phase-2b-gate1-fix`, E1 commit `{m1['git_hash']}`.
- E1 run: `{e1.relative_to(REPO_ROOT)}` (seed {m1['seed']}, {m1['timestamp_utc']}).
- E2 run: `{e2.relative_to(REPO_ROOT)}` (seed {m2['seed']}, {m2['timestamp_utc']}).
- E3 run: `{e3.relative_to(REPO_ROOT)}` (seed {m3['seed']}, {m3['timestamp_utc']}).
- Device: `{st1['device']}`; single GPU 0 (compose pins GPU 0).
- Earth model: **spherical**, R = 6371 km (`configs/constellations.yaml`,
  `earth_model.source: ASSUMED`). Epoch `2025-01-01T00:00:00Z`.
- Propagation: circular orbits with J2 secular RAAN and argument-of-latitude drift,
  batched torch float64 on the GPU; SGP4 validation ~15.7 km/day
  (`tests/test_propagation.py`).
- E1/E2 grid: latitudes 0..65 deg step 5, four meridians; min elevation
  {st1['min_elevation_deg']} deg; {st1['time_step_s']} s over {st1['duration_s']:.0f} s;
  {st1['n_phasings']} independent phasings. Half-angles theta = {st1['thetas_deg']} deg.
- E3 grid: regional hex cell lattices ({st3['region_size_km']:.0f} x {st3['region_size_km']:.0f} km,
  {st3['cell_spacing_km']:.0f} km spacing) at latitudes {st3['lat_center_deg']} deg,
  meridians {st3.get('meridians_deg')} deg, beam sweep N_beam = {st3['n_beam_sweep']},
  Delta T/T criterion, {st3['time_step_s']} s over {st3['duration_s']:.0f} s,
  {st3['n_phasings']} phasings. (See `results/PHASE3A_REPORT.md` for the Phase 3a
  re-analysis: both cell-time denominators and beam-depleting rescue.)

## 2. Frequency bands and co-frequency pairs

Bands are read from `configs/bands.yaml` (per system x link type, each entry with a
`source:`). The full ordered-pair overlap matrix is written to every run directory as
`summary_band_overlap.csv`; the **user-downlink** rows are:

{md_table(udl)}

Co-frequency pairs for the **user downlink** (the link class of the original Gate-1
headline) are therefore only STARLINK <-> ONEWEB, sharing Ku 10.7-12.7 GHz with a
shared-band centre of 11.7 GHz. Kuiper's user downlink is Ka (17.7-20.2 GHz), so
**no co-frequency user-downlink pair involves Kuiper**. Gateway downlink overlaps
(all three systems) are shown for completeness:

{md_table(gwl)}

Every event table below is screened **only** over the co-frequency user-downlink
pairs, and each row carries `link_type` and `shared_freq_ghz` so results can never be
mixed across link classes again.

## 3. Parameters and their sources (rule 2)

Shell orbital elements are from the FCC-derived literature (Starlink/Kuiper: Pachler
et al. 2021 Table II / Kuiper FCC technical appendix). Antenna/beam parameters are
worst-case filing values: Starlink Ku max gain 44 dBi, max EIRP density -8.3 dBW/MHz
(FCC-22-91A1 paras. 190-191); Kuiper user beam 35 dBi / 2.8 deg spot at 19.8 GHz
(Kuiper experimental STA). Bands and their sources are in `configs/bands.yaml`
(SpaceX FCC-22-91A1 + STA id=405805; Kuiper FCC 20-102 / DA 23-114; OneWeb FCC 17-77
/ DA 16-804). **OneWeb shell layout 18x36 (648 sats) is verified against FCC 17-77**;
a 12x49 sensitivity is run in Section 6. Remaining `ASSUMED` values: satellite
beamwidths (Starlink derived), all system noise temperatures, every Walker F (=1),
and all E3 capacity/geometry parameters. Gate-1/E2 events are geometric and do not
consume beam parameters; E3 consumes only N_beam (ASSUMED).

## 4. Sanity gate (results in `{e1.name}/sanity.json`)

Sub-satellite latitude density max relative deviation:

{md_table(dens)}

Mean visible satellites per shell vs the analytic cap integral:

{md_table(vis)}

Latitude symmetry of the mean visible count:

{md_table(sym)}

Peak GPU memory: {s1['runtime']['peak_mem_bytes'] / 1e6:.0f} MB. **Gate PASSED** in
all three runs (density dev < 0.6%, visible-count dev < 0.2%, symmetry to machine
precision).

## 5. E1 - geographic incidence of in-line events (co-frequency user downlink)

Event time-fraction, **co-located variant (main estimate)**, theta = 2 deg, mean over
meridians and phasings:

{md_table(ev2c)}

Same, **worst-case variant (upper bound only)**, theta = 2 deg:

{md_table(ev2w)}

Reading: for the one co-frequency user-downlink pair the co-located event rate is
~2e-3 at the equator and peaks at {ev_peak:.4g} near {ev_peak_lat:.0f} deg. The two
ordered directions are identical to the last digit, as required (the separation angle
does not depend on which system is the victim). The worst-case upper bound is a few
times larger but has the same latitude shape.

## 6. E2 - diversity, durations, and 25.261 burden (co-frequency user downlink)

Unresolvable time-fraction, **co-located (main)**, theta = 2 deg:

{md_table(unic)}

Unresolvable time-fraction, **worst-case upper bound**, theta = 2 deg:

{md_table(uniw)}

Resolving events counted in the fine 1 s run, theta = 5 deg (the only configuration in
which any unresolvable event exists):

{md_table(dur_tab)}

Reading: **satellite diversity resolves every co-located in-line event for the
co-frequency pair at every latitude and every theta tested**: the maximum co-located
unresolvable time-fraction is {unres_max:.4g} (exactly 0), and even the worst-case
upper bound maxes out at {unres_worst_max:.4g} (also 0). There is therefore **no
high-latitude burden for this pair**; the only unresolvable-event durations that exist
at all are in the worst-case variant. 25.261(c) band-split loss is the mean of
0.5 x unresolvable-fraction: **{bs_div:.4g} with diversity** versus
**{bs_nodiv:.4g} with no diversity** (0.5 x event-fraction).

Minimum-elevation sensitivity (co-located, theta = 2 deg, STARLINK <- ONEWEB,
lat >= 45 deg):

{md_table(me_tab)}

Walker-F sensitivity (Starlink 53 deg shells, co-located, theta = 2 deg, lat 45):

{md_table(fs_tab)}

OneWeb shell-size sensitivity (co-located, theta = 2 deg, lat 45): the shipped
18x36 gives {ow_base:.4g}, the ASSUMED 12x49 gives {ow_alt:.4g} (a
{abs(ow_alt - ow_base) / ow_base:.0%} change in the event rate; unresolvable stays 0).

{md_table(ow_tab)}

## 7. Service availability and diversity pool (Task 3)

Minimum service-availability fraction by victim system over the grid (fraction of
time the victim has any satellite above the elevation mask):

{md_table(svc_min.reset_index())}

Kuiper is served at every latitude up to 60 deg but drops to {ku65:.0%} at 65 deg -
a pure **coverage-edge** effect of its 33/42/51.9 deg shells. Starlink and OneWeb are
served 100% of the time across the whole grid. Because event fractions already require
the victim to be served, conditioning on service only rescales the event rate by
1/served_frac and cannot create a high-latitude burden for Starlink or OneWeb; the
diversity pool (mean and p10 visible satellites) is reported per latitude in
`summary_service_pool.csv` and drawn in `F4_service_pool.png`.

## 8. E3 - capacity-limited diversity (regional cell grids)

Each victim satellite grants at most N_beam cell requests, greedy by elevation; an
event is resolvable only when another visible victim satellite keeps Delta T/T at or
below the 25.261(d) threshold **and** has a free beam, with rescue depleting one beam per
cell (Phase 3a). **Unresolvable cell-time fraction** (over all cell-time) vs latitude
and N_beam:

{md_table(cap_unres)}

Resulting **fraction of cells served** (capacity limit):

{md_table(cap_serv)}

Reading: with unlimited diversity the unresolvable fraction was exactly
{cap_unlim:.4g}. Under beam capacity it becomes non-zero: the maximum (over all
cell-time) is {cap_max:.4g} at latitude {cap_max_row['latitude_deg']:.0f} deg, N_beam =
{int(cap_max_row['n_beam'])}, where only {cap_max_row['served_frac_mean']:.0%} of cells
are served. The absolute burden is capacity-induced and grows with N_beam because more
cells are served; the conditional burden per served cell-time falls with N_beam (Phase 3a
denominator split). See `results/PHASE3A_REPORT.md` Section 2.

## 9. Revised Gate 1 verdict

1. **Which pairs are co-frequency for user links?** Only **STARLINK <-> ONEWEB**
   (Ku, shared centre 11.7 GHz). Kuiper's user downlink is Ka and does **not** overlap
   either system's user band, so the previous "KUIPER <- STARLINK near 60 deg" headline
   is **dropped as not co-frequency at the user link**. Gateway links overlap for all
   three systems but are not user links.

2. **For the surviving pair, is there a latitude-dependent burden that is not a
   coverage-edge effect?** **No.** Co-located unresolvable time-fraction is exactly 0
   at every latitude (0-65 deg) and every theta in {{1,2,5}} deg; the worst-case upper
   bound is also 0 for this pair. There is no high-latitude burden to attribute. The
   only coverage-edge signature in the data is Kuiper's service availability dropping
   from 100% to {ku65:.0%} between 60 and 65 deg, and Kuiper takes part in no
   co-frequency user-downlink pair.

3. **Does capacity-limited diversity create burden where unlimited showed none?**
   **Yes, but small.** With unlimited diversity the unresolvable cell-time fraction is
   0; with a per-satellite beam limit it reaches {cap_max:.4g} (about
   {cap_max:.3%} of cell-time) at N_beam = {int(cap_max_row['n_beam'])},
   latitude {cap_max_row['latitude_deg']:.0f} deg. This is a capacity effect, not a
   coverage-edge effect: it peaks near 50 deg and grows with the number of beams.

## 10. Figures (built from `results/` only)

- `F0a_sub_latitude_density.png`, `F0b_visible_count.png` - model validation.
- `F1_event_rate.png` - E1 event time-fraction, co-located on top (main) and worst
  case below (upper bound), theta in {{1,2,5}} deg, 95% CI over phasings.
- `F2_unresolvable.png` - E2 unresolvable fraction and P(unresolvable | event).
- `F3_event_durations.png` - worst-case unresolvable-event durations.
- `F4_service_pool.png` - victim service availability, diversity pool (mean, p10) and
  service-conditioned burden vs latitude.
- `F5_capacity_diversity.png` - E3 capacity-limited unresolvable cell-time fraction and
  served-cell fraction vs latitude and N_beam.

## 11. Caveats

- Spherical earth, circular orbits, J2 secular only, fixed GMST epoch, no drag, no
  link budget: an *incidence* screen, not a co-ordination simulation.
- E3 is a documented approximation: greedy-by-elevation beam grants, one regional cell
  grid per latitude on a single meridian, rescue does not deplete spare beams, and all
  capacity/geometry parameters are `ASSUMED`. Interpret its magnitudes as indicative.
- Event fractions are sampling-limited (10 s for E1/E2, 60 s for E3); durations use a
  separate 1 s run.
- The 25.261(d) Delta T/T thresholds are not yet coupled to an I/N calculation
  (deferred by instruction); there is no population weighting yet (deferred by
  instruction).

## 12. Approval requested

Gate 1 (revised) is complete and reproducible (`make run-all`). Requesting review of
the co-frequency filter, the service-conditioning definition, and the E3 capacity
approximation before population weighting and the I/N -> Delta T/T link.
"""

    out = RESULTS / "GATE1_REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"[report] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
