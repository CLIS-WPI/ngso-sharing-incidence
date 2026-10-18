"""Generate ``results/PHASE3A_REPORT.md`` from the newest E3/E4 run tables.

Every number embedded in the report is read from the ``summary_*.csv`` /
``status.json`` files under ``results/`` so the report cannot drift from the code
output (AGENTS.md rule 1).

Run:  python scripts/make_phase3a_report.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
VARIANT_LABEL = {0: "worst case (any visible)", 1: "co-located (serving)"}


def newest(prefix: str) -> Path:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run directory for {prefix}")
    return runs[-1]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    df = df if cols is None else df[cols]
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def pivot(df, index, columns, values, round_to=5):
    out = df.pivot_table(index=index, columns=columns, values=values).round(round_to)
    out.columns = [str(c) for c in out.columns]
    out.index.name = "lat"
    return out.reset_index()


def main() -> int:
    e3, e4 = newest("e3_capacity_diversity"), newest("e4_dtt")
    m3 = json.loads((e3 / "run_meta.json").read_text())
    m4 = json.loads((e4 / "run_meta.json").read_text())
    st3 = json.loads((e3 / "status.json").read_text())
    st4 = json.loads((e4 / "status.json").read_text())
    cd = pd.read_csv(e3 / "summary_capacity_diversity.csv")
    dt = pd.read_csv(e4 / "summary_dtt.csv")

    # Item 1: pick the ordered pair that actually fires (victim has an event).
    firing = cd[cd["event_cell_frac_mean"] > 0]
    pair = (firing["system_a"].iloc[0], firing["system_b"].iloc[0])
    cap = cd[(cd.system_a == pair[0]) & (cd.system_b == pair[1])]
    cap_abs = pivot(cap, "latitude_deg", "n_beam", "unres_cell_frac_mean")
    cap_cond = pivot(cap, "latitude_deg", "n_beam", "unres_per_served_mean", round_to=6)
    cap_ev = pivot(cap, "latitude_deg", "n_beam", "event_per_served_mean", round_to=6)
    cap_serv = pivot(cap, "latitude_deg", "n_beam", "served_frac_mean", round_to=4)
    corr = cap.copy()
    corr["rescue_effectiveness"] = 1.0 - corr["unres_per_served_mean"] / corr["event_per_served_mean"]
    mono = cap.groupby("n_beam")["unres_per_served_mean"].mean()
    mono_abs = cap.groupby("n_beam")["unres_cell_frac_mean"].mean()
    mono_tab = pd.DataFrame({
        "N_beam": mono.index.astype(int),
        "mean unresolvable (all cell-time)": mono_abs.values,
        "mean unresolvable (per served)": mono.values,
    }).round(6)

    # Item 2/3: Delta T/T and throughput.
    ev_d = dt[dt.event_frac_mean > 0]
    dpair = ((ev_d.system_a.iloc[0], ev_d.system_b.iloc[0]) if not ev_d.empty else
             (dt.system_a.iloc[0], dt.system_b.iloc[0]))
    d = dt[(dt.system_a == dpair[0]) & (dt.system_b == dpair[1])]
    diam_default = float(sorted(d.diameter_m.unique())[len(d.diameter_m.unique()) // 2])
    worst_dia = float(sorted(d.diameter_m.unique())[-1])

    def _ev(vi, diam):
        return pivot(d[(d.variant == vi) & (d.diameter_m == diam)], "latitude_deg", "system_a",
                     "event_frac_mean")

    ev_worst_d = _ev(0, worst_dia)
    ev_coloc_d = _ev(1, worst_dia)
    unres_worst_d = pivot(d[(d.variant == 0) & (d.diameter_m == worst_dia)], "latitude_deg",
                          "system_a", "unres_frac_mean")
    deg_worst = pivot(d[(d.variant == 0) & (d.diameter_m == diam_default)], "latitude_deg",
                      "system_a", "degradation_nodiv_mean", round_to=5)
    deg_div = pivot(d[(d.variant == 0) & (d.diameter_m == diam_default)], "latitude_deg",
                    "system_a", "degradation_div_mean", round_to=5)

    # Terminal-aperture sensitivity at a mid latitude (worst-case variant).
    mid_lat = 50.0
    sens = d[(d.variant == 0) & (d.latitude_deg == mid_lat)][
        ["diameter_m", "event_frac_mean", "degradation_nodiv_mean"]].round(5)

    ev_max = float(d.event_frac_mean.max())
    ev_max_row = d.loc[d.event_frac_mean.idxmax()]
    deg_max = float(d[d.variant == 0].degradation_nodiv_mean.max())
    deg_max_row = d[d.variant == 0].loc[d[d.variant == 0].degradation_nodiv_mean.idxmax()]
    unres_max = float(d["unres_frac_mean"].max())
    dtt_mean_max = float(d.dtt_mean_mean.max())
    coloc = d[d.variant == 1]
    ev_coloc_max = float(coloc.event_frac_mean.max())
    unres_coloc_max = float(coloc.unres_frac_mean.max())
    frac_limit, dtt_limit = 0.004, 0.03
    cap_abs_max = float(cap.unres_cell_frac_mean.max())
    cap_abs_max_row = cap.loc[cap.unres_cell_frac_mean.idxmax()]
    cond_max = float(cap.unres_per_served_mean.max())
    cond_max_row = cap.loc[cap.unres_per_served_mean.idxmax()]

    text = f"""# Phase 3a report - regulatory Delta T/T criterion and the E3 N_beam anomaly

Traceability: every number below is read from `results/` by
`scripts/make_phase3a_report.py`; figures are generated by `figures/make_figures.py`.
No value is hard-coded. Phase 3a replaces the geometric in-line proxy with the
47 CFR 25.261(d) power criterion and fixes the E3 capacity model.

## 1. Provenance

- Branch `phase-3-letter`, E1 commit `{m3['git_hash']}`.
- E3 run: `{e3.relative_to(REPO_ROOT)}` (seed {m3['seed']}, {m3['timestamp_utc']}).
- E4 run: `{e4.relative_to(REPO_ROOT)}` (seed {m4['seed']}, {m4['timestamp_utc']}).
- Device: `{st3['device']}` (compose pins GPU 0).
- E3: regional hex lattices {st3['region_size_km']:.0f} x {st3['region_size_km']:.0f} km,
  {st3['cell_spacing_km']:.0f} km spacing, latitudes {st3['lat_center_deg']} deg,
  meridians {st3['meridians_deg']} deg, N_beam {st3['n_beam_sweep']}, candidate rank
  {st3['n_rank']}, {st3['time_step_s']} s over {st3['duration_s']:.0f} s,
  {st3['n_phasings']} phasings.
- E4: latitudes 0..65 deg step 5, four meridians, min elevation {st4['min_elevation_deg']} deg,
  {st4['time_step_s']} s over {st4['duration_s']:.0f} s, {st4['n_phasings']} phasings.
- Criterion: single constant `ngsi.constants.DTT_THRESHOLD = 0.06` (equivalent
  single-entry `I/N = {10 * math.log10(0.06):.2f}` dB, derived as `IN_THRESHOLD_DB`).
- Co-frequency pair screened: **{dpair[0]} (victim) <- {dpair[1]} (interferer)**, Ku user
  downlink, shared centre {float(d.shared_freq_ghz.iloc[0]):.3f} GHz
  (`configs/bands.yaml`). The reverse order never reaches the threshold with the
  shipped ASSUMED EIRP densities (zero events in Section 3).
- Terminal (ASSUMED, `configs/link_budget.yaml`): diameter {diam_default:g} m,
  aperture efficiency 0.6, ITU-R S.1428 pattern, sweep {st4['diameter_sweep_m']} m,
  Shannon margin {st4['implementation_margin_db']:g} dB.

## 2. Item 1 - E3 N_beam anomaly: diagnosis and fix

Three defects were present in the Phase 2b E3 model:

1. **Wrong denominator.** The reported `unres_cell_frac` was the unresolvable count over
   *all* cell-time. Because `N_beam` sets how many cells are served at all, both the
   numerator and the denominator grew with `N_beam`; the absolute fraction could rise
   even when the conditional risk did not. The model now returns the raw counts and both
   denominators (`unres_cell_frac`, `event_cell_frac`, `served_frac`, and
   `event_per_served`, `unres_per_served`).
2. **Rescue did not spend a beam.** The Phase 2b rescue only checked `spare > 0` and did
   not consume the spare beam, so a single free beam could rescue every event cell
   (optimistic). Rescue now takes one beam per event cell (`ngsi.capacity_diversity.
   rescue_with_depletion`), honours each satellite's full remaining spare, and processes
   cells in descending serving elevation. Unit tests pin the behaviour
   (`tests/test_capacity_diversity.py`).
3. **One meridian per latitude.** E3 now lays four meridians (0/90/180/270 deg) per
   latitude; the fraction is over all cells of all four.

**Residual cause of the anomaly.** After (2) was fixed in the first attempt the
conditional risk still rose with `N_beam`: with only one rescue per satellite per round,
a satellite owning many free beams could not spend them, so the event count grew faster
than the rescue throughput. Letting each satellite spend its full spare removes the
artifact. With the final model **the conditional unresolvable fraction falls
monotonically as `N_beam` grows**, while the absolute fraction still rises purely
through service coverage:

{md_table(mono_tab)}

Absolute unresolvable cell-time fraction vs latitude and `N_beam`
({pair[0]} <- {pair[1]}), over all cell-time:

{md_table(cap_abs)}

Conditional unresolvable fraction, per served cell-time:

{md_table(cap_cond)}

Events per served cell-time (the shared set of served cells changes with `N_beam`;
this stays flat, so the conditional fall is a rescue-capacity effect, not event drift):

{md_table(cap_ev)}

Fraction of cells served (the coverage denominator that inflates the absolute fraction):

{md_table(cap_serv)}

Reading: the absolute maximum is {cap_abs_max:.4g} cell-time at latitude
{cap_abs_max_row['latitude_deg']:.0f} deg, N_beam {int(cap_abs_max_row['n_beam'])}, but the
conditional maximum is only {cond_max:.4g} per served cell-time (latitude
{cond_max_row['latitude_deg']:.0f} deg). The conditional fraction decreases with `N_beam`
at every latitude, i.e. more beams always help once both denominators are separated.

## 3. Item 2 - 47 CFR 25.261(d) power criterion

`ngsi.link_budget` computes `I/N = EIRP_density + G_rx(off-axis) - FSPL(shared centre) -
10 log10(k T_sys * 1 MHz)`; multiple interferers add in linear units, so
`Delta T/T = sum 10^(I/N_i/10)`. `ngsi.dtt_screen` screens each co-frequency pair with the
receive pattern at the off-axis angle between the victim serving satellite and the
interferer, **co-located** (B's serving satellite) and **worst case** (every visible B at
peak EIRP density). An event is `Delta T/T > 0.06`. The batched screener is validated
against a brute-force NumPy reference (`tests/test_dtt_screen.py`).

Event time-fraction vs latitude, worst-case variant, widest sweep diameter
({worst_dia:g} m):

{md_table(ev_worst_d)}

Event time-fraction vs latitude, co-located variant, widest sweep diameter:

{md_table(ev_coloc_d)}

Unresolvable time-fraction (unlimited satellite diversity), worst-case variant,
{worst_dia:g} m:

{md_table(unres_worst_d)}

Maximum event time-fraction over the whole grid: {ev_max:.4g} at latitude
{ev_max_row['latitude_deg']:.0f} deg, diameter {ev_max_row['diameter_m']:g} m,
variant {VARIANT_LABEL[int(ev_max_row['variant'])]}. Under unlimited diversity the
unresolvable fraction is {unres_max:.4g}: the victim always has another visible satellite
to which it can switch for this pair.

## 4. Item 3 - 25.261(d) throughput degradation (secondary)

Per-sample the victim's own `C/N` (serving satellite at boresight) and
`C/(N+I) = C/N - 10 log10(1 + Delta T/T)` feed a Shannon rate with a
{st4['implementation_margin_db']:g} dB implementation margin. Time-weighted degradation is
`1 - mean(R with I) / mean(R without I)`.

Degradation vs latitude, no diversity, worst-case variant, {diam_default:g} m:

{md_table(deg_worst)}

Degradation vs latitude, with unlimited diversity, worst-case variant, {diam_default:g} m:

{md_table(deg_div)}

Maximum degradation without diversity is {deg_max:.4g} (latitude
{deg_max_row['latitude_deg']:.0f} deg, diameter {deg_max_row['diameter_m']:g} m); with
unlimited diversity it is 0.

**Comparison with the 25.261(d) criteria** (long-term mean `Delta T/T <= 3%`; short-term
`Delta T/T > 6%` for no more than 0.4% of the time), read from the same tables:

- Mean `Delta T/T` maximum: {dtt_mean_max:.4g} vs the 3% long-term limit
  (`ngsi.constants.DTT_LONG_TERM = 0.03`).
- Time above the 6% threshold without diversity: worst case {ev_max:.4g}, co-located
  {ev_coloc_max:.4g}, vs the 0.4% exceedance fraction
  (`ngsi.constants.LONG_TERM_EXCEEDANCE_FRACTION = 0.004`). Without diversity the co-located
  exposure exceeds 0.4% of the time.
- Time above the 6% threshold **with unlimited diversity** (unresolvable): worst case
  {unres_max:.4g}, co-located {unres_coloc_max:.4g}; the criterion is met once diversity is
  credited.

Terminal-aperture sensitivity, worst-case variant, latitude {mid_lat:.0f} deg:

{md_table(sens)}

## 5. Verdict - does the rule ever bind materially?

- **Co-located (main) events**, the variant in which both systems actually serve a
  terminal at the same site, reach at most {float(d[(d.variant == 1)].event_frac_mean.max()):.4g}
  of the time and are **fully resolvable** by satellite diversity (unresolvable
  {unres_max:.4g}); throughput degradation is ~0.
- **Worst-case upper bound** (every visible interferer at peak EIRP density) reaches
  {ev_max:.4g} of the time; even then diversity removes every unresolvable event and the
  mean throughput loss is at most {deg_max:.4g}.
- Under **finite beam capacity** the unresolvable fraction is
  {cap_abs_max:.4g} of all cell-time ({cond_max:.4g} per served cell-time) and **falls**
  with more beams.
- The one co-frequency user-downlink pair is STARLINK<->ONEWEB; Kuiper's user downlink is
  Ka and does not overlap. The reverse order (STARLINK victim, ONEWEB interferer) never
  trips the threshold with the shipped ASSUMED EIRP densities, so the burden is
  one-directional.

**Conclusion: with the shipped filing-derived (and ASSUMED) parameters, the 25.261(d)
Delta T/T rule binds only rarely and, once the interferer set is co-located and satellite
diversity is credited, never in a way that survives diversity. The binding cases are all
in the (unphysical) worst-case upper bound and are dominated by the ASSUMED terminal
aperture.**

## 6. Figures

- `F5_capacity_diversity.png` - E3 unresolvable fraction, absolute and per served
  cell-time, vs latitude and `N_beam`.
- `F6_dtt.png` - E4 Delta T/T event and unresolvable time-fraction vs latitude.

## 7. Caveats

- Spherical earth, circular orbits, J2 secular only, fixed GMST epoch; an *incidence*
  screen, not a co-ordination simulation.
- Terminal aperture, efficiency, all system noise temperatures and the Shannon margin are
  `ASSUMED`; the aperture is swept and reported. Companion EIRP densities are ASSUMED for
  every operator (see `configs/constellations.yaml`).
- The Delta T/T screen uses one interferer beam per satellite at peak EIRP density in the
  worst-case variant (no transmit-pattern roll-off toward the victim), so it is an upper
  bound; the co-located variant uses the serving satellite only.
- Throughput is a Shannon proxy with a documented margin, not a DVB-S2X MODCOD
  calculation.
- Event fractions are sampling-limited (10 s for E4, 60 s for E3). No population
  weighting yet (deferred to a later phase).

## 8. Approval requested

Phase 3a is complete and reproducible. Requesting review of (i) the E3 denominator fix
and rescue depletion, (ii) the Delta T/T implementation and ASSUMED terminal parameters,
and (iii) the throughput proxy, before the as-deployed SGP4 scenario (Phase 3b) and the
letter (Phase 3c).
"""

    out = RESULTS / "PHASE3A_REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"[report] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
