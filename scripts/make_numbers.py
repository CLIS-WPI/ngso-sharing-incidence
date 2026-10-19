"""Write ``paper/numbers.tex`` from the newest run directory of each experiment.

Every number the paper quotes is a ``\\newcommand`` here, read from ``results/``
(AGENTS.md rule 1: no number is invented, estimated, or hard-coded).  The LaTeX in
``paper/main.tex`` references only these macros, so the paper cannot drift from the
results.  Run before the LaTeX build (``scripts/build_paper.sh``).

Run:  python scripts/make_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "paper" / "numbers.tex"

# The binding (worst) direction is OneWeb-victim <- Starlink-interferer.
VICTIM = "ONEWEB"

# Headline terminal aperture (metres); the sensitivity set is lb.diameter_sweep_m.
HEAD_DIAM = 0.45
APERTURES = (0.45, 0.6, 0.75)
APERTURE_WORD = {0.45: "FourFive", 0.6: "SixZero", 0.75: "SevenFive"}


def newest(prefix: str) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def _num(v, fmt: str = "{:.4g}") -> str:
    return fmt.format(float(v))


def _pct(v, nd: int = 2) -> str:
    return f"{float(v) * 100.0:.{nd}f}"


def _peak(df: pd.DataFrame, value: str) -> pd.Series:
    return df.loc[df[value].idxmax()]


def main() -> int:
    macros: list[tuple[str, str]] = []
    sources: list[str] = []

    def add(name: str, value: str, run: Path | None = None) -> None:
        macros.append((name, value))
        if run is not None:
            sources.append(f"% {name} <- results/{run.name}/")

    # --- rule values (single source of truth in the library) ---
    from ngsi.constants import (
        DTT_LONG_TERM,
        DTT_THRESHOLD,
        LONG_TERM_EXCEEDANCE_FRACTION,
        THROUGHPUT_DEGRADE_MAX,
        UNAVAILABILITY_INCREASE_MAX,
    )

    add("DttThreshold", _num(DTT_THRESHOLD))
    add("DttLongTerm", _num(DTT_LONG_TERM))
    add("LongTermFraction", _num(LONG_TERM_EXCEEDANCE_FRACTION))
    add("DegradeLimitPct", _pct(THROUGHPUT_DEGRADE_MAX, 0))
    add("UnavailLimitPct", _num(UNAVAILABILITY_INCREASE_MAX * 100.0, "{:g}"))

    # --- E1: modelled incidence, geometric theta proxy vs latitude ---
    e1 = newest("e1_event_rate")
    if e1 is not None:
        ev = pd.read_csv(e1 / "summary_event_rate.csv")
        ov = pd.read_csv(e1 / "summary_band_overlap.csv")
        udl = ov[(ov["link_type"] == "user_downlink") & (ov["overlaps"] == True)]  # noqa: E712
        add("SharedFreqGhz", _num(udl["center_ghz"].iloc[0], "{:.1f}"), e1)
        add("EventNPhasings", str(int(ev["n_phasings"].max())), e1)
        th2 = ev[ev["theta_deg"] == 2.0]
        colo = th2[th2["variant"] == 1]
        worst = th2[th2["variant"] == 0]
        cp = _peak(colo, "event_frac_mean")
        wp = _peak(worst, "event_frac_mean")
        add("ThetaColoPeakPct", _pct(cp["event_frac_mean"]), e1)
        add("ThetaColoPeakLat", _num(cp["latitude_deg"], "{:.0f}"), e1)
        eq = colo[colo["latitude_deg"] == 0.0]
        add("ThetaColoEquatorPct", _pct(eq["event_frac_mean"].iloc[0]), e1)
        add("ThetaWorstPeakPct", _pct(wp["event_frac_mean"]), e1)

    # --- E2: satellite diversity (geometric) and min-elevation sensitivity ---
    e2 = newest("e2_diversity")
    if e2 is not None:
        u = pd.read_csv(e2 / "summary_unresolvable.csv")
        ua = u[(u["system_a"] == VICTIM) & (u["variant"] == 1)]
        add("UnresGeomWithDivPct", _pct(ua["unres_frac_mean"].max(), 3), e2)
        me = pd.read_csv(e2 / "summary_min_elevation.csv")
        mea = me[(me["system_a"] == VICTIM) & (me["variant"] == 1) & (me["theta_deg"] == 2.0)]
        peaks = mea.groupby("min_elevation_deg")["event_frac_mean"].max()
        add("MinElevLowDeg", _num(mea["min_elevation_deg"].min(), "{:.0f}"), e2)
        add("MinElevHighDeg", _num(mea["min_elevation_deg"].max(), "{:.0f}"), e2)
        add("MinElevPeakLowPct", _pct(peaks.min()), e2)
        add("MinElevPeakHighPct", _pct(peaks.max()), e2)

    # --- E3: capacity-limited diversity and the N_beam effect ---
    e3 = newest("e3_capacity_diversity")
    if e3 is not None:
        c = pd.read_csv(e3 / "summary_capacity_diversity.csv")
        ow = c[c["system_a"] == VICTIM]
        add("CapUnresMaxPct", _pct(ow["unres_cell_frac_mean"].max(), 4), e3)
        lat60 = ow[ow["latitude_deg"] == 60.0].sort_values("n_beam")
        nbeam_word = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}
        for r in lat60.itertuples():
            add(f"CapUnresCondNbeam{nbeam_word[int(r.n_beam)]}",
                _num(r.unres_per_served_mean), e3)

    # --- E4: 25.261 section (c)/(d) criteria, modelled (as-filed Walker) ---
    e4 = newest("e4_dtt")
    if e4 is not None:
        d = pd.read_csv(e4 / "summary_dtt.csv")
        bind = d[d["system_a"] == VICTIM]
        add("BindFilWorstPct", _pct(_peak(bind[bind["variant"] == 0], "event_frac_mean")[
            "event_frac_mean"]), e4)
        add("BindFilColoPct", _pct(_peak(bind[bind["variant"] == 1], "event_frac_mean")[
            "event_frac_mean"]), e4)
        add("BindFilWorstLat", _num(_peak(bind[bind["variant"] == 0], "event_frac_mean")[
            "latitude_deg"], "{:.0f}"), e4)
        add("BindWithDivPct", _pct(bind["unres_frac_mean"].max(), 3), e4)
        # Per-aperture worst-case throughput degradation, binding direction.
        for ap in APERTURES:
            w = bind[(bind["variant"] == 0) & (bind["diameter_m"] == ap)]
            add(f"DegrFilWorstPct{APERTURE_WORD[ap]}", _pct(w["degradation_nodiv_mean"].max()), e4)
        st = d[d["system_a"] == "STARLINK"]
        add("DegrFilStarPct", _pct(st["degradation_nodiv_mean"].max(), 3), e4)
        add("BaselineAvailStarMinPct", _pct(1.0 - st["unavail0_frac_mean"].max(), 0), e4)
        add("BaselineAvailOnePct", _pct(1.0 - bind["unavail0_frac_mean"].max(), 0), e4)
        inc = (bind["unavail_int_frac_mean"] - bind["unavail0_frac_mean"]).max()
        add("UnavailIncPct", _pct(max(inc, 0.0), 3), e4)

    # --- E5: as-deployed CelesTrak snapshot ---
    e5 = newest("e5_as_deployed")
    if e5 is not None:
        st = json.loads((e5 / "status.json").read_text())
        add("NStarlinkRaw", str(int(st["n_sats_raw"]["STARLINK"])), e5)
        add("NStarlinkOp", str(int(st["n_sats"]["STARLINK"])), e5)
        add("NOnewebRaw", str(int(st["n_sats_raw"]["ONEWEB"])), e5)
        add("NOnewebOp", str(int(st["n_sats"]["ONEWEB"])), e5)
        add("NStartTimes", str(int(st["n_start_times"])), e5)
        ddtt = pd.read_csv(e5 / "summary_dtt.csv")
        bind5 = ddtt[ddtt["system_a"] == VICTIM]
        wp = _peak(bind5[bind5["variant"] == 0], "event_frac_mean")
        cp = _peak(bind5[bind5["variant"] == 1], "event_frac_mean")
        add("BindDepWorstPct", _pct(wp["event_frac_mean"]), e5)
        add("BindDepWorstCi", _pct(wp["event_frac_ci95"]), e5)
        add("BindDepWorstLat", _num(wp["latitude_deg"], "{:.0f}"), e5)
        add("BindDepColoPct", _pct(cp["event_frac_mean"]), e5)
        add("BindDepColoCi", _pct(cp["event_frac_ci95"]), e5)
        add("BindDepColoLat", _num(cp["latitude_deg"], "{:.0f}"), e5)
        for ap in APERTURES:
            w = bind5[(bind5["variant"] == 0) & (bind5["diameter_m"] == ap)]
            add(f"DegrDepWorstPct{APERTURE_WORD[ap]}", _pct(w["degradation_nodiv_mean"].max()), e5)
        # Aperture sensitivity of the headline binding fraction (worst case).
        peak_by_ap = [bind5[(bind5["variant"] == 0) & (bind5["diameter_m"] == ap)][
            "event_frac_mean"].max() for ap in APERTURES]
        add("BindMinApertureDepPct", _pct(min(peak_by_ap)), e5)
        add("BindMaxApertureDepPct", _pct(max(peak_by_ap)), e5)
        add("DegrDepStarPct", _pct(ddtt[ddtt["system_a"] == "STARLINK"][
            "degradation_nodiv_mean"].max(), 3), e5)
        add("DegrWithDivPct", _pct(ddtt["degradation_div_mean"].max(), 3), e5)
        # as-deployed vs as-filed binding ratio (worst case, headline aperture).
        ev5 = pd.read_csv(e5 / "summary_event_rate.csv")
        w5 = ev5[(ev5["system_a"] == VICTIM) & (ev5["variant"] == 1) & (ev5["theta_deg"] == 2.0)]
        if e1 is not None:
            w1 = ev[(ev["system_a"] == VICTIM) & (ev["variant"] == 1) & (ev["theta_deg"] == 2.0)]
            add("GeomRatioEquator", _num(
                w5[w5["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
                / w1[w1["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0], "{:.2f}"), e5)
        if e4 is not None:
            a = d[(d["system_a"] == VICTIM) & (d["variant"] == 0) & (d["diameter_m"] == HEAD_DIAM)]
            b = bind5[(bind5["variant"] == 0) & (bind5["diameter_m"] == HEAD_DIAM)]
            latpk = b.loc[b["event_frac_mean"].idxmax(), "latitude_deg"]
            fil_eq = a[a["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
            dep_eq = b[b["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
            fil_pk = a[a["latitude_deg"] == latpk]["event_frac_mean"].iloc[0]
            dep_pk = b["event_frac_mean"].max()
            add("BindRatioEquator", _num(dep_eq / fil_eq, "{:.1f}"), e5)
            add("BindRatioPeak", _num(dep_pk / fil_pk, "{:.1f}"), e5)

    # --- E6: EIRP-density sweep (pre-paper check 1) ---
    e6 = newest("e6_eirp_sweep")
    if e6 is not None:
        s = pd.read_csv(e6 / "summary_eirp_sweep.csv")
        rev = s[(s["victim"] == "STARLINK") & (s["interferer"] == VICTIM)
                & (s["variant"] == 0) & (s["diameter_m"] == HEAD_DIAM)]
        fwd = s[(s["victim"] == VICTIM) & (s["variant"] == 0) & (s["diameter_m"] == HEAD_DIAM)]
        add("EirpSweepValues", ", ".join(_num(v, "{:g}") for v in sorted(s["eirp_dbw_per_mhz"].unique())), e6)
        add("EirpHeadDiam", _num(HEAD_DIAM, "{:g}"), e6)
        add("EirpFwdInvariantPct", _pct(fwd["event_frac_mean"].max()), e6)
        add("EirpRevMinPct", _pct(rev["event_frac_mean"].min(), 2), e6)
        add("EirpRevMaxPct", _pct(rev["event_frac_mean"].max(), 2), e6)

    # --- write ---
    header = [
        "% AUTOGENERATED by scripts/make_numbers.py -- do not edit by hand.",
        "% Every macro below is read from results/ (AGENTS.md rule 1).",
        "% Newest run per experiment:",
        *sources,
        "",
    ]
    body = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    print(f"[numbers] wrote {OUT} ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
