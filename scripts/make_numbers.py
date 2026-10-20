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

# Regulatory / filing text quoted by the paper (not model parameters).
PFD_LIMIT_25208B = -116.0          # dBW/m^2/MHz, 47 CFR 25.208(b)(2), 25-90 deg elevation
PFD_SCOPE_GHZ = (10.7, 11.7)       # band to which 47 CFR 25.208(b) applies
# Top of the filed Starlink Ku EIRP range, per 4 kHz (FCC-22-91A1 para. 42, unit-corrected;
# see configs/constellations.yaml STARLINK source).
STARLINK_FILED_TOP_DBW_PER_4KHZ = -8.3


def newest(prefix: str) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def newest_where(prefix: str, **status_kv) -> Path | None:
    """Newest run of ``prefix`` whose ``status.json`` matches every key/value.

    Used to pin the headline E4/E5 macros to the primary configuration
    (correct interferer-range diversity, per-beam steering: Starlink
    PFD-constant / OneWeb constant-EIRP) when sensitivity reruns
    (legacy ``diversity_fspl=victim``, forced ``steering``) share the prefix.
    """
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    for run in reversed(runs):
        try:
            st = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            continue
        if all(st.get(k) == v for k, v in status_kv.items()):
            return run
    return runs[-1] if runs else None


def _num(v, fmt: str = "{:.4g}") -> str:
    return fmt.format(float(v))


def _pct(v, nd: int = 2) -> str:
    # NaN = undefined, e.g. no station meets the 25.261(d) baseline-availability gate.
    return "n/a" if pd.isna(v) else f"{float(v) * 100.0:.{nd}f}"


def _at_max(g: pd.DataFrame, col: str, other: str) -> float:
    """``other`` in the row where ``col`` is largest (NaN if ``col`` is all-NaN)."""
    c = g[col].dropna()
    return float(g.loc[c.idxmax(), other]) if len(c) else float("nan")


def _peak(df: pd.DataFrame, value: str) -> pd.Series:
    return df.loc[df[value].idxmax()]


def _ci_at(df: pd.DataFrame, value: str, mode: str = "max") -> float:
    """95% CI of the cell that attains the extreme of ``<value>_mean``."""
    idx = df[value + "_mean"].idxmax() if mode == "max" else df[value + "_mean"].idxmin()
    return float(df.loc[idx, value + "_ci95"])


def _rep_worst(raw: pd.DataFrame, value: str, system_a: str, variant: int,
               diam: float = HEAD_DIAM) -> tuple[float, float]:
    """The single 25.261(d) aggregation used in the paper (same as E6 / Table I).

    Worst qualified station (baseline availability >= 99%) per replication
    (phasing or start time), then mean and 95% t-interval half-width across
    replications.
    """
    import numpy as np
    from ngsi.runner import t95

    r = raw[(raw["system_a"] == system_a) & (raw["variant"] == variant)
            & (raw["diameter_m"] == diam) & (raw["avail0_frac"] >= 0.99)]
    w = r.groupby("phasing")[value].max().dropna().to_numpy(dtype=np.float64)
    if w.size == 0:
        return float("nan"), float("nan")
    ci = float(t95(w.size) * np.std(w, ddof=1) / np.sqrt(w.size)) if w.size >= 2 else 0.0
    return float(w.mean()), ci


def _sign_bands(lat: pd.Series, err: pd.Series, sign: int) -> list[tuple[float, float]]:
    """Contiguous latitude runs (lo, hi) where ``sign * err > 0``, in latitude order."""
    bands: list[list[float]] = []
    prev_in = False
    for la, e in sorted(zip(lat, err)):
        now_in = sign * e > 0
        if now_in and prev_in:
            bands[-1][1] = la
        elif now_in:
            bands.append([la, la])
        prev_in = now_in
    return [(lo, hi) for lo, hi in bands]


def main() -> int:
    macros: list[tuple[str, str]] = []
    sources: list[str] = []
    oneweb_eirp: float | None = None
    screen_theta: float | None = None

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
    add("DttThresholdPct", _num(DTT_THRESHOLD * 100.0, "{:g}"))
    add("DttLongTerm", _num(DTT_LONG_TERM))
    add("LongTermFraction", _num(LONG_TERM_EXCEEDANCE_FRACTION))
    add("DegradeLimitPct", _pct(THROUGHPUT_DEGRADE_MAX, 0))
    add("UnavailLimitPct", _num(UNAVAILABILITY_INCREASE_MAX * 100.0, "{:g}"))
    add("CiPct", "95")
    add("RuleBaselineAvailPct", "99")
    add("CnGateDb", "0")

    # --- E1: modelled incidence, geometric theta proxy vs latitude ---
    e1 = newest("e1_event_rate")
    if e1 is not None:
        ev = pd.read_csv(e1 / "summary_event_rate.csv")
        ov = pd.read_csv(e1 / "summary_band_overlap.csv")
        udl = ov[(ov["link_type"] == "user_downlink") & (ov["overlaps"] == True)]  # noqa: E712
        add("SharedFreqGhz", _num(udl["center_ghz"].iloc[0], "{:.1f}"), e1)
        add("EventNPhasings", str(int(ev["n_phasings"].max())), e1)
        import yaml as _y1
        _e1cfg = e1 / "configs" / "e1_event_rate.yaml"
        if "screen_theta_deg" not in _y1.safe_load(_e1cfg.read_text())["params"]:
            _e1cfg = REPO_ROOT / "configs" / "experiments" / "e1_event_rate.yaml"
        _p1 = _y1.safe_load(_e1cfg.read_text())["params"]
        screen_theta = float(_p1["screen_theta_deg"])
        if screen_theta not in [float(t) for t in ev["theta_deg"].unique()]:
            raise SystemExit(f"screen_theta_deg {screen_theta} not in the E1 run's thetas")
        add("ThetaScreenDeg", _num(screen_theta, "{:g}"), e1)
        th2 = ev[ev["theta_deg"] == screen_theta]
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
        mea = me[(me["system_a"] == VICTIM) & (me["variant"] == 1) & (me["theta_deg"] == screen_theta)]
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
    # The 25.261(d) metrics are computed only over cases whose modelled baseline
    # availability is >= 99% (the domain in which the rule is defined);
    # `avail0_frac_mean` is the mean over phasings/longitudes of the fraction of
    # time the victim link keeps C/N >= 0 dB.
    e4 = newest_where("e4_dtt", diversity_fspl="interferer", steering=None)
    if e4 is not None:
        d = pd.read_csv(e4 / "summary_dtt.csv")
        st4 = json.loads((e4 / "status.json").read_text())
        oneweb_eirp = float(st4["beam_eirp_dbw_per_mhz"]["ONEWEB"])
        add("OnewebEirpDbw", _num(oneweb_eirp, "{:g}"), e4)
        add("StationMinElevDeg", _num(st4["min_elevation_deg"], "{:g}"), e4)
        add("ImplMarginDb", _num(st4["implementation_margin_db"], "{:g}"), e4)
        # Altitudes from the run's copied config (a file under results/).
        from ngsi.config import load_constellations
        _cons = load_constellations(e4 / "configs" / "constellations.yaml")
        add("StarlinkAltKm", _num(_cons["STARLINK"].shells[0].altitude_km, "{:g}"), e4)
        add("OnewebAltKm", _num(_cons["ONEWEB"].shells[0].altitude_km, "{:g}"), e4)
        # 25.208(b) anchor of the PFD-constant Starlink model and the filed-EIRP cross-check.
        import math
        from ngsi.analytic import pfd_constant_dbw_per_m2_per_mhz
        from ngsi.bands import load_bands, overlap
        _sh = _cons["STARLINK"].shells[0]
        _pfd = lambda e: pfd_constant_dbw_per_m2_per_mhz(  # noqa: E731
            e, _sh.altitude_km, _sh.beam.pfd_anchor_elevation_deg)
        _filed = STARLINK_FILED_TOP_DBW_PER_4KHZ + 10.0 * math.log10(1e6 / 4e3)
        add("PfdLimitDbw", _num(PFD_LIMIT_25208B, "{:g}"))
        add("PfdScopeLoGhz", _num(PFD_SCOPE_GHZ[0], "{:g}"))
        add("PfdScopeHiGhz", _num(PFD_SCOPE_GHZ[1], "{:g}"))
        add("PfdAnchorElevDeg", _num(_sh.beam.pfd_anchor_elevation_deg, "{:g}"), e4)
        add("PrimaryPfdDbw", _num(_pfd(_sh.beam.eirp_density_dbw_per_mhz), "{:.2f}"), e4)
        add("StarlinkFiledTopDbw", _num(_filed, "{:.2f}"))
        add("FiledPfdDbw", _num(_pfd(_filed), "{:.2f}"), e4)
        add("FiledPfdOffsetDb", _num(_pfd(_sh.beam.eirp_density_dbw_per_mhz) - _pfd(_filed),
                                     "{:.1f}"), e4)
        _b = load_bands(e4 / "configs" / "bands.yaml")
        _seg = overlap(_b["ONEWEB"], _b["STARLINK"], "user_downlink").segments_ghz[0]
        add("SharedBandLoGhz", _num(_seg[0], "{:g}"), e4)
        add("SharedBandHiGhz", _num(_seg[1], "{:g}"), e4)
        import yaml
        _g = yaml.safe_load((e4 / "configs" / "stations.yaml").read_text())["station_grid"]
        _lats = _g["latitudes_deg"]
        add("LatMinDeg", _num(min(_lats), "{:g}"), e4)
        add("LatMaxDeg", _num(max(_lats), "{:g}"), e4)
        add("LatStepDeg", _num(_lats[1] - _lats[0], "{:g}"), e4)
        add("NLongitudes", str(len(_g["longitudes_deg"])), e4)
        add("AperMaxM", _num(max(APERTURES), "{:g}"))
        add("InThresholdDb", _num(10.0 * math.log10(DTT_THRESHOLD), "{:.1f}"))
        # Parameter table (Table I): values read from the run's copied configs.
        for sysn, tag in (("STARLINK", "Star"), ("ONEWEB", "One")):
            bm = _cons[sysn].shells[0].beam
            add(f"{tag}PeakGainDbi", _num(bm.peak_gain_dbi, "{:g}"), e4)
            add(f"{tag}BeamwidthDeg", _num(bm.beamwidth_deg, "{:g}"), e4)
            add(f"{tag}NoiseK", _num(bm.system_noise_temp_k, "{:g}"), e4)
            add(f"{tag}BwMhz", _num(bm.bandwidth_mhz, "{:g}"), e4)
            add(f"{tag}EirpDbw", _num(bm.eirp_density_dbw_per_mhz, "{:g}"), e4)
        _lb = yaml.safe_load((e4 / "configs" / "link_budget.yaml").read_text())["user_terminal"]
        add("AperEff", _num(_lb["aperture_efficiency"], "{:g}"), e4)
        bind = d[d["system_a"] == VICTIM]
        rev = d[d["system_a"] != VICTIM]
        add("BindFilWorstPct", _pct(_peak(bind[bind["variant"] == 0], "event_frac_mean")[
            "event_frac_mean"]), e4)
        add("BindFilColoPct", _pct(_peak(bind[bind["variant"] == 1], "event_frac_mean")[
            "event_frac_mean"]), e4)
        add("BindFilWorstLat", _num(_peak(bind[bind["variant"] == 0], "event_frac_mean")[
            "latitude_deg"], "{:.0f}"), e4)
        add("BindWithDivPct", _pct(bind[bind["variant"].isin([0, 1])][
            "unres_frac_mean"].max(), 3), e4)
        add("BindFilStarWorstPct", _pct(_peak(rev[rev["variant"] == 0], "event_frac_mean")[
            "event_frac_mean"]), e4)
        add("BindFilStarColoPct", _pct(_peak(rev[rev["variant"] == 1], "event_frac_mean")[
            "event_frac_mean"]), e4)
        bw = bind[bind["variant"] == 0]["event_frac_mean"].max()
        sw = rev[rev["variant"] == 0]["event_frac_mean"].max()
        add("BurdenRatioFil", _num(bw / sw, "{:.1f}"), e4)
        add("UpperBoundEventPct", _pct(bind[bind["variant"] == 2][
            "event_frac_mean"].max()), e4)
        # Baseline C/N (median, p10) and availability per direction, worst latitude.
        for sysn, tag in ((VICTIM, "One"), ("STARLINK", "Star")):
            dd = d[d["system_a"] == sysn]
            add(f"BaselineCnMed{tag}Db", _num(dd["cn0_median_mean"].min(), "{:.1f}"), e4)
            add(f"BaselineCnPTen{tag}Db", _num(dd["cn0_p10_mean"].min(), "{:.1f}"), e4)
            add(f"BaselineAvail{tag}MinPct", _pct(dd["avail0_frac_mean"].min(), 0), e4)
            ref = dd[(dd["diameter_m"] == 0.6) & (dd["latitude_deg"] == 45.0)]
            if sysn == VICTIM and len(ref):
                add("CalRefCnMedDb", _num(ref["cn0_median_mean"].iloc[0], "{:.1f}"), e4)
                add("CalRefCnPTenDb", _num(ref["cn0_p10_mean"].iloc[0], "{:.1f}"), e4)
            tot = dd[["diameter_m", "variant", "latitude_deg"]].drop_duplicates().shape[0]
            qual = dd[dd["avail0_frac_mean"] >= 0.99]
            add(f"Qual{tag}Cases", str(int(qual[["diameter_m", "variant", "latitude_deg"]]
                                            .drop_duplicates().shape[0])), e4)
            add(f"Qual{tag}Total", str(int(tot)), e4)
        # 25.261(d) metrics: worst qualified station per phasing, then mean/CI
        # across phasings (_rep_worst; the same aggregation as Table I).
        raw4 = pd.read_parquet(e4 / "raw_dtt.parquet")
        for ap in APERTURES:
            add(f"DegrFilWorstPct{APERTURE_WORD[ap]}",
                _pct(_rep_worst(raw4, "degradation_nodiv", VICTIM, 0, ap)[0]), e4)
        for name, col, sysn, var, nd in (
            ("DegrFilMax", "degradation_nodiv", VICTIM, 0, 2),
            ("DegrFilColo", "degradation_nodiv", VICTIM, 1, 2),
            ("DegrFilDiv", "degradation_div", VICTIM, 0, 2),
            ("UnavailInc", "unavail_incr_frac", VICTIM, 0, 3),
            ("UnavailIncColo", "unavail_incr_frac", VICTIM, 1, 3),
            ("UnavailFilDiv", "unavail_incr_div_frac", VICTIM, 0, 3),
            ("DegrFilStar", "degradation_nodiv", "STARLINK", 0, 2),
        ):
            m, ci = _rep_worst(raw4, col, sysn, var)
            add(f"{name}Pct", _pct(m, nd), e4)
            add(f"{name}Ci", _pct(ci, nd), e4)

    # --- E5: as-deployed CelesTrak snapshot ---
    e5 = newest_where("e5_as_deployed", diversity_fspl="interferer", steering=None)
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
        # Per-aperture worst-case degradation, restricted to cases with >=99%
        # baseline availability (the 25.261(d) domain).
        raw5 = pd.read_parquet(e5 / "raw_as_deployed_dtt.parquet")
        for ap in APERTURES:
            add(f"DegrDepWorstPct{APERTURE_WORD[ap]}",
                _pct(_rep_worst(raw5, "degradation_nodiv", VICTIM, 0, ap)[0]), e5)
        for name, col, sysn, var, nd in (
            ("DegrDepMax", "degradation_nodiv", VICTIM, 0, 2),
            ("DegrDepColo", "degradation_nodiv", VICTIM, 1, 2),
            ("DegrDepDiv", "degradation_div", VICTIM, 0, 2),
            ("UnavailIncDep", "unavail_incr_frac", VICTIM, 0, 3),
            ("UnavailIncColoDep", "unavail_incr_frac", VICTIM, 1, 3),
            ("UnavailDepDiv", "unavail_incr_div_frac", VICTIM, 0, 3),
            ("DegrDepStar", "degradation_nodiv", "STARLINK", 0, 2),
        ):
            m, ci = _rep_worst(raw5, col, sysn, var)
            add(f"{name}Pct", _pct(m, nd), e5)
            add(f"{name}Ci", _pct(ci, nd), e5)
        add("BaselineAvailOneDepPct", _pct(bind5["avail0_frac_mean"].min(), 0), e5)
        add("BindDepStarWorstPct", _pct(_peak(
            ddtt[(ddtt["system_a"] == "STARLINK") & (ddtt["variant"] == 0)],
            "event_frac_mean")["event_frac_mean"]), e5)
        bwd = bind5[bind5["variant"] == 0]["event_frac_mean"].max()
        swd = ddtt[(ddtt["system_a"] == "STARLINK") & (ddtt["variant"] == 0)][
            "event_frac_mean"].max()
        add("BurdenRatioDep", _num(bwd / swd, "{:.1f}"), e5)
        # Aperture sensitivity of the headline binding fraction (worst case).
        peak_by_ap = [bind5[(bind5["variant"] == 0) & (bind5["diameter_m"] == ap)][
            "event_frac_mean"].max() for ap in APERTURES]
        add("BindMinApertureDepPct", _pct(min(peak_by_ap)), e5)
        add("BindMaxApertureDepPct", _pct(max(peak_by_ap)), e5)
        # as-deployed vs as-filed binding ratio (worst case, headline aperture).
        ev5 = pd.read_csv(e5 / "summary_event_rate.csv")
        w5 = ev5[(ev5["system_a"] == VICTIM) & (ev5["variant"] == 1) & (ev5["theta_deg"] == screen_theta)]
        if e1 is not None:
            w1 = ev[(ev["system_a"] == VICTIM) & (ev["variant"] == 1) & (ev["theta_deg"] == screen_theta)]
            add("GeomRatioEquator", _num(
                w5[w5["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
                / w1[w1["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0], "{:.2f}"), e5)
        if e4 is not None:
            a = d[(d["system_a"] == VICTIM) & (d["variant"] == 0) & (d["diameter_m"] == HEAD_DIAM)]
            b = bind5[(bind5["variant"] == 0) & (bind5["diameter_m"] == HEAD_DIAM)]
            latpk = _at_max(b, "event_frac_mean", "latitude_deg")
            fil_eq = a[a["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
            dep_eq = b[b["latitude_deg"] == 0.0]["event_frac_mean"].iloc[0]
            fil_pk = a[a["latitude_deg"] == latpk]["event_frac_mean"].iloc[0]
            dep_pk = b["event_frac_mean"].max()
            add("BindRatioEquator", _num(dep_eq / fil_eq, "{:.1f}"), e5)
            add("BindRatioPeak", _num(dep_pk / fil_pk, "{:.1f}"), e5)

    # --- E6: EIRP-density cases (pre-paper check 1) ---
    # The experiment is run over named cases (interferer EIRP densities and the
    # constant-EIRP vs PFD-constant beam-steering model), not a single one-system
    # sweep.  The macros the manuscript already uses are preserved by name and
    # remapped to the primary case / across-case extremes; the per-case macros
    # expose the sensitivity rows.
    e6 = newest("e6_eirp_sweep")
    if e6 is not None:
        s = pd.read_csv(e6 / "summary_eirp_sweep.csv")
        _st6 = json.loads((e6 / "status.json").read_text())
        add("SenStartTimes", str(int(_st6["n_start_times"])), e6)

        def cg(case, victim, variant, diam=HEAD_DIAM):
            return s[(s["case"] == case) & (s["victim"] == victim)
                     & (s["variant"] == variant) & (s["diameter_m"] == diam)]

        fwd0 = cg("primary", VICTIM, 0)          # binding direction, worst-case
        fwd1 = cg("primary", VICTIM, 1)          # binding direction, co-located
        rev0 = cg("primary", "STARLINK", 0)
        add("EirpHeadDiam", _num(HEAD_DIAM, "{:g}"), e6)
        _p0 = float(fwd0["starlink_eirp_dbw_per_mhz"].iloc[0])
        _p3 = float(cg("starlink_pfd_minus3", VICTIM, 0)["starlink_eirp_dbw_per_mhz"].iloc[0])
        _p6 = float(cg("starlink_gen1_design", VICTIM, 0)["starlink_eirp_dbw_per_mhz"].iloc[0])
        add("PfdOffsetThreeDb", _num(_p0 - _p3, "{:g}"), e6)
        add("PfdOffsetGenOneDb", _num(_p0 - _p6, "{:.2f}"), e6)
        # Gen1 design PFD and the OneWeb min-elevation case, from the run's E6 config.
        from ngsi.analytic import pfd_constant_dbw_per_m2_per_mhz as _pfdc
        from ngsi.config import load_constellations as _lc
        _sh6 = _lc(e6 / "configs" / "constellations.yaml")["STARLINK"].shells[0]
        add("GenOnePfdDbw", _num(_pfdc(_p6, _sh6.altitude_km,
                                        _sh6.beam.pfd_anchor_elevation_deg), "{:.2f}"), e6)
        add("GenOneEirpDbw", _num(_p6, "{:g}"), e6)
        import yaml as _yaml
        _c6 = {c["name"]: c for c in _yaml.safe_load(
            (e6 / "configs" / "e6_eirp_sweep.yaml").read_text())["params"]["cases"]}
        add("OnewebMinElevSensDeg",
            _num(_c6["oneweb_min_elev_55"]["min_elevation_deg"]["ONEWEB"], "{:g}"), e6)
        _g6 = _yaml.safe_load((e6 / "configs" / "stations.yaml").read_text())["station_grid"]
        add("NStations", str(len(_g6["latitudes_deg"]) * len(_g6["longitudes_deg"])), e6)
        add("MinElevQualStations",
            str(int(cg("oneweb_min_elev_55", VICTIM, 0)["n_qual_min"].min())), e6)
        add("NoiseAssumedOneK", _num(_c6["noise_assumed"]["noise_temp_k"]["ONEWEB"], "{:g}"), e6)
        add("NoiseAssumedStarK",
            _num(_c6["noise_assumed"]["noise_temp_k"]["STARLINK"], "{:g}"), e6)
        add("EirpSweepValues", ", ".join(
            _num(v, "{:g}") for v in sorted(s["starlink_eirp_dbw_per_mhz"].unique())), e6)
        add("EirpFwdInvariantPct", _pct(fwd0["event_frac_max"].max()), e6)
        add("EirpFwdColoPct", _pct(fwd1["event_frac_max"].max()), e6)
        add("EirpFwdUnresPct", _pct(fwd0["unres_frac_max"].max()), e6)
        add("EirpFwdBaseAvailPct", _pct(fwd0["avail0_min"].min(), 0), e6)
        rev_all = s[(s["victim"] == "STARLINK") & (s["variant"] == 0)
                    & (s["diameter_m"] == HEAD_DIAM)]
        add("EirpRevMinPct", _pct(rev_all["event_frac_max"].min(), 2), e6)
        add("EirpRevMaxPct", _pct(rev_all["event_frac_max"].max(), 2), e6)
        # 25.261(d) sensitivity: qualified throughput degradation, forward direction.
        fwd_all = s[(s["victim"] == VICTIM) & (s["variant"] == 0)]
        add("EirpDegrMinPct", _pct(fwd_all["degr_max_qual"].min()), e6)
        add("EirpDegrMaxPct", _pct(fwd_all["degr_max_qual"].max()), e6)
        add("EirpDegrMinCi", _pct(fwd_all.loc[fwd_all["degr_max_qual"].idxmin(), "degr_max_qual_ci95"]), e6)
        add("EirpDegrMaxCi", _pct(_at_max(fwd_all, "degr_max_qual", "degr_max_qual_ci95")), e6)
        add("EirpDegrAtCalPct", _pct(fwd0["degr_max_qual"].max()), e6)
        add("EirpDegrAtCalCi", _pct(_at_max(fwd0, "degr_max_qual", "degr_max_qual_ci95")), e6)
        add("EirpRevAtCalPct", _pct(rev0["event_frac_max"].max(), 2), e6)
        # Smallest OneWeb interferer EIRP at which the reverse (Starlink-victim)
        # trigger is still nonzero, i.e. where the reverse result still holds.
        rev_ok = rev_all[rev_all["event_frac_max"].fillna(0.0) > 0.0]
        if len(rev_ok):
            hold = rev_ok.loc[rev_ok["oneweb_eirp_dbw_per_mhz"].idxmin()]
            add("EirpRevHoldMinDbw", _num(hold["oneweb_eirp_dbw_per_mhz"], "{:g}"), e6)
            add("EirpRevHoldPct", _pct(hold["event_frac_max"], 2), e6)
        # Per-case headline sensitivity rows (letter-only names).
        case_tags = {
            "primary": "Prim",
            "starlink_low_end": "Low",
            "delportillo_common": "DpCommon",
            "oneweb_cn_calibrated": "CnCal",
            "starlink_constant_eirp": "ConstEirp",
            "starlink_pfd_minus3": "PfdMinusThree",
            "starlink_gen1_design": "GenOne",
            "noise_assumed": "NoiseAssumed",
            "oneweb_min_elev_55": "MinElevFiftyFive",
        }
        for case, tag in case_tags.items():
            f0 = cg(case, VICTIM, 0)
            f1 = cg(case, VICTIM, 1)
            r0 = cg(case, "STARLINK", 0)
            add(f"Eirp{tag}FwdWorstPct", _pct(f0["event_frac_max"].max()), e6)
            add(f"Eirp{tag}FwdUnresPct", _pct(f0["unres_frac_max"].max()), e6)
            add(f"Eirp{tag}FwdDegrPct", _pct(f0["degr_max_qual"].max()), e6)
            add(f"Eirp{tag}FwdDegrCi",
                _pct(_at_max(f0, "degr_max_qual", "degr_max_qual_ci95")), e6)
            add(f"Eirp{tag}ColoPct", _pct(f1["event_frac_max"].max()), e6)
            add(f"Eirp{tag}RevWorstPct", _pct(r0["event_frac_max"].max()), e6)

        # Table I: realistic variant (0), OneWeb victim, headline aperture, per
        # case; every threshold comparison carries its 95% CI (over start times).
        for case, tag in case_tags.items():
            g = cg(case, VICTIM, 0)
            add(f"Tab{tag}StarlinkDbw",
                _num(g["starlink_eirp_dbw_per_mhz"].iloc[0], "{:g}"), e6)
            add(f"Tab{tag}EventPct", _pct(g["event_frac_max"].max()), e6)
            add(f"Tab{tag}EventCi",
                _pct(_at_max(g, "event_frac_max", "event_frac_max_ci95")), e6)
            add(f"Tab{tag}DegrNoDivPct", _pct(g["degr_max_qual"].max()), e6)
            add(f"Tab{tag}DegrNoDivCi",
                _pct(_at_max(g, "degr_max_qual", "degr_max_qual_ci95")), e6)
            add(f"Tab{tag}DegrDivPct", _pct(g["degr_div_max_qual"].max()), e6)
            add(f"Tab{tag}DegrDivCi",
                _pct(_at_max(g, "degr_div_max_qual", "degr_div_max_qual_ci95")), e6)
            add(f"Tab{tag}UnavNoDivPct", _pct(g["unavail_max_qual"].max(), 3), e6)
            add(f"Tab{tag}UnavNoDivCi",
                _pct(_at_max(g, "unavail_max_qual", "unavail_max_qual_ci95"), 3), e6)
            add(f"Tab{tag}UnavDivPct", _pct(g["unavail_div_max_qual"].max(), 3), e6)
            add(f"Tab{tag}UnavDivCi",
                _pct(_at_max(g, "unavail_div_max_qual", "unavail_div_max_qual_ci95"), 3), e6)

    # --- E7/E8: analytic model and its diagnosis ---
    e7 = newest("e7_analytic")
    if e7 is not None:
        a7 = pd.read_csv(e7 / "summary_analytic.csv")
        t = a7[a7["victim"] == VICTIM]["theta_star_deg"].dropna()
        if len(t):
            add("AnalyticThetaStarDeg", _num(t.iloc[0], "{:.2f}"), e7)
            add("AnalyticThetaSpreadDeg", _num(t.max() - t.min(), "{:.2f}"), e7)
        c7 = pd.read_csv(e7 / "summary_compare.csv")
        if len(c7):
            # `abs_err` is signed (analytic - simulated): take its magnitude.
            add("AnalyticMaxAbsErrPct", _pct(c7["abs_err"].abs().max()), e7)
            add("AnalyticSignedMinPct", _pct(c7["abs_err"].min()), e7)
            add("AnalyticSignedMaxPct", _pct(c7["abs_err"].max()), e7)
            h7 = c7[(c7["victim"] == VICTIM) & (c7["diameter_m"] == HEAD_DIAM)]
            under = _sign_bands(h7["latitude_deg"], h7["abs_err"], -1)
            over = _sign_bands(h7["latitude_deg"], h7["abs_err"], +1)
            # Widest under-prediction band; over-prediction bands below / above it.
            ulo, uhi = max(under, key=lambda b: b[1] - b[0])
            olow = [b for b in over if b[1] < ulo]
            ohigh = [b for b in over if b[0] > uhi]
            add("AnalyticUnderLatLo", _num(ulo, "{:.0f}"), e7)
            add("AnalyticUnderLatHi", _num(uhi, "{:.0f}"), e7)
            if olow:
                add("AnalyticOverLowLatLo", _num(olow[0][0], "{:.0f}"), e7)
                add("AnalyticOverLowLatHi", _num(olow[-1][1], "{:.0f}"), e7)
            if ohigh:
                add("AnalyticOverHighLatLo", _num(ohigh[0][0], "{:.0f}"), e7)
                add("AnalyticOverHighLatHi", _num(ohigh[-1][1], "{:.0f}"), e7)
    e8 = newest("e8_analytic_diagnosis")
    if e8 is not None:
        js = json.loads((e8 / "diagnosis.json").read_text())
        add("DiagnosisOldMaxErrPct", _pct(js["max_abs_err_old_cond"]), e8)
        add("DiagnosisNewMaxErrPct", _pct(js["max_abs_err_new_cond"]), e8)
        add("DiagnosisIndepMaxErrPct", _pct(js["max_abs_err_indep_cond"]), e8)

    # --- E9: OneWeb 55 deg mask diagnosis (as-deployed E6 geometry) ---
    e9 = newest("e9_min_elevation_diagnosis")
    if e9 is not None:
        j9 = json.loads((e9 / "diagnosis.json").read_text())
        lt = pd.read_csv(e9 / "summary_min_elevation_latitude.csv")
        ql = lt[lt["n_qual_m55"] > 0]
        add("MinElevQualLatLo", _num(ql["latitude_deg"].min(), "{:.0f}"), e9)
        add("MinElevQualLatHi", _num(ql["latitude_deg"].max(), "{:.0f}"), e9)
        add("MinElevNAboveMin", _num(ql["n_above_mask_mean"].min(), "{:.2f}"), e9)
        add("MinElevNAboveMax", _num(ql["n_above_mask_mean"].max(), "{:.2f}"), e9)
        add("MinElevNAboveTenMin", _num(ql["n_above_mask_p10"].min(), "{:g}"), e9)
        eq = lt[lt["latitude_deg"] == lt["latitude_deg"].min()].iloc[0]
        add("MinElevNoneEquatorPct", _pct(eq["frac_none_above_mask"]), e9)
        w9 = j9["worst_degr_nodiv_m55"]
        add("MinElevWorstLat", _num(w9[0]["latitude_deg"], "{:.0f}"), e9)
        add("MinElevWorstNonePct", _pct(max(r["frac_none_above_mask"] for r in w9)), e9)
        add("BwCheckMhz", _num(j9["bandwidth_check_mhz"], "{:g}"), e9)
        add("BwMaxAbsDiffPct", _pct(max(j9["bandwidth_max_abs_diff"].values()), 3), e9)

    # --- write ---
    bad = [n for n, _ in macros if not n.isalpha()]
    if bad:
        raise SystemExit(
            f"macro names must be alphabetic LaTeX control words (no digits/other): {bad}"
        )
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
