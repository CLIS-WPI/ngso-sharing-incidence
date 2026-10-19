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


def newest(prefix: str) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def _num(v, fmt: str = "{:.4g}") -> str:
    return fmt.format(float(v))


def main() -> int:
    macros: list[tuple[str, str]] = []
    sources: list[str] = []

    def add(name: str, value: str, run: Path | None = None) -> None:
        macros.append((name, value))
        if run is not None:
            sources.append(f"% {name} <- results/{run.name}/")

    # --- constants (single source of truth in the library) ---
    from ngsi.constants import DTT_LONG_TERM, DTT_THRESHOLD, LONG_TERM_EXCEEDANCE_FRACTION

    add("DttThreshold", _num(DTT_THRESHOLD))
    add("DttLongTerm", _num(DTT_LONG_TERM))
    add("LongTermFraction", _num(LONG_TERM_EXCEEDANCE_FRACTION))

    # --- E1: modelled incidence, event time-fraction vs latitude ---
    e1 = newest("e1_event_rate")
    if e1 is not None:
        ev = pd.read_csv(e1 / "summary_event_rate.csv")
        ov = pd.read_csv(e1 / "summary_band_overlap.csv")
        udl = ov[(ov["link_type"] == "user_downlink") & (ov["overlaps"] == True)]  # noqa: E712
        add("SharedFreqGhz", _num(udl["center_ghz"].iloc[0], "{:.1f}"), e1)
        th2 = ev[ev["theta_deg"] == 2.0]
        for vi, tag in ((1, "Coloc"), (0, "Worst")):
            sub = th2[th2["variant"] == vi]
            mx = sub.loc[sub["event_frac_mean"].idxmax()]
            add(f"Event{tag}Max", _num(mx["event_frac_mean"], "{:.3f}"), e1)
            add(f"Event{tag}Lat", _num(mx["latitude_deg"], "{:.0f}"), e1)
        add("EventNPhasings", str(int(ev["n_phasings"].max())), e1)

    # --- E2: diversity and 25.261(c) band-split loss ---
    e2 = newest("e2_diversity")
    if e2 is not None:
        u = pd.read_csv(e2 / "summary_unresolvable.csv")
        main = u[(u["variant"] == 1) & (u["theta_deg"] == 2.0)]
        add("BandSplitNoDiv", _num(main["band_split_loss_no_diversity"].max(), "{:.3f}"), e2)
        add("BandSplitDiv", _num(main["band_split_loss_with_diversity"].max(), "{:.3f}"), e2)
        add("UnresColocMax", _num(u[u["variant"] == 1]["unres_frac_mean"].max(), "{:.3f}"), e2)
        add("UnresWorstMax", _num(u[u["variant"] == 0]["unres_frac_mean"].max(), "{:.3f}"), e2)

    # --- E3: capacity-limited diversity and the N_beam effect ---
    e3 = newest("e3_capacity_diversity")
    if e3 is not None:
        c = pd.read_csv(e3 / "summary_capacity_diversity.csv")
        ow = c[c["system_a"] == "ONEWEB"]
        add("CapUnresMax", _num(ow["unres_cell_frac_mean"].max(), "{:.2e}"), e3)
        add("CapEventMax", _num(ow["event_cell_frac_mean"].max(), "{:.3f}"), e3)
        lat60 = ow[ow["latitude_deg"] == 60.0].sort_values("n_beam")
        nbeam_word = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}
        for r in lat60.itertuples():
            add(f"CapUnresCondNbeam{nbeam_word[int(r.n_beam)]}",
                _num(r.unres_per_served_mean), e3)

    # --- E4: 25.261(d) Delta T/T criterion (modelled) ---
    e4 = newest("e4_dtt")
    if e4 is not None:
        d = pd.read_csv(e4 / "summary_dtt.csv")
        f = d[d["event_frac_mean"] > 0]
        mx = f.loc[f["event_frac_mean"].idxmax()]
        add("DttEventMax", _num(mx["event_frac_mean"], "{:.3f}"), e4)
        add("DttEventLat", _num(mx["latitude_deg"], "{:.0f}"), e4)
        add("DttEventDiam", _num(mx["diameter_m"], "{:.2f}"), e4)
        add("DttMeanMax", _num(d["dtt_mean_mean"].max(), "{:.3f}"), e4)
        add("DegradeNoDivMax", _num(d["degradation_nodiv_mean"].max(), "{:.3f}"), e4)
        add("DegradeDivMax", _num(d["degradation_div_mean"].max(), "{:.3f}"), e4)
        add("UnresDttMax", _num(d["unres_frac_mean"].max(), "{:.3f}"), e4)
        add("DttNPhasings", str(int(d["n_phasings"].max())), e4)

    # --- E5: as-deployed CelesTrak snapshot ---
    e5 = newest("e5_as_deployed")
    if e5 is not None:
        st = json.loads((e5 / "status.json").read_text())
        add("NStarlinkRaw", str(int(st["n_sats_raw"]["STARLINK"])), e5)
        add("NStarlinkOp", str(int(st["n_sats"]["STARLINK"])), e5)
        add("NOnewebRaw", str(int(st["n_sats_raw"]["ONEWEB"])), e5)
        add("NOnewebOp", str(int(st["n_sats"]["ONEWEB"])), e5)
        add("NStartTimes", str(int(st["n_start_times"])), e5)
        dep = pd.read_csv(e5 / "summary_event_rate.csv")
        depth = dep[dep["theta_deg"] == 2.0]
        for vi, tag in ((1, "Coloc"), (0, "Worst")):
            sub = depth[depth["variant"] == vi]
            add(f"DepEvent{tag}Max", _num(sub["event_frac_mean"].max(), "{:.3f}"), e5)
        ddtt = pd.read_csv(e5 / "summary_dtt.csv")
        add("DepDttEventMax", _num(ddtt[ddtt["event_frac_mean"] > 0]["event_frac_mean"].max(),
                                   "{:.3f}"), e5)

    # --- E6: EIRP-density sweep (pre-paper check 1) ---
    e6 = newest("e6_eirp_sweep")
    if e6 is not None:
        s = pd.read_csv(e6 / "summary_eirp_sweep.csv")
        rev = s[(s["victim"] == "STARLINK") & (s["interferer"] == "ONEWEB")
                & (s["variant"] == 0) & (s["diameter_m"] == 0.6)]
        fwd = s[(s["victim"] == "ONEWEB") & (s["variant"] == 0) & (s["diameter_m"] == 0.6)]
        add("EirpSweepValues", ", ".join(_num(v, "{:g}") for v in sorted(s["eirp_dbw_per_mhz"].unique())), e6)
        rev_assumed = rev[rev["eirp_dbw_per_mhz"] == -20.0]
        rev_equal = rev[rev["eirp_dbw_per_mhz"] == rev["eirp_dbw_per_mhz"].max()]
        add("EirpRevAssumed", _num(rev_assumed["event_frac_mean"].iloc[0], "{:.2e}"), e6)
        add("EirpRevEqual", _num(rev_equal["event_frac_mean"].iloc[0], "{:.3f}"), e6)
        add("EirpFwdMax", _num(fwd["event_frac_mean"].max(), "{:.3f}"), e6)

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
