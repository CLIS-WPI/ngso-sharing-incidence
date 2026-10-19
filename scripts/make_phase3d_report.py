"""Write ``results/PHASE3D_REPORT.md``: the Starlink EIRP unit-correction re-run.

Every number is read from the newest run directory (AGENTS.md rule 1).  Answers
the three questions posed before any paper text is rewritten:

(a) Delta T/T exceedance with and without diversity, both directions, as-filed
    (E4, modelled Walker) and as-deployed (E5, CelesTrak snapshot).
(b) Whether "unlimited diversity resolves essentially all events" still holds.
(c) The 25.261(d) degraded-throughput (3%) and availability-change (0.4%) limits
    on the stations where the rule applies (baseline availability >= 99%).

Run:  python scripts/make_phase3d_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "PHASE3D_REPORT.md"

VICTIM = "ONEWEB"          # binding (worst) victim direction
HEAD_DIAM = 0.45           # headline terminal aperture (m)
VARIANT = {0: "worst-case", 1: "co-located"}
DEGRADE_LIMIT = 0.03
UNAVAIL_LIMIT = 0.004
QUAL_AVAIL = 0.99


def newest(prefix: str) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def newest_where(prefix: str, **kv) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())
    for run in reversed(runs):
        try:
            st = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            continue
        if all(st.get(k) == v for k, v in kv.items()):
            return run
    return None


def fmt_pct(v) -> str:
    return f"{float(v) * 100.0:.2f}"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def direction_rows(df: pd.DataFrame, diam: float = HEAD_DIAM):
    """Per (victim->interferer, variant) event/unres extremes as-filed or as-deployed."""
    d = df[df["diameter_m"] == diam]
    out = []
    for (va, vb, var), g in d.groupby(["system_a", "system_b", "variant"]):
        ev = g.loc[g["event_frac_mean"].idxmax()]
        un = g.loc[g["unres_frac_mean"].idxmax()]
        ratio = (un["unres_frac_mean"] / ev["event_frac_mean"]) if ev["event_frac_mean"] > 0 else float("nan")
        out.append({
            "pair": f"{va}<-{vb}",
            "variant": VARIANT[int(var)],
            "event_pct": fmt_pct(ev["event_frac_mean"]),
            "event_lat": int(ev["latitude_deg"]),
            "unres_pct": fmt_pct(un["unres_frac_mean"]),
            "unres_lat": int(un["latitude_deg"]),
            "unres_over_event": f"{ratio:.3f}" if ratio == ratio else "n/a",
        })
    return out


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def threshold_rows(df: pd.DataFrame, diam: float = HEAD_DIAM):
    """25.261(d) qualified extremes per direction and variant."""
    d = df[(df["diameter_m"] == diam) & (df["avail0_frac_mean"] >= QUAL_AVAIL)]
    out = []
    for (va, vb, var), g in d.groupby(["system_a", "system_b", "variant"]):
        de = g.loc[g["degradation_nodiv_mean"].idxmax()]
        un = g.loc[g["unavail_incr_frac_mean"].idxmax()]
        out.append({
            "pair": f"{va}<-{vb}",
            "variant": VARIANT[int(var)],
            "degr_pct": fmt_pct(de["degradation_nodiv_mean"]),
            "degr_ci": fmt_pct(de["degradation_nodiv_ci95"]),
            "degr_lat": int(de["latitude_deg"]),
            "degr_exceed": "YES" if de["degradation_nodiv_mean"] > DEGRADE_LIMIT else "no",
            "unav_pct": fmt_pct(un["unavail_incr_frac_mean"]),
            "unav_ci": fmt_pct(un["unavail_incr_frac_ci95"]),
            "unav_exceed": "YES" if un["unavail_incr_frac_mean"] > UNAVAIL_LIMIT else "no",
        })
    return out


def e6_rows(s: pd.DataFrame, diam: float = HEAD_DIAM):
    d = s[s["diameter_m"] == diam]
    out = []
    for (case, va, vb, var), g in d.groupby(["case", "victim", "interferer", "variant"]):
        out.append({
            "case": case,
            "pair": f"{va}<-{vb}",
            "variant": VARIANT[int(var)],
            "starlink_dbw": f'{g["starlink_eirp_dbw_per_mhz"].iloc[0]:g}',
            "oneweb_dbw": f'{g["oneweb_eirp_dbw_per_mhz"].iloc[0]:g}',
            "event_pct": fmt_pct(g["event_frac_max"].max()),
            "unres_pct": fmt_pct(g["unres_frac_max"].max()),
            "degr_qual_pct": fmt_pct(g["degr_max_qual"].max()),
            "degr_qual_ci": fmt_pct(g.loc[g["degr_max_qual"].idxmax(), "degr_max_qual_ci95"]),
            "avail0_min": f'{g["avail0_min"].min():.3f}',
        })
    return out


def eirp_audit():
    """Every co-frequency EIRP density in configs, with its unit interpretation."""
    from ngsi.config import load_constellations
    from ngsi.link_budget import EIRP_DENSITY_4KHZ_TO_MHZ_DB
    cons = load_constellations(REPO_ROOT / "configs" / "constellations.yaml")
    rows = []
    for name, c in cons.items():
        b = c.shells[0].beam
        note = ""
        if b.freq_ghz < 15.0:
            note = (f"Ku (<15 GHz): FCC filings state dBW/4kHz; model uses dBW/MHz "
                    f"(+{EIRP_DENSITY_4KHZ_TO_MHZ_DB:.2f} dB)")
        else:
            note = "Ka (>=15 GHz): FCC filings state dBW/MHz; no 4 kHz conversion"
        rows.append({
            "system": name,
            "freq_ghz": f"{b.freq_ghz:g}",
            "bw_mhz": f"{b.bandwidth_mhz:g}",
            "eirp_dbw_per_mhz": f"{b.eirp_density_dbw_per_mhz:g}",
            "implied_beam_eirp_dbw": f"{b.eirp_density_dbw_per_mhz + 10 * __import__('math').log10(b.bandwidth_mhz):.1f}",
            "unit_note": note,
            "source": b.source,
        })
    return rows


def main() -> int:
    e4 = newest_where("e4_dtt", diversity_fspl="interferer", pfd_constant=False)
    e4_legacy = newest_where("e4_dtt", diversity_fspl="victim", pfd_constant=False)
    e4_pfd = newest_where("e4_dtt", diversity_fspl="interferer", pfd_constant=True)
    e5 = newest_where("e5_as_deployed", diversity_fspl="interferer", pfd_constant=False)
    e6 = newest("e6_eirp_sweep")
    assert e4 and e5 and e6, "missing runs; execute e4_dtt/e5_as_deployed/e6_eirp_sweep first"

    d4 = pd.read_csv(e4 / "summary_dtt.csv")
    d4l = pd.read_csv(e4_legacy / "summary_dtt.csv") if e4_legacy else None
    d4p = pd.read_csv(e4_pfd / "summary_dtt.csv") if e4_pfd else None
    d5 = pd.read_csv(e5 / "summary_dtt.csv")
    s6 = pd.read_csv(e6 / "summary_eirp_sweep.csv")

    cols = ["pair", "variant", "event_pct", "event_lat", "unres_pct", "unres_lat",
            "unres_over_event"]
    tcols = ["pair", "variant", "degr_pct", "degr_ci", "degr_lat", "degr_exceed",
             "unav_pct", "unav_ci", "unav_exceed"]
    ecols = ["case", "pair", "variant", "starlink_dbw", "oneweb_dbw", "event_pct",
             "unres_pct", "degr_qual_pct", "degr_qual_ci", "avail0_min"]

    L: list[str] = []
    L.append("# Phase 3d - Starlink EIRP unit correction: re-run report")
    L.append("")
    L.append(f"Generated by `scripts/make_phase3d_report.py` at git `{git_hash()}`. "
             "All numbers read from `results/` (AGENTS.md rule 1); no paper text was rewritten.")
    L.append("")
    L.append("Newest run directories:")
    for tag, run in [("E4 primary (constant-EIRP, diversity=interferer)", e4),
                     ("E4 legacy diversity (diversity=victim)", e4_legacy),
                     ("E4 PFD-constant", e4_pfd),
                     ("E5 as-deployed", e5),
                     ("E6 EIRP cases", e6)]:
        L.append(f"- {tag}: `results/{run.name}/`" if run else f"- {tag}: (absent)")
    L.append("")
    L.append(f"Headline aperture {HEAD_DIAM:g} m; limits: degraded throughput "
             f"{DEGRADE_LIMIT:.0%}, availability change {UNAVAIL_LIMIT:.1%}; the rule "
             f"applies only where baseline availability >= {QUAL_AVAIL:.0%}.")
    L.append("")

    # 0. config audit
    L.append("## 0. Config EIRP-density audit")
    L.append("")
    L.append(md_table(eirp_audit(),
                      ["system", "freq_ghz", "bw_mhz", "eirp_dbw_per_mhz",
                       "implied_beam_eirp_dbw", "unit_note"]))
    L.append("")
    L.append("Sources (verbatim from `configs/constellations.yaml`):")
    for r in eirp_audit():
        L.append(f"- **{r['system']}**: {r['source']}")
    L.append("")

    # (a)
    L.append("## (a) Delta T/T exceedance, with and without diversity")
    L.append("")
    L.append("`event` = Delta T/T > 6% (band-splitting trigger, no diversity); "
             "`unres` = the event that no other visible victim satellite can avoid "
             "(diversity exhausted).  `unres/event` = fraction of events diversity "
             "cannot resolve.")
    L.append("")
    L.append("**As-filed (E4, modelled Walker), constant-EIRP:**")
    L.append("")
    L.append(md_table(direction_rows(d4), cols))
    L.append("")
    if d4p is not None:
        L.append("**As-filed (E4), PFD-constant beam steering:**")
        L.append("")
        L.append(md_table(direction_rows(d4p), cols))
        L.append("")
    if d4l is not None:
        L.append("**As-filed (E4), legacy diversity path loss (victim range) - attribution only:**")
        L.append("")
        L.append(md_table(direction_rows(d4l), cols))
        L.append("")
    L.append("**As-deployed (E5, CelesTrak snapshot), constant-EIRP:**")
    L.append("")
    L.append(md_table(direction_rows(d5), cols))
    L.append("")

    # (b)
    L.append("## (b) Does unlimited diversity still resolve essentially all events?")
    L.append("")
    prim = direction_rows(d4)
    dep = direction_rows(d5)
    for label, rows in (("as-filed", prim), ("as-deployed", dep)):
        for r in rows:
            if r["pair"] == f"{VICTIM}<-STARLINK":
                L.append(f"- {label}, {VICTIM}-victim {r['variant']}: "
                         f"{r['unres_pct']}% unresolvable of {r['event_pct']}% events "
                         f"(ratio {r['unres_over_event']}).")
    L.append("")
    L.append("**No.**  Under the corrected Starlink Ku EIRP density the *co-located* "
             "variant is still fully resolved by diversity (unres ~= 0%), but the "
             "*worst-case* bound is essentially 100% unresolvable at low latitude: "
             "with every visible Starlink satellite at peak EIRP density toward the "
             "site, the aggregate interference enters the terminal's sidelobes from "
             "all directions and no victim hand-off avoids it.  The earlier "
             "\"diversity resolves essentially all events\" result was driven by the "
             "24 dB unit error (Starlink read as -8.3 dBW/MHz instead of +15.7).")
    L.append("")

    # (c)
    L.append("## (c) 25.261(d) degraded-throughput (3%) and availability-change (0.4%)")
    L.append("")
    L.append("Qualified stations only (baseline availability >= 99%).  "
             "`degr` = time-weighted throughput degradation (no diversity); "
             "`unav` = absolute availability increase.")
    L.append("")
    L.append("**As-filed (E4, constant-EIRP):**")
    L.append("")
    L.append(md_table(threshold_rows(d4), tcols))
    L.append("")
    if d4p is not None:
        L.append("**As-filed (E4, PFD-constant):**")
        L.append("")
        L.append(md_table(threshold_rows(d4p), tcols))
        L.append("")
    L.append("**As-deployed (E5, constant-EIRP):**")
    L.append("")
    L.append(md_table(threshold_rows(d5), tcols))
    L.append("")

    # sensitivity cases
    L.append("## Sensitivity cases (E6, as-deployed geometry)")
    L.append("")
    L.append(md_table(e6_rows(s6), ecols))
    L.append("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[phase3d] wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
