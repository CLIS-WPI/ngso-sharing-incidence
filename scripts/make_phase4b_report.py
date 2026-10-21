"""Phase 4b report: as-deployed re-run with non-overlapping 24 h windows.

Writes ``results/PHASE4B_REPORT.md`` from

- ``paper/numbers_overlapping.tex`` (the macros of the overlapping 5 h-spaced runs,
  preserved before the re-run) and the regenerated ``paper/numbers.tex``;
- the newest ``e12_propagation_check_*`` run (SGP4 vs the J2 secular model);
- the newest ``e13_daily_replications_*`` run (per-day values, dependence tests);
- the paper text (``paper/main_overleaf4.tex`` if present, else ``paper/main.tex``),
  for the line numbers of the sentences that depend on each verdict.

Every number is read from those files (AGENTS.md rule 1).

Usage (in the container):
    python scripts/make_phase4b_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402

from ngsi.constants import THROUGHPUT_DEGRADE_MAX, UNAVAILABILITY_INCREASE_MAX  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "PHASE4B_REPORT.md"
OLD_TEX = REPO / "paper" / "numbers_overlapping.tex"
NEW_TEX = REPO / "paper" / "numbers.tex"
PAPERS = [p for p in (REPO / "paper" / "main_overleaf4.tex", REPO / "paper" / "main.tex")
          if p.exists()]
DLIM, ULIM = 100 * THROUGHPUT_DEGRADE_MAX, 100 * UNAVAILABILITY_INCREASE_MAX
SKIP = ("Event", "Alt", "Unres", "SwitchOnly", "Switches", "Vis")


# --------------------------------------------------------------------- macros
def read_macros(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    txt = path.read_text(encoding="utf-8")
    val = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt))
    src = dict(re.findall(r"^% ([A-Za-z]+) <- results/(\S+?)/\s*$", txt, flags=re.M))
    return val, src


def limit_of(base: str) -> float | None:
    if any(s in base for s in SKIP):
        return None
    if "Degr" in base:
        return DLIM
    if "Unav" in base:
        return ULIM
    return None


def verdict(m: float, ci: float, lim: float) -> str:
    if m - ci > lim:
        return "exceeds"
    if m + ci < lim:
        return "within"
    return "indistinguishable"


def vof(val: dict, base: str) -> str | None:
    lim = limit_of(base)
    try:
        m, ci = float(val[base + "Pct"]), float(val[base + "Ci"])
    except (KeyError, ValueError):
        return None
    return verdict(m, ci, lim) if lim is not None else None


def pm(val: dict, base: str) -> str:
    return f"{val.get(base + 'Pct', 'n/a')} +/- {val.get(base + 'Ci', 'n/a')}"


def asdep(src: dict, name: str) -> bool:
    return src.get(name, "").split("/")[-1].startswith(("e5_as_deployed", "e6_eirp_sweep"))


# ------------------------------------------------------------------ paper text
def paper_lines() -> dict[Path, list[str]]:
    return {p: p.read_text(encoding="utf-8").splitlines() for p in PAPERS}


def macro_lines(lines: list[str], name: str) -> list[int]:
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    return [i + 1 for i, s in enumerate(lines) if pat.search(s)]


def phrase_lines(lines: list[str], pattern: str) -> list[int]:
    """Lines matching ``pattern`` on the text joined line by line (spans two lines)."""
    pat = re.compile(pattern, flags=re.I)
    out = []
    for i in range(len(lines)):
        joined = lines[i] + " " + (lines[i + 1] if i + 1 < len(lines) else "")
        if pat.search(joined) and (not out or out[-1] != i + 1):
            if pat.search(lines[i]) or not (i + 2 <= len(lines) and pat.search(lines[i + 1])):
                out.append(i + 1)
    return out


def where(texts: dict, macros: list[str] = (), pattern: str | None = None) -> str:
    parts = []
    for p, lines in texts.items():
        ln = set()
        for m in macros:
            ln.update(macro_lines(lines, m))
        if pattern:
            ln.update(phrase_lines(lines, pattern))
        if ln:
            parts.append(f"`{p.relative_to(REPO).as_posix()}` " + ", ".join(map(str, sorted(ln))))
    return "; ".join(parts) or "not in the text"


# ------------------------------------------------------------------- tables
def md_table(rows: list[dict]) -> str:
    if not rows:
        return "(none)"
    cols = list(rows[0])
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(L)


def newest(prefix: str) -> Path | None:
    runs = sorted(p for p in RESULTS.glob(f"{prefix}_*") if (p / "status.json").exists())
    return runs[-1] if runs else None


def headline_runs(src: dict) -> list[dict]:
    seen: dict[str, int] = {}
    for name, run in src.items():
        seen[run] = seen.get(run, 0) + 1
    rows = []
    for run, n in sorted(seen.items()):
        st = {}
        f = RESULTS / run / "status.json"
        if f.exists():
            st = json.loads(f.read_text())
        rows.append({"run": f"`{run}`", "macros": n,
                     "generation": st.get("starlink_generation", ""),
                     "start spacing h": (f"{st['start_offset_s'] / 3600:g}"
                                         if "start_offset_s" in st else ""),
                     "start times": st.get("n_start_times", ""),
                     "handoff policies": ", ".join(q["name"] for q in
                                                   st.get("handoff_policies", []))})
    return rows


# ------------------------------------------------------------------ sections
def section_runs(oldsrc: dict, newsrc: dict) -> list[str]:
    L = ["## 1. Runs compared", "",
         "Overlapping-window macros: `paper/numbers_overlapping.tex` (copied from "
         "`paper/numbers.tex` before the re-run).  Source runs (as-deployed only; the "
         "as-filed E4 and E1-E3/E7-E9 runs are shared by both):", ""]
    old = {k: v for k, v in oldsrc.items() if asdep(oldsrc, k)}
    new = {k: v for k, v in newsrc.items() if asdep(newsrc, k)}
    L += ["**Overlapping windows (5 h apart)**", "", md_table(headline_runs(old)), "",
          "**Non-overlapping windows (24 h apart)**", "", md_table(headline_runs(new)), ""]
    return L


def section_propagation() -> list[str]:
    run = newest("e12_propagation_check")
    L = ["## 2. Orbit-propagation validity over the 8-day window", ""]
    if run is None:
        return L + ["(no e12_propagation_check run)", ""]
    st = json.loads((run / "status.json").read_text())
    se = pd.read_csv(run / "summary_sample_errors.csv")
    L += [f"Run `results/{run.name}/` (`scripts/check_propagation.py`).  Sample: "
          f"{int(se[se.system == 'STARLINK'].n_sats.iloc[0])} Gen2 Starlink and "
          f"{int(se[se.system == 'ONEWEB'].n_sats.iloc[0])} OneWeb operational satellites "
          f"(seed 0), SGP4 on the same TLEs vs the production J2 secular circular model, "
          f"{st['window_s'] / 3600:g} h at {st['step_s']:g} s from day d after the "
          "simulation epoch (the latest TLE epoch).  Look-angle error: angle between the "
          "two station-to-satellite directions over the station grid, samples with SGP4 "
          f"elevation >= {st['min_elevation_deg']:g} deg.", ""]
    rows = []
    for _, r in se.iterrows():
        rows.append({"system": r.system, "day": int(r.day),
                     "position error median / p90 / max km":
                         f"{r.pos_err_median_km:.1f} / {r.pos_err_p90_km:.1f} / {r.pos_err_max_km:.1f}",
                     "radial error median km": f"{r.radial_err_median_km:.2f}",
                     "look-angle error median / p90 / max deg":
                         f"{r.look_err_median_deg:.2f} / {r.look_err_p90_deg:.2f} / "
                         f"{r.look_err_max_deg:.2f}",
                     "elevation error median deg": f"{r.elev_err_median_deg:.2f}",
                     "look samples": int(r.look_err_n), "SGP4 errors": int(r.n_sgp4_errors)})
    L += [md_table(rows), ""]

    raw = pd.read_parquet(run / "raw_daily_geometry.parquet")
    s = pd.read_csv(run / "summary_daily_geometry.csv")
    L += [f"**Per-day statistics, every operational satellite** ({st['n_sats']['STARLINK']} "
          f"Gen2 Starlink, {st['n_sats']['ONEWEB']} OneWeb; {st['daily_step_s']:g} s step, "
          f"full 24 h per day; means over the {raw.station.nunique()} stations).  In-line "
          f"proxy: fraction of served time with a visible Starlink satellite within "
          f"{st['theta_deg']:g} deg of the highest-elevation OneWeb satellite (a geometric "
          "screen, not the Delta T/T criterion).", ""]
    rows = []
    for d, g in s.groupby("day"):
        j, q = g[g.propagator == "j2"].iloc[0], g[g.propagator == "sgp4"].iloc[0]
        rows.append({"day": int(d),
                     "Starlink visible J2 / SGP4": f"{j.vis_starlink_mean:.2f} / {q.vis_starlink_mean:.2f}",
                     "OneWeb visible J2 / SGP4": f"{j.vis_oneweb_mean:.3f} / {q.vis_oneweb_mean:.3f}",
                     "in-line proxy % J2 / SGP4": f"{100 * j.inline_frac:.3f} / {100 * q.inline_frac:.3f}"})
    L += [md_table(rows), ""]
    rows = []
    for col, lab in (("vis_starlink_mean", "Starlink visible"), ("vis_oneweb_mean", "OneWeb visible"),
                     ("inline_frac", "in-line proxy")):
        for prop in ("j2", "sgp4", "j2 - sgp4"):
            if prop == "j2 - sgp4":
                y = (s[s.propagator == "j2"].set_index("day")[col]
                     - s[s.propagator == "sgp4"].set_index("day")[col])
            else:
                y = s[s.propagator == prop].set_index("day")[col]
            lr = stats.linregress(y.index.to_numpy(float), y.to_numpy(float))
            rel = lr.slope / abs(y.mean()) if prop != "j2 - sgp4" and y.mean() else np.nan
            rows.append({"quantity": lab, "propagator": prop,
                         "slope per day": f"{lr.slope:.3g}",
                         "slope / mean per day": ("" if np.isnan(rel) else f"{rel:.2%}"),
                         "trend p": f"{lr.pvalue:.3f}"})
    L += ["Linear trend across days (OLS on the station-mean daily values):", "",
          md_table(rows), ""]
    return L


def section_daily() -> list[str]:
    run = newest("e13_daily_replications")
    L = ["## 3. Dependence of the daily replications", ""]
    if run is None:
        return L + ["(no e13_daily_replications run)", ""]
    d = pd.read_csv(run / "summary_daily_stats.csv")
    L += [f"Run `results/{run.name}/` (`scripts/analyze_daily_replications.py`).  Value per "
          "day = worst qualified station of that day (event fraction: mean over meridians "
          "at the peak latitude); lag-1 r with the rough +/-2/sqrt(n) bound; OLS trend "
          "p-value and Kendall tau across days.  CI basis: t-interval unless |r| >= "
          "2/sqrt(n) or trend p < 0.05, then the daily range.  With 8 (or 4) days these "
          "tests have little power, so passing them does not establish independence.", ""]

    def fmt(r, scale):
        vals = " ".join(f"{scale * float(v):.3f}" for v in str(r["values"]).split())
        return {"metric": r["metric"], "days": int(r["n_days"]),
                "daily values %": vals,
                "mean +/- t-CI %": f"{scale * r['mean']:.3f} +/- {scale * r['ci95']:.3f}",
                "min-max %": f"{scale * r['min']:.3f} - {scale * r['max']:.3f}",
                "lag-1 r": ("" if pd.isna(r["lag1_r"]) else f"{r['lag1_r']:.2f}"),
                "bound": f"{r['lag1_crit']:.2f}",
                "trend slope %/day": f"{scale * r['trend_slope_per_day']:.4f}",
                "trend p": f"{r['trend_p']:.3f}", "Kendall p": f"{r['kendall_p']:.3f}",
                "CI basis": r["ci_basis"]}

    for gen in ("gen2", "gen1", "all"):
        g = d[(d.experiment == "e5") & (d.generation == gen)]
        if len(g):
            L += [f"**E5, {gen}**", "", md_table([fmt(r, 100) for _, r in g.iterrows()]), ""]
    g = d[d.experiment == "e6"]
    if len(g):
        rows = []
        for _, r in g.iterrows():
            x = fmt(r, 100)
            rows.append({"case": r["case"], **{k: x[k] for k in
                         ("metric", "mean +/- t-CI %", "min-max %", "lag-1 r", "trend p",
                          "CI basis")}})
        L += ["**E6 (Table I cases), Gen2, event-avoiding**", "", md_table(rows), ""]
    n_dep = int((d.ci_basis == "daily range").sum())
    L += [f"{n_dep} of {len(d)} metric series fail the independence screen "
          "(listed with CI basis 'daily range'); "
          f"{int((d.ci_basis == 'constant').sum())} are constant across days.", ""]
    return L


def section_verdicts(old: dict, new: dict, oldsrc: dict, newsrc: dict,
                     texts: dict) -> tuple[list[str], list[str]]:
    bases = sorted({k[:-3] for k in set(old) | set(new) if k.endswith("Pct")})
    rows, changed = [], []
    for b in bases:
        if limit_of(b) is None or not (asdep(oldsrc, b + "Pct") or asdep(newsrc, b + "Pct")):
            continue
        vo, vn = vof(old, b), vof(new, b)
        if vo is None and vn is None:
            continue
        r = {"macro": b, "overlapping": f"{pm(old, b)} [{vo}]",
             "non-overlapping": f"{pm(new, b)} [{vn}]",
             "over (old -> new)": (f"{old.get(b + 'Over', '')} -> {new.get(b + 'Over', '')}"
                                   if (b + "Over") in old or (b + "Over") in new else ""),
             "paper lines": where(texts, [b + "Pct", b + "Ci", b + "Over"])}
        rows.append(r)
        if vo != vn and vo is not None and vn is not None:
            changed.append(b)
    L = ["## 4. Verdicts, overlapping vs non-overlapping windows", "",
         f"Limit {DLIM:g}% (degradation), {ULIM:g}% (availability change); 'exceeds' / "
         "'within' only when the whole printed interval is above / below the limit, else "
         "'indistinguishable' (the paper's rule).  Every as-deployed limit macro pair:", ""]
    chg = [r for r in rows if r["macro"] in changed]
    L += ["### 4.1 Verdicts that change", "", md_table(chg) if chg else "- none", ""]
    over = []
    for k in sorted(set(old) | set(new)):
        if k.endswith("Over") and asdep(newsrc, k):
            a, b = old.get(k), new.get(k)
            if a != b and ((a == "0") != (b == "0")):
                over.append({"macro": k, "overlapping": a, "non-overlapping": b,
                             "paper lines": where(texts, [k])})
    L += ["### 4.2 Exceedance counts that move to or from zero", "",
          md_table(over) if over else "- none", ""]
    L += ["### 4.3 All as-deployed limit verdicts", "", md_table(rows), ""]
    return L, changed


def _unres_unconstrained() -> list[dict]:
    rows = []
    for gen in ("gen2", "gen1", "all"):
        for pol in ("full", "full_keep_serving", "legacy"):
            run = mn.run_with_policy("e5_as_deployed", gen, pol)
            if run is None:
                continue
            ho = pd.read_parquet(run / "raw_handoff.parquet")
            ho = ho[(ho.policy == pol) & (ho.system_a == mn.VICTIM) & (ho.variant == 0)
                    & (ho.diameter_m == mn.HEAD_DIAM)]
            q = ho[ho.avail0_frac >= mn.QUAL]
            rows.append({"generation": gen, "policy": pol, "run": run.name,
                         "max unres_frac": float(q.unres_frac.max()),
                         "max unavail_incr_div_frac": float(q.unavail_incr_div_frac.max())})
    return rows


def section_claims(old: dict, new: dict, texts: dict) -> list[str]:
    """Paper sentences whose claim rests on a limit verdict, tested on both macro sets."""

    def all_v(val, bases, want):
        got = {b: vof(val, b) for b in bases}
        return all(v == want for v in got.values()), got

    def mk(label, pattern, macros, test):
        ok_o, det_o = test(old)
        ok_n, det_n = test(new)
        return {"claim": label, "paper lines": where(texts, macros, pattern),
                "holds (overlapping)": "yes" if ok_o else "**no**",
                "holds (non-overlapping)": "yes" if ok_n else "**no**",
                "evidence (non-overlapping)": det_n}

    def verd(bases, want):
        def t(val):
            ok, got = all_v(val, bases, want)
            return ok, "; ".join(f"{b} {pm(val, b)} [{v}]" for b, v in got.items())
        return t

    def both(*tests):
        def t(val):
            r = [f(val) for f in tests]
            return all(x[0] for x in r), "; ".join(x[1] for x in r)
        return t

    def count0(names):
        def t(val):
            got = {n: val.get(n) for n in names}
            return all(v == "0" for v in got.values()), "; ".join(f"{n} {v}" for n, v in got.items())
        return t

    def not_within(base):
        def t(val):
            v = vof(val, base)
            return v != "within", f"{base} {pm(val, base)} [{v}]"
        return t

    def only_gen1_both_within(val):
        cases = sorted({k[3:-len("DegrNoDivPct")] for k in val
                        if k.startswith("Tab") and k.endswith("DegrNoDivPct")})
        met = [c for c in cases if vof(val, f"Tab{c}DegrNoDiv") == "within"
               and vof(val, f"Tab{c}UnavNoDiv") == "within"]
        return met == ["GenOne"], "Table I cases with both no-handoff limits met: " + (
            ", ".join(met) or "none")

    rate = [f"{g}{t}RateMaxDegr" for g in ("Fil", "Dep") for t in ("GenTwo", "GenOne", "GenAll")]
    ho_un = [f"{g}{t}{r}UnavOver" for g in ("Fil", "Dep") for t in ("GenTwo", "GenOne", "GenAll")
             for r in ("EvAv", "RateMax", "Legacy")]
    claims = [
        mk("No handoff: both limits broken in both geometries (abstract; Sec. Results; Concl.)",
           r"exceeds both protection limits|both limits are broken|maximum exceeds both limits",
           ["FilGenTwoNoHoDegrPct", "DepGenTwoNoHoDegrPct", "DepGenTwoNoHoUnavPct"],
           verd(["FilGenTwoNoHoDegr", "FilGenTwoNoHoUnav", "DepGenTwoNoHoDegr",
                 "DepGenTwoNoHoUnav"], "exceeds")),
        mk("PFD -3 dB: throughput meets its limit, availability does not",
           r"already meets its limit", ["TabPfdMinusThreeDegrNoDivPct"],
           both(verd(["TabPfdMinusThreeDegrNoDiv"], "within"),
                not_within("TabPfdMinusThreeUnavNoDiv"))),
        mk("'only at the Gen1 design PFD are both met' (no handoff; also Conclusion)",
           r"only at the Gen1 design PFD|only its lower Gen1 design power",
           ["TabGenOneDegrNoDivPct", "TabGenOneUnavNoDivPct"], only_gen1_both_within),
        mk("Handoff 'always' restores availability; 'every unconstrained policy' "
           "(abstract, Results, Conclusion)",
           r"always restores the availability|Handoff always fixes availability|"
           r"every unconstrained|restores availability as long",
           [], count0(ho_un)),
        mk("Event-avoiding breaks the throughput limit as deployed (Gen2)",
           r"still breaks|still exceeds the limit|leaving every interfered satellite exceeds",
           ["DepGenTwoEvAvDegrPct"], verd(["DepGenTwoEvAvDegr"], "exceeds")),
        mk("Rate-maximizing meets it as deployed, 'none over'",
           r"none over", ["DepGenTwoRateMaxDegrPct"],
           both(verd(["DepGenTwoRateMaxDegr"], "within"), count0(["DepGenTwoRateMaxDegrOver"]))),
        mk("As filed, event-avoiding meets the limit 'only narrowly'",
           r"meets the limit only", ["FilGenTwoEvAvDegrPct"],
           verd(["FilGenTwoEvAvDegr"], "within")),
        mk("Gen1-only and all-Starlink: event-avoiding also breaks the limit",
           r"does not\s+hinge|against Gen1 alone",
           ["DepGenOneEvAvDegrPct", "DepGenAllEvAvDegrPct"],
           verd(["DepGenOneEvAvDegr", "DepGenAllEvAvDegr"], "exceeds")),
        mk("Rate-maximizing 'meets it in every case' (all generations, both geometries)",
           r"meets it in every case", ["DepGenAllRateMaxDegrPct"], verd(rate, "within")),
        mk("K = 2: availability change 'still within its limit'",
           r"still within its limit", ["DepKTwoUnavPct"], verd(["DepKTwoUnav"], "within")),
        mk("N = 60 s dwell 'breaks both limits'; 'slow handoff ... breaks the availability "
           "limit' (abstract, contributions)",
           r"breaks both limits|Slow handoff|slow handoff",
           ["DepNSixtyDegrPct", "DepNSixtyUnavPct"],
           verd(["DepNSixtyDegr", "DepNSixtyUnav"], "exceeds")),
        mk("55 deg mask: event-avoiding exceeds both limits; 'a high elevation mask also "
           "breaks the availability limit'",
           r"high elevation mask|exceeds both limits\s*$",
           ["TabMinElevFiftyFiveDegrDivPct", "TabMinElevFiftyFiveUnavDivPct"],
           verd(["TabMinElevFiftyFiveDegrDiv", "TabMinElevFiftyFiveUnavDiv"], "exceeds")),
    ]
    L = ["## 5. Change log: paper sentences that depend on a verdict", "",
         "Each claim of the text is tested against the overlapping and the non-overlapping "
         "macro sets with the paper's own rule (Sec. 4).  Line numbers in: "
         + ", ".join(f"`{p.relative_to(REPO).as_posix()}`" for p in texts) + ".", "",
         md_table(claims), ""]
    un = _unres_unconstrained()
    L += ["'No admissible alternate was ever missing' (unconstrained policies, qualified "
          "station-days, headline aperture):", "",
          md_table([{**r, "max unres_frac": f"{r['max unres_frac']:.3g}",
                     "max unavail_incr_div_frac": f"{r['max unavail_incr_div_frac']:.3g}"}
                    for r in un]),
          "", "Paper lines: " + where(texts, [], r"ever missing"), ""]
    rows = []
    for k in ("BurdenRatioDep", "BindDepWorstPct", "BindDepWorstCi", "BindDepWorstLat",
              "BindDepStarWorstPct", "StartSpacingH", "NStartTimes", "SenStartTimes"):
        rows.append({"macro": k, "overlapping": old.get(k), "non-overlapping": new.get(k),
                     "paper lines": where(texts, [k])})
    L += ["Burden ratio, event peak and timing macros:", "", md_table(rows), ""]
    return L


def section_basis() -> list[str]:
    """CI basis and the propagation limitation, from the e12 / e13 runs."""
    L = ["## 6. Basis of the as-deployed intervals and recorded limitations", ""]
    e12, e13 = newest("e12_propagation_check"), newest("e13_daily_replications")
    if e13 is not None:
        d = pd.read_csv(e13 / "summary_daily_stats.csv")
        n_dep = int((d.ci_basis == "daily range").sum())
        n_var = int((d.ci_basis != "constant").sum())
        basis = ("the t-interval is kept" if n_dep == 0 else
                 "the daily range (min-max) is the interval for the series flagged "
                 "'daily range' in Sec. 3")
        L += [f"- **CI basis.** {n_var - n_dep} of the {n_var} non-constant daily series pass "
              "the lag-1 and trend screen and " f"{n_dep} fail it, so {basis}.  The "
              "interval is a 95% t-interval over consecutive days and describes "
              "day-to-day (temporal) variability of one TLE snapshot propagated forward, "
              "conditional on that snapshot and on the propagation model; it is not a "
              "sampling interval over constellation states or snapshots.  With 8 (Gen2) or "
              "4 (Gen1-only, all-Starlink) days the screen has little power, so the "
              "independence it supports is weak.  The re-run removes the window overlap; it "
              "is not a complete fix of CI validity.", ""]
    if e12 is not None:
        se = pd.read_csv(e12 / "summary_sample_errors.csv")
        sg = pd.read_csv(e12 / "summary_daily_geometry.csv")
        st = se[se.system == "STARLINK"].set_index("day")
        ow = se[se.system == "ONEWEB"].set_index("day")
        j = sg[sg.propagator == "j2"].set_index("day")
        q = sg[sg.propagator == "sgp4"].set_index("day")
        dmax = int(st.index.max())
        off = (j.vis_starlink_mean / q.vis_starlink_mean - 1.0)
        L += ["- **Propagation.** Against SGP4 on the same TLEs, the J2 secular model's "
              f"median Starlink position error grows from {st.pos_err_median_km.iloc[0]:.1f} km "
              f"(day {int(st.index[0])}) to {st.pos_err_median_km.loc[dmax]:.1f} km (day "
              f"{dmax}), a median look-angle error of {st.look_err_median_deg.loc[dmax]:.1f} deg; "
              f"OneWeb stays near {ow.pos_err_median_km.median():.1f} km.  Individual "
              "Starlink positions in the later days are therefore not a forecast of the real "
              "constellation (SGP4 itself omits manoeuvres): those days act as further "
              "phasings of the observed shells.  The statistics the results use are stable: "
              f"mean visible Gen2 satellites {j.vis_starlink_mean.min():.2f}-"
              f"{j.vis_starlink_mean.max():.2f} (J2) and {q.vis_starlink_mean.min():.2f}-"
              f"{q.vis_starlink_mean.max():.2f} (SGP4) over the days, J2 above SGP4 by "
              f"{100 * off.min():.1f}-{100 * off.max():.1f}%, and the geometric in-line proxy "
              f"{100 * j.inline_frac.min():.2f}-{100 * j.inline_frac.max():.2f}% (J2) vs "
              f"{100 * q.inline_frac.min():.2f}-{100 * q.inline_frac.max():.2f}% (SGP4); trend "
              "tests in Sec. 2.", ""]
    return L


def main() -> int:
    old, oldsrc = read_macros(OLD_TEX)
    new, newsrc = read_macros(NEW_TEX)
    texts = paper_lines()
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = "unknown"
    L = ["# Phase 4b report: non-overlapping 24 h as-deployed windows", "",
         f"Generated by `scripts/make_phase4b_report.py` at `{head}`.  Do not edit by hand.",
         "",
         "## 0. What changed", "",
         "- E5/E6 start times are now 24 h apart (`start_offset_s: 86400` = `duration_s`), "
         "so the 24 h windows are consecutive days from the snapshot epoch and do not "
         "overlap; before they were 5 h apart and overlapped.  Same 10 s step, float32, "
         "snapshot and Gen1/Gen2 classification.",
         "- This removes the overlap only.  All windows still come from one TLE snapshot "
         "and one J2 propagation of it, so the intervals describe day-to-day variability "
         "conditional on that snapshot (and on the propagation model, Sec. 2), not "
         "sampling uncertainty over constellation states.  It is not a complete fix of CI "
         "validity.",
         "- `paper/main.tex` is not edited.", ""]
    L += section_runs(oldsrc, newsrc)
    L += section_propagation()
    L += section_daily()
    v, _changed = section_verdicts(old, new, oldsrc, newsrc, texts)
    L += v
    L += section_claims(old, new, texts)
    L += section_basis()
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
