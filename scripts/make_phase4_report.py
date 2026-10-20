"""Write ``results/PHASE4_REPORT.md``: WCNC review items (Gen1/Gen2, handoff, compliance).

Every number is read from ``results/`` (AGENTS.md rule 1).  Sections:

1. Gen1 / Gen2 separation: the modelled shells, the as-deployed classification
   rule and counts, and the headline 25.261(d) metrics side by side for the
   pre-Phase-4 runs (the numbers currently in the paper), Gen2 only, Gen1 only
   and all Starlink; which limit comparisons change.
2. Handoff (satellite diversity): the definition and the constrained policies
   (K highest-elevation candidates, minimum time between switches).
3. Compliance reporting: station-replications above each limit and the largest
   observed value, the qualifying-case count, and the numbers behind Figs. 1-2.

Run:  python scripts/make_phase4_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ngsi.constants import THROUGHPUT_DEGRADE_MAX, UNAVAILABILITY_INCREASE_MAX
from ngsi.runner import t95

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "PHASE4_REPORT.md"

VICTIM = "ONEWEB"
HEAD_DIAM = 0.45
QUAL = 0.99
GENS = ("gen2", "gen1", "all")
GEN_LABEL = {None: "pre-Phase-4 (paper)", "gen2": "Gen2 only", "gen1": "Gen1 only",
             "all": "all Starlink"}
LIMITS = {"degradation": THROUGHPUT_DEGRADE_MAX, "unavail": UNAVAILABILITY_INCREASE_MAX}


# --------------------------------------------------------------------- helpers
def runs(prefix: str) -> list[Path]:
    return sorted(p for p in RESULTS.glob(f"{prefix}_*") if p.is_dir())


def status(run: Path) -> dict:
    return json.loads((run / "status.json").read_text())


def newest_where(prefix: str, **kv) -> Path | None:
    for run in reversed(runs(prefix)):
        try:
            st = status(run)
        except (OSError, ValueError):
            continue
        if all(st.get(k) == v for k, v in kv.items()):
            return run
    return None


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pct(v, nd: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v) * 100.0:.{nd}f}"


def pm(m, ci, nd: int = 2) -> str:
    return "n/a" if pd.isna(m) else f"{pct(m, nd)} +/- {pct(ci, nd)}"


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "(no rows)"
    cols = cols or list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def rep_worst(raw: pd.DataFrame, col: str, system_a: str = VICTIM, variant: int = 0,
              diam: float = HEAD_DIAM) -> tuple[float, float, int]:
    """Worst qualified station per replication, then mean / 95% t-CI (the paper's aggregation)."""
    r = raw[(raw["system_a"] == system_a) & (raw["variant"] == variant)
            & (raw["diameter_m"] == diam) & (raw["avail0_frac"] >= QUAL)]
    w = r.groupby("phasing")[col].max().dropna().to_numpy(dtype=np.float64)
    if w.size == 0:
        return float("nan"), float("nan"), 0
    ci = float(t95(w.size) * np.std(w, ddof=1) / np.sqrt(w.size)) if w.size >= 2 else 0.0
    return float(w.mean()), ci, int(w.size)


def exceed(raw: pd.DataFrame, col: str, limit: float, system_a: str = VICTIM,
           variant: int = 0, diam: float = HEAD_DIAM) -> dict:
    """Station-replications (station x replication) above ``limit`` and the largest value."""
    r = raw[(raw["system_a"] == system_a) & (raw["variant"] == variant)
            & (raw["diameter_m"] == diam)]
    q = r[r["avail0_frac"] >= QUAL]
    v = q[col].dropna()
    return {"n_total": int(len(r)), "n_qual": int(len(q)), "n_exceed": int((v > limit).sum()),
            "max": float(v.max()) if len(v) else float("nan")}


def verdict(m: float, ci: float, limit: float) -> str:
    """Limit comparison with the CI: exceeds / within / indistinguishable."""
    if pd.isna(m):
        return "n/a"
    if m - ci > limit:
        return "exceeds"
    if m + ci < limit:
        return "within"
    return "indistinguishable"


def raw_dtt(run: Path) -> pd.DataFrame:
    name = "raw_dtt.parquet" if run.name.startswith("e4_") else "raw_as_deployed_dtt.parquet"
    return pd.read_parquet(run / name)


# ------------------------------------------------------------- run selection
def select_runs() -> dict:
    """{experiment: {generation: run}}; None = pre-Phase-4 (no generation key)."""
    out: dict = {}
    for prefix in ("e4_dtt", "e5_as_deployed"):
        out[prefix] = {g: newest_where(prefix, diversity_fspl="interferer", steering=None,
                                       starlink_generation=g) for g in (None, *GENS)}
    out["e6_eirp_sweep"] = {g: newest_where("e6_eirp_sweep", starlink_generation=g)
                            for g in (None, *GENS)}
    return out


# ------------------------------------------------------------------ section 1
def shells_section(sel: dict) -> list[str]:
    L = ["### 1.1 Modelled (as-filed) Starlink shells", ""]
    run = sel["e4_dtt"].get("gen2") or sel["e4_dtt"].get(None)
    if run is None:
        return L + ["(no E4 run)", ""]
    cons = yaml.safe_load((run / "configs" / "constellations.yaml").read_text())
    rows = []
    for c in cons["constellations"]:
        if not c["name"].startswith("STARLINK"):
            continue
        gen = "Gen2" if c["name"] == "STARLINK_GEN2" else "Gen1"
        for sh in c["shells"]:
            rows.append({"generation": gen, "shell": sh["name"],
                         "altitude_km": f"{sh['altitude_km']:g}",
                         "inclination_deg": f"{sh['inclination_deg']:g}",
                         "planes": sh["planes"], "sats_per_plane": sh["sats_per_plane"],
                         "n_sats": sh["planes"] * sh["sats_per_plane"],
                         "source": " ".join(str(sh["source"]).split())})
    L += [f"From `results/{run.name}/configs/constellations.yaml`.", "", md_table(rows), ""]
    for g in GENS:
        r = sel["e4_dtt"].get(g)
        if r is not None:
            L.append(f"- E4 `{g}`: STARLINK = {status(r)['n_sats']['STARLINK']} satellites "
                     f"(`results/{r.name}/`).")
    L.append("")
    return L


def classification_section(sel: dict) -> list[str]:
    L = ["### 1.2 As-deployed Gen1 / Gen2 classification", ""]
    run = next((sel["e5_as_deployed"][g] for g in GENS if sel["e5_as_deployed"].get(g)), None)
    if run is None:
        return L + ["(no Phase 4 E5 run)", ""]
    tm = json.loads((run / "tle_meta.json").read_text())
    gm = tm.get("starlink_generation")
    if gm is None:
        return L + ["(run has no classification)", ""]
    L.append(f"From `results/{run.name}/` (`tle_meta.json`, `summary_starlink_generation.csv`). "
             "Rule (`starlink_generation_rule` in the E5/E6 config, first match wins; a "
             "satellite matches when |inclination - i| <= tol and the circular altitude is "
             "in [min, max)):")
    L.append("")
    L.append(md_table([{"order": i + 1, "generation": e["generation"],
                        "shell": e.get("shell", ""),
                        "inclination_deg": f"{e['inclination_deg']:g}",
                        "tol_deg": f"{e['inclination_tol_deg']:g}",
                        "altitude_km": f"[{e['altitude_min_km']:g}, {e['altitude_max_km']:g})",
                        "source": " ".join(str(e.get("source", "")).split())}
                       for i, e in enumerate(gm["rule"])]))
    L.append("")
    c = gm["counts"]
    n_op = tm["groups"]["STARLINK"]["n_operational"]
    L.append(f"Operational Starlink satellites (altitude window "
             f"{tm['groups']['STARLINK']['operational_band_km']} km): **{n_op}** = "
             f"Gen1 **{c['gen1']}** + Gen2 **{c['gen2']}** + unclassified "
             f"**{c['unclassified']}** (raw snapshot {tm['groups']['STARLINK']['n_sats_raw']}).")
    L.append("")
    sh = pd.read_csv(run / "summary_starlink_generation.csv")
    L.append(md_table([{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()}
                       for r in sh.to_dict("records")]))
    L.append("")
    return L


POLICY_LABEL = {
    "full": "event-avoiding handoff",
    "full_keep_serving": "rate-maximizing handoff",
    "legacy": "legacy handoff (pre-Phase-4)",
    "full_k2": "event-avoiding, K = 2",
    "full_k3": "event-avoiding, K = 3",
    "full_n10": "event-avoiding, N = 10 s",
    "full_n60": "event-avoiding, N = 60 s",
}
HEADLINE_POLICIES = ("full", "full_keep_serving", "legacy")
PAPER_COMMIT = "45b4aa8"   # numbers.tex as committed before Phase 4 regenerated it
PAPER_LABEL = "paper now (Phase 3 numbers.tex)"


def paper_macros() -> dict[str, str]:
    """Macros of the pre-Phase-4 numbers.tex (the numbers the paper prints now)."""
    import re

    try:
        txt = subprocess.check_output(["git", "show", f"{PAPER_COMMIT}:paper/numbers.tex"],
                                      cwd=REPO_ROOT, text=True)
    except Exception:  # noqa: BLE001
        return {}
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt))


def run_with_policy(prefix: str, gen, policy: str) -> Path | None:
    """Newest primary-configuration run of ``gen`` that evaluated ``policy``."""
    for run in reversed(runs(prefix)):
        try:
            st = status(run)
        except (OSError, ValueError):
            continue
        if (st.get("starlink_generation") == gen and st.get("steering") is None
                and st.get("diversity_fspl") == "interferer"
                and policy in [q["name"] for q in st.get("handoff_policies", [])]
                and (run / "raw_handoff.parquet").exists()):
            return run
    return None


def cell(raw: pd.DataFrame, col: str, nd: int) -> str:
    """mean +/- CI [verdict]; station-replications over the limit; largest value."""
    lim = LIMITS["degradation" if "degradation" in col else "unavail"]
    m, ci, _ = rep_worst(raw, col)
    e = exceed(raw, col, lim)
    return (f"{pm(m, ci, nd)} [{verdict(m, ci, lim)}]; {e['n_exceed']}/{e['n_qual']} over; "
            f"max {pct(e['max'], nd)}")


def headline_rows(sel: dict, prefix: str) -> list[dict]:
    """Headline 25.261(d) numbers per generation (OneWeb victim, realistic, 0.45 m).

    Both handoff rules (event-avoiding, rate-maximizing) are headline rows; the
    legacy rule is shown for attribution.  The first column is the paper as it
    stands (pre-Phase-4 numbers.tex for E5; the pre-Phase-4 run for E4).
    """
    cols: dict[str, dict] = {}
    pm_ = paper_macros()
    if prefix == "e5_as_deployed" and pm_:
        g = lambda k: pm_.get(k, "n/a")  # noqa: E731
        cols[PAPER_LABEL] = {
            "run": f"numbers.tex @ {PAPER_COMMIT} (2026-10-18 snapshot, all Starlink)",
            "Starlink satellites": g("NStarlinkOp"),
            "event peak % (lat)": f"{g('BindDepWorstPct')} +/- {g('BindDepWorstCi')} "
                                  f"({g('BindDepWorstLat')})",
            "degradation, no handoff %": f"{g('DegrDepMaxPct')} +/- {g('DegrDepMaxCi')}",
            "avail. change, no handoff %": f"{g('UnavailIncDepPct')} +/- {g('UnavailIncDepCi')}",
            f"degradation, {POLICY_LABEL['legacy']} %": f"{g('DegrDepDivPct')} +/- "
                                                         f"{g('DegrDepDivCi')}",
            f"avail. change, {POLICY_LABEL['legacy']} %": f"{g('UnavailDepDivPct')} +/- "
                                                           f"{g('UnavailDepDivCi')}",
            "Starlink-victim degradation, no handoff % (comparison)":
                f"{g('DegrDepStarPct')} +/- {g('DegrDepStarCi')}",
        }
    for gen in (None, *GENS):
        run = sel[prefix].get(gen)
        if run is None or (gen is None and prefix == "e5_as_deployed"):
            continue
        raw = raw_dtt(run)
        st = status(run)
        d = pd.read_csv(run / "summary_dtt.csv")
        b = d[(d["system_a"] == VICTIM) & (d["variant"] == 0) & (d["diameter_m"] == HEAD_DIAM)]
        pk = b.loc[b["event_frac_mean"].idxmax()]
        vals = {
            "run": f"`{run.name}`",
            "Starlink satellites": str(st.get("n_sats", {}).get("STARLINK", "n/a")),
            "event peak % (lat)": f"{pct(pk['event_frac_mean'])} +/- "
                                  f"{pct(pk['event_frac_ci95'])} ({pk['latitude_deg']:g})",
            "degradation, no handoff %": cell(raw, "degradation_nodiv", 2),
            "avail. change, no handoff %": cell(raw, "unavail_incr_frac", 3),
        }
        if gen is None:
            vals[f"degradation, {POLICY_LABEL['legacy']} %"] = cell(raw, "degradation_div", 2)
            vals[f"avail. change, {POLICY_LABEL['legacy']} %"] = cell(raw, "unavail_incr_div_frac",
                                                                     3)
        else:
            for pol in HEADLINE_POLICIES:
                pr = run_with_policy(prefix, gen, pol)
                if pr is None:
                    continue
                ho = pd.read_parquet(pr / "raw_handoff.parquet")
                ho = ho[ho["policy"] == pol]
                vals[f"degradation, {POLICY_LABEL[pol]} %"] = cell(ho, "degradation_div", 2)
                vals[f"avail. change, {POLICY_LABEL[pol]} %"] = cell(ho, "unavail_incr_div_frac", 3)
        m, ci, _ = rep_worst(raw, "degradation_nodiv", system_a="STARLINK")
        vals["Starlink-victim degradation, no handoff % (comparison)"] = pm(m, ci)
        cols[GEN_LABEL[gen]] = vals
    if not cols:
        return []
    keys: list[str] = []
    for v in cols.values():
        keys += [k for k in v if k not in keys]
    return [{"metric": k, **{c: v.get(k, "-") for c, v in cols.items()}} for k in keys]


def table1_rows(sel: dict) -> list[dict]:
    """E6 Table I side by side: worst-qualified-station mean +/- CI per case."""
    data = {}
    for g in (None, *GENS):
        run = sel["e6_eirp_sweep"].get(g)
        if run is None:
            continue
        s = pd.read_csv(run / "summary_eirp_sweep.csv")
        if "handoff_policy" in s.columns:
            s = s[s["handoff_policy"] == status(run)["primary_handoff"]]
        data[g] = s[(s["victim"] == VICTIM) & (s["variant"] == 0)
                    & (s["diameter_m"] == HEAD_DIAM)].set_index("case")
    if not data:
        return []
    rows = []
    cases = list(next(iter(data.values())).index)
    for case in cases:
        for lab, col, nd, lim in (("degr no div", "degr_max_qual", 2, THROUGHPUT_DEGRADE_MAX),
                                  ("degr div", "degr_div_max_qual", 2, THROUGHPUT_DEGRADE_MAX),
                                  ("unav no div", "unavail_max_qual", 3,
                                   UNAVAILABILITY_INCREASE_MAX),
                                  ("unav div", "unavail_div_max_qual", 3,
                                   UNAVAILABILITY_INCREASE_MAX)):
            row = {"case": case, "metric %": lab}
            for g, s in data.items():
                if case not in s.index:
                    row[GEN_LABEL[g]] = "n/a"
                    continue
                m, ci = s.loc[case, col], s.loc[case, col + "_ci95"]
                row[GEN_LABEL[g]] = f"{pm(m, ci, nd)} [{verdict(m, ci, lim)}]"
            rows.append(row)
    return rows


def flips(rows: list[dict], ref: str = GEN_LABEL[None], new: str = GEN_LABEL["gen2"]) -> list[str]:
    """Rows whose limit verdict differs between the paper's numbers and Gen2 only."""
    out = []
    for r in rows:
        a, b = r.get(ref, ""), r.get(new, "")
        va = a[a.find("[") + 1:a.find("]")] if "[" in a else None
        vb = b[b.find("[") + 1:b.find("]")] if "[" in b else None
        if va and vb and va != vb:
            label = " / ".join(str(r[k]) for k in r if k in ("case", "metric", "metric %"))
            out.append(f"- {label}: {ref} **{va}** ({a.split(' [')[0]}) -> {new} **{vb}** "
                       f"({b.split(' [')[0]})")
    return out


# ------------------------------------------------------------------ section 2
HANDOFF_DEFINITION = """\
Implemented in `ngsi.dtt_screen` (`HandoffPolicy`, `_evaluate_policies`,
`rate_limited_choice`); unit tests with hand-checkable sequences in
`tests/test_handoff.py`.

- **Candidate set.** The visible OneWeb satellites (elevation >= the victim mask)
  among the `k_a` highest-elevation ones (`k_a` = the largest number ever visible at
  any station, from the visibility pass), optionally limited to the K
  highest-elevation satellites (rank 0, the serving satellite, included).  An
  alternate (rank >= 1) is admissible only if its own aggregate Delta T/T <= 6% (it
  resolves the event).
- **Trigger.** The terminal stays on the highest-elevation satellite (the same
  policy as the baseline) and considers a handoff only while that link is in
  event (aggregate Delta T/T > 6%).  Not a continuous best-C/(N+I) choice.
- **Two selection rules, both reported as headline rows (neither is chosen).**
  - **Event-avoiding handoff** (`full`, the code's primary): during an event the
    terminal always leaves the triggered satellite, for the admissible
    (non-triggered) alternate with the highest C/(N+I).  If none is admissible
    (unresolvable), it stays on the highest-elevation satellite with its degraded
    SINR.
  - **Rate-maximizing handoff** (`full_keep_serving`): the same, except the
    terminal stays on the triggered satellite when its degraded C/(N+I) is higher
    than every admissible alternate's.
- **New link's C/N (full link budget).** `C/N_k = EIRP_OneWeb + G_rx,peak -
  FSPL(r_k) - N`, with `r_k` the slant range to candidate k (the terminal re-points,
  so the receive gain toward it is the peak gain); the interference is the
  aggregate of the same interferer satellites at the off-axis angles of the
  re-pointed antenna (main beam of Starlink's serving satellite plus the S.1528
  sidelobes of the others, at the interferer's slant ranges and PFD-constant EIRP).
  `C/(N+I)_k = C/N_k - 10 log10(1 + Delta T/T_k)`.
- **Legacy rule (pre-Phase-4, `legacy`).** The admissible alternate with the lowest
  Delta T/T, but with the *serving* satellite's clear-sky C/N (the alternate's own
  slant range was not used).  Retained only to attribute the change.
- **Rate limit (`full_n10`, `full_n60`).** Stateful: the terminal keeps its
  current satellite until at least N s have passed since its last switch, unless
  that satellite leaves the candidate set (forced switch).  Every change of
  satellite counts, including geometric handovers to a new highest-elevation
  satellite.  N = 10 s equals the time step, so it reproduces the memoryless rule
  (checked in `tests/test_handoff.py`).
- **Baseline for degradation.** `1 - R_used / R_0`, with `R_0` the
  interference-free Shannon rate of the highest-elevation satellite (the
  rule-free reference link, no switching limit).  For the rate-limited policies the
  report also gives the loss of the same rate limit without any interferer
  (`base_loss`) and the degradation against that baseline (`degr_vs_base`).
- **Availability change.** Fraction of time with baseline C/N >= 0 dB and
  C/(N+I) < 0 dB on the link actually used.
"""


def policy_rows(run: Path) -> list[dict]:
    ho = pd.read_parquet(run / "raw_handoff.parquet")
    rows = []
    for pol, g in ho.groupby("policy", sort=False):
        md, cd, n = rep_worst(g, "degradation_div")
        mu, cu, _ = rep_worst(g, "unavail_incr_div_frac")
        mb, cb, _ = rep_worst(g, "base_loss")
        mv, cv, _ = rep_worst(g, "degradation_div_vs_base")
        h = g[(g["system_a"] == VICTIM) & (g["variant"] == 0) & (g["diameter_m"] == HEAD_DIAM)
              & (g["avail0_frac"] >= QUAL)]
        k = g["k"].iloc[0]
        ni = g["min_interval_s"].iloc[0]
        rows.append({
            "policy": f"{POLICY_LABEL.get(pol, pol)} (`{pol}`)", "link C/N": g["link_cn"].iloc[0],
            "K": "all" if pd.isna(k) else f"{k:g}", "N_s": "-" if pd.isna(ni) else f"{ni:g}",
            "degr with div %": f"{pm(md, cd)} [{verdict(md, cd, THROUGHPUT_DEGRADE_MAX)}]",
            "avail. change with div %": f"{pm(mu, cu, 3)} "
                                        f"[{verdict(mu, cu, UNAVAILABILITY_INCREASE_MAX)}]",
            "base_loss %": pm(mb, cb, 3), "degr_vs_base %": pm(mv, cv),
            "unresolved % (mean)": pct(h["unres_frac"].mean(), 3),
            "time on alternate % (mean)": pct(h["diverted_frac"].mean()),
            "switches/h (mean)": ("-" if h["switches_per_hour"].isna().all()
                                  else f"{h['switches_per_hour'].mean():.1f}"),
            "replications": n,
        })
    return rows


# ------------------------------------------------------------------ section 3
def exceed_rows_raw(run: Path) -> list[dict]:
    ho = pd.read_parquet(run / "raw_handoff.parquet")
    rows = []
    for pol, g in ho.groupby("policy", sort=False):
        if pol == ho["policy"].iloc[0]:
            for lab, col, lim in (("degr no div", "degradation_nodiv", THROUGHPUT_DEGRADE_MAX),
                                  ("avail. change no div", "unavail_incr_frac",
                                   UNAVAILABILITY_INCREASE_MAX)):
                e = exceed(g, col, lim)
                rows.append({"policy": "(none)", "metric": lab, "limit %": pct(lim, 1),
                             "station-reps": e["n_total"], "qualified": e["n_qual"],
                             "exceeding": e["n_exceed"], "max observed %": pct(e["max"], 3)})
        for lab, col, lim in (("degr with div", "degradation_div", THROUGHPUT_DEGRADE_MAX),
                              ("avail. change with div", "unavail_incr_div_frac",
                               UNAVAILABILITY_INCREASE_MAX)):
            e = exceed(g, col, lim)
            rows.append({"policy": pol, "metric": lab, "limit %": pct(lim, 1),
                         "station-reps": e["n_total"], "qualified": e["n_qual"],
                         "exceeding": e["n_exceed"], "max observed %": pct(e["max"], 3)})
    return rows


def exceed_rows_e6(run: Path) -> list[dict]:
    s = pd.read_csv(run / "summary_eirp_sweep.csv")
    s = s[(s["victim"] == VICTIM) & (s["variant"] == 0) & (s["diameter_m"] == HEAD_DIAM)]
    rows = []
    for r in s.itertuples():
        rows.append({
            "case": r.case, "policy": r.handoff_policy,
            "station-reps": int(r.n_station_reps), "qualified": int(r.n_qual_station_reps),
            "degr>3% no div": int(r.n_exceed_degr_nodiv),
            "max degr no div %": pct(r.degr_max_qual_obs),
            "degr>3% div": int(r.n_exceed_degr_div),
            "max degr div %": pct(r.degr_div_max_qual_obs),
            "unav>0.4% no div": int(r.n_exceed_unav_nodiv),
            "max unav no div %": pct(r.unavail_max_qual_obs, 3),
            "unav>0.4% div": int(r.n_exceed_unav_div),
            "max unav div %": pct(r.unavail_div_max_qual_obs, 3),
        })
    return rows


def latitude_rows(e4: Path, e5: Path) -> list[dict]:
    """Event fraction and visible-satellite pools per latitude (Fig. 1 support)."""
    rows = []
    tabs = {}
    for tag, run in (("filed", e4), ("deployed", e5)):
        d = pd.read_csv(run / "summary_dtt.csv")
        b = d[(d["system_a"] == VICTIM) & (d["variant"] == 0) & (d["diameter_m"] == HEAD_DIAM)]
        pool = (pd.read_csv(run / "summary_service_pool.csv")
                if (run / "summary_service_pool.csv").exists() else None)
        tabs[tag] = (b.set_index("latitude_deg"), pool)
    lats = sorted(tabs["filed"][0].index)
    for la in lats:
        row = {"latitude_deg": f"{la:g}"}
        for tag, (b, pool) in tabs.items():
            row[f"event % {tag}"] = pct(b.loc[la, "event_frac_mean"]) if la in b.index else "n/a"
            row[f"OneWeb C/N median dB {tag}"] = (f"{b.loc[la, 'cn0_median_mean']:.1f}"
                                                  if la in b.index else "n/a")
            for sysn in ("STARLINK", "ONEWEB"):
                v = "n/a"
                if pool is not None:
                    q = pool[(pool["system"] == sysn) & (pool["latitude_deg"] == la)]
                    if len(q):
                        v = f"{q['pool_mean_mean'].iloc[0]:.2f}"
                row[f"{sysn} visible {tag}"] = v
        rows.append(row)
    return rows


def main() -> int:
    sel = select_runs()
    L = ["# Phase 4 - WCNC review: Gen1/Gen2 separation, handoff, compliance reporting", "",
         f"Generated by `scripts/make_phase4_report.py` at git `{git_hash()}`. Every number "
         "is read from `results/` (AGENTS.md rule 1).  The paper's framing is unchanged.", "",
         "## Runs", ""]
    rr = []
    for prefix, d in sel.items():
        for g, run in d.items():
            rr.append({"experiment": prefix, "starlink_generation": str(g),
                       "run": f"`results/{run.name}/`" if run else "(absent)"})
    L += [md_table(rr), ""]
    L.append("Limit verdicts in brackets compare mean +/- 95% CI with the 25.261(d) limit "
             "(3% degradation, 0.4% availability change): **exceeds** if mean - CI > limit, "
             "**within** if mean + CI < limit, otherwise **indistinguishable**.  Metric: "
             "worst qualified station (baseline availability >= 99%) per replication, then "
             "mean and CI across replications (the paper's single aggregation); OneWeb "
             "victim, realistic variant, 0.45 m.")
    L.append("")

    L += ["## 1. Gen1 / Gen2 separation", ""]
    L += shells_section(sel)
    L += classification_section(sel)
    for prefix, title in (("e4_dtt", "1.3 As filed (E4, Walker)"),
                          ("e5_as_deployed", "1.4 As deployed (E5, CelesTrak snapshot)")):
        rows = headline_rows(sel, prefix)
        L += [f"### {title}", "", md_table(rows), ""]
        f = flips(rows)
        L += ["Verdicts that change from the paper's numbers to Gen2 only:", ""]
        L += (f or ["- none"]) + [""]
    rows6 = table1_rows(sel)
    L += ["### 1.5 Table I (E6 sensitivity cases, as deployed)", "", md_table(rows6), ""]
    f = flips(rows6)
    L += ["Verdicts that change from the paper's numbers to Gen2 only:", ""]
    L += (f or ["- none"]) + [""]

    L += ["## 2. Handoff definition and constrained diversity", "", HANDOFF_DEFINITION, ""]
    for prefix, title in (("e5_as_deployed", "as deployed (E5)"), ("e4_dtt", "as filed (E4)")):
        for g in GENS:
            run = sel[prefix].get(g)
            if run is None or not (run / "raw_handoff.parquet").exists():
                continue
            L += [f"### 2.x Policies, {title}, {GEN_LABEL[g]} (`results/{run.name}/`)", "",
                  md_table(policy_rows(run)), ""]

    L += ["## 3. Compliance reporting", ""]
    L += ["### 3.1 Station-replications above each limit (E4 / E5, per handoff policy)", "",
          "station-reps = stations x replications at the headline aperture; qualified = "
          "baseline availability >= 99% (the 25.261(d) domain).", ""]
    for prefix in ("e4_dtt", "e5_as_deployed"):
        for g in GENS:
            run = sel[prefix].get(g)
            if run is None or not (run / "raw_handoff.parquet").exists():
                continue
            L += [f"**{prefix}, {GEN_LABEL[g]}** (`results/{run.name}/`)", "",
                  md_table(exceed_rows_raw(run)), ""]
    L += ["### 3.2 Station-replications above each limit (E6 cases)", ""]
    for g in GENS:
        run = sel["e6_eirp_sweep"].get(g)
        if run is None:
            continue
        L += [f"**{GEN_LABEL[g]}** (`results/{run.name}/`)", "", md_table(exceed_rows_e6(run)),
              ""]

    L += ["### 3.3 The qualifying-case count", ""]
    e4p = sel["e4_dtt"].get(None)
    e4g = sel["e4_dtt"].get("gen2")
    for g, run in ((None, e4p), ("gen2", e4g)):
        if run is None:
            continue
        d = pd.read_csv(run / "summary_dtt.csv")
        dd = d[d["system_a"] == VICTIM]
        cells = dd[["diameter_m", "variant", "latitude_deg"]].drop_duplicates()
        qc = dd[dd["avail0_frac_mean"] >= QUAL][["diameter_m", "variant",
                                                  "latitude_deg"]].drop_duplicates()
        raw = raw_dtt(run)
        q = raw[(raw["system_a"] == VICTIM) & (raw["variant"] == 0)
                & (raw["diameter_m"] == HEAD_DIAM)]
        st = status(run)
        L.append(f"- {GEN_LABEL[g]} (`results/{run.name}/`): the old macro counted "
                 f"(aperture x variant x latitude) cells of `summary_dtt.csv` whose "
                 f"*latitude-mean* baseline availability is >= 99%: {len(qc)} of {len(cells)} "
                 f"= {dd['diameter_m'].nunique()} apertures x {dd['variant'].nunique()} variants "
                 f"x {dd['latitude_deg'].nunique()} latitudes.  The 25.261(d) domain is "
                 f"station-replications: {int((q['avail0_frac'] >= QUAL).sum())} of {len(q)} "
                 f"= {q['latitude_deg'].nunique() * q['longitude_deg'].nunique()} stations x "
                 f"{st['n_phasings']} phasings (headline aperture, realistic variant; the "
                 f"baseline does not depend on the variant).")
    L.append("")

    L += ["### 3.4 Numbers behind Fig. 1 (event fraction vs latitude) and Fig. 2", ""]
    e4h = sel["e4_dtt"].get("gen2")
    e5h = sel["e5_as_deployed"].get("gen2")
    if e4h is not None and e5h is not None:
        L += [f"Gen2-only headline runs `results/{e4h.name}/` (filed) and "
              f"`results/{e5h.name}/` (deployed); per-latitude means over meridians and "
              "replications; visible = mean number of satellites above the 25 deg mask.", "",
              md_table(latitude_rows(e4h, e5h)), ""]
        d5 = pd.read_csv(e5h / "summary_dtt.csv")
        b5 = d5[(d5["system_a"] == VICTIM) & (d5["variant"] == 0)
                & (d5["diameter_m"] == HEAD_DIAM)]
        L.append("Fig. 2 (as deployed, Gen2 only) largest 95% CI half-width over latitude: "
                 f"degradation {pct(b5[['degradation_nodiv_ci95', 'degradation_div_ci95']].max().max())}"
                 f" pp, availability change "
                 f"{pct(b5[['unavail_incr_frac_ci95', 'unavail_incr_div_frac_ci95']].max().max(), 3)}"
                 " pp.")
        L.append("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[phase4] wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
