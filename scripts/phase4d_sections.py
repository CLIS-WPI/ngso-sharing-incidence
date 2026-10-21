"""Phase 4d report sections (A-D of the second review), used by make_phase4c_report.py.

A. both handoff rules under the same constraints (K = 2, dwell 30 / 60 s, 55 deg mask);
B. interruption cost per excess switch (results/e14_interruption_cost_*);
C. 2 s vs 10 s time step on the same days (results/sensitivity/);
D. Gen2 = 43 deg shell only vs 43 + 53 deg (results/sensitivity/);
and the ``paper/main.tex`` sentences whose verdict depends on A-D.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import make_numbers as mn
import make_phase4b_report as p4b
from ngsi.constants import THROUGHPUT_DEGRADE_MAX as DL
from ngsi.constants import UNAVAILABILITY_INCREASE_MAX as UL

RULE_LABEL = {"full": "event-avoiding", "full_keep_serving": "rate-maximizing"}
CONSTRAINTS = (("unconstrained", "full", "full_keep_serving"),
               ("K = 2", "full_k2", "full_keep_serving_k2"),
               ("dwell N = 30 s", "full_n30", "full_keep_serving_n30"),
               ("dwell N = 60 s", "full_n60", "full_keep_serving_n60"))


def _verdict(m, ci, lim):
    return p4b.verdict(100 * m, 100 * ci, 100 * lim)


def stats(df: pd.DataFrame, col: str, lim: float) -> dict:
    m, ci = mn._rep_worst(df, col, mn.VICTIM, 0)
    n_over, n_q, mx = mn._over(df, col, lim)
    return {"m": m, "ci": ci, "max": mx, "over": n_over, "n": n_q, "v": _verdict(m, ci, lim)}


def fmt(s: dict, nd: int) -> str:
    return (f"{100 * s['m']:.{nd}f} +/- {100 * s['ci']:.{nd}f} (max {100 * s['max']:.{nd}f}, "
            f"{s['over']}/{s['n']} over) [{s['v']}]")


def _policy_frame(run: Path, pol: str) -> pd.DataFrame:
    ho = pd.read_parquet(run / "raw_handoff.parquet")
    return ho[ho["policy"] == pol]


def section_a() -> tuple[list[str], dict]:
    run = mn.headline_run("e5_as_deployed", "gen2")
    out: dict = {}
    rows = []
    for label, ev, rm in CONSTRAINTS:
        r = {"constraint": label}
        for tag, pol in (("event-avoiding", ev), ("rate-maximizing", rm)):
            pr = mn.run_with_policy("e5_as_deployed", "gen2", pol)
            if pr is None or pr != run:
                r[f"{tag} degradation %"] = r[f"{tag} availability change %"] = "not in run"
                continue
            h = _policy_frame(pr, pol)
            d, u = stats(h, "degradation_div", DL), stats(h, "unavail_incr_div_frac", UL)
            out[(label, tag)] = (d, u)
            r[f"{tag} degradation %"] = fmt(d, 2)
            r[f"{tag} availability change %"] = fmt(u, 3)
        rows.append(r)
    e6 = mn.newest_where("e6_eirp_sweep", starlink_generation="gen2")
    if e6 is not None:
        s6 = pd.read_csv(e6 / "summary_eirp_sweep.csv")
        g = s6[(s6["case"] == "oneweb_min_elev_55") & (s6["victim"] == mn.VICTIM)
               & (s6["variant"] == 0) & (s6["diameter_m"] == mn.HEAD_DIAM)]
        r = {"constraint": "OneWeb 55 deg mask (E6)"}
        for pol, tag in RULE_LABEL.items():
            x = g[g["handoff_policy"] == pol] if "handoff_policy" in g.columns else g.iloc[0:0]
            if x.empty:
                r[f"{tag} degradation %"] = r[f"{tag} availability change %"] = "not in run"
                continue
            x = x.iloc[0]
            n_q = int(x["n_qual_station_reps"])
            d = {"m": x["degr_div_max_qual"], "ci": x["degr_div_max_qual_ci95"],
                 "max": x["degr_div_max_qual_obs"], "over": int(x["n_exceed_degr_div"]), "n": n_q}
            u = {"m": x["unavail_div_max_qual"], "ci": x["unavail_div_max_qual_ci95"],
                 "max": x["unavail_div_max_qual_obs"], "over": int(x["n_exceed_unav_div"]),
                 "n": n_q}
            d["v"], u["v"] = _verdict(d["m"], d["ci"], DL), _verdict(u["m"], u["ci"], UL)
            out[("55 deg mask", tag)] = (d, u)
            r[f"{tag} degradation %"] = fmt(d, 2)
            r[f"{tag} availability change %"] = fmt(u, 3)
        rows.append(r)
    L = ["## 5. A: both handoff rules under the same constraints", "",
         f"All E5 policies share one run (`results/{run.name}/`), the same 8 days and "
         "stations, one baseline (interference-free, highest-elevation satellite, R_0) and a "
         "full link budget (own slant range, own C/N, own aggregate Delta T/T) for every "
         "candidate satellite.  The 55 deg mask rows share one E6 run "
         f"(`results/{e6.name if e6 is not None else '-'}/`, case `oneweb_min_elev_55`) with "
         "the same baseline definition under that mask.  Entries: worst qualified station "
         "per day, mean +/- 95% t-interval over days (max observed station-day, "
         "station-days over the limit) [verdict].", "", p4b.md_table(rows), ""]
    return L, out


def section_b() -> tuple[list[str], dict]:
    run = mn.newest("e14_interruption_cost")
    if run is None:
        return ["## 6. B: interruption cost", "", "(no e14 run)", ""], {}
    si = pd.read_csv(run / "summary_interruption.csv")
    mt = pd.read_csv(run / "summary_max_tau.csv")
    st = json.loads((run / "status.json").read_text())
    rows, changed = [], []
    for pol, g in si.groupby("policy", sort=False):
        g = g.sort_values("tau_s")
        v0 = g.iloc[0]
        for _, r in g.iterrows():
            rows.append({"policy": pol, "tau s": f"{r.tau_s:g}",
                         "switches/h (baseline / excess)":
                             f"{r.switches_per_hour_mean:.1f} ({r.base_switches_per_hour_mean:.1f}"
                             f" / {r.excess_switches_per_hour_mean:.1f})",
                         "degradation %": f"{100 * r.degradation_mean:.2f} +/- "
                                          f"{100 * r.degradation_ci95:.2f} [{r.degradation_verdict}]",
                         "availability change %": f"{100 * r.availability_mean:.3f} +/- "
                                                  f"{100 * r.availability_ci95:.3f} "
                                                  f"[{r.availability_verdict}]",
                         "over (degr / avail)":
                             f"{int(r.degradation_n_over)} / {int(r.availability_n_over)}"})
            for k in ("degradation", "availability"):
                if r[f"{k}_verdict"] != v0[f"{k}_verdict"]:
                    changed.append({"policy": pol, "limit": k, "tau s": f"{r.tau_s:g}",
                                    "tau = 0": v0[f"{k}_verdict"], "with tau": r[f"{k}_verdict"]})

    def tau_txt(v):
        return ("not met at tau = 0" if v < 0 else f"> {st['tau_max_s']:g}"
                if not np.isfinite(v) else f"{v:.2f}")

    mrows = [{"policy": r.policy, "largest tau meeting degradation limit s":
              tau_txt(r.max_tau_degradation_s),
              "largest tau meeting availability limit s": tau_txt(r.max_tau_availability_s)}
             for _, r in mt.iterrows()]
    L = ["## 6. B: handoff interruption cost (post-processing)", "",
         f"Run `results/{run.name}/` (`scripts/interruption_cost.py`) on "
         f"`results/{st['source_run']}/`; no re-simulation.", "",
         "### 6.1 Exact definition", "",
         "For station g, day d (one row of `raw_handoff.parquet`), T_d = 86 400 s, step "
         "dt = 10 s, served(t) = the victim has a satellite above its mask:", "",
         "- Policy switches: S_p(g,d) = #{t : s_p(t) != s_p(t - dt), both steps served} / "
         "(T_d / 3600) [1/h], where s_p(t) is the satellite the policy uses "
         "(`switches_all_per_hour`; `ngsi.dtt_screen.count_switches`).  A gap (an unserved "
         "step) between two satellites is not counted as a switch.",
         "- Baseline switches: S_0(g,d), the same count for s_0(t) = the highest-elevation "
         "satellite (the unconstrained interference-free baseline that defines R_0), "
         "`base_switches_per_hour`.  It is the same baseline for every policy, including the "
         "dwell-constrained ones.",
         "- Extra switches: E(g,d) = max(0, S_p - S_0).  This is a **count difference**, not "
         "a match of switch instants: a policy switch at a time step where the baseline also "
         "switches is not removed individually; only the excess of the totals is charged.",
         "- Interrupted fraction of time: f(g,d; tau) = E(g,d) * tau / 3600.",
         "- Availability change: dU_tau(g,d) = dU(g,d) + f(g,d; tau), with dU = "
         "#{t : served, C/N_0(t) >= gamma_0 and gamma_p(t) < gamma_0} / n (gamma_0 = 0 dB; "
         "C/N_0 is the baseline satellite's clear-sky C/N, gamma_p the policy's SINR; n the "
         "number of steps), the simulated `unavail_incr_div_frac`.  **Overlap is not removed**: an interruption "
         "is added even if it falls on a step that is already unavailable (gamma < gamma_0), "
         "baseline-unavailable (C/N_0 < 0 dB) or unserved, so dU_tau is an upper bound on "
         "the added unavailability for the stated tau.",
         "- Throughput degradation: D_tau(g,d) = 1 - (1 - D(g,d)) (1 - f(g,d; tau)), i.e. "
         "R_tau = R_p (1 - f) with R_p the policy's **time-averaged** rate over the day "
         "(including its own outages), not the instantaneous rate at the switch instant; "
         "D = 1 - R_p / R_0 is the simulated `degradation_div` against the same baseline R_0.",
         "- Order: tau is applied to every station-day first; then, per day, the worst "
         "qualified station (baseline availability >= 99%) is taken, and the reported value "
         "is the mean +/- 95% t-interval of those daily worsts over the days, as for every "
         "other 25.261(d) number.  'Over' counts station-days above the limit.",
         "- Largest tau meeting a limit: bisection on [0, "
         f"{st['tau_max_s']:g}] s for the condition mean + CI < limit (both D_tau and "
         "dU_tau increase with tau).", "",
         "### 6.2 Results", "",
         p4b.md_table(rows), "",
         "Verdicts that change with tau:", "", p4b.md_table(changed) if changed else "- none", "",
         "Largest tau (bisection on [0, "
         f"{st['tau_max_s']:g}] s, whole interval below the limit):", "",
         p4b.md_table(mrows), ""]
    return L, {"changed": changed, "max_tau": mt}


def _sens(pred) -> Path | None:
    root = mn.RESULTS / "sensitivity"
    runs = sorted(p for p in root.glob("e5_as_deployed_*") if (p / "status.json").exists()) \
        if root.exists() else []
    runs = [r for r in runs if pred(r)]
    return runs[-1] if runs else None


def _three(raw: pd.DataFrame, ho: pd.DataFrame) -> dict:
    out = {"no handoff": (stats(raw, "degradation_nodiv", DL),
                          stats(raw, "unavail_incr_frac", UL), None)}
    for pol, lab in RULE_LABEL.items():
        h = ho[ho["policy"] == pol]
        q = mn._head(h)
        q = q[q["avail0_frac"] >= mn.QUAL]
        sw = (q["switches_all_per_hour"].mean() if "switches_all_per_hour" in q.columns
              else np.nan)
        out[lab] = (stats(h, "degradation_div", DL), stats(h, "unavail_incr_div_frac", UL), sw)
    return out


def section_c() -> tuple[list[str], dict]:
    r2 = _sens(lambda r: float(json.loads((r / "status.json").read_text())
                               .get("time_step_s", 10.0)) < 10.0)
    head = mn.headline_run("e5_as_deployed", "gen2")
    if r2 is None or head is None:
        return ["## 7. C: time-step check", "", "(no run)", ""], {}
    st = json.loads((r2 / "status.json").read_text())
    n = int(st["n_start_times"])
    res = {}
    for key, run in (("fine", r2), ("10 s", head)):
        raw = pd.read_parquet(run / "raw_as_deployed_dtt.parquet")
        ho = pd.read_parquet(run / "raw_handoff.parquet")
        res[key] = _three(raw[raw["phasing"] < n], ho[ho["phasing"] < n])
    rows = []
    for lab in res["fine"]:
        a, b = res["10 s"][lab], res["fine"][lab]
        rows.append({"policy": lab,
                     "degradation % 10 s": fmt(a[0], 2),
                     f"degradation % {st['time_step_s']:g} s": fmt(b[0], 2),
                     "avail. change % 10 s": fmt(a[1], 3),
                     f"avail. change % {st['time_step_s']:g} s": fmt(b[1], 3),
                     "switches/h 10 s": "" if a[2] is None else f"{a[2]:.1f}",
                     f"switches/h {st['time_step_s']:g} s": "" if b[2] is None else f"{b[2]:.1f}"})
    L = ["## 7. C: time-step check", "",
         f"`results/sensitivity/{r2.name}/` ({st['time_step_s']:g} s step, days 0-{n - 1}) vs "
         f"the headline `results/{head.name}/` restricted to the same {n} days.  With "
         f"{n} days the t-interval is wide (t95 for {n - 1} dof); compare the means.", "",
         p4b.md_table(rows), ""]
    return L, res


def section_d() -> tuple[list[str], dict]:
    r43 = _sens(lambda r: (r / "configs" / "e5_gen2_43only.yaml").exists())
    head = mn.headline_run("e5_as_deployed", "gen2")
    if r43 is None or head is None:
        return ["## 8. D: Gen2 attribution", "", "(no run)", ""], {}
    res = {}
    for key, run in (("43 + 53 deg (headline)", head), ("43 deg only", r43)):
        res[key] = _three(pd.read_parquet(run / "raw_as_deployed_dtt.parquet"),
                          pd.read_parquet(run / "raw_handoff.parquet"))
    n43 = json.loads((r43 / "status.json").read_text())["n_sats"]["STARLINK"]
    nh = json.loads((head / "status.json").read_text())["n_sats"]["STARLINK"]
    rows = []
    for lab in res["43 deg only"]:
        a, b = res["43 + 53 deg (headline)"][lab], res["43 deg only"][lab]
        rows.append({"policy": lab, f"degradation % 43+53 ({nh} sats)": fmt(a[0], 2),
                     f"degradation % 43 only ({n43} sats)": fmt(b[0], 2),
                     "avail. change % 43+53": fmt(a[1], 3), "avail. change % 43 only": fmt(b[1], 3)})
    L = ["## 8. D: Gen2 attribution (43 deg shell only)", "",
         f"`results/sensitivity/{r43.name}/` "
         "(`configs/experiments/sensitivity/e5_gen2_43only.yaml`: the post-grant 53 deg "
         "satellites are left out of both generations) vs the headline "
         f"`results/{head.name}/`; same 8 days.", "", p4b.md_table(rows), ""]
    return L, res


def section_sentences(lines: list[str], a: dict, b: dict, c: dict, d: dict) -> list[str]:
    """main.tex sentences whose verdict depends on A-D, with what A-D find."""
    def lines_of(pat):
        return ", ".join(map(str, p4b.phrase_lines(lines, pat))) or "-"

    def v(x):
        return x["v"] if x else "n/a"

    rows = []
    ua = a.get(("unconstrained", "rate-maximizing"))
    for label, key in (("K = 2", "K = 2"), ("dwell N = 30 s", "dwell N = 30 s"),
                       ("dwell N = 60 s", "dwell N = 60 s"), ("55 deg mask", "55 deg mask")):
        ev, rm = a.get((key, "event-avoiding")), a.get((key, "rate-maximizing"))
        rows.append({"task": "A", "constraint": label,
                     "event-avoiding degr / avail": f"{v(ev[0]) if ev else 'n/a'} / "
                                                    f"{v(ev[1]) if ev else 'n/a'}",
                     "rate-maximizing degr / avail": f"{v(rm[0]) if rm else 'n/a'} / "
                                                     f"{v(rm[1]) if rm else 'n/a'}"})
    L = ["## 9. `paper/main.tex` sentences whose verdict depends on A-D", "",
         "### 9.1 What A-D find (verdict: exceeds / within / indistinguishable)", "",
         p4b.md_table(rows), ""]
    if ua:
        L += [f"Unconstrained rate-maximizing (reference): degradation {v(ua[0])}, "
              f"availability {v(ua[1])}.", ""]
    if b.get("changed"):
        L += ["B (interruption) verdict changes: " + "; ".join(
            f"{r['policy']} {r['limit']} at tau {r['tau s']} s: {r['tau = 0']} -> {r['with tau']}"
            for r in b["changed"]), ""]
    for key, res, ref in (("C", c, "10 s"), ("D", d, "43 + 53 deg (headline)")):
        if not res:
            continue
        other = [k for k in res if k != ref][0]
        ch = [f"{lab} {m}: {res[ref][lab][i]['v']} -> {res[other][lab][i]['v']}"
              for lab in res[ref] for i, m in ((0, "degradation"), (1, "availability"))
              if res[ref][lab][i]["v"] != res[other][lab][i]["v"]]
        L += [f"{key} verdict changes ({ref} -> {other}): " + ("; ".join(ch) or "none"), ""]
    sent = [
        ("A, B, C", "Abstract: both policies meet availability under a 25 deg mask and "
         "unrestricted switching; rate-maximizing meets throughput; dwell or 55 deg mask "
         "breaks availability", r"compare meet the availability|remains the better one meets it|"
         r"dwell time on each satellite"),
        ("A, B, C", "Contributions: same claims", r"still the better one meets it;|minimum dwell "
         r"time or a high elevation mask"),
        ("A", "Rate-maximizing definition (candidate set)", r"rate-maximizing\} handoff picks"),
        ("D", "Gen2 classification by inclination and launch", r"assigned to Gen2 or Gen1|"
         r"misassigned"),
        ("B, C", "Results: availability restored under every unconstrained policy; event-"
         "avoiding exceeds, rate-maximizing meets ('none over')", r"every unconstrained|"
         r"rate-maximizing handoff meets it|none over"),
        ("D", "'does not hinge on the generation split' (Gen1-only, all Starlink)",
         r"hinge on the generation split|stays below it in every case"),
        ("A, B", "K = 2 'still within its limit'", r"still within its limit"),
        ("A, B", "Dwell 60 s 'breaks both limits'; 'tested only these two settings'",
         r"breaks both limits|tested only these two settings|at which compliance is lost"),
        ("A", "55 deg mask: event-avoiding exceeds both limits",
         r"event-avoiding handoff exceeds both limits"),
        ("B, C", "Limitation (h): instantaneous handoff at the 10 s step",
         r"Handoff is instantaneous|step of the simulation"),
        ("D", "Limitation (g): launch-date classification at 53 deg",
         r"launch-date classification"),
        ("A, B, C, D", "Conclusion: both policies restore availability; only rate-maximizing "
         "meets throughput", r"both handoff policies we compare|only the rate-maximizing policy|"
         r"still loses more than the limit"),
    ]
    find = _findings(a, b, c, d)
    L += ["### 9.2 Sentences", "",
          p4b.md_table([{"depends on": t, "sentence": s_, "main.tex lines": lines_of(pat),
                         "finding (from A-D)": find.get(s_[:14], "")}
                        for t, s_, pat in sent]), ""]
    return L


def _findings(a: dict, b: dict, c: dict, d: dict) -> dict:
    """One-line finding per sentence group, keyed by the sentence label's first 14 chars."""
    def v(key, rule, i):
        x = a.get((key, rule))
        return x[i]["v"] if x else "n/a"

    mt = b.get("max_tau")
    tau = {}
    if mt is not None:
        for _, r in mt.iterrows():
            tau[r.policy] = (r.max_tau_degradation_s, r.max_tau_availability_s)

    def t(pol, i):
        x = tau.get(pol, (np.nan, np.nan))[i]
        return "not met at 0" if x < 0 else f"{x:.2f} s"

    def same(res, ref):
        if not res:
            return "not run"
        other = [k for k in res if k != ref][0]
        ch = [lab for lab in res[ref] for i in (0, 1) if res[ref][lab][i]["v"] != res[other][lab][i]["v"]]
        return "no verdict changes" if not ch else "verdicts change for " + ", ".join(sorted(set(ch)))

    av = (f"availability met by both rules only for tau <= {t('full', 1)} (event-avoiding) / "
          f"{t('full_keep_serving', 1)} (rate-maximizing) per excess switch")
    rm = (f"rate-maximizing throughput: unconstrained {v('unconstrained', 'rate-maximizing', 0)}, "
          f"K=2 {v('K = 2', 'rate-maximizing', 0)}, N=30 s {v('dwell N = 30 s', 'rate-maximizing', 0)}, "
          f"N=60 s {v('dwell N = 60 s', 'rate-maximizing', 0)}, 55 deg {v('55 deg mask', 'rate-maximizing', 0)}; "
          f"met for tau <= {t('full_keep_serving', 0)}")
    dw = (f"availability under dwell: N=30 s {v('dwell N = 30 s', 'event-avoiding', 1)} / "
          f"{v('dwell N = 30 s', 'rate-maximizing', 1)}, N=60 s {v('dwell N = 60 s', 'event-avoiding', 1)} / "
          f"{v('dwell N = 60 s', 'rate-maximizing', 1)} (event-avoiding / rate-maximizing); 55 deg mask "
          f"{v('55 deg mask', 'event-avoiding', 1)} / {v('55 deg mask', 'rate-maximizing', 1)}")
    cs = "C (2 s step): " + same(c, "10 s")
    ds = "D (43 deg only): " + same(d, "43 + 53 deg (headline)")
    return {
        "Abstract: both": f"{av}; {rm}; {dw}; {cs}",
        "Contributions:": f"{rm}; {dw}",
        "Rate-maximizin": "definition only; A shows rate-maximizing also loses the throughput limit "
                        "under dwell >= 30 s and the 55 deg mask",
        "Gen2 classific": ds,
        "Results: avail": f"{av}; {rm}; {cs}",
        "'does not hing": ds,
        "K = 2 'still w": (f"K=2 availability: event-avoiding {v('K = 2', 'event-avoiding', 1)}, "
                         f"rate-maximizing {v('K = 2', 'rate-maximizing', 1)}; met for tau <= "
                         f"{t('full_k2', 1)} / {t('full_keep_serving_k2', 1)}"),
        "Dwell 60 s 'br": (f"{dw}; the sweep answers 'tested only these two settings' (see Sec. 2 "
                         "for the smallest N exceeding each limit)"),
        "55 deg mask: e": (f"55 deg: event-avoiding {v('55 deg mask', 'event-avoiding', 0)} / "
                         f"{v('55 deg mask', 'event-avoiding', 1)}, rate-maximizing "
                         f"{v('55 deg mask', 'rate-maximizing', 0)} / {v('55 deg mask', 'rate-maximizing', 1)} "
                         "(degradation / availability)"),
        "Limitation (h)": f"{av}; {cs}",
        "Limitation (g)": ds,
        "Conclusion: bo": f"{av}; {rm}; {ds}",
    }
