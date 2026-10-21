"""Phase 4c report: current-scenario re-runs (E1 screen, E7/E8, E9) and the dwell sweep.

Writes ``results/PHASE4C_REPORT.md`` from

- ``paper/numbers.tex`` now and at the Phase 4b commit (``--base``, git), to list
  every macro whose value or source changed and the ``paper/main.tex`` lines that
  use it;
- the newest E5 Gen2 run (dwell-time sweep, both handoff rules);
- the newest E7 / E8 / E9 runs (scenario of each);
- ``paper/main.tex``, for the sentences whose wording depends on the scenario.

Every number is read from those files (AGENTS.md rule 1).

Usage (in the container):
    python scripts/make_phase4c_report.py [--base 22dc0ce]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402
import make_phase4b_report as p4b  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "PHASE4C_REPORT.md"
PAPER = REPO / "paper" / "main.tex"
DWELL = (10, 20, 30, 40, 60)
RULES = (("event-avoiding", "full"), ("rate-maximizing", "full_keep_serving"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def snippet(lines: list[str], n: int, width: int = 110) -> str:
    t = lines[n - 1].strip().replace("|", "\\|")
    return t if len(t) <= width else t[:width] + " ..."


def status(run: Path) -> dict:
    for name in ("status.json", "diagnosis.json"):
        f = run / name
        if f.exists():
            return json.loads(f.read_text())
    return {}


def section_runs() -> list[str]:
    rows = []
    for prefix in ("e5_as_deployed", "e7_analytic", "e8_analytic_diagnosis",
                   "e9_min_elevation_diagnosis"):
        run = (mn.headline_run(prefix, "gen2") if prefix == "e5_as_deployed"
               else mn.newest(prefix))
        if run is None:
            continue
        st = status(run)
        meta = json.loads((run / "run_meta.json").read_text())
        rows.append({"run": f"`{run.name}`", "Starlink": st.get("starlink_generation", ""),
                     "start spacing h": (f"{st['start_offset_s'] / 3600:g}"
                                         if "start_offset_s" in st else ""),
                     "start times": st.get("n_start_times", ""),
                     "git": meta.get("git_hash", "")[:7],
                     "dirty": (meta.get("git_worktree") or {}).get("dirty", ""),
                     "modified files": ", ".join((meta.get("git_worktree") or {})
                                                 .get("modified_files") or []) or "-",
                     "dtype": meta.get("compute_dtype", "")})
    return ["## 1. New runs", "", p4b.md_table(rows), "",
            "A modified file listed for a run was not imported by that run's code path "
            "(`scripts/make_numbers.py` is read only when macros are written).  Sensitivity "
            "runs (C, D) are under `results/sensitivity/` (Secs. 7-8).", ""]


def section_dwell() -> tuple[list[str], dict]:
    from ngsi.constants import THROUGHPUT_DEGRADE_MAX as DL
    from ngsi.constants import UNAVAILABILITY_INCREASE_MAX as UL

    rows, first = [], {}
    for label, base in RULES:
        first[label] = {"degradation": None, "availability": None}
        for n_s in DWELL:
            pol = f"{base}_n{n_s}"
            run = mn.run_with_policy("e5_as_deployed", "gen2", pol)
            if run is None:
                continue
            ho = pd.read_parquet(run / "raw_handoff.parquet")
            ho = ho[ho["policy"] == pol]
            q = mn._head(ho)
            q = q[q["avail0_frac"] >= mn.QUAL]
            r = {"rule": label, "N s": n_s}
            for name, col, lim in (("degradation", "degradation_div", DL),
                                   ("availability", "unavail_incr_div_frac", UL)):
                m, ci = mn._rep_worst(ho, col, mn.VICTIM, 0)
                n_over, n_q, _ = mn._over(ho, col, lim)
                v = p4b.verdict(100 * m, 100 * ci, 100 * lim)
                nd = 2 if name == "degradation" else 3
                r[f"{name} %"] = f"{100 * m:.{nd}f} +/- {100 * ci:.{nd}f}"
                r[f"{name} verdict"] = v
                r[f"{name} over"] = f"{n_over}/{n_q}"
                if v == "exceeds" and first[label][name] is None:
                    first[label][name] = n_s
            r["switches/h"] = f"{q['switches_per_hour'].mean():.1f}"
            r["run"] = f"`{run.name}`"
            rows.append(r)
    L = ["## 2. Dwell-time sweep (Gen2 as deployed, 8 non-overlapping days)", "",
         "OneWeb victim, realistic variant, 0.45 m; worst qualified station per day, mean "
         "+/- 95% t-interval over days; 'over' = station-days above the limit.  Verdict: "
         "'exceeds' / 'within' only when the whole interval is above / below the limit.  "
         "N = 10 s equals the simulation step.", "", p4b.md_table(rows), "",
         "Smallest tested N at which each limit is exceeded:", ""]
    for label, f in first.items():
        L.append(f"- {label}: degradation {f['degradation'] or 'not exceeded up to 60'} s, "
                 f"availability change {f['availability'] or 'not exceeded up to 60'} s")
    return L + [""], first


def section_changed(old: dict, oldsrc: dict, new: dict, newsrc: dict,
                    lines: list[str]) -> list[str]:
    used, unused = [], []
    for k in sorted(set(old) | set(new)):
        if old.get(k) == new.get(k) and oldsrc.get(k) == newsrc.get(k):
            continue
        ln = p4b.macro_lines(lines, k)
        r = {"macro": k, "before": old.get(k, "(new)"), "now": new.get(k, "(removed)"),
             "source before -> now": f"{oldsrc.get(k, '-')} -> {newsrc.get(k, '-')}"}
        if ln:
            used.append({**r, "main.tex lines": ", ".join(map(str, ln))})
        else:
            unused.append(r)
    L = ["## 3. Macros whose value or source changed", "",
         f"### 3.1 Used in `paper/main.tex` ({len(used)})", "", p4b.md_table(used), "",
         f"### 3.2 Not used in the text ({len(unused)}; new macros marked '(new)')", ""]
    L += [p4b.md_table(unused), ""]
    # The sentences behind 3.1, one row per line.
    seen = sorted({int(x) for r in used for x in r["main.tex lines"].split(", ")})
    rows = [{"line": n, "macros changed": ", ".join(r["macro"] for r in used
                                                   if str(n) in r["main.tex lines"].split(", ")),
             "text": snippet(lines, n)} for n in seen]
    L += ["### 3.3 `paper/main.tex` lines that print a changed value", "",
          p4b.md_table(rows), ""]
    return L


def section_wording(lines: list[str], newsrc: dict, first: dict) -> list[str]:
    e7, e9 = mn.newest("e7_analytic"), mn.newest("e9_min_elevation_diagnosis")
    st7 = status(e7) if e7 else {}
    st9 = status(e9) if e9 else {}
    e5 = mn.headline_run("e5_as_deployed", "gen2")
    snap5 = json.loads((e5 / "tle_meta.json").read_text())["common_epoch_jd"] if e5 else None
    snap9 = st9.get("common_epoch_jd")
    theta_src = newsrc.get("ThetaColoPeakPct", "")
    checks = [
        ("E7/E8 validated on Gen1 Walker shells",
         r"Gen1 Walker geometry on which it was validated",
         f"E7 run `{e7.name if e7 else '-'}` uses Starlink {st7.get('starlink_generation')} "
         f"shells, compared with E4 run `{st7.get('e4_run')}`"),
        ("Geometric screen 'on the filed Walker shells of Starlink Gen1'",
         r"filed Walker shells of Starlink Gen1",
         f"ThetaColoPeakPct now from `{theta_src}` (as-deployed Gen2, 8 days)"),
        ("Dwell: 'We tested only these two settings ... remains open'",
         r"tested only these two settings|at which compliance is lost",
         "dwell sweep N = " + ", ".join(map(str, DWELL)) + " s now run; first exceedance "
         + "; ".join(f"{k}: degradation {v['degradation'] or 'none'}, availability "
                     f"{v['availability'] or 'none'}" for k, v in first.items())),
        ("Limitation (g): 'the analytic model and the geometric-screen comparison use the "
         "Gen1 Walker shells'",
         r"geometric-screen comparison use the Gen1 Walker",
         "removable" if st7.get("starlink_generation") == "gen2"
         and theta_src.startswith("e5_as_deployed") else "keep"),
        ("Limitation (g): '55 deg visibility diagnosis uses an earlier snapshot'",
         r"earlier snapshot",
         ("removable: E9 epoch JD {:.5f} equals the E5 headline snapshot epoch".format(snap9)
          if snap9 is not None and snap5 is not None and abs(snap9 - snap5) < 1e-9
          else f"E9 snapshot epoch {snap9} vs E5 {snap5}")
         + f"; E9 Starlink {st9.get('starlink_generation')}, "
           f"{st9.get('n_start_times')} start times "
           f"{(st9.get('start_offset_s') or 0) / 3600:g} h apart"),
    ]
    rows = []
    for label, pat, status_ in checks:
        ln = p4b.phrase_lines(lines, pat)
        rows.append({"wording": label, "main.tex lines": ", ".join(map(str, ln)) or "-",
                     "text": " / ".join(snippet(lines, n, 90) for n in ln),
                     "now": status_})
    return ["## 4. Sentences whose wording depends on the scenario", "", p4b.md_table(rows), ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="22dc0ce", help="git ref of the previous numbers.tex")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        base_tex = Path(td) / "numbers.tex"
        base_tex.write_text(git("show", f"{args.base}:paper/numbers.tex"), encoding="utf-8")
        old, oldsrc = p4b.read_macros(base_tex)
    new, newsrc = p4b.read_macros(p4b.NEW_TEX)
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    texts = {PAPER: lines}
    head = git("rev-parse", "--short", "HEAD").strip()
    L = ["# Phase 4c/4d report: current-scenario re-runs, dwell sweep, second-review checks A-D", "",
         f"Generated by `scripts/make_phase4c_report.py` at `{head}`, compared with "
         f"`paper/numbers.tex` at `{args.base}` (Phase 4b).  Do not edit by hand.", "",
         "## 0. Scope", "",
         "- Geometric 2 deg screen (`ThetaColo*`, `ThetaWorstPeakPct`): now read from the "
         "as-deployed Gen2 E5 run (8 non-overlapping days) instead of E1 (Gen1 Walker).",
         "- E7 (analytic model) and E8 (its diagnosis) on the FCC 22-91 Gen2 Walker shells, "
         "compared with the as-filed Gen2 E4 run.",
         "- E9 (OneWeb 55 deg mask diagnosis) on the 2026-10-20 snapshot, Gen2 Starlink, "
         "8 days 24 h apart.",
         "- E5 Gen2 re-run with the dwell-time sweep added (N = 20/30/40 s for event-"
         "avoiding; N = 10..60 s for rate-maximizing).",
         "- Fig. 2: no handoff, event-avoiding and rate-maximizing curves; right panel from "
         "-0.05.  The Fig. 2 caption is the only text change in `paper/main.tex`.", ""]
    L += section_runs()
    dl, first = section_dwell()
    L += dl
    L += section_changed(old, oldsrc, new, newsrc, lines)
    L += section_wording(lines, newsrc, first)
    v, _ = p4b.section_verdicts(old, new, oldsrc, newsrc, texts)
    L += [x.replace("overlapping vs non-overlapping windows", "Phase 4b vs now")
           .replace("| overlapping |", "| Phase 4b |").replace("| non-overlapping |", "| now |")
          for x in v]
    L += [x.replace("(overlapping)", "(Phase 4b)").replace("(non-overlapping)", "(now)")
          for x in p4b.section_claims(old, new, texts)]
    import phase4d_sections as p4d

    la, ra = p4d.section_a()
    lb, rb = p4d.section_b()
    lc, rc = p4d.section_c()
    ld, rd = p4d.section_d()
    L += la + lb + lc + ld + p4d.section_sentences(lines, ra, rb, rc, rd)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
