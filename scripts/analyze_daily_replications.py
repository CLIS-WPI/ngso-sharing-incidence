"""Dependence of the daily as-deployed replications (Phase 4b).

E5/E6 replicate over consecutive non-overlapping 24 h windows from one TLE
snapshot.  For each headline metric this writes the per-day values (worst
qualified station per day, the same aggregation as ``paper/numbers.tex``), their
mean and 95% t-interval, min / max, lag-1 autocorrelation, an OLS trend across
days (slope, p-value) and Kendall's tau, and the CI basis that follows:

- ``t-interval`` when neither |lag-1 r| >= 2/sqrt(n) nor a trend at p < 0.05;
- ``daily range`` otherwise.

With n = 8 (or 4) days these tests have little power; passing them does not
establish independence.

Usage (in the container):
    python scripts/analyze_daily_replications.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402

from ngsi.io import new_run_dir  # noqa: E402
from ngsi.runner import t95  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POLICIES = {"full": "event-avoiding", "full_keep_serving": "rate-maximizing",
            "legacy": "legacy", "full_k2": "event-avoiding K=2",
            "full_k3": "event-avoiding K=3", "full_n10": "event-avoiding N=10 s",
            "full_n60": "event-avoiding N=60 s"}


def _head(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["system_a"] == mn.VICTIM) & (df["variant"] == 0)
              & (df["diameter_m"] == mn.HEAD_DIAM)]


def worst_per_day(df: pd.DataFrame, col: str) -> pd.Series:
    q = _head(df)
    q = q[q["avail0_frac"] >= mn.QUAL]
    return q.groupby("phasing")[col].max()


def lag1(x: np.ndarray) -> float:
    d = x - x.mean()
    den = float(np.sum(d * d))
    return float(np.sum(d[:-1] * d[1:]) / den) if den > 0 else float("nan")


def describe(values: pd.Series) -> dict:
    x = values.sort_index().to_numpy(dtype=np.float64)
    day = values.sort_index().index.to_numpy(dtype=np.float64)
    n = x.size
    out = {"n_days": n, "mean": x.mean(), "ci95": t95(n) * x.std(ddof=1) / np.sqrt(n),
           "min": x.min(), "max": x.max(), "lag1_r": lag1(x),
           "lag1_crit": 2.0 / np.sqrt(n)}
    if np.ptp(x) > 0:
        lr = stats.linregress(day, x)
        kt = stats.kendalltau(day, x)
        out.update(trend_slope_per_day=lr.slope, trend_p=lr.pvalue,
                   kendall_tau=kt.statistic, kendall_p=kt.pvalue)
    else:
        out.update(trend_slope_per_day=0.0, trend_p=1.0, kendall_tau=np.nan, kendall_p=1.0)
    dep = (abs(out["lag1_r"]) >= out["lag1_crit"]) or out["trend_p"] < 0.05
    out["ci_basis"] = ("constant" if np.ptp(x) == 0 else
                       "daily range" if dep else "t-interval")
    out["values"] = " ".join(f"{v:.6g}" for v in x)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", type=Path, default=REPO / "results")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="where the run directory goes (default: --results-root)")
    args = ap.parse_args()
    mn.RESULTS = args.results_root.resolve()

    series: list[tuple[dict, pd.Series]] = []
    used: dict[str, str] = {}
    for gen in ("gen2", "gen1", "all"):
        run = mn.headline_run("e5_as_deployed", gen)
        if run is None:
            continue
        used[f"e5 {gen}"] = run.name
        raw = pd.read_parquet(run / "raw_as_deployed_dtt.parquet")
        h = _head(raw)
        s = pd.read_csv(run / "summary_dtt.csv")
        s = _head(s)
        pk = float(s.loc[s["event_frac_mean"].idxmax(), "latitude_deg"])
        base = {"experiment": "e5", "generation": gen, "run": run.name}
        series.append(({**base, "metric": f"event fraction at peak latitude {pk:g} deg"},
                       h[h["latitude_deg"] == pk].groupby("phasing")["event_frac"].mean()))
        series.append(({**base, "metric": "degradation, no handoff"},
                       worst_per_day(raw, "degradation_nodiv")))
        series.append(({**base, "metric": "availability change, no handoff"},
                       worst_per_day(raw, "unavail_incr_frac")))
        for pol, lab in POLICIES.items():
            pr = mn.run_with_policy("e5_as_deployed", gen, pol)
            if pr is None:
                continue
            used[f"e5 {gen} {pol}"] = pr.name
            ho = pd.read_parquet(pr / "raw_handoff.parquet")
            ho = ho[ho["policy"] == pol]
            b = {**base, "run": pr.name}
            series.append(({**b, "metric": f"degradation, {lab}"},
                           worst_per_day(ho, "degradation_div")))
            series.append(({**b, "metric": f"availability change, {lab}"},
                           worst_per_day(ho, "unavail_incr_div_frac")))

    e6 = mn.newest_where("e6_eirp_sweep",
                         starlink_generation=mn.primary_generation("e6_eirp_sweep"))
    if e6 is not None:
        used["e6"] = e6.name
        r6 = pd.read_parquet(e6 / "raw_eirp_sweep.parquet")
        r6 = r6[(r6["victim"] == mn.VICTIM) & (r6["variant"] == 0)
                & (r6["diameter_m"] == mn.HEAD_DIAM)]
        for case, g in r6.groupby("case"):
            g = g.set_index("phasing")
            b = {"experiment": "e6", "generation": "gen2", "run": e6.name, "case": case}
            for col, lab in (("degr_max_qual", "degradation, no handoff"),
                             ("degr_div_max_qual", "degradation, event-avoiding"),
                             ("unavail_max_qual", "availability change, no handoff"),
                             ("unavail_div_max_qual", "availability change, event-avoiding")):
                series.append(({**b, "metric": lab}, g[col]))

    run_dir = new_run_dir(args.out_root or args.results_root, "e13_daily_replications", 0, repo_root=REPO)
    daily, rows = [], []
    for meta, v in series:
        v = v.dropna()
        if v.empty:
            continue
        rows.append({**meta, **describe(v)})
        for d, x in v.sort_index().items():
            daily.append({**meta, "day": int(d), "value": float(x)})
    pd.DataFrame(daily).to_csv(run_dir / "summary_daily.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "summary_daily_stats.csv", index=False)
    (run_dir / "status.json").write_text(json.dumps(
        {"experiment": "e13_daily_replications", "status": "ok", "runs": used}, indent=2))
    print(f"[e13] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
