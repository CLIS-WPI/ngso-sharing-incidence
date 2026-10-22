"""Handoff interruption cost, post-processed from an E5 run (Phase 4d).

The simulation treats a handoff as instantaneous.  Here every switch beyond the
interference-free baseline's own switches (highest-elevation satellite, which a
terminal makes anyway) costs an interruption of ``tau`` seconds with rate 0,
counted as unavailable.  Per station and day, with ``f = excess switches per
hour * tau / 3600`` (the interrupted fraction of time):

    degradation_tau  = 1 - (1 - degradation_div) * (1 - f)
    unavail_incr_tau = unavail_incr_div_frac + f

"Excess switches" is a count difference per station-day,
``max(0, switches_all_per_hour - base_switches_per_hour)``, not a match of switch
instants.  The first line assumes the rate during an interruption would have
equalled the policy's time-averaged rate (not the instantaneous rate at the
switch); the second counts every interrupted second as unavailable, including
seconds already unavailable or unserved (no overlap removal, an upper bound).
Both are approximations stated in the report.  Aggregation as in the paper: worst
qualified station per day, mean and 95% t-interval over days.  For each policy the
largest tau at which a limit is still met (whole interval below it) is found by
bisection on [0, ``--tau-max``] s.

Usage (in the container):
    python scripts/interruption_cost.py [--taus 0.1,0.5,1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402

from ngsi.constants import THROUGHPUT_DEGRADE_MAX, UNAVAILABILITY_INCREASE_MAX  # noqa: E402
from ngsi.io import new_run_dir  # noqa: E402
from ngsi.runner import t95  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LIMITS = {"degradation": THROUGHPUT_DEGRADE_MAX, "availability": UNAVAILABILITY_INCREASE_MAX}


def worst(q: pd.DataFrame, col: str) -> tuple[float, float, float, int, int]:
    w = q.groupby("phasing")[col].max().dropna().to_numpy(np.float64)
    ci = float(t95(w.size) * w.std(ddof=1) / np.sqrt(w.size)) if w.size >= 2 else 0.0
    return float(w.mean()), ci, float(q[col].max()), int(w.size), int(len(q))


def with_tau(q: pd.DataFrame, tau: float) -> pd.DataFrame:
    excess = (q["switches_all_per_hour"] - q["base_switches_per_hour"]).clip(lower=0.0)
    f = excess * tau / 3600.0
    return q.assign(degradation=1.0 - (1.0 - q["degradation_div"]) * (1.0 - f),
                    availability=q["unavail_incr_div_frac"] + f)


def verdict(m: float, ci: float, lim: float) -> str:
    return "exceeds" if m - ci > lim else "within" if m + ci < lim else "indistinguishable"


def max_tau(q: pd.DataFrame, key: str, tau_max: float) -> float:
    """Largest tau (s) with the whole interval below the limit; -1 if not met at 0."""
    lim = LIMITS[key]

    def ok(t):
        m, ci, *_ = worst(with_tau(q, t), key)
        return m + ci < lim

    if not ok(0.0):
        return -1.0
    if ok(tau_max):
        return float("inf")
    lo, hi = 0.0, tau_max
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", type=Path, default=REPO / "results")
    ap.add_argument("--taus", default="0.1,0.5,1.0")
    ap.add_argument("--tau-max", type=float, default=60.0)
    args = ap.parse_args()
    mn.RESULTS = args.results_root.resolve()
    taus = [0.0] + [float(x) for x in args.taus.split(",")]

    run = mn.headline_run("e5_as_deployed", "gen2")
    if run is None:
        raise SystemExit("no Gen2 E5 headline run")
    ho = pd.read_parquet(run / "raw_handoff.parquet")
    if "switches_all_per_hour" not in ho.columns:
        raise SystemExit(f"{run.name} has no per-policy switch counts (re-run E5)")
    rows, limits = [], []
    for pol, g in ho.groupby("policy", sort=False):
        q = mn._head(g)
        q = q[q["avail0_frac"] >= mn.QUAL]
        exc = (q["switches_all_per_hour"] - q["base_switches_per_hour"]).clip(lower=0.0)
        for tau in taus:
            t = with_tau(q, tau)
            r = {"policy": pol, "tau_s": tau,
                 "switches_per_hour_mean": float(q["switches_all_per_hour"].mean()),
                 "base_switches_per_hour_mean": float(q["base_switches_per_hour"].mean()),
                 "excess_switches_per_hour_mean": float(exc.mean()),
                 "excess_switches_per_hour_max": float(exc.max())}
            for key, lim in LIMITS.items():
                m, ci, mx, n_days, n_q = worst(t, key)
                r.update({f"{key}_mean": m, f"{key}_ci95": ci, f"{key}_max_obs": mx,
                          f"{key}_n_over": int((t[key] > lim).sum()),
                          f"{key}_verdict": verdict(m, ci, lim)})
            r.update(n_days=n_days, n_station_days=n_q)
            rows.append(r)
        limits.append({"policy": pol, **{f"max_tau_{k}_s": max_tau(q, k, args.tau_max)
                                         for k in LIMITS}})
    out = new_run_dir(args.results_root, "e14_interruption_cost", 0, repo_root=REPO)
    pd.DataFrame(rows).to_csv(out / "summary_interruption.csv", index=False)
    pd.DataFrame(limits).to_csv(out / "summary_max_tau.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "experiment": "e14_interruption_cost", "status": "ok", "source_run": run.name,
        "taus_s": taus, "tau_max_s": args.tau_max}, indent=2))
    print(f"[e14] run complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
