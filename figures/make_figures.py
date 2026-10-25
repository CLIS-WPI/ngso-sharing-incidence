"""Build the paper figures from files under ``results/`` only.

The paper includes two figures, both drawn from the ``summary_*.csv``
tables of the headline E4 (as-filed) and E5 (as-deployed) runs; no physical
quantity is hard-coded here:

- Fig. 1 ``output/figures/B1_events_both.png``  (:func:`fig_b1_events`)
- Fig. 2 ``output/figures/B2_limits.png``       (:func:`fig_b2_limits`)

Run:  python figures/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PAPER_FIG_DIR = REPO_ROOT / "output" / "figures"
VICTIM = "ONEWEB"
HEAD_DIAM = 0.45


def newest_run(results_root: Path, prefix: str) -> Path | None:
    runs = sorted(p for p in results_root.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def newest_primary(results_root: Path, prefix: str) -> Path | None:
    """Newest run whose status.json shows primary (per-beam) steering.

    The constant-EIRP sensitivity runs override ``steering``; the primary
    conclusions use the PFD-constant run, so paper figures must read that one.
    """
    import json

    for run_dir in reversed(sorted(results_root.glob(f"{prefix}_*"))):
        status = run_dir / "status.json"
        if not run_dir.is_dir() or not status.exists():
            continue
        try:
            data = json.loads(status.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (data.get("steering") in (None, "config", "null")
                and data.get("starlink_generation") == primary_generation(prefix)):
            return run_dir
    return newest_run(results_root, prefix)


def primary_generation(prefix: str) -> str | None:
    """Headline Starlink generation: ``starlink_generation`` in the current config.

    Runs made with another generation (the Gen1-only / all-Starlink sensitivity
    runs) or before Phase 4 (no key) are then never picked as the headline.
    """
    import yaml

    cfg = REPO_ROOT / "configs" / "experiments" / f"{prefix}.yaml"
    if not cfg.exists():
        return None
    return yaml.safe_load(cfg.read_text()).get("params", {}).get("starlink_generation")


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / f"summary_{name}.csv")


def fig_b1_events(e4: Path, e5: Path) -> Path | None:
    """Paper Fig. 1: realistic-variant Delta T/T event fraction, both directions."""
    try:
        d4 = _read(e4, "dtt")
        d5 = _read(e5, "dtt")
    except FileNotFoundError:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for d, ls, tag in ((d4, "-", "as-filed"), (d5, "--", "as-deployed")):
        for sysn, color, lab in ((VICTIM, "#1f77b4", "OneWeb victim"),
                                 ("STARLINK", "#d62728", "Starlink victim")):
            s = d[(d["system_a"] == sysn) & (d["variant"] == 0) &
                  (d["diameter_m"] == HEAD_DIAM)].sort_values("latitude_deg")
            if s.empty:
                continue
            ax.plot(s["latitude_deg"], s["event_frac_mean"] * 100.0, ls, color=color,
                    lw=1.5, label=f"{lab}, {tag}")
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel(r"$\Delta T/T>6\%$ time fraction [%]")
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.8, loc="upper left", ncol=1)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "B1_events_both.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_b2_limits(e5: Path) -> Path | None:
    """Paper Fig. 2: degradation and availability change vs latitude, no handoff and
    the event-avoiding and rate-aware handoff rules, 95% bands across days."""
    try:
        d = _read(e5, "dtt")
        h = _read(e5, "handoff")
    except FileNotFoundError:
        return None

    def _sel(df):
        return df[(df["system_a"] == VICTIM) & (df["variant"] == 0) &
                  (df["diameter_m"] == HEAD_DIAM)].sort_values("latitude_deg")

    s = _sel(d)
    if s.empty:
        return None
    curves = [  # (frame, degradation column, availability column, label, style, color)
        (s, "degradation_nodiv", "unavail_incr_frac", "No handoff", "-", "#1f77b4"),
    ]
    for pol, label, ls, color in (("full", "Event-avoiding", "--", "#2ca02c"),
                                  ("full_keep_serving", "Rate-aware", "-.", "#d62728")):
        hp = _sel(h[h["policy"] == pol])
        if not hp.empty:
            curves.append((hp, "degradation_div", "unavail_incr_div_frac", label, ls, color))
    # 2.75 in tall: room for the two-line x labels without shrinking the panels.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75), sharex=True)

    for df, cd, cu, label, ls, color in curves:
        lat = df["latitude_deg"]
        for ax, col in ((ax1, cd), (ax2, cu)):
            # 95% t-interval across start times (the summary's ``_ci95`` column).
            m, hw = df[col + "_mean"] * 100.0, df[col + "_ci95"] * 100.0
            ax.plot(lat, m, ls, color=color, lw=1.6, label=label)
            ax.fill_between(lat, m - hw, m + hw, color=color, alpha=0.25, lw=0)
    ax1.axhline(3.0, color="k", ls=":", lw=1.1)
    # Right end of the limit line (curves are low there); legend lower left, below them.
    ax1.text(64, 3.10, "3% limit", fontsize=6, ha="right")
    # Panel labels as a second x-label line: centred, same size as the axis labels.
    ax1.set_xlabel("latitude [deg]\n(a) Throughput degradation")
    ax1.set_ylabel("throughput degradation [%]")
    ax1.set_xlim(0, 65)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=6, loc="lower left")
    ax2.axhline(0.4, color="k", ls=":", lw=1.1)
    ax2.text(1, 0.45, "0.4% limit", fontsize=6)
    ax2.set_xlabel("latitude [deg]\n(b) Availability change")
    ax2.set_ylabel("availability change [%]")
    ax2.set_ylim(bottom=-0.05)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "B2_limits.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    e4 = newest_primary(args.results_root, "e4_dtt")
    e5 = newest_primary(args.results_root, "e5_as_deployed")
    if e4 is None or e5 is None:
        print(f"[figures] no e4_dtt / e5_as_deployed run directories under {args.results_root}.")
        return 1
    for out in (fig_b1_events(e4, e5), fig_b2_limits(e5)):
        if out is not None:
            print(f"[figures] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
