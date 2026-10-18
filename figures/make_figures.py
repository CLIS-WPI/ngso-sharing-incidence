"""Build every figure from files under ``results/`` only (AGENTS.md rule 4).

Figures are drawn exclusively from the ``summary_*.csv`` tables written by
``experiments/e1_event_rate.py`` and ``experiments/e2_diversity.py``; no physical
quantity is hard-coded here.  Each figure is also copied into the run directory it
came from, for provenance.

Run:  python figures/make_figures.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

VARIANT_LABEL = {0: "worst case (any visible)", 1: "co-located (serving)"}
THETA_COLORS = {1.0: "#1f77b4", 2.0: "#ff7f0e", 5.0: "#2ca02c"}


def newest_run(results_root: Path, prefix: str) -> Path | None:
    runs = sorted(p for p in results_root.glob(f"{prefix}_*") if p.is_dir())
    return runs[-1] if runs else None


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / f"summary_{name}.csv")


def fig_f0a(e1: Path) -> Path | None:
    try:
        df = _read(e1, "sub_latitude_density")
    except FileNotFoundError:
        return None
    systems = list(dict.fromkeys(df["system"]))
    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4), squeeze=False)
    for ax, sysname in zip(axes[0], systems):
        sub = df[df["system"] == sysname]
        for inc, g in sub.groupby("inclination_deg"):
            g = g.sort_values("lat_center_deg")
            ax.plot(g["lat_center_deg"], g["sim_density_per_deg"], lw=1.6,
                    label=f"sim {inc:g}°")
            ax.plot(g["lat_center_deg"], g["analytic_density_per_deg"], "--", lw=1.2,
                    label=f"analytic {inc:g}°")
        ax.set_title(f"{sysname} sub-satellite latitude density")
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("density [1/deg]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "F0a_sub_latitude_density.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f0b(e2: Path) -> Path | None:
    try:
        df = _read(e2, "shell_visible_analytic")
    except FileNotFoundError:
        return None
    shells = list(dict.fromkeys(df["shell"]))
    ncol = 3
    nrow = int(np.ceil(len(shells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, sh in zip(axes.ravel(), shells):
        g = df[df["shell"] == sh].sort_values("latitude_deg")
        ax.plot(g["latitude_deg"], g["sim_mean"], "-o", ms=3, label="simulation")
        ax.plot(g["latitude_deg"], g["analytic"], "--", label="analytic")
        ax.set_title(sh, fontsize=8)
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("mean visible")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(shells):]:
        ax.axis("off")
    fig.tight_layout()
    out = FIGURES_DIR / "F0b_visible_count.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f1(e1: Path) -> Path | None:
    try:
        df = _read(e1, "event_rate")
    except FileNotFoundError:
        return None
    pairs = list(dict.fromkeys(zip(df["system_a"], df["system_b"])))
    fig, axes = plt.subplots(2, len(pairs), figsize=(3.6 * len(pairs), 7), squeeze=False)
    for j, (a, b) in enumerate(pairs):
        for vi in (1, 0):  # co-located (main estimate) first, worst case below
            ax = axes[0 if vi == 1 else 1][j]
            sub = df[(df["system_a"] == a) & (df["system_b"] == b) & (df["variant"] == vi)]
            for th, g in sub.groupby("theta_deg"):
                g = g.sort_values("latitude_deg")
                m = g["event_frac_mean"].to_numpy()
                c = g["event_frac_ci95"].to_numpy()
                ax.plot(g["latitude_deg"], m, color=THETA_COLORS.get(th), label=f"θ={th:g}°")
                ax.fill_between(g["latitude_deg"], m - c, m + c,
                                color=THETA_COLORS.get(th), alpha=0.2)
            ax.set_title(f"{a} ← {b}\n{VARIANT_LABEL[vi]}", fontsize=8)
            ax.set_xlabel("latitude [deg]")
            ax.set_ylabel("event time-fraction")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("E1 in-line event time-fraction vs latitude (mean ± 95% CI over phasings); "
                 "co-located = main estimate, worst case = upper bound")
    fig.tight_layout()
    out = FIGURES_DIR / "F1_event_rate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f2(e2: Path) -> Path | None:
    try:
        df = _read(e2, "unresolvable")
    except FileNotFoundError:
        return None
    pairs = list(dict.fromkeys(zip(df["system_a"], df["system_b"])))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, b in pairs:
        for vi, ls in ((1, "-"), (0, "--")):  # co-located solid (main), worst case dashed
            sub = df[(df["system_a"] == a) & (df["system_b"] == b) & (df["variant"] == vi) &
                     (df["theta_deg"] == 5.0)].sort_values("latitude_deg")
            if sub.empty:
                continue
            label = f"{a}←{b} {'colo' if vi == 1 else 'worst'}"
            axes[0].plot(sub["latitude_deg"], sub["unres_frac_mean"], ls, label=label)
            axes[1].plot(sub["latitude_deg"], sub["unres_conditional"], ls, label=label)
    axes[0].set_title("Unresolvable time-fraction (θ=5°)")
    axes[1].set_title("P(unresolvable | event) (θ=5°)")
    for ax in axes:
        ax.set_xlabel("latitude [deg]")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "F2_unresolvable.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f3(e2: Path) -> Path | None:
    try:
        df = _read(e2, "durations")
    except FileNotFoundError:
        return None
    sub = df[(df["kind"] == "unres") & (df["theta_deg"] == 5.0) & (df["variant"] == 0)]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (a, b), g in sub.groupby(["system_a", "system_b"]):
        g = g.sort_values("latitude_deg")
        axes[0].plot(g["latitude_deg"], g["median_s"], "-o", ms=3, label=f"{a}←{b}")
        axes[1].plot(g["latitude_deg"], g["p90_s"], "-o", ms=3, label=f"{a}←{b}")
    axes[0].set_title("Median unresolvable-event duration (θ=5°, worst case upper bound)")
    axes[1].set_title("p90 unresolvable-event duration")
    for ax in axes:
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("duration [s]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "F3_event_durations.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f4(e2: Path) -> Path | None:
    """Service availability, diversity pool size and service-conditioned burden."""
    try:
        svc = _read(e2, "service_pool")
        un = _read(e2, "unresolvable")
    except FileNotFoundError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for sysname, g in svc.groupby("system"):
        g = g.sort_values("latitude_deg")
        axes[0].plot(g["latitude_deg"], g["served_frac_mean"], "-o", ms=3, label=sysname)
        axes[1].plot(g["latitude_deg"], g["pool_mean_mean"], "-o", ms=3, label=f"{sysname} mean")
        axes[1].plot(g["latitude_deg"], g["pool_p10_mean"], "--", lw=1.2,
                     label=f"{sysname} p10")
    pairs = list(dict.fromkeys(zip(un["system_a"], un["system_b"])))
    for a, b in pairs:
        sub = un[(un["system_a"] == a) & (un["system_b"] == b) & (un["variant"] == 1) &
                 (un["theta_deg"] == 2.0)].sort_values("latitude_deg")
        if sub.empty:
            continue
        axes[2].plot(sub["latitude_deg"], sub["unres_when_served"], "-o", ms=3,
                     label=f"{a}←{b}")
    axes[0].set_title("Victim service availability")
    axes[0].set_ylabel("fraction of time served")
    axes[1].set_title("Diversity pool size (mean and p10)")
    axes[1].set_ylabel("visible victim satellites")
    axes[2].set_title("P(unresolvable | event, victim served), θ=2°, co-located")
    axes[2].set_ylabel("fraction of events")
    for ax in axes:
        ax.set_xlabel("latitude [deg]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "F4_service_pool.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f5(e3: Path) -> Path | None:
    """Capacity-limited unresolvable cell-time fraction vs latitude and n_beam."""
    try:
        df = _read(e3, "capacity_diversity")
    except FileNotFoundError:
        return None
    pairs = list(dict.fromkeys(zip(df["system_a"], df["system_b"])))
    if not pairs:
        return None
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.2 * len(pairs), 8), squeeze=False)
    for j, (a, b) in enumerate(pairs):
        sub = df[(df["system_a"] == a) & (df["system_b"] == b)]
        for nb, g in sub.groupby("n_beam"):
            g = g.sort_values("latitude_deg")
            axes[0][j].plot(g["latitude_deg"], g["unres_cell_frac_mean"], "-o", ms=3,
                            label=f"N_beam={int(nb)}")
            axes[1][j].plot(g["latitude_deg"], g["served_frac_mean"], "-o", ms=3,
                            label=f"N_beam={int(nb)}")
        axes[0][j].set_title(f"{a} ← {b}: unresolvable cell-time", fontsize=8)
        axes[1][j].set_title(f"{a} ← {b}: cells served (capacity)", fontsize=8)
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xlabel("latitude [deg]")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("E3 capacity-limited diversity (θ=2°, co-located, co-frequency user downlink)")
    fig.tight_layout()
    out = FIGURES_DIR / "F5_capacity_diversity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    e1 = newest_run(args.results_root, "e1_event_rate")
    e2 = newest_run(args.results_root, "e2_diversity")
    e3 = newest_run(args.results_root, "e3_capacity_diversity")
    if e1 is None and e2 is None and e3 is None:
        print(f"[figures] no e1/e2/e3 run directories under {args.results_root}.")
        return 0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    produced: list[tuple[Path, Path]] = []
    if e1 is not None:
        for fn in (fig_f0a, fig_f1):
            out = fn(e1)
            if out:
                produced.append((out, e1))
    if e2 is not None:
        for fn in (fig_f0b, fig_f2, fig_f3, fig_f4):
            out = fn(e2)
            if out:
                produced.append((out, e2))
    if e3 is not None:
        out = fig_f5(e3)
        if out:
            produced.append((out, e3))

    for out, run_dir in produced:
        shutil.copy2(out, run_dir / out.name)
        print(f"[figures] wrote {out} (copied to {run_dir.name})")
    if not produced:
        print("[figures] no plottable summary tables found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
