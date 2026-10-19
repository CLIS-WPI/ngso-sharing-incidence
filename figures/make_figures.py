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
        if data.get("steering") in (None, "config", "null"):
            return run_dir
    return newest_run(results_root, prefix)


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
    """Capacity-limited unresolvable fraction vs latitude and n_beam (both denominators)."""
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
            axes[1][j].plot(g["latitude_deg"], g["unres_per_served_mean"], "-o", ms=3,
                            label=f"N_beam={int(nb)}")
        axes[0][j].set_title(f"{a} ← {b}: unresolvable cell-time\n(all cell-time)", fontsize=8)
        axes[1][j].set_title(f"{a} ← {b}: unresolvable per served cell-time", fontsize=8)
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xlabel("latitude [deg]")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("E3 capacity-limited diversity (ΔT/T > 6%, co-located, co-frequency user downlink); "
                 "absolute rises with N_beam via service coverage, conditional falls")
    fig.tight_layout()
    out = FIGURES_DIR / "F5_capacity_diversity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f6(e4: Path) -> Path | None:
    """47 CFR 25.261(d) Delta T/T event fraction and throughput degradation vs latitude."""
    try:
        df = _read(e4, "dtt")
    except FileNotFoundError:
        return None
    diam = sorted(df["diameter_m"].unique())[len(df["diameter_m"].unique()) // 2]
    pairs = list(dict.fromkeys(zip(df["system_a"], df["system_b"])))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, b in pairs:
        for vi, ls in ((1, "-"), (0, "--")):
            sub = df[(df["system_a"] == a) & (df["system_b"] == b) & (df["variant"] == vi) &
                     (df["diameter_m"] == diam)].sort_values("latitude_deg")
            if sub.empty:
                continue
            axes[0].plot(sub["latitude_deg"], sub["event_frac_mean"], ls,
                         label=f"{a}←{b} {'colo' if vi == 1 else 'worst'}")
            axes[1].plot(sub["latitude_deg"], sub["unres_frac_mean"], ls,
                         label=f"{a}←{b} {'colo' if vi == 1 else 'worst'}")
    axes[0].set_title(f"ΔT/T > 6% event time-fraction (diameter={diam:g} m)")
    axes[1].set_title("Unresolvable time-fraction (unlimited diversity)")
    for ax in axes:
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("fraction of time")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("E4 47 CFR 25.261(d) power criterion, co-frequency user downlink (mean ± 95% CI)")
    fig.tight_layout()
    out = FIGURES_DIR / "F6_dtt.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f7(e5: Path) -> Path | None:
    """E5 as-deployed event time-fraction vs latitude (real TLE snapshot)."""
    try:
        df = _read(e5, "event_rate")
    except FileNotFoundError:
        return None
    pairs = list(dict.fromkeys(zip(df["system_a"], df["system_b"])))
    if not pairs:
        return None
    fig, axes = plt.subplots(2, len(pairs), figsize=(3.6 * len(pairs), 7), squeeze=False)
    for j, (a, b) in enumerate(pairs):
        for vi in (1, 0):
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
    fig.suptitle("E5 as-deployed (CelesTrak TLE) in-line event time-fraction vs latitude "
                 "(mean ± 95% CI over start times)")
    fig.tight_layout()
    out = FIGURES_DIR / "F7_as_deployed_event_rate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_f8(e5: Path, e1: Path) -> Path | None:
    """As-deployed vs modelled Walker event rate (co-located, θ=2°)."""
    try:
        d5 = _read(e5, "event_rate")
        d1 = _read(e1, "event_rate")
    except FileNotFoundError:
        return None
    pairs = list(dict.fromkeys(zip(d5["system_a"], d5["system_b"])))
    if not pairs:
        return None
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
    for j, (a, b) in enumerate(pairs):
        ax = axes[0][j]
        s5 = d5[(d5["system_a"] == a) & (d5["system_b"] == b) & (d5["variant"] == 1) &
                (d5["theta_deg"] == 2.0)].sort_values("latitude_deg")
        s1 = d1[(d1["system_a"] == a) & (d1["system_b"] == b) & (d1["variant"] == 1) &
                (d1["theta_deg"] == 2.0)].sort_values("latitude_deg")
        if not s5.empty:
            ax.plot(s5["latitude_deg"], s5["event_frac_mean"], "-o", ms=3,
                    label="as-deployed (TLE)")
        if not s1.empty:
            ax.plot(s1["latitude_deg"], s1["event_frac_mean"], "--s", ms=3,
                    label="modelled Walker")
        ax.set_title(f"{a} ← {b}", fontsize=9)
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel("event time-fraction")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("E5 vs E1: as-deployed TLE snapshot against the modelled Walker pattern "
                 "(co-located, θ=2°)")
    fig.tight_layout()
    out = FIGURES_DIR / "F8_as_deployed_vs_walker.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
VICTIM = "ONEWEB"
HEAD_DIAM = 0.45


def fig_p1_binding(e4: Path, e5: Path) -> Path | None:
    """Fig. 1: 25.261(c) binding fraction (Delta T/T > 6%) vs latitude."""
    try:
        d4 = _read(e4, "dtt")
        d5 = _read(e5, "dtt")
    except FileNotFoundError:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for d, color, tag in ((d4, "#1f77b4", "as-filed"), (d5, "#d62728", "as-deployed")):
        for vi, ls, vt in ((1, "-", "co-located"), (0, "--", "worst case")):
            s = d[(d["system_a"] == VICTIM) & (d["variant"] == vi) &
                  (d["diameter_m"] == HEAD_DIAM)].sort_values("latitude_deg")
            if s.empty:
                continue
            ax.plot(s["latitude_deg"], s["event_frac_mean"] * 100.0, ls, color=color, lw=1.5,
                    label=f"{tag}, {vt}")
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel(r"binding time fraction [%]")
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    out = PAPER_FIG_DIR / "P1_binding_vs_latitude.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_p2_throughput(e4: Path, e5: Path) -> Path | None:
    """Fig. 2: time-weighted throughput degradation vs latitude per aperture."""
    try:
        d5 = _read(e5, "dtt")
        d4 = _read(e4, "dtt")
    except FileNotFoundError:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    colors = {0.45: "#1f77b4", 0.6: "#2ca02c", 0.75: "#9467bd"}
    for diam in sorted(d5["diameter_m"].unique()):
        s = d5[(d5["system_a"] == VICTIM) & (d5["variant"] == 0) &
               (d5["diameter_m"] == diam)].sort_values("latitude_deg")
        ax.plot(s["latitude_deg"], s["degradation_nodiv_mean"] * 100.0, "-",
                color=colors.get(diam), lw=1.5, label=f"as-deployed, {diam:g} m")
    s4 = d4[(d4["system_a"] == VICTIM) & (d4["variant"] == 0) &
            (d4["diameter_m"] == HEAD_DIAM)].sort_values("latitude_deg")
    ax.plot(s4["latitude_deg"], s4["degradation_nodiv_mean"] * 100.0, "--",
            color="#1f77b4", lw=1.2, label="as-filed, 0.45 m")
    ax.axhline(3.0, color="k", ls=":", lw=1.2)
    ax.text(1, 3.15, "3% limit (25.261(d))", fontsize=6)
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel("throughput degradation [%]")
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "P2_throughput_degradation.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_p3_capacity(e3: Path) -> Path | None:
    """Fig. 3: capacity-limited conditional unresolvable fraction vs N_beam."""
    try:
        d = _read(e3, "capacity_diversity")
    except FileNotFoundError:
        return None
    s = d[(d["system_a"] == VICTIM) & (d["latitude_deg"] == 60.0)].sort_values("n_beam")
    if s.empty:
        s = d[d["system_a"] == VICTIM].sort_values("n_beam")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.errorbar(s["n_beam"], s["unres_per_served_mean"] * 1e4,
                yerr=s["unres_per_served_ci95"] * 1e4, marker="o", ms=4, lw=1.5,
                color="#1f77b4", capsize=3)
    ax.set_xlabel(r"$N_{\text{beam}}$ (served beams per cell)")
    ax.set_ylabel(r"unresolvable per served [$\times 10^{-4}$]")
    ax.set_xticks(sorted(s["n_beam"].astype(int)))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "P3_capacity_vs_nbeam.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_p4_theta(e1: Path, e5: Path) -> Path | None:
    """Fig. 4: supporting geometric theta-proxy event fraction vs latitude."""
    try:
        d1 = _read(e1, "event_rate")
        d5 = _read(e5, "event_rate")
    except FileNotFoundError:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for d, color, tag in ((d1, "#1f77b4", "as-filed"), (d5, "#d62728", "as-deployed")):
        for vi, ls, vt in ((1, "-", "co-located"), (0, "--", "worst case")):
            s = d[(d["system_a"] == VICTIM) & (d["variant"] == vi) &
                  (d["theta_deg"] == 2.0)].sort_values("latitude_deg")
            if s.empty:
                continue
            ax.plot(s["latitude_deg"], s["event_frac_mean"] * 100.0, ls, color=color, lw=1.4,
                    label=f"{tag}, {vt}")
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel(r"$\theta{=}2^\circ$ event fraction [%]")
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    out = PAPER_FIG_DIR / "P4_theta_proxy.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_p5_analytic(e7: Path) -> Path | None:
    """Fig. 5: analytic vs simulated co-located event fraction vs latitude."""
    try:
        d = _read(e7, "compare")
    except FileNotFoundError:
        return None
    if d.empty:
        return None
    colors = {0.45: "#1f77b4", 0.6: "#2ca02c", 0.75: "#9467bd"}
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for diam in sorted(d["diameter_m"].unique()):
        s = d[d["diameter_m"] == diam].sort_values("latitude_deg")
        ax.plot(s["latitude_deg"], s["analytic_frac"] * 100.0, "-", color=colors.get(diam),
                lw=1.5, label=f"analytic, {diam:g} m")
        ax.plot(s["latitude_deg"], s["sim_frac"] * 100.0, "o", ms=3,
                color=colors.get(diam), mfc="none", label=f"simulated, {diam:g} m")
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel(r"co-located $\Delta T/T>6\%$ [% of time]")
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    out = PAPER_FIG_DIR / "P5_analytic_validation.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


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
    """Paper Fig. 2: degradation and availability change, without vs with diversity."""
    try:
        d = _read(e5, "dtt")
    except FileNotFoundError:
        return None
    s = d[(d["system_a"] == VICTIM) & (d["variant"] == 0) &
          (d["diameter_m"] == HEAD_DIAM)].sort_values("latitude_deg")
    if s.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharex=True)
    ax1.plot(s["latitude_deg"], s["degradation_nodiv_mean"] * 100.0, "-",
             color="#1f77b4", lw=1.6, label="no diversity")
    ax1.plot(s["latitude_deg"], s["degradation_div_mean"] * 100.0, "--",
             color="#2ca02c", lw=1.6, label="with diversity")
    ax1.axhline(3.0, color="k", ls=":", lw=1.1)
    ax1.text(1, 3.10, "3% limit", fontsize=6)
    ax1.set_xlabel("latitude [deg]")
    ax1.set_ylabel("throughput degradation [%]")
    ax1.set_xlim(0, 65)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=6, loc="upper left")
    ax2.plot(s["latitude_deg"], s["unavail_incr_frac_mean"] * 100.0, "-",
             color="#1f77b4", lw=1.6)
    ax2.plot(s["latitude_deg"], s["unavail_incr_div_frac_mean"] * 100.0, "--",
             color="#2ca02c", lw=1.6)
    ax2.axhline(0.4, color="k", ls=":", lw=1.1)
    ax2.text(1, 0.45, "0.4% limit", fontsize=6)
    ax2.set_xlabel("latitude [deg]")
    ax2.set_ylabel("availability change [%]")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "B2_limits.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_b3_pfd(e6: Path) -> Path | None:
    """Paper Fig. 3: no-diversity degradation vs Starlink PFD offset."""
    try:
        d = _read(e6, "eirp_sweep")
    except FileNotFoundError:
        return None
    cases = ("primary", "starlink_pfd_minus3", "starlink_gen1_design")
    s = d[(d["victim"] == VICTIM) & (d["variant"] == 0) &
          (d["diameter_m"] == HEAD_DIAM) & (d["case"].isin(cases))]
    if s.empty:
        return None
    # Offset = PFD change = EIRP_max change under PFD-constant steering.
    p0 = s.loc[s["case"] == "primary", "starlink_eirp_dbw_per_mhz"].iloc[0]
    off = s["starlink_eirp_dbw_per_mhz"] - p0
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.errorbar(off, s["degr_max_qual"] * 100.0, yerr=s["degr_max_qual_ci95"] * 100.0,
                fmt="o", ms=5, color="#1f77b4", capsize=3, lw=1.4)
    # CI half-widths are below the marker size, so print mean +- CI per point.
    names = {"primary": "Gen2 filed max.", "starlink_pfd_minus3": "",
             "starlink_gen1_design": "Gen1 design"}
    for x, (_, r) in zip(off, s.iterrows()):
        lab = f"{r['degr_max_qual'] * 100:.2f}$\\pm${r['degr_max_qual_ci95'] * 100:.2f}"
        if names[r["case"]]:
            lab = names[r["case"]] + "\n" + lab
        below = r["degr_max_qual"] * 100.0 < 3.0  # keep labels clear of the limit line
        ax.annotate(lab, (x, r["degr_max_qual"] * 100.0), textcoords="offset points",
                    xytext=(0, -8 if below else 7), ha="center",
                    va="top" if below else "bottom", fontsize=5.5)
    ax.axhline(3.0, color="k", ls=":", lw=1.1)
    ax.text(0.8, 2.75, "3% limit", fontsize=6, va="top")
    ax.set_xlabel("Starlink PFD offset from limit [dB]")
    ax.set_ylabel("throughput degradation [%]")
    ax.set_xticks(sorted(off.round(2).unique()))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.set_xlim(1, -7)
    ax.set_ylim(0, 1.3 * float((s["degr_max_qual"] * 100.0).max()))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIG_DIR / "B3_pfd_offset.png"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    e1 = newest_run(args.results_root, "e1_event_rate")
    e2 = newest_run(args.results_root, "e2_diversity")
    e3 = newest_run(args.results_root, "e3_capacity_diversity")
    e4 = newest_primary(args.results_root, "e4_dtt")
    e5 = newest_primary(args.results_root, "e5_as_deployed")
    e7 = newest_run(args.results_root, "e7_analytic")
    e6 = newest_run(args.results_root, "e6_eirp_sweep")
    if e1 is None and e2 is None and e3 is None and e4 is None and e5 is None:
        print(f"[figures] no e1/e2/e3/e4/e5 run directories under {args.results_root}.")
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
    if e4 is not None:
        out = fig_f6(e4)
        if out:
            produced.append((out, e4))
    if e5 is not None:
        for fn in (fig_f7,):
            out = fn(e5)
            if out:
                produced.append((out, e5))
        if e1 is not None:
            out = fig_f8(e5, e1)
            if out:
                produced.append((out, e5))

    for out, run_dir in produced:
        shutil.copy2(out, run_dir / out.name)
        print(f"[figures] wrote {out} (copied to {run_dir.name})")

    # Letter figures (paper/figures/): read from results/ only (AGENTS.md rule 4).
    if e3 is not None and e4 is not None and e5 is not None:
        for fn in (fig_p1_binding(e4, e5), fig_p2_throughput(e4, e5),
                   fig_p3_capacity(e3), fig_p4_theta(e1, e5) if e1 is not None else None):
            if fn is not None:
                print(f"[figures] wrote {fn} (paper)")

    if e7 is not None:
        out = fig_p5_analytic(e7)
        if out is not None:
            print(f"[figures] wrote {out} (paper)")

    # Part B paper figures.
    if e4 is not None and e5 is not None:
        for fn in (fig_b1_events(e4, e5), fig_b2_limits(e5)):
            if fn is not None:
                print(f"[figures] wrote {fn} (paper)")
    if e6 is not None:
        out = fig_b3_pfd(e6)
        if out is not None:
            print(f"[figures] wrote {out} (paper)")

    if not produced:
        print("[figures] no plottable summary tables found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
