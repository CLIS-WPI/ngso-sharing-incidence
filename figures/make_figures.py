"""Build every figure from files under ``results/`` only (AGENTS.md rule 4).

STUB (phase 1): discovers summary tables in ``results/`` and, when none exist,
exits cleanly.  Figures are never drawn from hard-coded arrays.

Run:  python figures/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"


def find_summaries(results_root: Path) -> list[Path]:
    """Return candidate summary tables (CSV) committed under ``results/``."""
    if not results_root.exists():
        return []
    return sorted(results_root.glob("**/summary*.csv"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    summaries = find_summaries(args.results_root)
    if not summaries:
        print(f"[figures] no summary tables under {args.results_root}; nothing to plot.")
        return 0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figures] found {len(summaries)} summary table(s); rendering not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
