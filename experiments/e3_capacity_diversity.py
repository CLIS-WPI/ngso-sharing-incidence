"""Experiment 3 - capacity-limited satellite diversity.

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner`.  Builds regional
hex cell grids, applies a per-satellite beam capacity, and reports the
unresolvable cell-time fraction vs latitude and victim beam count.

Run:  python experiments/e3_capacity_diversity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e3_capacity_diversity.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    run_dir = run_e3(args.config, args.results_root)
    print(f"[e3] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
