"""Experiment 7 - analytic validation of the co-located in-line event fraction.

Thin entrypoint: the model lives in :mod:`ngsi.analytic`, the driver in
:func:`ngsi.runner.run_e7`.

Run:  python experiments/e7_analytic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e7


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e7_analytic.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--starlink-generation", choices=("gen1", "gen2", "all"),
                        default=None, help="Starlink shells (default: the config's)")
    args = parser.parse_args(argv)

    run_dir = run_e7(args.config, args.results_root,
                     starlink_generation=args.starlink_generation)
    print(f"[e7] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
