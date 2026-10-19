"""Experiment 6 - pre-paper check 1: sweep the ASSUMED OneWeb EIRP density.

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner` (``run_e6``).  Holds
the as-deployed CelesTrak snapshot fixed and sweeps one system's downlink EIRP
density over the configured values, reporting both ordered user-downlink
Delta T/T exceedance fractions.

Run:  python experiments/e6_eirp_sweep.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e6_eirp_sweep.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    run_dir = run_e6(args.config, args.results_root)
    print(f"[e6] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
