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
    parser.add_argument("--starlink-generation", choices=("gen1", "gen2", "all"), default=None,
                        help="Starlink shells used as interferer/victim (default: config)")
    parser.add_argument("--handoff-policies", default=None,
                        help="comma-separated subset of the config's handoff_policies")
    parser.add_argument("--n-start-times", type=int, default=None,
                        help="number of start times (default: config)")
    args = parser.parse_args(argv)

    run_dir = run_e6(args.config, args.results_root,
                     starlink_generation=args.starlink_generation,
                     handoff_policies=(args.handoff_policies.split(",")
                                       if args.handoff_policies else None),
                     n_start_times=args.n_start_times)
    print(f"[e6] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
