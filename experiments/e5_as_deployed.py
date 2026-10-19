"""Experiment 5 - as-deployed constellation from a CelesTrak TLE snapshot (Phase 3b).

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner` (``run_e5``).  The
snapshot is downloaded (and cached under ``data/raw/tle/``); its URL and date are
recorded in ``data/raw/README.md``.  Replication is over independent simulation
start times.

Run:  python experiments/e5_as_deployed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e5_as_deployed.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--diversity-fspl", choices=("interferer", "victim"), default=None,
                        help="diversity path loss: 'interferer' (default) or legacy 'victim'")
    parser.add_argument("--steering", choices=("config", "constant_eirp", "pfd_constant"),
                        default=None,
                        help="interferer beam steering: 'config' (default, per-beam), "
                             "'constant_eirp' or 'pfd_constant'")
    args = parser.parse_args(argv)

    steering = None if args.steering in (None, "config") else args.steering
    run_dir = run_e5(args.config, args.results_root, diversity_fspl=args.diversity_fspl,
                     steering=steering)
    print(f"[e5] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
