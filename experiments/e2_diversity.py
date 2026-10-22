"""Experiment 2 - satellite diversity and the burden of 25.261 rules.

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner`.  Adds the
unresolvable rate, event durations, the minimum-elevation sweep, the Starlink
53 deg Walker-F sensitivity, and the 25.261(c) band-split spectrum loss.

Run:  python experiments/e2_diversity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e2_diversity.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    run_dir = run_e2(args.config, args.results_root)
    print(f"[e2] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
