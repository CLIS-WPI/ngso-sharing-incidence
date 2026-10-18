"""Experiment 4 - 47 CFR 25.261(d) Delta T/T link-budget screening.

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner` (``run_e4``).  Screens
the co-frequency user-downlink pairs with the regulatory power criterion
(aggregate Delta T/T > 0.06) and reports the event / unresolvable time fraction
and the throughput degradation vs latitude.

Run:  python experiments/e4_dtt.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e4_dtt.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    run_dir = run_e4(args.config, args.results_root)
    print(f"[e4] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
