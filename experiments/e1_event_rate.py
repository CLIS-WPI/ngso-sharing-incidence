"""Experiment 1 - geographic incidence of NGSO-NGSO in-line interference events.

Thin entrypoint: all heavy logic lives in :mod:`ngsi.runner`.  Writes a
provenance-stamped run directory with the sanity gate, raw per-phasing tables and
the summary event-rate profile.

Run:  python experiments/e1_event_rate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ngsi.runner import REPO_ROOT, run_e1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e1_event_rate.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    run_dir = run_e1(args.config, args.results_root)
    print(f"[e1] run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
