"""Experiment 2 - satellite diversity and the burden of the 25.261 rules.

STUB (phase 1): validates configuration and creates a provenance-stamped run
directory.  No physical quantities are produced yet (AGENTS.md rule 1).

Run:  python experiments/e2_diversity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ngsi import io
from ngsi.config import load_constellations, load_experiment, load_stations
from ngsi.rules import DTT_LONG_TERM, DTT_SHORT_TERM

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e2_diversity.yaml",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    exp = load_experiment(args.config)
    constellations = load_constellations(REPO_ROOT / "configs" / "constellations.yaml")
    stations = load_stations(REPO_ROOT / "configs" / exp.stations)

    missing = [name for name in exp.constellations if name not in constellations]
    if missing:
        raise SystemExit(f"experiment references unknown constellations: {missing}")

    run_dir = io.new_run_dir(
        args.results_root,
        experiment=exp.name,
        seed=exp.seed,
        config_paths=[
            args.config,
            REPO_ROOT / "configs" / "constellations.yaml",
            REPO_ROOT / "configs" / exp.stations,
        ],
        repo_root=REPO_ROOT,
    )

    status = {
        "experiment": exp.name,
        "status": "not_implemented",
        "dtt_long_term": DTT_LONG_TERM,
        "dtt_short_term": DTT_SHORT_TERM,
        "n_constellations": len(exp.constellations),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[e2] validated config and created run dir: {run_dir}")
    print("[e2] diversity/rule pipeline not implemented yet (stub).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
