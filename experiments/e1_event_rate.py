"""Experiment 1 - geographic incidence of NGSO-NGSO in-line interference events.

STUB (phase 1): this entrypoint validates the configuration and creates a
provenance-stamped run directory, but does not yet run the screening pipeline.
It deliberately emits no physical quantities (AGENTS.md rule 1).

Run:  python experiments/e1_event_rate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ngsi import io
from ngsi.config import load_constellations, load_experiment, load_stations

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "e1_event_rate.yaml",
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
        "n_constellations": len(exp.constellations),
        "n_latitudes": len(stations.latitudes_deg),
        "n_longitudes": len(stations.longitudes_deg),
        "min_elevation_deg": stations.min_elevation_deg,
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[e1] validated config and created run dir: {run_dir}")
    print("[e1] screening pipeline not implemented yet (stub).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
