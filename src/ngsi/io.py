"""Result I/O: create run directories with full provenance and write raw tables.

Every run writes into ``results/<experiment>_<timestamp>_seed<seed>/`` containing:

* ``run_meta.json`` - git hash, seed, UTC timestamp, command line;
* ``configs/``       - a verbatim copy of the config files used;
* raw per-phasing tables as Parquet (preferred) or CSV.

AGENTS.md rules 1 and 4: figures/tables are built only from these files, and any
number quoted in a report must be traceable to one of them.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def git_hash(repo_root: str | Path = ".") -> str:
    """Return the current git commit hash, or ``'unknown'`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def new_run_dir(
    results_root: str | Path,
    experiment: str,
    seed: int,
    config_paths: Iterable[str | Path] = (),
    repo_root: str | Path = ".",
) -> Path:
    """Create a fresh run directory, write ``run_meta.json`` and copy configs."""
    root = Path(results_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{experiment}_{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "experiment": experiment,
        "seed": seed,
        "timestamp_utc": ts,
        "git_hash": git_hash(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if config_paths:
        cfg_dir = run_dir / "configs"
        cfg_dir.mkdir()
        for path in config_paths:
            src = Path(path)
            if src.exists():
                shutil.copy2(src, cfg_dir / src.name)
    return run_dir


def write_records(
    records: Iterable[dict],
    run_dir: str | Path,
    name: str,
    fmt: str = "parquet",
) -> Path:
    """Write an iterable of record dicts to ``run_dir/name.<ext>`` and return the path."""
    df = pd.DataFrame(list(records))
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")
    return path
