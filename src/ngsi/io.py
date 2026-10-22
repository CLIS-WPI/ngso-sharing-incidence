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


def _compute_dtype_name() -> str:
    from .precision import compute_dtype_name

    return compute_dtype_name()


def git_dirty(repo_root: str | Path = ".") -> dict:
    """Uncommitted tracked changes at run time: file list and a hash of ``git diff HEAD``.

    The commit hash alone does not identify the code that ran when the tree is
    dirty; the diff hash does (``git diff HEAD | sha256sum``).
    """
    import hashlib

    try:
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=repo_root, capture_output=True, text=True, check=True)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root, capture_output=True,
                              check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"dirty": None}
    files = [ln[3:] for ln in st.stdout.splitlines() if ln.strip()]
    return {"dirty": bool(files), "modified_files": files,
            "diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if files else None}


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
        "git_worktree": git_dirty(repo_root),
        "compute_dtype": _compute_dtype_name(),
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
