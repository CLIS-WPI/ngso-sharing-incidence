"""Compare a float32 run against the same run in float64 (``ngsi.precision``).

For one experiment run made twice (``NGSI_COMPUTE_DTYPE=float64`` and ``float32``,
same config) this reports

* the maximum absolute difference of every numeric column of every
  ``summary_*.csv`` (and of the raw per-station tables), and
* every paper macro (``scripts/make_numbers.py`` run on each results root alone)
  compared as printed strings, i.e. at the precision the paper prints.

The decision rule (Phase 4): float32 may be used for the final runs only if every
printed paper number is identical.  The verdict and all differences are written to
``results/e11_precision_check_<utc>_seed0/precision_check.json``.

Run:  python scripts/compare_precision.py --f64 <results-root-64> --f32 <results-root-32>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ngsi.io import new_run_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _only_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise SystemExit(f"{root}: expected exactly one run directory, found {len(runs)}")
    return runs[0]


def _table_diffs(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return {"comparable": False}
    out = {}
    for c in a.columns:
        if not pd.api.types.is_numeric_dtype(a[c]):
            out[c] = {"equal": bool((a[c].astype(str) == b[c].astype(str)).all())}
            continue
        x = a[c].to_numpy(dtype=np.float64)
        y = b[c].to_numpy(dtype=np.float64)
        nan = np.isnan(x) & np.isnan(y)
        d = np.where(nan, 0.0, np.abs(x - y))
        out[c] = {"max_abs_diff": float(np.nanmax(d)) if d.size else 0.0,
                  "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum())}
    return {"comparable": True, "columns": out}


def _macros(root: Path, out: Path) -> dict[str, str]:
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_numbers.py"),
                    "--results-root", str(root), "--out", str(out)], check=True)
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", out.read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f64", type=Path, required=True, help="results root of the float64 run")
    ap.add_argument("--f32", type=Path, required=True, help="results root of the float32 run")
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args(argv)
    r64, r32 = _only_run(args.f64), _only_run(args.f32)
    meta = {k: json.loads((r / "run_meta.json").read_text()).get("compute_dtype")
            for k, r in (("f64", r64), ("f32", r32))}
    if meta != {"f64": "float64", "f32": "float32"}:
        raise SystemExit(f"run_meta compute_dtype mismatch: {meta}")

    tables = {}
    for p in sorted(r64.glob("summary_*.csv")) + sorted(r64.glob("raw_*.parquet")):
        q = r32 / p.name
        if not q.exists():
            tables[p.name] = {"comparable": False, "missing_in_f32": True}
            continue
        rd = pd.read_csv if p.suffix == ".csv" else pd.read_parquet
        tables[p.name] = _table_diffs(rd(p), rd(q))

    m64 = _macros(args.f64, args.f64 / "numbers_f64.tex")
    m32 = _macros(args.f32, args.f32 / "numbers_f32.tex")
    names = sorted(set(m64) | set(m32))
    differing = {n: {"f64": m64.get(n), "f32": m32.get(n)} for n in names
                 if m64.get(n) != m32.get(n)}
    passed = not differing
    run_dir = new_run_dir(args.results_root, "e11_precision_check", 0, repo_root=REPO_ROOT)
    (run_dir / "precision_check.json").write_text(json.dumps({
        "f64_run": str(r64), "f32_run": str(r32), "n_macros": len(names),
        "printed_paper_numbers_identical": passed, "differing_macros": differing,
        "macros": {n: m64.get(n) for n in names}, "tables": tables}, indent=2))
    worst = {}
    for name, t in tables.items():
        if t.get("comparable"):
            cols = {c: v["max_abs_diff"] for c, v in t["columns"].items() if "max_abs_diff" in v}
            if cols:
                c = max(cols, key=cols.get)
                worst[name] = (c, cols[c])
    for name, (c, v) in worst.items():
        print(f"[precision] {name}: max |f32 - f64| = {v:.3g} ({c})")
    print(f"[precision] {len(names)} paper macros, {len(differing)} differ at printed precision")
    for n, v in differing.items():
        print(f"    {n}: f64 {v['f64']}  f32 {v['f32']}")
    print(f"[precision] {'PASS' if passed else 'FAIL'} -> {run_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
