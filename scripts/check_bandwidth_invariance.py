"""Assert that the channel-bandwidth change left every reported metric unchanged.

Compares the summary tables of a run made with the new channel bandwidths (250 MHz
for both STARLINK and ONEWEB, del Portillo Table 3) against a reference run of the
same experiment made with the earlier ASSUMED values (240 / 125 MHz).  Every
numeric column of every ``summary_*.csv`` common to both runs is compared row by
row; the maximum absolute difference per file is written to
``results/e10_bandwidth_invariance_<utc>_seed0/bandwidth_invariance.json``.

The 25.261(d) metrics are bandwidth-invariant by construction (I/N and C/N are
per-MHz densities and the degradation is a ratio of rates that are both
proportional to B), so any difference above the tolerance fails the check.

Run:  python scripts/check_bandwidth_invariance.py --new results/<run> --ref results/<run> \
          [--new ... --ref ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ngsi.io import new_run_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
# Bookkeeping columns that legitimately differ between runs (none are metrics).
SKIP = {"n_phasings"}


def _bandwidths(run: Path) -> dict:
    cons = yaml.safe_load((run / "configs" / "constellations.yaml").read_text())
    out = {}
    for c in cons["constellations"]:
        out[c["name"]] = float(c["shells"][0]["beam"]["bandwidth_mhz"])
    return out


def compare(new: Path, ref: Path) -> dict:
    files = sorted({p.name for p in new.glob("summary_*.csv")}
                   & {p.name for p in ref.glob("summary_*.csv")})
    per_file = {}
    for name in files:
        a = pd.read_csv(new / name)
        b = pd.read_csv(ref / name)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            per_file[name] = {"comparable": False, "shape_new": list(a.shape),
                              "shape_ref": list(b.shape)}
            continue
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c not in SKIP]
        other = [c for c in a.columns if c not in num and c not in SKIP]
        keys_equal = bool((a[other].astype(str).values == b[other].astype(str).values).all())
        diff = (a[num].to_numpy(dtype=np.float64) - b[num].to_numpy(dtype=np.float64))
        both_nan = np.isnan(a[num].to_numpy(dtype=np.float64)) & np.isnan(
            b[num].to_numpy(dtype=np.float64))
        diff = np.where(both_nan, 0.0, np.abs(diff))
        worst = int(np.nanargmax(diff)) if diff.size else 0
        per_file[name] = {
            "comparable": True, "keys_equal": keys_equal, "n_values": int(diff.size),
            "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
            "nan_mismatch": int(np.isnan(diff).sum()),
            "worst_column": num[worst % len(num)] if num else None,
        }
    return {"new": f"results/{new.name}/", "ref": f"results/{ref.name}/",
            "bandwidth_new_mhz": _bandwidths(new), "bandwidth_ref_mhz": _bandwidths(ref),
            "files": per_file}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path, action="append", required=True)
    ap.add_argument("--ref", type=Path, action="append", required=True)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args(argv)
    if len(args.new) != len(args.ref):
        raise SystemExit("give one --ref per --new")
    pairs = [compare(n.resolve(), r.resolve()) for n, r in zip(args.new, args.ref)]
    ok = True
    for pr in pairs:
        for name, f in pr["files"].items():
            good = f["comparable"] and f["keys_equal"] and f["nan_mismatch"] == 0 \
                and f["max_abs_diff"] <= TOL
            ok &= good
            print(f"[bw] {pr['new']} vs {pr['ref']} {name}: "
                  + (f"max|diff|={f['max_abs_diff']:.3g} ({f['worst_column']})"
                     if f["comparable"] else "NOT COMPARABLE") + ("" if good else "  FAIL"))
    run_dir = new_run_dir(args.results_root, "e10_bandwidth_invariance", 0,
                          config_paths=[REPO_ROOT / "configs" / "constellations.yaml"],
                          repo_root=REPO_ROOT)
    (run_dir / "bandwidth_invariance.json").write_text(
        json.dumps({"tolerance": TOL, "passed": bool(ok), "pairs": pairs}, indent=2))
    print(f"[bw] {'PASS' if ok else 'FAIL'} -> {run_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
