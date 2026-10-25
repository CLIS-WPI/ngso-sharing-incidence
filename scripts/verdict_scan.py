"""Verdict-sentence scan: every paper/main.tex line that prints a limit metric, with its verdict.

A limit metric is a numbers.tex macro ``...Degr...Pct`` or ``...Unav...Pct`` with a
matching ``...Ci`` macro; its verdict uses the rule of ``make_phase4_report.verdict``
(exceeds / within / indistinguishable) against ``\\DegradeLimitPct`` or
``\\UnavailLimitPct``.  Comment lines and ``\\vcell`` table cells (``check_vcell.py``)
are skipped.  The output is for reading each sentence against its verdicts; the
script does not judge the wording.

Usage:  python scripts/verdict_scan.py [PAPER_DIR]
"""

from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path


def main() -> int:
    paper = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
    lines = (paper / "main.tex").read_text(encoding="utf-8").splitlines()
    vals = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
                           (paper / "numbers.tex").read_text(encoding="utf-8")))
    limits = {"Degr": Decimal(vals["DegradeLimitPct"]), "Unav": Decimal(vals["UnavailLimitPct"])}

    def verdict(name: str) -> str | None:
        kind = "Degr" if "Degr" in name else "Unav" if "Unav" in name else None
        ci = name[:-3] + "Ci" if name.endswith("Pct") else None
        if kind is None or ci not in vals:
            return None
        m, c, limit = Decimal(vals[name]), Decimal(vals[ci]), limits[kind]
        v = "EXCEEDS" if m - c > limit else "within" if m + c < limit else "INDISTINGUISHABLE"
        return f"{name}={m}+-{c} vs {limit}: {v}"

    n = 0
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("%") or "\\vcell{" in line:
            continue
        hits = [v for v in map(verdict, re.findall(r"\\([A-Za-z]+)", line)) if v]
        if hits:
            n += 1
            print(f"L{i}: {line.strip()}")
            for h in hits:
                print(f"      {h}")
    print(f"{n} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
