"""Table II check: the formatting TeX gives each ``\\vcell`` vs the verdict from numbers.tex.

``\\vcell{mean}{ci}{limit}`` in ``paper/main.tex`` prints bold if the whole CI is above
the limit, italic if the CI contains it, plain if below.  This script

1. writes a LaTeX probe that inputs ``paper/numbers.tex`` and defines the same branch
   test with ``\\typeout`` instead of formatting, one call per ``\\vcell`` in main.tex;
2. runs pdflatex and reads the branch TeX took for each cell;
3. computes the verdict in exact decimals with the rule of
   ``make_phase4_report.verdict`` (exceeds -> bold, within -> plain,
   indistinguishable -> italic) and reports any disagreement (exit 1).

At main.tex sha256 b9724c61...: 38 cells agree, 18 bold, 20 plain, 0 italic.

Usage (in the container):  python scripts/check_vcell.py [PAPER_DIR]
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

STYLE = {"exceeds": "bold", "within": "plain", "indistinguishable": "italic"}


def main() -> int:
    paper = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
    main_tex = (paper / "main.tex").read_text(encoding="utf-8")
    numbers = (paper / "numbers.tex").read_text(encoding="utf-8")
    # The probe copies this definition; fail if main.tex changed it.
    assert r"\ifdim\dimexpr #1pt-#2pt\relax>#3pt\relax\textbf{#1$\pm$#2}" in main_tex
    assert r"\else\ifdim\dimexpr #1pt+#2pt\relax<#3pt\relax#1$\pm$#2" in main_tex

    vals = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", numbers))
    calls = re.findall(r"\\vcell\{\\([A-Za-z]+)\}\{\\([A-Za-z]+)\}\{\\([A-Za-z]+)\}",
                       main_tex)
    assert len(calls) == main_tex.count("\\vcell{"), "a \\vcell call does not use three macros"

    probe = [r"\documentclass{article}", r"\input{numbers}", r"\newcommand{\vprobe}[4]{%",
             r"\ifdim\dimexpr #1pt-#2pt\relax>#3pt\relax\typeout{VCELL #4 bold}%",
             r"\else\ifdim\dimexpr #1pt+#2pt\relax<#3pt\relax\typeout{VCELL #4 plain}%",
             r"\else\typeout{VCELL #4 italic}\fi\fi}", r"\begin{document}"]
    probe += [rf"\vprobe{{\{p}}}{{\{c}}}{{\{lim}}}{{{i}}}" for i, (p, c, lim) in enumerate(calls)]
    probe += ["x", r"\end{document}"]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "probe.tex").write_text("\n".join(probe) + "\n", encoding="utf-8")
        (Path(d) / "numbers.tex").write_text(numbers, encoding="utf-8")
        out = subprocess.run(["pdflatex", "-interaction=nonstopmode", "probe.tex"], cwd=d,
                             capture_output=True, text=True).stdout
    tex = dict(re.findall(r"VCELL (\d+) (\w+)", out))
    assert len(tex) == len(calls), f"TeX reported {len(tex)} of {len(calls)} cells"

    bad = 0
    for i, (p, c, lim) in enumerate(calls):
        m, ci, limit = Decimal(vals[p]), Decimal(vals[c]), Decimal(vals[lim])
        v = "exceeds" if m - ci > limit else "within" if m + ci < limit else "indistinguishable"
        ok = STYLE[v] == tex[str(i)]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {p}={m}+-{ci} vs {lim}={limit}: {v} / TeX {tex[str(i)]}")
    counts = {s: list(tex.values()).count(s) for s in ("bold", "plain", "italic")}
    print(f"{len(calls) - bad}/{len(calls)} agree; TeX branches {counts}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
