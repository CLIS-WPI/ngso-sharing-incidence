"""Run the ai-research-writing skill's record_build.py with \\graphicspath support.

The skill's ``paper_contract.resolve_graphics`` looks for ``\\includegraphics``
files only next to the .tex file; ``paper/main.tex`` sets
``\\graphicspath{{figures/}}``.  This wrapper adds the declared graphics paths to
the search (same file suffixes, same fail-closed error) and then runs
``record_build.main()`` unchanged.  The skill itself is not modified.

Usage (in the container):
    python scripts/record_build_graphicspath.py SKILL_SCRIPTS_DIR PROJECT_DIR [--run]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    skill_dir = Path(sys.argv[1]).resolve()
    sys.argv = [str(skill_dir / "record_build.py")] + sys.argv[2:]
    sys.path.insert(0, str(skill_dir))
    import paper_contract as pc
    import record_build as rb

    gp_re = re.compile(r"\\graphicspath\{((?:\{[^{}]*\})+)\}")

    def resolve_graphics(main_tex: Path) -> list[Path]:
        resolved: list[Path] = []
        for tex_path in pc.collect_tex_files(main_tex):
            text = pc.strip_comments(pc.read_utf8(tex_path))
            dirs = [tex_path.parent]
            for m in gp_re.finditer(text):
                dirs += [tex_path.parent / d for d in re.findall(r"\{([^{}]*)\}", m.group(1))]
            for match in pc.GRAPHICS_RE.finditer(text):
                raw = match.group(1).strip()
                cands = [(d / f"{raw}{s}").resolve() for d in dirs for s in pc.GRAPHIC_SUFFIXES]
                found = next((c for c in cands if c.is_file()), None)
                if found is None:
                    raise pc.ContractError(f"Missing graphic declared by {tex_path}: {raw}")
                resolved.append(found)
        return sorted(set(resolved))

    pc.resolve_graphics = resolve_graphics
    return rb.main()


if __name__ == "__main__":
    raise SystemExit(main())
