"""Compare paper/refs.bib with the resolved metadata in paper/citation_lock.json.

``citation_lock.json`` is written by the ai-research-writing skill's
``verify_citations.py`` (Crossref / DataCite / arXiv).  This script compares each
locked record with its refs.bib entry (title, year, first-author surname, volume,
pages, DOI) and, for entries with no DOI / arXiv identifier, checks that the URL
in refs.bib answers (HTTP status).  It writes ``paper/citation_check.md``.

Usage:  python scripts/check_bib_against_lock.py [--no-network]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "paper" / "refs.bib"
LOCK = REPO / "paper" / "citation_lock.json"
MAIN = REPO / "paper" / "main.tex"
OUT = REPO / "paper" / "citation_check.md"


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {"_type": kind}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[f.group(1).lower()] = f.group(2)
        entries[key] = fields
    return entries


def norm(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", s or "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.casefold() if c.isalnum())


def surname(author_field: str) -> str:
    first = (author_field or "").split(" and ")[0].strip()
    if "," in first:
        return norm(first.split(",")[0])
    return norm(first.split()[-1]) if first else ""


def meta_surname(meta: dict) -> str:
    a = (meta.get("authors") or [""])[0]
    if isinstance(a, dict):
        a = a.get("family") or a.get("name") or ""
    a = str(a)
    return norm(a.split(",")[0] if "," in a else (a.split()[-1] if a.split() else ""))


def url_status(url: str) -> str:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0 (ngsi audit)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"


def md(rows: list[dict]) -> str:
    if not rows:
        return "(none)"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()
    bib = parse_bib(BIB.read_text(encoding="utf-8"))
    cited = set()
    for m in re.finditer(r"\\cite\w*(?:\[[^\]]*\])?\{([^}]*)\}", MAIN.read_text(encoding="utf-8")):
        cited.update(k.strip() for k in m.group(1).split(","))
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {"records": []}
    rows, mism = [], []
    for rec in lock["records"]:
        key = rec["key"].removesuffix("-arxiv")
        b, meta = bib.get(key, {}), rec.get("metadata") or {}
        checks = {
            "title": norm(b.get("title")) == norm(meta.get("title")),
            "year": str(b.get("year", "")) == str(meta.get("year", "")),
            "first author": surname(b.get("author")) == meta_surname(meta),
        }
        if rec["identifier"].lower().startswith("doi:"):
            checks["doi"] = (b.get("doi", "").lower() == rec["identifier"].split(":", 1)[1].lower())
            for f in ("volume", "pages"):
                if b.get(f) and meta.get(f):
                    checks[f] = norm(b[f]) == norm(str(meta[f]))
        bad = [k for k, ok in checks.items() if not ok]
        rows.append({"key": rec["key"], "identifier": rec["identifier"], "lock status": rec["status"],
                     "provider": rec.get("provider", ""),
                     "resolved title": str(meta.get("title", ""))[:80],
                     "resolved year": meta.get("year", ""),
                     "fields checked": ", ".join(checks),
                     "mismatch": ", ".join(bad) or "none"})
        if bad:
            mism.append({"key": rec["key"], "field": ", ".join(bad),
                         "refs.bib": "; ".join(f"{k}={b.get(k.split()[0] if k != 'first author' else 'author', '')}"
                                               for k in bad)[:160],
                         "resolved": "; ".join(f"{k}={meta.get(k.split()[0] if k != 'first author' else 'authors', '')}"
                                               for k in bad)[:160]})
    locked = {r["key"].removesuffix("-arxiv") for r in lock["records"]}
    other = []
    for key in sorted(bib):
        if key in locked:
            continue
        b = bib[key]
        url = b.get("url") or (re.search(r"\\url\{([^}]*)\}", b.get("howpublished", "")) or [None, None])[1]
        url = url.replace("\\_", "_") if url else None
        other.append({"key": key, "type": b["_type"], "cited": "yes" if key in cited else "**no**",
                      "identifier": "none (no DOI/arXiv)", "url": url or "-",
                      "url status": ("not checked" if args.no_network or not url else url_status(url))})
    L = ["# Citation check: paper/refs.bib", "",
         "Generated by `scripts/check_bib_against_lock.py` from `paper/citation_lock.json` "
         "(ai-research-writing-skill `verify_citations.py`: Crossref / DataCite / arXiv).  "
         "Metadata identity only; whether each source supports its sentence is not checked "
         "(claim-support status 'unverified' in `paper/citation_requests.json`).", "",
         "## 1. Entries with a DOI or arXiv identifier", "", md(rows), "",
         "## 2. Mismatches between refs.bib and the resolved metadata", "", md(mism), "",
         "## 3. Entries with no DOI / arXiv identifier (not verifiable via Crossref/arXiv)", "",
         md(other), ""]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}: {len(rows)} resolved, {len(mism)} with mismatches, {len(other)} without identifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
