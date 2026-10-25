"""Download a NEW CelesTrak TLE snapshot into data/raw/tle/, replacing the pinned one.

Every as-deployed result (E5, E6, E9, E15 and the sensitivity runs) uses the pinned
snapshot recorded in ``data/raw/README.md``; the experiments never download
(``ngsi.tle.fetch_group`` fails if the snapshot is missing or its sha256 differs).
Run this only to move to a new snapshot on purpose, then record the printed
sha256 values, URL and date in ``data/raw/README.md`` and re-run every
as-deployed experiment.  Never called by ``scripts/run_all.sh``.

Usage (in the container):
    python scripts/fetch_tle.py --fetch-tle [--groups starlink,oneweb]
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ngsi.tle import celestrak_url, fetch_group

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "raw" / "tle"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch-tle", action="store_true",
                    help="required: confirm replacing the pinned snapshot")
    ap.add_argument("--groups", default="starlink,oneweb")
    args = ap.parse_args()
    if not args.fetch_tle:
        ap.error("refusing to replace the pinned TLE snapshot without --fetch-tle")
    for group in args.groups.split(","):
        fetch_group(group, CACHE, fetch=True)
        path = CACHE / f"{group}.tle"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(REPO)}  {celestrak_url(group)}  "
              f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}  sha256 {digest}")
    print("Record these in data/raw/README.md before running any experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
