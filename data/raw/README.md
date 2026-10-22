# Raw data

`tle/` is the CelesTrak TLE snapshot every as-deployed result in the paper uses
(E5, E6, E9 and the sensitivity runs). It is committed byte-exact (`.gitattributes`
marks `*.tle` as `-text`, so the CRLF line endings CelesTrak serves are kept and the
sha256 values below hold). `ngsi.tle.fetch_group` reads `tle/<group>.tle` when it
exists and downloads only when it is missing, so re-running the experiments uses
this snapshot, not today's sky. Everything else under `data/raw/` is gitignored.

| file | source URL | download date (UTC) | objects | sha256 |
|------|------------|---------------------|---------|--------|
| `tle/starlink.tle` (= `tle/starlink_20261020T192548Z.tle`) | https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle | 2026-10-20T19:25:48Z | 10,649 | `7f3f66521005e216dd94ea5f9cd543594be5014c1ce289b3a1d21bce2eeacace` |
| `tle/oneweb.tle` (= `tle/oneweb_20261020T192553Z.tle`) | https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle | 2026-10-20T19:25:53Z | 651 | `0d8c70cc30dc436e81914b4a4da74492c262e57c8215e6723276a66ccd0491d9` |

CelesTrak GP data (TLE format), public. Starlink TLE epochs 2026-09-21 05:49 to
2026-10-20 16:38 UTC (median 2026-10-20 05:34); OneWeb TLE epochs 2026-10-19 03:37 to
2026-10-20 12:59 UTC (median 2026-10-20 06:33). The timestamped files are identical
copies kept under their download time.

Check: `sha256sum data/raw/tle/*.tle`.
