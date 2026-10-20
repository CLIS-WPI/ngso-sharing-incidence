# Raw data

This directory holds input data that is **not** committed (see `.gitignore`).
For every file downloaded, record here:

| file | source URL | download date (UTC) | notes / licence |
|------|------------|---------------------|-----------------|
| `tle/starlink.tle` (= `tle/starlink_20261020T192548Z.tle`) | https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle | 2026-10-20T19:25:48Z | CelesTrak GP data (TLE format), public. 10,649 objects; TLE epochs 2026-09-21 05:49 to 2026-10-20 16:38 UTC (median 2026-10-20 05:34). sha256 7f3f66521005e216dd94ea5f9cd543594be5014c1ce289b3a1d21bce2eeacace. Phase 4 snapshot: all as-deployed results (E5/E6) from Phase 4 on use it. |
| `tle/oneweb.tle` (= `tle/oneweb_20261020T192553Z.tle`) | https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle | 2026-10-20T19:25:53Z | CelesTrak GP data (TLE format), public. 651 objects; TLE epochs 2026-10-19 03:37 to 2026-10-20 12:59 UTC (median 2026-10-20 06:33). sha256 0d8c70cc30dc436e81914b4a4da74492c262e57c8215e6723276a66ccd0491d9. Phase 4 snapshot. |

Earlier snapshot (Phases 3b-3e, not in this clone): CelesTrak `starlink` / `oneweb` groups downloaded 2026-10-18 (10,651 / 651 objects), used by every as-deployed result before Phase 4.

Expected sources (fill in when downloaded, do not guess URLs):

- **TLE snapshots** for the modelled NGSO constellations (e.g. Celestrak), for
  validating the analytic two-body propagation against SGP4.
- **Population / Broadband-Serviceable-Location (BSL)** rasters for the
  population-weighting phase of the incidence study.

Do not hard-code values derived from these files; process them into
`data/processed/` and cite the run directory in `results/`.
