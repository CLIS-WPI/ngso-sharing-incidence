# Raw data

This directory holds input data that is **not** committed (see `.gitignore`).
For every file downloaded, record here:

| file | source URL | download date (UTC) | notes / licence |
|------|------------|---------------------|-----------------|
| `tle/oneweb.tle` | https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle | 2026-10-18 | CelesTrak GP data, public; snapshot of the OneWeb constellation (651 sats), used by E5 (Phase 3b). Cached; CelesTrak refreshes once every 2 h. |
| `tle/starlink.tle` | https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle | 2026-10-18 | CelesTrak GP data, public; snapshot of the Starlink constellation, used by E5 (Phase 3b). Cached; CelesTrak refreshes once every 2 h. |

Expected sources (fill in when downloaded, do not guess URLs):

- **TLE snapshots** for the modelled NGSO constellations (e.g. Celestrak), for
  validating the analytic two-body propagation against SGP4.
- **Population / Broadband-Serviceable-Location (BSL)** rasters for the
  population-weighting phase of the incidence study.

Do not hard-code values derived from these files; process them into
`data/processed/` and cite the run directory in `results/`.
