# Raw data

This directory holds input data that is **not** committed (see `.gitignore`).
For every file downloaded, record here:

| file | source URL | download date (UTC) | notes / licence |
|------|------------|---------------------|-----------------|
| _(none yet)_ | | | |

Expected sources (fill in when downloaded, do not guess URLs):

- **TLE snapshots** for the modelled NGSO constellations (e.g. Celestrak), for
  validating the analytic two-body propagation against SGP4.
- **Population / Broadband-Serviceable-Location (BSL)** rasters for the
  population-weighting phase of the incidence study.

Do not hard-code values derived from these files; process them into
`data/processed/` and cite the run directory in `results/`.
