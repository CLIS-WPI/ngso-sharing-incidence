# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The rules, commands, environment and gotchas in `AGENTS.md` (imported above) apply
in full. Its "Current status" section is **stale**: the project is past the Phase 1
scaffold. Notes below cover what it does not.

## Current state

- Work is in Phase 3 (IEEE Communications Letters draft in `paper/`); see
  `git log` for the phase history (Gate-1, 3a DTT screen, 3b as-deployed TLEs,
  3c paper, 3d/3e PFD work). Check the newest `results/PHASE*_REPORT.md` for context.
- Experiments E1-E7 are implemented. `README.md` "Status" and "Layout" are also stale.

## Architecture

- `src/ngsi/runner.py` holds nearly all orchestration: `run_e1` ... `run_e7`,
  `screen_phasings`, sanity gates, summaries. `experiments/eN_*.py` only parse
  CLI flags (e.g. `e4_dtt.py --steering`, `--diversity-fspl`) and call `run_eN`.
- Pipeline per experiment: `ngsi.config` loads/validates YAML -> `ngsi.systems` /
  `constellation` / `tle` build systems -> `propagation` (batched torch on GPU) ->
  `geometry`/`frames` look angles -> `screening` (in-line events) or `dtt_screen`
  (25.261(d) Delta T/T link budget via `link_budget`, `antennas`, `bands`) ->
  aggregation across phasings with t-interval CIs -> `ngsi.io.new_run_dir`.
- `ngsi.analytic` is a closed-form cross-check of the simulation (E7);
  `scripts/diagnose_analytic.py` writes `e8_analytic_diagnosis_*` runs.
- Downstream consumers find inputs by **newest run directory** for a prefix
  (`newest()` / `newest_where()` in `scripts/make_numbers.py`, `newest_run` in
  `figures/make_figures.py`). `newest_where` filters on a run's `status.json` so
  sensitivity reruns sharing a prefix don't replace the primary configuration.
  A new sensitivity run must write the distinguishing keys to `status.json`, or it
  will silently become the headline result.
- Reports (`scripts/make_*_report.py` -> `results/*_REPORT.md`) and
  `paper/numbers.tex` are generated; edit the scripts, not the outputs.

## Phase 4 conventions

- `STARLINK` in `configs/constellations.yaml` is the Gen1 system; `STARLINK_GEN2`
  holds the FCC 22-91 shells. Runs build the `STARLINK` interferer from
  `starlink_generation` (gen1 | gen2 | all; config key or `--starlink-generation`).
  As deployed, TLEs are labelled by `starlink_generation_rule` (E5/E6 configs).
  Headline selection (`make_numbers.py`, figures, reports) only takes runs whose
  `status.json` generation equals the configured one.
- Handoff policies (`ngsi.dtt_screen.HandoffPolicy`) are listed per experiment;
  `primary_handoff` feeds `degradation_div`; every policy goes to
  `raw_handoff.parquet` / `summary_handoff.csv`. `--handoff-policies a,b` runs a
  subset of them.
- `NGSI_COMPUTE_DTYPE=float32` switches the heavy torch geometry to float32
  (default float64; recorded in `run_meta.json`). Some reference tests are
  float64-only.
- On Windows: set `MSYS_NO_PATHCONV=1` before `docker exec -w /workspace ...` in
  Git Bash, and write files with LF endings.

## Commands not in AGENTS.md

- `scripts/run_all.sh` runs only E1-E6, figures, the Gate-1/3a/3b reports and the paper.
  It does **not** run `experiments/e7_analytic.py`, `scripts/diagnose_analytic.py` (E8),
  `scripts/diagnose_min_elevation.py` (E9, OneWeb 55 deg mask / bandwidth invariance),
  `scripts/calibrate_oneweb_eirp.py` or `scripts/make_phase3d_report.py`; run those
  directly in the container when needed.
- Rebuild the paper only: `make paper` (runs `make_numbers.py` then `latexmk` via
  `scripts/build_paper.sh`).
- Every experiment entrypoint accepts `--config` and `--results-root`. Pointing
  `--results-root` at a scratch directory keeps an exploratory run from becoming the
  "newest" run that reports and `paper/numbers.tex` pick up.
- `tests/test_paper_numbers.py` checks that every capitalised macro in
  `paper/main.tex` is defined in `paper/numbers.tex`. After you add a number to the
  paper, add it to `scripts/make_numbers.py` and regenerate, or the suite fails.
- The Makefile uses `$(shell id -u)`, so on this Windows host run `make` from Git Bash
  (or call `docker compose -f docker/docker-compose.yml ...` directly).
