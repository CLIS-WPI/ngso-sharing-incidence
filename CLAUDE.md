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

## Commands not in AGENTS.md

- `scripts/run_all.sh` does **not** run `e7_analytic.py`, `scripts/diagnose_analytic.py`
  or `scripts/make_phase3d_report.py`; run those directly in the container when needed.
- Rebuild the paper only: `make paper` (runs `make_numbers.py` then `latexmk` via
  `scripts/build_paper.sh`).
