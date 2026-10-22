#!/usr/bin/env bash
# Re-run every experiment the paper cites, then regenerate numbers, figures, paper.
# Intended to run *inside* the dev container:  make run-all
#
# Each command reproduces one run directory under results/ that paper/numbers.tex
# reads (same arguments, same order as recorded in that run's run_meta.json).
# E1-E3 ran in float64 (the default); E4-E9 and the sensitivity runs in float32
# (NGSI_COMPUTE_DTYPE, recorded in run_meta.json). New run directories get new
# timestamps, so make_numbers.py / make_figures.py pick them up as the newest.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== e1_event_rate / e2_diversity / e3_capacity_diversity (float64) =="
python experiments/e1_event_rate.py
python experiments/e2_diversity.py
python experiments/e3_capacity_diversity.py

export NGSI_COMPUTE_DTYPE=float32

echo "== headline: e4_dtt, e5_as_deployed, e6_eirp_sweep (Starlink Gen2) =="
python experiments/e4_dtt.py --starlink-generation gen2
python experiments/e5_as_deployed.py --starlink-generation gen2
python experiments/e6_eirp_sweep.py --starlink-generation gen2

echo "== e7 analytic cross-check, e8 analytic diagnosis =="
python experiments/e7_analytic.py
python scripts/diagnose_analytic.py

echo "== Starlink generation sensitivity (Gen1 only, all Starlink) =="
python experiments/e4_dtt.py --starlink-generation gen1 --handoff-policies full,full_keep_serving,legacy
python experiments/e4_dtt.py --starlink-generation all --handoff-policies full,full_keep_serving,legacy
python experiments/e5_as_deployed.py --starlink-generation gen1 \
    --handoff-policies full,full_keep_serving,legacy --n-start-times 4
python experiments/e5_as_deployed.py --starlink-generation all \
    --handoff-policies full,full_keep_serving,legacy --n-start-times 4

echo "== sensitivity: Gen2 = 43 deg shell only; 2 s time step (results/sensitivity) =="
python experiments/e5_as_deployed.py --config configs/experiments/sensitivity/e5_gen2_43only.yaml \
    --starlink-generation gen2 --handoff-policies full,full_keep_serving \
    --results-root results/sensitivity
python experiments/e5_as_deployed.py --starlink-generation gen2 --time-step-s 2 --n-start-times 2 \
    --handoff-policies full,full_keep_serving --results-root results/sensitivity

echo "== e9 OneWeb 55 deg mask diagnosis =="
python scripts/diagnose_min_elevation.py

echo "== e14 interruption cost (post-processes the e5 raw tables) =="
NGSI_COMPUTE_DTYPE=float64 python scripts/interruption_cost.py

echo "== paper numbers, figures, paper =="
python scripts/make_numbers.py
python figures/make_figures.py
bash scripts/build_paper.sh

echo "done."
