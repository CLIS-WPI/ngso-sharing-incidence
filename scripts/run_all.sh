#!/usr/bin/env bash
# Regenerate every result and figure with one command.
# Intended to run *inside* the dev container:  make run-all
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== e1_event_rate =="
python experiments/e1_event_rate.py

echo "== e2_diversity =="
python experiments/e2_diversity.py

echo "== e3_capacity_diversity =="
python experiments/e3_capacity_diversity.py

echo "== e4_dtt =="
python experiments/e4_dtt.py

echo "== figures =="
python figures/make_figures.py

echo "== gate1 report =="
python scripts/make_gate1_report.py

echo "== phase 3a report =="
python scripts/make_phase3a_report.py

echo "done."
