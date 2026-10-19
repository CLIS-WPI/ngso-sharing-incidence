#!/usr/bin/env bash
# Build paper/main.pdf with latexmk (runs pdflatex + bibtex as needed).
# Runs inside the dev container, where the TeX Live toolchain is installed.
set -euo pipefail

cd "$(dirname "$0")/../paper"

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

if grep -q "Output written" main.log; then
  grep "Output written" main.log
fi
echo "[paper] wrote $(pwd)/main.pdf"
