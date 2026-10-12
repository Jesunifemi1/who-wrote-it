#!/usr/bin/env bash
# Rebuild every derived artefact from the per-run result files: summary, figures, PDF report.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f results/models/cnn_output_s0.pt ] && python src/cnn_filters.py
python src/make_figures.py
python report/build_report.py
