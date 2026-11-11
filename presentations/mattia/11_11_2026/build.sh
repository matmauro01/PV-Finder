#!/usr/bin/env bash
# Build the November 11 additional-studies deck (TL2017 pdflatex, as v4).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .build
for pass in 1 2; do
    pdflatex -halt-on-error -interaction=nonstopmode -output-directory=.build \
        -jobname=slides slides.tex >.build/slides.stdout 2>&1 || { tail -40 .build/slides.stdout; exit 1; }
done
cp .build/slides.pdf PVFinder_additional_studies_2026-11-11.pdf
grep -E 'Overfull|undefined' .build/slides.log || true
pdfinfo PVFinder_additional_studies_2026-11-11.pdf | grep '^Pages:'
