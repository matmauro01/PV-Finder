#!/usr/bin/env bash
# Build the EB deck: two pdflatex passes into .build/, copy the PDF here.
# Regenerate the numbers first: python scripts/eb_numbers.py --out numbers.tex
set -euo pipefail
cd "$(dirname "$0")"
engine="${PDFLATEX:-pdflatex}"
name=PVFinder_EB_2026-11-13
mkdir -p .build
for pass in 1 2; do
    if ! "$engine" -halt-on-error -interaction=nonstopmode -output-directory=.build \
        -jobname="$name" slides.tex >".build/$name.stdout" 2>&1; then
        tail -40 ".build/$name.stdout"; exit 1
    fi
done
cp ".build/$name.pdf" "$name.pdf"
if grep -E 'Overfull|undefined references|Undefined control' ".build/$name.log"; then exit 1; fi
pdfinfo "$name.pdf" | grep '^Pages:'
