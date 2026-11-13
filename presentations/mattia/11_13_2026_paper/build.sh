#!/usr/bin/env bash
# Build the paper deck: sync the figures listed in images.txt, then two
# pdflatex passes into .build/, and copy the PDF here.
#
# Figures: for each line of images.txt the first existing source (PDF
# preferred, else PNG) is copied to images/<name>.<ext>. images_manifest.tex
# records which figures exist, so slides.tex draws a framed placeholder with
# the expected path for the others. Re-run after the recompute agents finish.
#
# Numbers: every number is a macro in numbers.tex; red TBD marks the ones
# still to be filled (each with its source in a comment).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cd "$here"
engine="${PDFLATEX:-pdflatex}"
name=PVFinder_paper_deck_2026-11-13

tex_path() { sed -e 's#_#\\_\\allowbreak{}#g' -e 's#/#/\\allowbreak{}#g' <<<"$1"; }

manifest=images_manifest.tex
: >"$manifest.tmp"
missing=0
while read -r img srcs; do
    [[ -z "$img" || "$img" == \#* ]] && continue
    found=""
    for src in $srcs; do
        for ext in pdf png; do
            if [[ -f "$repo/$src.$ext" ]]; then found="$src.$ext"; break 2; fi
        done
    done
    first=$(awk '{print $1}' <<<"$srcs")
    if [[ -n "$found" ]]; then
        ext="${found##*.}"
        rm -f "images/$img.pdf" "images/$img.png"
        cp "$repo/$found" "images/$img.$ext"
        printf '\\imgsrc{%s}{%s}{%s}\n' "$img" "$(tex_path "$found")" "$ext" >>"$manifest.tmp"
    else
        kept=""
        for ext in pdf png; do
            [[ -f "images/$img.$ext" ]] && { kept=$ext; break; }
        done
        if [[ -n "$kept" ]]; then
            # source not readable here (e.g. a clone without outputs/): keep the committed copy
            printf '\\imgsrc{%s}{%s}{%s}\n' "$img" "$(tex_path "$first")" "$kept" >>"$manifest.tmp"
            echo "kept committed copy: $img.$kept"
        else
            printf '\\imgsrc{%s}{%s}{}\n' "$img" "$(tex_path "$first")" >>"$manifest.tmp"
            echo "placeholder: $img  (expected $first.{pdf,png})"
            missing=$((missing + 1))
        fi
    fi
done <images.txt
mv "$manifest.tmp" "$manifest"

mkdir -p .build
for pass in 1 2; do
    if ! "$engine" -halt-on-error -interaction=nonstopmode -output-directory=.build \
        -jobname="$name" slides.tex >".build/$name.stdout" 2>&1; then
        tail -40 ".build/$name.stdout"; exit 1
    fi
done
cp ".build/$name.pdf" "$name.pdf"
if grep -E 'Overfull|undefined references|Undefined control' ".build/$name.log"; then exit 1; fi
echo "figures still placeholders: $missing"
echo "TBD macros: $(grep -c '{\\TBD}' numbers.tex)"
pdfinfo "$name.pdf" | grep '^Pages:'
