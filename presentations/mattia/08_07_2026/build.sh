#!/usr/bin/env bash
# Build both versions of the deck from the single source, slides.tex.
#
#   slides.pdf       reading version, every explanation kept
#   slides_talk.pdf  presentation version, dense prose suppressed
#
# The only difference is the \talkbuild flag. Anything wrapped in
# \readingonly{...} is dropped from the talk build; anything in \talkonly{...}
# appears only there. Numbers live in one place and cannot drift apart.
#
# Two passes each, because the "More in the Backup" slide uses \pageref.
set -eu
cd "$(dirname "$0")"

echo "== reading version =="
pdflatex -interaction=nonstopmode slides.tex >/dev/null
pdflatex -interaction=nonstopmode slides.tex >/dev/null

echo "== talk version =="
pdflatex -interaction=nonstopmode -jobname=slides_talk \
    "\def\talkbuild{1}\input{slides.tex}" >/dev/null
pdflatex -interaction=nonstopmode -jobname=slides_talk \
    "\def\talkbuild{1}\input{slides.tex}" >/dev/null

for j in slides slides_talk; do
    printf '%-12s %s pages, %s overfull, %s underfull\n' "$j" \
        "$(pdfinfo $j.pdf | awk '/^Pages/{print $2}')" \
        "$(grep -c Overfull $j.log || true)" \
        "$(grep -c Underfull $j.log || true)"
done
