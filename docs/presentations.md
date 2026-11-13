# Presentations

The November 13, 2026 **paper deck** for the colleague finishing the PV-Finder
paper draft is
[`presentations/mattia/11_13_2026_paper`](../presentations/mattia/11_13_2026_paper/slides.tex)
(Philadelphia theme, preamble copied from the EB deck). It has seven parts:
the frozen configuration of both regimes, the datasets, the metric
definitions with their code, answers to the 29 questions of the 11 November
draft (page and line), answers to Mattia's own questions, every paper figure
and number (main and backup marked), and placeholders for the internal-note
edits (`noteedits.tex`).
- Every number is a macro in `numbers.tex`. Numbers from the final recompute
  that are not in yet are `\TBD` (red), each with its source doc and JSON key
  in a comment, so the fill is mechanical.
- Figures are listed in `images.txt` (name, expected source path, fallbacks).
  `build.sh` copies each existing source into `images/` and writes
  `images_manifest.tex`; a missing figure is drawn as a framed placeholder
  naming its expected path. Re-running `bash build.sh` after the recompute picks
  up the final figures. The build fails on any overfull box.
- The low pile-up pile-up distributions come from
  `scripts/lowpu_mu_distributions.py` (outputs in
  `outputs/11_12_2026_output/paper_deck/lowpu_mu/`).

The November 11, 2026 deck on the reviewer-requested additional studies is
[`presentations/mattia/11_11_2026`](../presentations/mattia/11_11_2026/slides.tex)
(16 pages: overview, one slide per study, next steps). Build with
`bash presentations/mattia/11_11_2026/build.sh`; it writes
`PVFinder_additional_studies_2026-11-11.pdf`. Images come from
`outputs/11_09_2026_output/` and are copied into its `images/` folder locally.

The latest Mattia deck is
[`presentations/mattia/09_14_2026`](../presentations/mattia/09_14_2026/README.md).
It refreshes the August 7 deck for a September 14, 2026 presentation, retaining
its visual theme and section order with less history and more emphasis on the
method and results.

One source produces a lean presentation PDF (`slides_talk.pdf`) and an expanded
reading PDF (`slides.pdf`). Both have matching slide order and backup material.
The build also produces clearly named `PVFinder_2026-09-14_Presentation_FINAL.pdf`
and `PVFinder_2026-09-14_Reading_FINAL.pdf` copies in the same directory. These
56-page PDFs have no review annotations; `slides_talk_comments.pdf` is the
preserved 54-page annotated original.

The [fact-check record](../presentations/mattia/09_14_2026/FACT_CHECK.md)
documents corrections taken from the current canonical technical note,
including its September 7 local revisions. In particular, use it before
reusing old slides about timing, matching windows, GNN diagnostic bounds,
independent samples, low-pile-up AMVF categories, or fake-filter performance.
Some older wiki entries retain superseded numbers and conventions.

The September 10 [professor-comment review](../presentations/mattia/09_14_2026/COMMENTS_REVIEW.md)
records all 45 annotation objects and the targeted changes applied to both decks.
Original main-slide and backup numbering is retained, with two diagnostic
backups appended (56 PDF pages total). New statistical checks and signed-residual
fits use saved evaluations; no model inference was rerun. All 45 annotations
are addressed, including the full AMVF reclassification at 50% purity on the
same 1,920 events: 60.17% Clean/reco and 52.91% Clean/truth, after Fake/Split
rules. The saved 70% baseline is reproduced event by event. Slide 33 gives
the purity-cut caveat and backup slide 56 shows both cuts with statistical
errors; [AMVF_PURITY_RESULTS.json](../presentations/mattia/09_14_2026/AMVF_PURITY_RESULTS.json)
records the results and source hashes. The Athena timing and component
validation status follow Mattia's explicit update.
