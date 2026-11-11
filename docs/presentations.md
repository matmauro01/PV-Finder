# Presentations

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
