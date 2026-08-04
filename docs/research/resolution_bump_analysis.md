# Resolution Bump Analysis (2026-04-23): SUPERSEDED

**Do not cite this page.** Its conclusions about the pairwise-Δz bump are wrong.
The current account is [pairwise_dz_bump](pairwise_dz_bump.md).

The original text has been removed rather than kept alongside the correct
version, because it was being cited from `docs/evaluation/vertex_finding.md` and
from slide decks and was actively propagating an inverted result.

## What it claimed, and what is actually true

| 2026-04-23 claim | status |
|---|---|
| "AMVF has a 6× larger bump than PV-Finder" | **inverted.** In the 0.3–0.7 mm band PV-Finder is at **+12.5 %** and AMVF at **−14.5 %**. AMVF's excess is a shoulder at 0.9–1.1 mm, not a larger bump. |
| "PV-Finder's bump is mostly noise, only 2 bins above 2σ" | **wrong.** Bootstrap over events puts the band excess at ~16σ. |
| "the bump is 85 % genuine physics, 15 % sidelobes" | **inverted.** Pairs of two truth-matched peaks show a *negative* band excess; the whole excess is carried by surplus (unmatched) peaks. |
| "the bump is the natural consequence of merging unresolvable close vertices, shared by any algorithm" | **wrong mechanism.** Merging redistributes pairs and cannot lift the integral above the plateau. The excess is created by extra peaks. |
| "PV-Finder has better resolution than AMVF" | **survives.** σ_vtx-vtx is still ~20 % better; that part was never in question. |
| NMS is counterproductive at PU200 | **survives qualitatively.** NMS and pre-smoothing remain off by default; the specific numbers are v1-era and are not reproducible on the current sample. |

## Why it got the answer wrong

Three reasons, all worth remembering:

1. **Different model and different data.** v1 end-to-end at epoch 100 on the
   |η| < 2.5 production. Both have since been replaced. The bump is
   model-dependent and data-dependent (see the successor page), so a v1
   measurement was never going to describe v6.
2. **Under-powered.** 2500 events, no bootstrap, significance judged by counting
   bins above 2σ on a distribution whose bins are correlated. The real effect
   was below its sensitivity and was reported as noise.
3. **The decomposition was done on the wrong denominator.** Classifying *close
   pairs* by type answers "what are close pairs made of", which at PU200 is
   dominated by genuine vertices simply because there are so many of them. The
   question that matters is "what is the excess *over the plateau* made of",
   which requires subtracting the combinatorial baseline first. Without that
   subtraction any close-pair census at PU200 reads as "mostly genuine physics".

The surviving quantitative material from this study (truth vertex spacing at
PU200, fake-peak amplitude and width distributions, the NMS trade-off table)
was all measured on the superseded sample. Regenerate before reuse.
