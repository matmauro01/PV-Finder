# The pairwise-Δz bump — what it is, and why it is not a defect

Investigated 2026-08-04 on the extended-|η| PU200 sample. Tool:
`src/pv_finder/diagnostics/pairwise_dz_comparison.py`. Outputs in
`outputs/08_04_2026_output/pairwise_dz/`.

## The observation

At the 240-bin binning now used for the σ_vtx-vtx fit (see
[evaluation/vertex_finding](../evaluation/vertex_finding.md)), the pairwise-Δz
distribution of reconstructed PVs is not a clean sigmoid. Immediately outside
the resolution dip there is a **+14 % excess at |Δz| ≈ 0.57 mm**, decaying by
about 1 mm. At the old 60-bin binning this was smeared away and invisible.

The excess is real. Across models (**cross-sample — different files, pile-up
and training states, so read the trend, not the differences**):

| model / data | peak excess | at \|Δz\| | mean excess 0.3–0.7 mm |
|---|---|---|---|
| v4b, old \|η\|<2.5, r16438 μ=200 | 8.8 % | 0.42 mm | 1.9 % |
| v5, old data, corrected widths | 16.5 % | 0.48 mm | 9.5 % |
| new all-η, held-out r16443 μ=192.6 | 13.7 % | 0.57 mm | 7.6 % |

The jump from v4b to v5 is the cleanest of these — same file, same events, only
the target widths differ.

> **Pitfall.** `eval_results.pkl` stores every event the eval *read*, not the
> subset it summarises. On a flat-μ held-out file only ~8 % of events are
> PU200-like, so slicing the first N events compares PV-Finder at ⟨μ⟩≈100
> against AMVF/truth at ⟨μ⟩≈192 — four times fewer pairs, a much lower plateau
> and far noisier curves. `pairwise_dz_comparison.py` now selects on the stored
> per-event μ. An earlier version of this page quoted 20.9 % from that mistake.

## The three candidate explanations, and the controls that separate them

Measured on the **same 1900 μ-matched held-out events** (⟨μ⟩ = 192.6):

| | plateau | dip depth | peak excess | at \|Δz\| | band 0.3–0.7 mm |
|---|---|---|---|---|---|
| **Truth** (nTrk≥2) | 6665 | +0.5 % | 3.3 % | 0.57 mm | **+0.5 %** |
| **AMVF** | 4941 | −99.8 % | 8.9 % | 0.92 mm | **−14.5 %** |
| **PV-Finder** | 5760 | −99.6 % | 13.7 % | 0.57 mm | **+7.6 %** |

This settles it:

1. **It is not the data.** The truth vertex-spacing distribution is flat to
   within ±1 % across the whole region. There is no physical excess of truth
   vertices at 0.3–0.7 mm to reconstruct.
2. **It is not a PV-Finder pathology.** AMVF shows the same qualitative
   structure — a deep dip followed by an excess above the plateau. AMVF's
   excess peaks at ~0.9 mm rather than ~0.57 mm.
3. **The excess sits immediately outside each algorithm's own resolution
   limit.** PV-Finder's dip recovers to the plateau by ≈0.35 mm and its excess
   peaks at 0.57 mm; AMVF's dip is far wider, recovering only by ≈0.85 mm, and
   its excess peaks at ~0.9 mm.

## Mechanism

A marginally-resolvable pair is only split into two vertices when the density
between the two peaks dips enough for the algorithm to see two objects. When
that happens, each reconstructed position is the amplitude-weighted mean of its
own side of the valley, so both are pulled **outward**, away from the shared
minimum. The reconstructed separation of a marginally-resolved pair is
therefore biased high.

The result is a deficit just below the resolution limit and a matching excess
just above it — exactly the deficit-then-excess shape observed, for both
algorithms, each at its own scale. The same argument explains why the excess
**grew** when the target widths were corrected (v4b → v5 on identical data,
8.8 % → 16.5 %): narrower targets make more pairs marginally resolvable, so
more pairs receive the outward bias.

## Why this is a good sign, not a bad one

In the 0.3–0.7 mm band PV-Finder sits at **+7.6 %** while AMVF sits at
**−14.5 %**. AMVF has essentially no vertex pairs at these separations at all;
PV-Finder does. The bump is the visible signature of PV-Finder resolving pairs
that AMVF merges, which is the same conclusion the σ_vtx-vtx fits reach
independently (0.224 mm vs 0.284 mm).

The one genuine caveat is the **outward position bias on marginally-resolved
pairs**. It is a real systematic on close-pair positions, it is not corrected
anywhere, and it is not currently quantified per-vertex. It does not affect
counting metrics (efficiency, fake rate), only the reconstructed separation of
pairs within ~1 mm.

## Relation to the earlier study

[resolution_bump_analysis](resolution_bump_analysis.md) (2026-04-23, old data)
reached a compatible but incomplete conclusion: it classified close reco pairs
and found them 84.6 % "genuine" (both matched to distinct truth), and noted
that AMVF's bump was 6× larger than PV-Finder's. What it did not have was the
truth control, which is what rules out the vertex-spacing explanation, and it
predated the 240-bin binning, so it under-resolved the shape.

Note that the pair-classification numbers in that study are sensitive to the
matching window: with a 0.22–0.24 mm window, a pair separated by 0.3–0.7 mm
can have one member fall outside the window and be counted "unmatched" even
when it is a real vertex. Classification fractions from that method should be
read as indicative, not exact.

## Reproducing

```bash
python -u src/pv_finder/diagnostics/pairwise_dz_comparison.py \
    --pkl outputs/<date>/eval_.../eval_results.pkl \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --mu-min 185 --mu-max 215 --n-events 1500 \
    --output-dir outputs/<date>/pairwise_dz
```
