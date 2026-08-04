# The pairwise-Δz bump — a per-peak satellite artefact

Status: **the bump is dominated by surplus (unmatched) peaks clustered around
real vertices.** It is *not* evidence of PV-Finder resolving genuine close
pairs. An earlier version of this page claimed the opposite; see
[What this page got wrong](#what-this-page-got-wrong).

Primary evidence: `outputs/08_01_2026_output/bump_study/` (2026-08-01).
Supporting three-way control: `src/pv_finder/diagnostics/pairwise_dz_comparison.py`,
`outputs/08_04_2026_output/pairwise_dz/`.

## The observation

At the 240-bin binning now used for the σ_vtx-vtx fit, the pairwise-Δz
distribution of reconstructed PVs is not a clean sigmoid: immediately outside
the resolution dip there is a **+14 % excess at |Δz| ≈ 0.5 mm**, decaying by
about 1 mm. At the old 60-bin binning it was smeared away and invisible.

## What it is not

Three-way control on the same 1900 μ-matched held-out events (⟨μ⟩ = 192.6):

| | plateau | dip depth | peak excess | at \|Δz\| | band 0.3–0.7 mm |
|---|---|---|---|---|---|
| **Truth** (nTrk≥2) | 6665 | +0.5 % | 3.3 % | 0.57 mm | **+0.5 %** |
| **AMVF** | 4941 | −99.8 % | 8.9 % | 0.92 mm | **−14.5 %** |
| **PV-Finder** | 5760 | −99.6 % | 13.7 % | 0.57 mm | **+7.6 %** |

- **Not the sample.** Truth vertex spacing is flat to ±0.5 %. There is no
  physical excess of truth vertices at these separations.
- **Not unique to PV-Finder.** AMVF shows the same structure, displaced to
  ~0.9 mm, i.e. just outside *its* (wider) resolution dip.

## What it is

Three independent measurements, all from the 2026-08-01 study, identify it as
**satellite peaks**: surplus reconstructed vertices sitting a characteristic
distance from a real one.

**1. The excess lives entirely in pairs containing an unmatched peak.**
Splitting the distribution by whether both members are truth-matched
(`bump_decomposition.png`, greedy 1-to-1 matching, 0.5 mm window):

| pair class | PV-Finder peak | AMVF peak |
|---|---|---|
| both vertices truth-matched | **1.06** (≈ flat) | 1.04 |
| ≥1 vertex is a surplus peak | **1.56** at 0.45 mm | 1.25 at 1.0 mm |
| all pairs | 1.17 | 1.08 |

Pairs of two genuine vertices show essentially **no** excess. Remove the
unmatched peaks and the bump disappears.

**2. Surplus peaks cluster around real ones; matched peaks do not.**
Cross-correlating each peak against the nearest truth-matched reconstructed
vertex (`satellite_crosscorrelation.png`): surplus-peak density reaches
**1.54× baseline at 0.35–0.5 mm** for PV-Finder and ~1.3× at ~0.95 mm for AMVF,
while truth-matched peaks stay flat at 1.0 for both.

**3. The excess scales as 1/n, not as a constant.**
(`excess_vs_pileup_and_floor.png`, middle panel.) Genuine two-vertex physics
would give an excess *fraction* independent of the number of reconstructed
vertices n, because signal and baseline pairs both scale as n². The measured
excess instead follows **∝ 1/n** across n ≈ 10–100, the signature of a
**per-peak** effect: each real peak spawns a roughly fixed number of
satellites, so satellite pairs scale as n while total pairs scale as n². The
left panel shows the same as a function of pile-up (excess 4.8× at μ = 1–25
falling to 1.2× at μ = 185–215).

## What about the outward bias on close pairs?

It is real, and it was measured (`pair_separation_repulsion.png`): for truth
pairs where both vertices are found, the reconstructed separation is
systematically larger than the true one below ~0.6 mm (PV-Finder) and ~1.0 mm
(AMVF). The fraction with Δz_reco > Δz_truth peaks at **0.86 at a true
separation of ~0.15–0.2 mm**, falling to 0.5 by ~0.7 mm.

But this **moves pairs, it does not create them**. Redistribution alone cannot
lift the distribution above its own large-|Δz| plateau. The repulsion explains
the *shape* of the dip edge; it does not explain the excess.

## Operating-point consequence

A minimum-height floor suppresses the excess **2.5× faster than the baseline**
(`excess_vs_pileup_and_floor.png`, right panel): the excess/baseline integral
falls 5.23 → 4.66 → 3.39 mm for floors 0.0 → 0.03 → 0.05. Satellites are
systematically lower-amplitude than real peaks — the same property the height
floor exploits for fake suppression.

The bump is therefore a visible, quantitative handle on the surplus-peak
population, and an argument for a non-zero floor at the production operating
point.

## What this page got wrong

The first version of this page (2026-08-04, commit `b4a508e`) concluded the
bump was "PV-Finder resolving pairs AMVF merges, not a pathology", from the
truth/AMVF control plus PV-Finder sitting at +7.6 % in the 0.3–0.7 mm band
where AMVF sits at −14.5 %. That control is sound and is kept above, but the
conclusion drawn from it was not: it never tested whether the excess pairs
contain *real* vertices. They largely do not.

The decomposition, cross-correlation and 1/n scaling all **predate** that claim
in the repository — the 2026-08-01 study had already settled the question and
was not consulted before writing.

The narrower claim that does survive: PV-Finder's *dip* is genuinely narrower
than AMVF's (recovering by ~0.35 mm vs ~0.85 mm), consistent with σ_vtx-vtx of
0.224 mm vs 0.284 mm. That is a real two-vertex resolution advantage — it is
just not what the bump measures.

## Relation to the earlier study

[resolution_bump_analysis](resolution_bump_analysis.md) (2026-04-23, old data)
classified close reco pairs as 84.6 % "genuine". That fraction is over all
pairs within 2 mm, dominated by the flat baseline, so it does not conflict with
the excess being satellite-driven: the excess is a small perturbation on a
large baseline. Read it as a statement about the bulk, not about the bump.

## Reproducing

```bash
python -u src/pv_finder/diagnostics/pairwise_dz_comparison.py \
    --pkl outputs/<date>/eval_.../eval_results.pkl \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --mu-min 185 --mu-max 215 --n-events 1900 \
    --output-dir outputs/<date>/pairwise_dz
```

> **Pitfall.** `eval_results.pkl` stores every event the eval *read*, not the
> μ-window subset it summarises. On a flat-μ held-out file only ~8 % of events
> are PU200-like, so slicing the first N compares PV-Finder at ⟨μ⟩≈100 against
> AMVF/truth at ⟨μ⟩≈192. The tool selects on the stored per-event μ.
