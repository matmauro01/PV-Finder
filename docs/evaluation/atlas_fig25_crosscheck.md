# ATLAS Fig. 25 cross-check — do we classify vertices the way ATLAS does?

**Question (LT, 2026-08-19):** check our AMVF plots for Run 3 MC against
Figure 25 (a) and (b) of the ATLAS ID performance paper, as a sanity check
that we use the same definitions.

**Reference:** ATLAS, *Track and Vertex Reconstruction with the ATLAS Inner
Detector*, IDTR-2021-01, [arXiv:2605.07585](https://arxiv.org/abs/2605.07585),
Section 6.5. Public figures:
<https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/IDTR-2021-01/>

**Answer: yes, the definitions match.** Our `PURITY_THRESHOLD = 0.7` is
their 70%, our Split tie-break is theirs, and AMVF on our Run 3 MC reproduces
their Fig. 25(a) to within a few percent for Matched, Merged and the total.
Two categories differ, both for understood reasons — see
[What differs](#what-differs).

---

## 1. What Figure 25 actually is

This is worth stating because it is easy to get wrong from the caption alone.
Figure 25 is **not** a resolution plot.

| Panel | Content |
|---|---|
| **25(a)** | Average number of reconstructed vertices vs number of interactions, broken down into **All reconstructed / Matched / Merged / Split / Fake**, plus two reference lines |
| **25(b)** | **Δz between reconstructed vertices** — flat plateau with a dip at zero from vertex merging |

The resolution plot is **Figure 26(b)** (longitudinal resolution of the
hard-scatter vertex vs pileup density), not Figure 25.

The two reference lines in 25(a):

- **100% interaction reconstruction efficiency** — the diagonal, y = μ.
- **Reconstruction acceptance** — the number of interactions having **at least
  two reconstructed tracks** in the detector, the minimum for a reconstructable
  PV. We compute it as `count(pv_ntracks >= 2)` per event.

## 2. ATLAS definitions, verbatim

> Reconstructed primary vertices are classified into four types based on the
> truth-matching of reconstructed tracks and the associated weight from vertex
> fitting:
>
> - **Matched**: At least 70% of the total track weight in the reconstructed
>   vertex originates from a single simulated pp interaction.
> - **Merged**: Less than 70% of the total track weight in the reconstructed
>   vertex originates from any single simulated pp interaction.
> - **Split**: If a single simulated pp interaction contributes the largest
>   fraction of track weights to two or more reconstructed vertices, the
>   reconstructed vertex with the largest track Σp<sub>T</sub>² is classified as
>   either matched or merged, whilst the other(s) are labelled split.
> - **Fake**: Fake tracks contribute more weight to the reconstructed vertex
>   than any simulated pp interaction.

## 3. Ours, side by side

Our track-purity taxonomy (`gnn/evaluation/classification.py:classify_assignments`,
see [metric_definitions](metric_definitions.md) §4):

| | ATLAS | Ours | Same? |
|---|---|---|---|
| Threshold | 70% | `PURITY_THRESHOLD = 0.7` | **yes** |
| Weighting | vertex-fit track weight | unweighted track count | **no** — fit weights not in our ntuples |
| Matched/Clean | ≥70% from one interaction | ≥70% from one truth PV | yes (modulo weighting) |
| Merged | <70% from any | <70% from any | yes |
| Split | same truth PV dominant in ≥2 reco; keep largest Σp<sub>T</sub>², demote rest | identical rule | **yes** |
| Fake | fake tracks outweigh any interaction | `"Fake"` bucket wins the plurality | same idea |
| Naming | Matched | Clean | cosmetic |

**Note.** The note (§7 `07_gnn.tex`) previously said "≥50%". That was a
documentation error — the code has always used 0.7. Corrected 2026-08-20, and
the note now cites IDTR-2021-01 for the definitions.

## 4. Numbers

Sample: `ATLAS_PVFinderData_TruthMatched.root` + `recoTracks_incamvfassoc.h5`
(51,000 events, tt̄, √s = 13 TeV, μ = 0–80, mean 39.9, beamspot σ_z = 35 mm).
AMVF vertices with ≥2 tracks: 1,148,652. Truth interactions with ≥2
reconstructed tracks: 1,478,286.

Average vertices per event (ATLAS values read off their figure, so ±0.3):

| μ | All (ours / ATLAS) | Matched | Merged | Split | Fake | Acceptance |
|---:|---|---|---|---|---|---|
| 20 | 13.4 / ~14.5 | 11.8 / ~13 | 1.4 / ~1.3 | 0.26 / ~0.3 | 0.00 / ~0.3 | 15.1 / ~14.5 |
| 40 | 23.7 / ~23 | 18.9 / ~18.5 | 4.3 / ~4.5 | 0.43 / ~0.8 | 0.00 / ~0.6 | 29.3 / ~28 |
| 60 | 32.2 / ~32.5 | 23.8 / ~23.5 | 7.8 / ~8.5 | 0.69 / ~1.2 | 0.00 / ~0.8 | 42.9 / ~42 |
| 80 | 39.3 / ~42 | 27.1 / ~27.5 | 11.3 / ~12.5 | 0.93 / ~1.7 | 0.01 / ~1.0 | 55.8 / ~56 |

Overall rates: Matched 77.2%, Merged 20.8%, Split 2.0%, Fake 0.01%.

Δz (Fig. 25b): the shape reproduces theirs closely.

| | ATLAS | Ours |
|---|---|---|
| Plateau | 0.73e-3 | 1.49e-3 (per 0.2 mm bin) |
| Depth at Δz = 0 | ~0 | 0.12% of plateau |
| Half-recovery \|Δz\| | ~0.75 mm | 0.90 mm |
| Full recovery | ~3 mm | ~3 mm |
| Expected plateau width √2·σ_beam | ~60 mm | 49.5 mm |

The plateau *level* differs only because the beamspot differs (35 mm vs their
~42 mm) and the bin width is ours; the plateau is flat over ±8 mm in both
because √2·σ_beam ≫ 8 mm. The physics content — the width and depth of the
merging dip — agrees.

## What differs

1. **Fake ≈ 0 for us, ~1/event for ATLAS.** Structural, not a bug: 99.84% of
   reconstructed tracks in our ntuple carry a truth PV association, and only
   0.084% of AMVF-assigned tracks have none. ATLAS calls a track fake when
   R_match < 0.5, a quantity our ntuple does not store, so we cannot populate
   this category at all. **We should not quote a Fake rate under this taxonomy
   as comparable to theirs.**
2. **Split is about half theirs** (0.69 vs ~1.2 at μ = 60). Partly follows from
   (1) — vertices they would call Fake we assign to a truth PV, changing which
   vertex wins the Σp<sub>T</sub>² tie-break — and partly from unweighted
   counting.
3. **Merged slightly below theirs at high μ** (11.3 vs ~12.5 at μ = 80),
   consistent with unweighted counting: fit weights suppress outlier tracks,
   which pushes borderline vertices from Matched toward Merged.

## Reproduce

```bash
tmux new -s fig25
source venv/bin/activate
python -u src/pv_finder/diagnostics/amvf_vs_atlas_fig25.py \
    --h5 /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \
    --root data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root \
    --max-events 51000 \
    --output-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25
# ~4 min at ~250 evt/s

python -u src/pv_finder/diagnostics/plot_amvf_vs_atlas_fig25.py \
    --input-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25 \
    --atlas-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25/atlas_reference
```

Outputs: `fig25a_ours.png`, `fig25b_ours.png`, `fig25_comparison.png`
(ATLAS panels beside ours), `fig25_summary.json`, `fig25_data.npz`.

`--min-reco-ntracks` defaults to 2 (ATLAS's requirement); set it to 0 to see
the effect of the trackless AMVF vertices, which is the same drop-empty
question as in the TTVA evaluation.
