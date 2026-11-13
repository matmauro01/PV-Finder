# HL-LHC resolution versus pile-up, 11 November 2026

> **Superseded for results by the frozen model (s8):**
> `resolution_vs_mu_2026-11-12_s8.md`. This page keeps the v6 reference and the
> method and physics discussion.

Before the GNN. Two resolution measures in bins of the number of interactions
mu, for PV-Finder and AMVF on the same events, with per-bin statistical errors
capped:

- **sigma_vtx-vtx**: half-width of the dip in the signed pairwise Delta z of
  each algorithm's own vertices. Same sigmoid fit as the production
  evaluation: 300 bins over ±6 mm, `run2_vs_run4.metrics`, verified bit for
  bit against `amvf_fair_comparison.scan.fit_sigma_vtx_vtx`.
- **68% residual**: 68th percentile of |z_reco - z_true| over the optimal
  one-to-one assignment within 1 mm, truth vertices with nTracks >= 2.

Errors are event bootstraps (200 resamples).

## Configuration

- **Density model:** v6 (`hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth`,
  sha256 e0c67038...). Rerunning with the frozen model is one command:
  `python -u scripts/resolution_vs_mu.py all --out DIR --model CKPT --unet-channels N --latent L --device D`
- **Peaks:** the single HL-LHC operating point, `working_points.HLLHC_PEAKS`:
  amplitude 0.01, width 3, integral 0.30, height 0.03, local centroid ±3.
  The sigma fit uses the same peaks as counting; there is no separate 0.50.
- **AMVF:** stored vertices with nTracks >= 2, as in the production evaluation.
- **Samples:** the flat-mu files r16443 (100,000 entries) and r16638 (99,800),
  mu from 0 to 200, about 5,000 events per 10 units of mu per file.
  - The density model was trained on other files (r16438, r16633, 601237) and
    never saw either file, so every entry of both is usable here.
  - The GNN uses r16443 entries >= 25,000 and r16638 for its own training.
    That is irrelevant for this before-GNN study.
- **Selection:** fixed per-bin event counts, chosen before looking at results
  from a pilot on 25,000 cached events. Within each file, entries are taken in
  increasing entry number; nothing depends on the outcome.
  - mu 0-10 and 10-20: 10,000 events each (all events in both files)
  - mu 20-30: 6,000 events
  - mu 30-40: 3,000 events
  - each 20-unit bin above 40: 2,000 events
  - 44,999 events in total; one mu < 10 event has no AMVF vertex and is
    dropped by the standard loader.
- **Reproducibility check:** freshly computed histograms are bit-identical to
  the September cache (`outputs/09_23_2026_output/hs_fullrange_050`) for the
  same events (max |Delta| = 0).
- **Provenance:**
  - Inference was launched at git ee8a127, before the step-1 working-point
    commit 6b6cf7d. The model helpers it uses (`load_ckpt`,
    `build_subevent_inputs`, `run_inference`) are unchanged by step 1, and the
    histograms do not depend on any working point.
  - The analysis was rerun after step 1. `results.json` records its git hash.

## Results

| mu | events | PVF sigma [um] | AMVF sigma [um] | PVF chi2/ndf | AMVF chi2/ndf | merge length PVF / AMVF [mm] | PVF p68 [um] | AMVF p68 [um] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-10 | 9999 | 58.3 ± 1.1 (1.9%) | 139.0 ± 6.8 (4.9%) | 56.8 | 56.9 | -8.748 / -8.466 | 33.7 ± 0.3 | 32.1 ± 0.3 |
| 10-20 | 10000 | 101.2 ± 0.4 (0.4%) | 160.0 ± 3.1 (1.9%) | 34.8 | 31.9 | -2.437 / -2.577 | 43.2 ± 0.2 | 42.6 ± 0.3 |
| 20-30 | 6000 | 111.1 ± 3.7 (3.3%) | 182.5 ± 2.8 (1.5%) | 15.6 | 16.0 | -1.118 / -1.285 | 45.6 ± 0.3 | 45.4 ± 0.3 |
| 30-40 | 3000 | 122.0 ± 2.4 (2.0%) | 197.5 ± 2.9 (1.5%) | 6.6 | 7.2 | -0.712 / -0.788 | 46.7 ± 0.3 | 47.2 ± 0.3 |
| 40-60 | 2000 | 142.5 ± 3.0 (2.1%) | 224.6 ± 3.5 (1.5%) | 4.4 | 4.0 | -0.331 / -0.336 | 48.2 ± 0.3 | 48.8 ± 0.4 |
| 60-80 | 2000 | 167.2 ± 2.6 (1.6%) | 244.3 ± 2.8 (1.1%) | 4.0 | 3.3 | -0.138 / -0.106 | 48.8 ± 0.3 | 49.9 ± 0.3 |
| 80-100 | 2000 | 179.5 ± 2.4 (1.3%) | 263.6 ± 2.4 (0.9%) | 4.9 | 5.7 | -0.069 / -0.002 | 49.7 ± 0.3 | 51.2 ± 0.3 |
| 100-120 | 2000 | 194.8 ± 1.7 (0.8%) | 275.1 ± 2.1 (0.7%) | 5.5 | 8.4 | +0.031 / +0.086 | 50.5 ± 0.3 | 52.1 ± 0.2 |
| 120-140 | 2000 | 204.5 ± 1.9 (0.9%) | 287.8 ± 1.9 (0.6%) | 6.6 | 14.4 | +0.066 / +0.149 | 51.7 ± 0.2 | 53.1 ± 0.3 |
| 140-160 | 2000 | 206.5 ± 1.4 (0.7%) | 294.4 ± 1.8 (0.6%) | 10.8 | 12.9 | +0.111 / +0.186 | 53.0 ± 0.3 | 54.8 ± 0.2 |
| 160-180 | 2000 | 210.6 ± 1.4 (0.7%) | 298.6 ± 1.7 (0.6%) | 9.4 | 15.9 | +0.121 / +0.205 | 53.6 ± 0.2 | 55.4 ± 0.2 |
| 180-200 | 2000 | 216.2 ± 1.6 (0.7%) | 304.3 ± 1.6 (0.5%) | 19.0 | 29.0 | +0.138 / +0.230 | 54.9 ± 0.2 | 56.5 ± 0.2 |

The merge length is the model-free size of the dip: the integral of
(1 - density / plateau) over |Delta z| < 3 mm, with the plateau taken at
3-6 mm. Negative means more close pairs than the plateau predicts.

- **Errors:** the relative statistical error on sigma is ≤ 2.1% in every bin
  except PV-Finder at mu 20-30 (3.3%) and AMVF at mu 0-10 (4.9%), and ≤ 0.9%
  for mu >= 100. The 68% residual has errors ≤ 0.4 um everywhere.
- **Check at mu 180-200:** these bins agree with the PU200 headline on the
  1,920 events at the same operating point (0.218 mm PV-Finder,
  0.305 mm AMVF).

Figures in `outputs/11_11_2026_output/resolution_vs_mu/`:
- `resolution_vs_mu.{png,pdf}`: sigma_vtx-vtx on top, 68% residual below.
- `per_bin_fits.pdf`: one page per bin, both algorithms, data and fit with
  chi2/ndf.

## Physics: where sigma_vtx-vtx means vertex separation

The per-bin fits show a correlated excess of close reconstructed pairs within
about ±2 mm on top of the flat pile-up plateau. At low mu it dominates: a
broad peak with a narrow notch at Delta z = 0. At high mu it remains as small
shoulders beside the dip.

- **It is made by reconstruction.** Truth vertex pairs (nTracks >= 2) are
  flat: close-pair rate over flat expectation 1.01 at mu < 20. PV-Finder (1.61)
  and AMVF (1.60) both show the excess.
- **The HS neighbourhood is the largest source.** Removing every vertex within
  3 mm of the true hard scatter halves the reconstructed pairs at mu < 20 and
  reduces the excess to 1.31 / 1.33. The rest comes from close fake and split
  vertices elsewhere.
- **Why it matters at low mu:** this component is roughly constant per event,
  while random pile-up pairs grow as mu². So it dominates at low mu.
- **Effect on the fit:** the flat-plateau sigmoid cannot describe the excess.
  - chi2/ndf is 4-57 in every bin, rising with the bin statistics. The quoted
    errors are statistical only and do not cover the model uncertainty.
  - Where the merge length is negative (PV-Finder mu < 100, AMVF mu < 100),
    the fitted sigma measures the notch inside the correlated peak, not the
    separation of independent interactions.

Treatment for the paper and note:
1. **The 68% residual is the resolution versus mu.** It is defined at every mu
   and the two algorithms agree within 1.8 um in every bin. Both rise from
   about 33 um (mu < 10) to 55-57 um (mu 180-200).
2. **sigma_vtx-vtx only for mu >= 100,** where the dip dominates (merge
   length > 0 for both algorithms). The plot shows lower bins with open
   markers. In that range AMVF / PV-Finder is constant at 1.41-1.43.
3. **sigma_vtx-vtx is not stable in mu** for either algorithm: it rises from
   195 to 216 um (PV-Finder) and from 275 to 304 um (AMVF) between mu 100 and
   200. The expectation that it would be flat does not hold. Quote it per mu
   bin, or at PU200, not as a single number for all mu.

## Code

- `src/pv_finder/diagnostics/resolution_vs_mu.py`: mu binning (top bin closed),
  outcome-blind event selection, per-bin summary and fit chi2. Tests:
  `tests/test_resolution_vs_mu.py`.
- `scripts/resolution_vs_mu.py`: stages select, infer, analyze, plot, all.
- `scripts/resolution_vs_mu_plots.py`: per-bin fit PDF and summary figure.
