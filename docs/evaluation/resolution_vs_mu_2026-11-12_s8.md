# HL-LHC resolution versus pile-up, frozen density model (s8), 12 November 2026

Rerun of the 11 November study (`resolution_vs_mu_2026-11-11.md`, v6) at the
frozen HL-LHC working point, on the same events. Method, binning and the
physics discussion are unchanged; only the model and its cuts differ.

## Configuration

- **Working point:** `get_regime("hllhc")` with no overrides
  (`results.json`: `regime_overrides` empty, `at_frozen_working_point` true).
  - Density model s8, `hllhc_alleta_v6_s8_1p1ep_phase2_step_450000_fullstate.pth`
    (sha256 29e11c4b...), v2, UNet 80 channels, 1 latent, hidden 128x5.
  - Peaks: amplitude 0.01, width 3, integral 0.184, height 0.0184, centroid ±3.
    sigma_vtx-vtx is fitted on the counted peaks, 300 bins over ±6 mm.
- **Events:** the v6 `selection.json`, copied byte for byte (sha256 d321ef09...).
  This gives the same 44,999 events in every bin, with identical ids, mu, truth
  and AMVF lists (asserted).
- **Fix:** `scripts/resolution_vs_mu.py` used to take its cuts from
  `HLLHC_PEAKS`, the old v6 set (0.30 / 0.03). The model, its architecture and
  the cuts now default to `get_regime("hllhc")`. Explicit flags override them
  and are recorded. `results.json` records the git hash, the model sha256 and
  `get_regime("hllhc").to_dict()`.
- **Outputs:** `outputs/11_12_2026_output/hllhc_final/resolution_vs_mu/`
  (README.md there has the exact commands).

## Results (s8 and AMVF)

Errors are event bootstraps (200 resamples).

| mu | events | PVF sigma [um] | AMVF sigma [um] | PVF chi2/ndf | AMVF chi2/ndf | merge length PVF / AMVF [mm] | PVF p68 [um] | AMVF p68 [um] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-10 | 9999 | fit fails (see below) | 139.0 ± 6.8 (4.9%) | - | 56.9 | -8.478 / -8.466 | 33.7 ± 0.3 | 32.1 ± 0.3 |
| 10-20 | 10000 | 101.2 ± 0.3 (0.3%) | 160.0 ± 3.1 (1.9%) | 31.5 | 31.9 | -2.413 / -2.577 | 43.5 ± 0.2 | 42.6 ± 0.3 |
| 20-30 | 6000 | 101.9 ± 0.5 (0.5%) | 182.5 ± 2.8 (1.5%) | 14.4 | 16.0 | -1.136 / -1.285 | 45.9 ± 0.3 | 45.4 ± 0.3 |
| 30-40 | 3000 | 117.8 ± 3.0 (2.6%) | 197.5 ± 2.9 (1.5%) | 6.6 | 7.2 | -0.698 / -0.788 | 46.8 ± 0.3 | 47.2 ± 0.3 |
| 40-60 | 2000 | 141.1 ± 4.0 (2.9%) | 224.6 ± 3.5 (1.5%) | 5.7 | 4.0 | -0.327 / -0.336 | 48.5 ± 0.4 | 48.8 ± 0.4 |
| 60-80 | 2000 | 169.4 ± 2.3 (1.4%) | 244.3 ± 2.8 (1.1%) | 5.6 | 3.3 | -0.134 / -0.106 | 49.3 ± 0.3 | 49.9 ± 0.3 |
| 80-100 | 2000 | 184.4 ± 2.2 (1.2%) | 263.6 ± 2.4 (0.9%) | 5.4 | 5.7 | -0.037 / -0.002 | 50.2 ± 0.3 | 51.2 ± 0.3 |
| 100-120 | 2000 | 196.3 ± 2.2 (1.1%) | 275.1 ± 2.1 (0.7%) | 6.9 | 8.4 | +0.036 / +0.086 | 51.1 ± 0.2 | 52.1 ± 0.2 |
| 120-140 | 2000 | 207.0 ± 2.1 (1.0%) | 287.8 ± 1.9 (0.6%) | 14.9 | 14.4 | +0.078 / +0.149 | 52.0 ± 0.2 | 53.1 ± 0.3 |
| 140-160 | 2000 | 209.5 ± 1.9 (0.9%) | 294.4 ± 1.8 (0.6%) | 16.0 | 12.9 | +0.119 / +0.186 | 53.4 ± 0.2 | 54.8 ± 0.2 |
| 160-180 | 2000 | 213.9 ± 1.6 (0.8%) | 298.6 ± 1.7 (0.6%) | 23.7 | 15.9 | +0.128 / +0.205 | 54.4 ± 0.2 | 55.4 ± 0.2 |
| 180-200 | 2000 | 216.9 ± 1.7 (0.8%) | 304.3 ± 1.6 (0.5%) | 29.9 | 29.0 | +0.143 / +0.230 | 55.5 ± 0.2 | 56.5 ± 0.2 |

- **Quoting rule holds:** the merge length is >= 0 exactly for mu >= 100, for
  both algorithms, as in the v6 study. sigma_vtx-vtx is quoted only there; the
  plot draws the lower bins with open markers.
- **68% residual:** PV-Finder is better than AMVF for mu >= 30, by up to
  1.4 um. It is worse at mu < 30, by 1.6 / 0.9 / 0.5 um at mu 0-10 / 10-20 / 20-30.
- **Check against the PU200 headline:** the 180-200 bin (216.9 ± 1.7 um,
  55.45 um) agrees with the 1,920-event headline at cuts 0.182
  (217.2 ± 3.4 um, 55.60 um; `model_freeze_candidates_2026-11-12.md`).
- **AMVF:** identical to the v6 study. All 132 AMVF values (11 fields × 12
  bins, errors and fit parameters included) are equal bit for bit (asserted).

## s8 minus v6, same events

Paired event bootstrap (200 resamples, the same resample applied to both
models).

| mu | sigma v6 | sigma s8 | d sigma [um] | p68 v6 | p68 s8 | d p68 [um] |
|---|---:|---:|---:|---:|---:|---:|
| 0-10 | 58.3 | fit fails | - | 33.66 | 33.66 | -0.00 ± 0.21 |
| 10-20 | 101.2 | 101.2 | +0.1 ± 0.5 | 43.18 | 43.54 | +0.35 ± 0.16 |
| 20-30 | 111.1 | 101.9 | -9.3 ± 3.7 | 45.57 | 45.87 | +0.30 ± 0.15 |
| 30-40 | 122.0 | 117.8 | -4.2 ± 3.5 | 46.73 | 46.79 | +0.06 ± 0.21 |
| 40-60 | 142.5 | 141.1 | -1.3 ± 4.3 | 48.21 | 48.47 | +0.25 ± 0.22 |
| 60-80 | 167.2 | 169.4 | +2.2 ± 2.8 | 48.84 | 49.27 | +0.43 ± 0.18 |
| 80-100 | 179.5 | 184.4 | +4.9 ± 2.2 | 49.70 | 50.21 | +0.50 ± 0.19 |
| 100-120 | 194.8 | 196.3 | +1.5 ± 2.0 | 50.52 | 51.07 | +0.55 ± 0.15 |
| 120-140 | 204.5 | 207.0 | +2.6 ± 2.1 | 51.73 | 52.03 | +0.30 ± 0.17 |
| 140-160 | 206.5 | 209.5 | +3.1 ± 1.8 | 53.05 | 53.40 | +0.35 ± 0.16 |
| 160-180 | 210.6 | 213.9 | +3.4 ± 1.5 | 53.62 | 54.41 | +0.80 ± 0.16 |
| 180-200 | 216.2 | 216.9 | +0.7 ± 1.7 | 54.92 | 55.45 | +0.53 ± 0.15 |

- **p68:** s8 is 0.3-0.8 um (about 1%) worse than v6 at every mu >= 10. That
  is 2-5 standard deviations per bin. At mu 0-10 the two models are equal. This
  matches the freeze study (+0.55 ± 0.17 um at PU200).
- **sigma at mu >= 100:** s8 is +0.7 to +3.4 um above v6, each bin within
  about 2 standard deviations, about +2 um on average. This is compatible with
  the freeze study (-0.9 ± 1.5 / +3.2 ± 1.6 um on the two files).
- **Vertices per event:** within about 1% of v6 in every bin.

## Odd things

- **mu 0-10, s8 sigma fit fails.** curve_fit exceeds its 20,000 evaluations
  from the production starting point, so results.json has NaN; the bootstrap
  error (5.7 um) comes from the resamples that converged.
  - Other starting points give 63 or 84 um, so the value depends on the start.
  - The pairwise counts at 0.2 < |dz| < 2 mm are 5 times the plateau, so the
    sigmoid does not describe the bin.
  - v6 converges (58-59 um from every start) because it has 5 times more pairs
    at |dz| ≈ 0.06 mm (95 vs 20), which pins a narrow notch.
  - The bin is not quoted; the production fit is left unchanged.
- **mu 10-30, s8 sigma is flat:** 101.2 and 101.9 um, while v6 jumps to 111.1
  at 20-30 (v6 had a 3.3% error there). Both sit in the negative-merge-length
  region, where the fit measures the notch inside the close-pair peak.
- **chi2/ndf of the s8 fit at high mu** is larger than v6: 14.9-29.9 vs
  6.6-19.0 for mu >= 120. The s8 pairwise distribution deviates more from the
  sigmoid, probably through more shoulder structure beside the dip. The quoted
  errors are statistical only.
- **The fit_err of the 10-20 bin is degenerate** (covariance ~4.6 m). The
  bootstrap error is the one quoted.

## Code

- `scripts/resolution_vs_mu.py`: regime defaults with recorded overrides,
  `--reuse-selection`, provenance (git, model sha256, selection sha256,
  regime). `load_chunk` reads each npz array once; it used to re-read the
  whole array per event; peak finding took over 25 minutes without
  finishing, against 9.5 minutes after the fix (identical results).
- `scripts/resolution_vs_mu_compare.py`: identical-event and AMVF-identity
  asserts, reproduction check, paired per-bin differences.
- `scripts/resolution_vs_mu_plots.py`: paper plot `resolution_vs_mu_p68`; the
  two-panel figure has p68 on top and sigma below. Open sigma markers are
  bins with mu < 100 (the quoting rule), with a printed note if the
  merge-length sign disagrees.
- `src/pv_finder/diagnostics/resolution_vs_mu.py`: `paired_bin_diff`,
  `markdown_table`. Tests in `tests/test_resolution_vs_mu.py`.
