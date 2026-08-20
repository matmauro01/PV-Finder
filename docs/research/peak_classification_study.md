# Peak Classification Study — Post-hoc Fake Rejection

**Current section is the v6 one immediately below.** The v4b and v1 sections are
kept for provenance; both sit on the superseded `|eta|<2.5` pool.

---

## Update — 2026-08-20 (v6 model, extended-|eta|, retrained and redeployed)

Re-measured on the production **v6** model and the extended-|eta| samples,
because the v4b numbers below were trained and validated on files inside the
old `|eta|<2.5` pool.

- **Training candidates**: 340,541 peaks from 6,000 events of `r16443` at
  `--min-height 0.0`, **11.6% fake** (39,546 / 340,541). `r16443` is held out
  from v6 finder training. Artifacts:
  `outputs/08_20_2026_output/gbt_v6_train_r16443/`.
- **AUC**: GBT 23-feature **0.9580**, MLP 0.9550, GBT track-only **0.9455**,
  GBT histogram-only **0.9528**. Model + plots:
  `outputs/08_20_2026_output/peak_classifier_v6/`.
- **Ablation holds, and is stronger than on v4b.** Histogram-only beats
  track-only by 0.007 here (0.9528 vs 0.9455) against 0.002 on v4b, and lands
  within 0.005 of the full 23-feature model. The fake/real signal is in the
  histogram; the tracks were already integrated to build it.
- **Importances**: `local_integral` **0.7368**, `nearest_peak_dz` 0.0638,
  `peak_height` 0.0523, `n_tracks_2mm` 0.0344.
- **Operating points** (test split, full GBT): thr 0.3 → keep 99.0% real /
  remove 30.4% fake; 0.5 → 96.1% / 63.2%; 0.7 → 92.1% / 83.1%. Histogram-only
  at 0.5 → 95.4% / 62.3%.
- **Height floor, measured like-for-like on v6** (same model, same 384
  in-window events of `r16443`, floor 0.0 vs 0.03): the floor removes
  **1.16 fake/event for 0.06 percentage points** of efficiency
  (17.115 → 15.956 fake/evt, 87.10% → 87.04%). Cheaper than the v4b-era
  estimate of "~1 fake/evt for 0.3 pp".
- **Fakes are not tight sidelobes** (re-checked on v6): only **0.4%** lie within
  0.3 mm of a real peak, median distance to the nearest real peak **1.06 mm**.
  But **12.7% are inside 0.6 mm**, and those are the satellites of
  [pairwise_dz_bump](pairwise_dz_bump.md) — the two findings are consistent, the
  satellite band sits at ~0.57 mm, not below 0.3 mm.
- **Deployed** on the second held-out tag `r16638` (independent of both the
  finder training and the classifier training), gate on top of the v6 operating
  point at threshold 0.3: see `outputs/08_20_2026_output/v6_r16638_gbt03/`.
  On the 268 in-window events it removes **1.52 fake/event** (15.993 → 14.478,
  −9.5%) for **0.15 pp** of efficiency (87.29% → 87.14%). σ_vtx-vtx unchanged
  within the fit error (0.2325 ± 0.0060 vs 0.2278 ± 0.0032).
- **The gate adds much less on top of the floor than its ROC suggests.** It
  removes 30% of fakes from the *raw* candidate set but only 9.5% once the
  0.03 height floor has run first, because both read peak amplitude/integral
  and the floor has already taken the easy fakes. Quote the stacked number,
  not the standalone one.
- **Neither handle is in the headline v6 numbers**, which are floor-only.

**Conclusion unchanged and now better supported**: a post-hoc gate re-reads the
histogram the model already produced, so its ceiling is the histogram itself.
The lever that matters is an in-model objectness head or a fake-aware loss.

---

## Update — 2026-06-09 (v4b model, 23 features, deployed gate) — superseded

The May study below used the v1 model and an 18-feature classifier. It was repeated
on the current **v4b** model (`peak_classifier_v2.py`, 23 features adding pT/KDE
kinematics and histogram skewness) and the histogram-only classifier was then wired
into the evaluation as a deployable gate.

- **Dataset**: v4b ep3, 2500 evt on `r16438` (241k peaks, 10.1% fake; real = within
  0.5 mm of a TruthVertex). Validated on the **independent** file `r16633`.
- **AUC**: GBT (23 feat) **0.9651**, MLP 0.9623.
- **Ablation (the key result)**: GBT track-only (feat 0–14) = **0.9569**,
  histogram-only (feat 15–22) = **0.9593**, all = **0.9651**. The two groups are
  largely redundant — each alone ≈ the full AUC — so the fake/real signal is mostly
  **already in the histogram**. Dominant feature: `local_integral` (±0.5 mm windowed
  area) importance **0.72**, then `peak_height` 0.08, `nearest_peak_dz` 0.06.
- **Operating points** (test split): thr 0.3 → keep 99.1% real / remove 38% fake
  (~−0.9 pp eff); thr 0.5 → 96.8% / 64.6% (~−3 pp); thr 0.7 → 93.9% / 81.9%. Better
  fake/efficiency trade than the 0.03 height floor.
- **Deployment**: the histogram-only GBT (`gbt_hist_model`) is loaded in
  `run_eval_pvf_run3.py` via `--gbt-filter-model` / `--gbt-threshold`; its
  `_hist_features` match `peak_classifier_v2` features 15–22 bit-for-bit. On the
  independent file `r16633` (v4b, thr 0.3, `--min-height 0.0`): Eff = 0.927,
  ~11.3 fake/evt, σ_vtx_vtx = 0.282 mm (vs ~14 fake/evt with no gate). Filtered
  peaks are excluded from both the category counts and the σ fit.
- **Implication**: a post-hoc gate re-reads signal the model already produced, so its
  ceiling is the histogram itself. The larger gain is an in-model objectness head /
  fake-aware loss that makes the network emit a cleaner histogram. See JOURNAL
  2026-06-09 and [evaluation/vertex_finding](../evaluation/vertex_finding.md).

---

## Motivation

At PU200, the model produces ~4.8% fake peaks (9,892 / 208,082). These are
small bumps in the predicted histogram that pass peak-finding thresholds but
don't correspond to truth vertices. Can we reject fakes using track information
and histogram shape features, without retraining the model?

## Method

For each predicted peak, we compute 18 features in three categories:

**Track features (0-10)**: n_tracks within 0.5/1/2mm, density contrast
(n_05/n_2), mean z0_err, min z0_err, std of z0, mean |d0|, mean and max d0
significance, precision-weighted track count (sum 1/z0_err).

**Histogram features (11-15)**: peak height, local integral (+-0.5mm),
FWHM, curvature (second derivative), relative height vs background.

**Neighborhood features (16-17)**: distance to nearest other peak, height
ratio to nearest peak.

Peaks within 0.5mm of a truth PV are labeled "real" (198,190), others "fake"
(9,892). 70/30 train/test split, stratified.

## Results

### AUC Comparison

| Model | Features | AUC |
|---|---|---|
| GBT (100 trees, depth 4) | All 18 | **0.9604** |
| MLP (32-16 hidden) | All 18 | 0.9587 |
| GBT | Track only (0-10) | 0.9396 |
| GBT | Histogram only (11-17) | **0.9584** |

**Key finding: Track features add only +0.002 AUC over histogram-only.**

### GBT Feature Importance

| Feature | Importance |
|---|---|
| peak_height | **0.533** |
| local_integral | 0.179 |
| nearest_peak_dz | 0.066 |
| density_contrast | 0.046 |
| min_z0_err | 0.023 |
| fwhm_mm | 0.020 |
| mean_abs_d0 | 0.016 |

Peak height alone accounts for 53% of discriminative power.

### Operating Points

| Filter | Real kept | Fake removed |
|---|---|---|
| height >= 0.05 (deterministic) | 96.1% | 50.1% |
| height >= 0.1 (deterministic) | 83.0% | 94.9% |
| trk >= 3 (0.5mm) | 93.3% | 43.1% |
| GBT thr=0.5 | 99.1% | 29.0% |
| GBT thr=0.7 | 96.6% | 62.7% |

The GBT at threshold 0.7 is the best operating point: keeps 96.6% of real
peaks while removing 62.7% of fakes. However, a simple height threshold of
0.05 achieves similar real-kept rate (96.1%) with 50.1% fake removal —
demonstrating that the ML classifier provides only modest improvement over
a tuned deterministic threshold.

## Why Track Features Don't Help at PU200

At PU200 with ~200 truth PVs and ~700 tracks per event:

1. **Uniform track density**: 700 tracks / 480mm = ~1.5 tracks/mm baseline.
   A +-0.5mm window captures ~1.5 background tracks regardless of position.

2. **Low-multiplicity vertices dominate**: 17% of truth PVs have <=2 reco
   tracks within 0.5mm. These are indistinguishable from fakes by track count.

3. **Track quality is similar**: Both real and fake peaks have tracks with
   similar z0_err and d0 distributions — the nearby tracks exist at both
   positions, they just come from different vertices.

4. **Fake peaks are already well-characterized by shape**: Height alone
   separates most fakes (median height 0.053) from reals (median 0.280).

## Conclusions

1. **Post-hoc track-based classification is NOT the right approach** for
   PU200 fake rejection. The information gain from tracks is negligible
   beyond what histogram shape already provides.

2. **A simple peak height threshold** is nearly optimal for deterministic
   fake rejection.

3. **The real opportunity is in the model itself**: train with a loss that
   penalizes false peaks, or add a confidence head, or use peak-aware
   architecture. The histogram output is where the fakes are created — that's
   where they should be eliminated.

## Output artifacts

- `outputs/05_04_2026_output/peak_classifier_full/` — full 2500-event results
- `feature_distributions.png` — real vs fake distributions for all 18 features
- `classifier_performance.png` — GBT precision-recall and efficiency curves
- `peak_classifier_results.pkl` — all results, AUCs, importances
