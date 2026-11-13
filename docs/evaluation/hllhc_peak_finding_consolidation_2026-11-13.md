# HL-LHC peak finding: consolidation of the s8 working point (13 November 2026)

Mattia, after the EB talk: the working-point slide had too many parameters;
consolidate them, especially the position estimator, and see whether one cut
is enough. This page answers both, on tuning data only, and proposes a change
to `working_points.py` (not applied).

## Result

| | Current (frozen 12 Nov) | Proposed |
|---|---|---|
| Amplitude ("on" bins) | 0.01 | 0.01 |
| Minimum width | 3 bins | 3 bins |
| Integral | 0.184 | **0.186** |
| Minimum height | 0.0184 | **none** |
| Position | local centroid ±3 bins | local centroid ±3 bins (unchanged) |

- **One cut: the region integral.** The proposal has the structure of the
  low pile-up point (0.01 / 3 / 0.50 / none / ±3): the two regimes differ only
  in the integral value.
- **It loses nothing measurable.** On the report half at PU200, against the
  current cuts (paired):
  - efficiency -0.00 ± 0.00 pt;
  - fakes +0.01 ± 0.01 per event;
  - splits -0.001 ± 0.001 per event;
  - p68 +0.01 ± 0.03 um;
  - sigma_vtx-vtx +0.6 ± 0.2 um (0.3%).

  Every mu bin agrees to 0.01 pt and 0.02 fakes per event (tables below).
  On a second file (r16443 GNN-validation entries, 229 PU200 events), with the
  same values: -0.00 ± 0.01 pt and +0.02 ± 0.03 fakes per event.
- **The other single knobs match only at PU200.** Height only (0.0265) and
  amplitude only (0.0269) reach the same PU200 point. But they add 0.04-0.14
  splits per event, lose 0.1-0.3 pt below mu = 100, and miss the fake rate by
  +0.2 per event on the second file.
- **Why it works:** at the current cuts the height floor binds on 1,359 of
  1,385,727 peaks (0.1%, all 25,000 r16638 events); the width cut binds on
  420. The current point is already an integral cut in practice.
- **Position: keep the local centroid ±3.** Its p68 is within 0.3 um (0.5%)
  of the best estimator (3-point parabola). The parabola costs +18 um in
  sigma_vtx-vtx and +0.36 splits per event, and its advantage is not seen on
  the second file. Centroid ±2 ties in p68 but costs +12 um in sigma_vtx-vtx;
  max bin, ±1, ±5 and the full region are 0.5-1.5 um worse.
- **Mattia's "height 0.5", taken literally,** is far off the operating point
  in s8 units: efficiency 57.6% (-25.6 pt), 0.23 fakes per event. The height
  cut alone can match the fake rate, at 0.0265 (below).

## Data and method

- **Tuning data only; the final test set (r16443 entries 0-24,999) is never read.**
  - r16638 entries 0-24,999: s8 densities saved by the frozen-point evaluation
    (`outputs/11_12_2026_output/hllhc_final/pre_gnn/s8/r16638`). The density
    models never saw this file.
    - Even entries are the **tuning half**: the matched cut values come from
      it, at PU200 (944 events).
    - Odd entries are the **report half** (944 at PU200). The cut numbers
      above come from it. The estimator comparison (section 1) uses all
      entries, because no value is tuned there.
  - Confirmation on a second file: the r16443 GNN-validation entries in
    25,000-49,999, from a new s8 evaluation on GPU 3
    (`outputs/11_13_2026_output/peak_finding/eval_s8_r16443_e25000_50000`).
    See section 3.
- **Note on the current values.** 0.184 / 0.0184 is the mean of the matched
  values on r16638 and on r16443 entries 0-24,999 (12 Nov), so it was partly
  tuned on the test set. The proposed 0.186 comes from r16638 even entries
  only.
- **Replay, not re-inference.** Peak finding is redone offline on the saved
  densities.
  - `scripts/peak_consolidation_lib.py` records every region of the
    amplitude-0.01 scan once, with its width, integral, height and the
    position under each estimator. Any cut set or estimator is then a mask
    and a column.
  - The region segmentation does not depend on the width, integral or height
    cut, so this is exact.
  - A different amplitude means a new segmentation, so it is re-extracted.
- **Matched fake rate.** The target is the fakes per event of the current cuts
  on the tuning half at PU200 (18.82; v6's rate, as on 12 Nov). For each
  family, one knob is bisected to that target, rounded to 3 significant
  figures, then frozen and applied to the report half and to r16443.
- **Metrics:** as in `model_freeze_candidates_2026-11-12.md`.
  - Efficiency, Eq. (1): truth vertices with n_trk >= 2 that have a
    reconstructed vertex within 0.17 mm, over all such truth vertices.
  - Fakes: reconstructed vertices with no truth vertex within 0.17 mm.
  - Splits and merged truth: from the production matcher.
  - p68: 68% of |z_reco - z_true|, optimal one-to-one assignment within 1 mm.
  - sigma_vtx-vtx: sigmoid fit of the pairwise Delta-z of the same peaks.
  - HS: true hard scatter within 0.1 mm.
  - mu bins: `MU_BINS` of `gnn.evaluation.chain_t_summary`.
  - AMVF: `RecoVertex` with n_trk >= 2, on the same events.
- **Errors:** paired event bootstrap against the current cuts (or against
  centroid ±3), with the same resampled events for both arms. 2,000 replicas
  for counts; 200 for p68 and sigma (PU200 and inclusive bins only).

### Checks passed before any number was taken

- **Bit-identical peaks.** At the frozen cuts, the candidate replay with
  centroid ±3 reproduces the saved `pred_pvs_mm` exactly (float32) in all
  25,000 r16638 events.
- **Each estimator against production.** For half-widths 1, 2, 3, 5 and the
  full region, the replay equals a fresh `pv_locations_updated_res` call on
  400 events.
- **Matcher identities.** Found truth equals truth with any reco within the
  window, and fakes equal reco with no truth within the window. Both are
  checked against `compare_res_reco` in every event, for every configuration.
- **Replay equals saved counts.** Found, fake and split counts match the saved
  per-event counts. AMVF found counts match the saved evaluation.
- **Cross-check with 12 Nov.** At the frozen cuts on r16638 PU200 (all
  entries), the replay gives p68 54.84 um and sigma 217.7 um. The freeze
  study, at 0.186 / 0.0186, gave 54.77 and 218.0.
- **Low pile-up, same question.** On the saved e400 Run 2 and Run 3 data
  histograms (2,500 events each, at 0.50), the width cut removes 1 of 132,183
  peaks. The replay is bit-identical to the saved peaks there too.

## 1. Position estimator (s8, current cuts, same peaks)

All r16638 entries, PU200 (1,888 events). Differences are paired against
centroid ±3.

| Estimator | p68 [um] | d p68 [um] | p68 incl. [um] | sigma [um] | d sigma [um] | Eff [%] | dEff [pt] | Fakes/evt | Splits/evt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max bin | 56.07 | +1.23 ± 0.10 | 53.70 | 226.2 | +8.4 ± 1.5 | 83.01 | -0.21 ± 0.02 | 18.33 | 0.86 |
| centroid ±1 | 55.33 | +0.50 ± 0.08 | 52.82 | 227.2 | +9.4 ± 1.1 | 83.06 | -0.16 ± 0.02 | 18.39 | 0.78 |
| centroid ±2 | 54.83 | -0.01 ± 0.06 | 52.26 | 229.4 | +11.7 ± 1.0 | 83.13 | -0.08 ± 0.01 | 18.46 | 0.68 |
| **centroid ±3 (current)** | **54.84** | | **52.19** | **217.7** | | **83.22** | | **18.57** | **0.54** |
| centroid ±5 | 55.30 | +0.46 ± 0.07 | 52.49 | 210.5 | -7.3 ± 1.0 | 83.35 | +0.14 ± 0.02 | 18.76 | 0.32 |
| full-region centroid | 56.31 | +1.48 ± 0.11 | 53.36 | 219.7 | +2.0 ± 1.1 | 83.39 | +0.17 ± 0.02 | 18.92 | 0.25 |
| parabola (3 pt) | 54.54 | -0.29 ± 0.08 | 51.99 | 235.6 | +17.8 ± 1.6 | 83.05 | -0.17 ± 0.02 | 18.25 | 0.90 |
| Gaussian (3 pt, log) | 54.53 | -0.30 ± 0.07 | 51.97 | 236.3 | +18.6 ± 1.6 | 83.04 | -0.17 ± 0.02 | 18.24 | 0.92 |
| AMVF | 55.40 | +0.57 ± 0.20 | 52.99 | 298.6 | +80.8 ± 2.1 | 80.13 | -3.08 ± 0.06 | 19.53 | 0.19 |

![estimators](../../outputs/11_13_2026_output/peak_finding/r16638/estimators_r16638.png)

**Pick: local centroid ±3.**
- It is within 0.3 um of the best single-vertex estimator, in every mu bin.
- The 3-point fits read only the maximum and its two neighbours. On the
  satellite peaks of a conjoined region this places them closer to the main
  peak, so splits rise from 0.54 to 0.90 per event and sigma_vtx-vtx from 218
  to 236 um.

**sigma_vtx-vtx moves with the estimator alone,** on identical peaks, by
-7 to +19 um. Wider windows clipped to the region push the two peaks of a
conjoined pair apart. This lowers the fitted sigma and the split count, but
costs p68 (centroid ±5: +0.5 um) and fakes (+0.2 per event). So
sigma_vtx-vtx is only comparable between configurations that use the same
estimator. The single-vertex residual (p68) is the estimator metric; splits
are the guard.

**Max bin** is the simplest estimator, but loses 1.2 um (2.2%) in p68,
0.21 pt in efficiency, and has 0.32 more splits per event.

## 2. Cut configurations at the matched fake rate

Knob values come from the tuning half. Rows are the report half, PU200
(944 events); differences are paired against the current cuts.

| Configuration | Knob | Eff [%] | dEff [pt] | Fakes/evt | dFakes | Splits/evt | dSplits | Truth merged/evt | p68 [um] | d p68 [um] | sigma [um] | d sigma [um] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **current: integral 0.184 + height 0.0184** | | **83.19** | | **18.32** | | **0.524** | | 10.68 | **55.15** | | **219.5** | |
| **integral only (width 3)** | **0.186** | **83.19** | -0.00 ± 0.00 | **18.33** | +0.01 ± 0.01 | **0.523** | -0.001 ± 0.001 | 10.68 | **55.15** | +0.01 ± 0.03 | **220.0** | +0.6 ± 0.2 |
| integral only (no width) | 0.187 | 83.18 | -0.01 ± 0.01 | 18.30 | -0.02 ± 0.01 | 0.541 | +0.017 ± 0.005 | 10.68 | 55.12 | -0.02 ± 0.03 | 219.2 | -0.3 ± 0.3 |
| height only (width 3) | 0.0265 | 83.20 | +0.01 ± 0.02 | 18.32 | +0.01 ± 0.05 | 0.659 | +0.135 ± 0.012 | 10.65 | 55.26 | +0.11 ± 0.09 | 200.2 | -19.3 ± 1.2 |
| height only (no width) | 0.0268 | 83.20 | +0.01 ± 0.02 | 18.31 | -0.01 ± 0.05 | 0.701 | | 10.65 | 55.24 | +0.09 ± 0.08 | 196.5 | -23.0 ± 1.4 |
| amplitude only (no width) | 0.0269 | 83.18 | -0.01 ± 0.02 | 18.30 | -0.02 ± 0.05 | 0.596 | +0.072 ± 0.012 | 10.66 | 55.03 | -0.12 ± 0.09 | 201.8 | -17.7 ± 1.3 |
| amplitude + width 3 | 0.0238 | 83.20 | +0.00 ± 0.02 | 18.28 | -0.03 ± 0.04 | 0.572 | +0.048 ± 0.010 | 10.67 | 55.12 | -0.02 ± 0.07 | 207.1 | -12.4 ± 1.0 |
| height 0.5 (literal), integral 0.184 | | 57.58 | -25.61 ± 0.15 | 0.23 | -18.09 ± 0.14 | 0.076 | | 9.67 | 31.61 | | 217.3 | |
| AMVF | | 80.16 | -3.04 ± 0.08 | 19.51 | +1.20 ± 0.13 | 0.195 | | 10.84 | 55.69 | +0.54 ± 0.30 | 297.6 | +78.1 ± 3.0 |

Scaling both current cuts together matches at k = 0.9998, so it rounds to the
current values.

![trade-off](../../outputs/11_13_2026_output/peak_finding/r16638/tradeoff_r16638.png)

**At PU200 every knob traces the same efficiency-fake curve,** within the
line width, from 5 to 26 fakes per event. AMVF (19.5 fakes, 80.2%) lies
3.0 pt below it at its own fake rate. The single knobs differ only in what
the curve does not show: splits, and the behaviour away from PU200.

**vs mu, report half (efficiency [%] / fakes per event; cut matched at PU200 only):**

| Configuration | 0-20 | 20-50 | 50-100 | 100-150 | 150-200 | PU200 | inclusive |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 85.59 / 2.27 | 84.20 / 4.92 | 83.79 / 9.03 | 83.18 / 13.56 | 83.18 / 17.19 | 83.19 / 18.32 | 83.37 / 10.85 |
| integral only (width 3) | 85.58 / 2.28 | 84.19 / 4.92 | 83.80 / 9.03 | 83.18 / 13.58 | 83.18 / 17.21 | 83.19 / 18.33 | 83.37 / 10.86 |
| height only (width 3) | 85.40 / 2.31 | 84.05 / 4.91 | 83.67 / 8.97 | 83.12 / 13.48 | 83.16 / 17.15 | 83.20 / 18.32 | 83.31 / 10.82 |
| amplitude only | 85.37 / 2.31 | 84.04 / 4.91 | 83.64 / 8.93 | 83.10 / 13.45 | 83.14 / 17.12 | 83.18 / 18.30 | 83.29 / 10.79 |
| AMVF | 80.94 / 2.17 | 79.87 / 4.94 | 80.25 / 9.44 | 80.15 / 14.48 | 80.12 / 18.42 | 80.16 / 19.51 | 80.14 / 11.48 |

Paired dEff against the current cuts [pt]:

| Configuration | 0-20 | 20-50 | 50-100 | 100-150 | 150-200 | PU200 | inclusive |
|---|---:|---:|---:|---:|---:|---:|---:|
| integral only (width 3) | -0.01 ± 0.02 | -0.01 ± 0.01 | +0.01 ± 0.00 | +0.00 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 | +0.00 ± 0.00 |
| height only (width 3) | -0.19 ± 0.05 | -0.15 ± 0.03 | -0.12 ± 0.02 | -0.06 ± 0.01 | -0.02 ± 0.01 | +0.01 ± 0.02 | -0.06 ± 0.01 |
| amplitude only | -0.23 ± 0.06 | -0.15 ± 0.03 | -0.15 ± 0.02 | -0.08 ± 0.01 | -0.04 ± 0.01 | -0.01 ± 0.02 | -0.08 ± 0.01 |

![vs mu](../../outputs/11_13_2026_output/peak_finding/r16638/cuts_vs_mu_r16638.png)

- **Integral only equals the current cuts at every mu.** Fakes differ by at
  most +0.02 ± 0.01 per event.
- **Height only and amplitude only match at PU200 but not below it.** They
  lose 0.1-0.2 pt at mu < 100, more splits come at PU200 (+0.07 to
  +0.14 per event), and sigma_vtx-vtx falls by 12-23 um. That fall is the
  artefact of section 1 (extra close peaks), not better separation.
- **Dropping the width cut too** (integral 0.187) costs +0.017 ± 0.005 splits
  per event at PU200 and nothing else. Width 3 stays: it is shared with low
  pile-up, and it removes the 2-bin spikes that become splits.

## 3. Confirmation on r16443 (GNN-validation entries)

- **Sample:** 2,464 events (229 at PU200), entries 25,006-49,999. This is a
  different file from the tuning data.
- **Knobs:** taken unchanged from the r16638 tuning half.
- **Checks:** the same as for r16638, all passed (bit-identical peaks,
  matcher identities, saved counts, AMVF counts).
- **Tables:** `outputs/11_13_2026_output/peak_finding/r16443val/tables.md`.

**PU200 (paired against the current cuts):**

| Configuration | Eff [%] | dEff [pt] | Fakes/evt | dFakes | Splits/evt | dSplits | p68 [um] | d p68 [um] | sigma [um] | d sigma [um] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 82.89 | | 18.94 | | 0.524 | | 54.81 | | 214.9 | |
| **integral only 0.186** | 82.89 | -0.00 ± 0.01 | 18.96 | +0.02 ± 0.03 | 0.520 | -0.004 ± 0.004 | 54.81 | +0.00 ± 0.04 | 215.4 | +0.5 ± 0.2 |
| height only 0.0265 | 82.93 | +0.04 ± 0.04 | 19.19 | +0.25 ± 0.11 | 0.616 | +0.092 ± 0.020 | 55.16 | +0.35 ± 0.17 | 196.5 | -18.4 ± 1.8 |
| amplitude only 0.0269 | 82.93 | +0.04 ± 0.04 | 19.16 | +0.22 ± 0.10 | 0.568 | +0.044 ± 0.022 | 54.96 | +0.15 ± 0.18 | 194.1 | -20.8 ± 2.2 |
| AMVF | 79.92 | -2.97 ± 0.18 | 19.70 | +0.76 ± 0.26 | 0.201 | | 56.21 | +1.40 ± 0.60 | 303.3 | +88.4 ± 4.1 |

**Inclusive mu, integral only:** dEff +0.00 ± 0.00 pt and dFakes
+0.02 ± 0.01 per event. Per mu bin, dEff is within ±0.01 pt and dFakes is at
most +0.04 ± 0.01 per event.

- **Integral only transfers exactly.**
- **Height only and amplitude only do not.** Their fake rate at the
  r16638-matched value is +0.22 to +0.25 per event (2 standard deviations)
  above the current one. They also have more splits again, and lose 0.24-0.33 pt at
  mu 20-50.
- **The position result holds.** Centroid ±3 has the best p68 (54.81 um).
  The parabola's 0.3 um advantage on r16638 is not seen here:
  +0.06 ± 0.21 um. Max bin and the full-region centroid are +1.8 and +2.0 um
  worse. Estimator tables are in `r16443val/tables.md`.

## Proposal for `working_points.py` (not applied)

```python
HLLHC_S8_PEAKS = PeakFinding(
    amplitude=0.01, min_width=3, integral=0.186, min_height=0.0, centroid_halfwidth=3
)
```

- **Effect:** none measurable on the vertex-finding numbers before the GNN
  (section 2).
- **Downstream:** 0.17% of the peaks change (r16638, all 25,000 events:
  1,024 removed and 1,263 added, out of 1,385,727), so the GNN working
  point does not need retuning. The final chain numbers would be recomputed
  once on the test set with the new cuts, as for any working-point change.
- **On the slide,** the HL-LHC and low pile-up rows then read:
  amplitude 0.01, width 3 bins, integral 0.186 (HL-LHC) or 0.50 (low pile-up),
  local centroid ±3 bins.

## Files

- **Code (git HEAD in `GIT_HEAD.txt` next to each output):**
  - `scripts/peak_consolidation_lib.py`: candidate extraction, estimators,
    identity checks.
  - `scripts/peak_consolidation.py`: driver.
  - `scripts/peak_consolidation_plots.py`: figures and `tables.md`.
  - Run order, inside tmux:
    1. `bash scripts/peak_consolidation_eval_r16443.sh` (GPU 3, about 50 min);
    2. `python -u scripts/peak_consolidation.py --samples r16638 --out .../r16638`;
    3. `python -u scripts/peak_consolidation.py --samples r16443val --knobs-from .../r16638/consolidation.json --out .../r16443val`;
    4. `python scripts/peak_consolidation_plots.py --dir <each output dir>`.
- **Outputs:** `outputs/11_13_2026_output/peak_finding/`.
  - `r16638/`: `consolidation.json`, `tables.md`, figures, `run.log`.
  - `r16443val/`: same, from the knobs matched on r16638.
  - `eval_s8_r16443_e25000_50000/`: the s8 evaluation with histograms, made
    by `scripts/peak_consolidation_eval_r16443.sh`.
  - The outputs are not in git (`.gitignore`); the figures are on sneezy.
