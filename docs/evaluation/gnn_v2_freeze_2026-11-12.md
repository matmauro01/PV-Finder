# GNN v2 freeze: checkpoint, t scan and full chain (12 November 2026)

Step 2b of the [11 November plan](../additional_studies/meeting_2026-11-11_plan.md)
(D3, D4). It does three things:

- chooses the GNN v2 checkpoint;
- scans the score cut t in mu bins;
- runs the full chain (v6 or s8 → GNN v2) against AMVF on the same events.

**t is not chosen here.** Mattia decides from the options below.

All outputs are in `outputs/11_12_2026_output/gnn_v2_freeze/`.

## 1. Checkpoint: v2 epoch 527

- **Rule:** lowest validation-monitor loss. The monitor is v1's 1,500 events,
  and v1 and v2 use the same loss.
- **Choice:** epoch 527, loss 0.255807. The last epoch (539) is 0.255833. The
  difference, 2.7e-5, is below the 4e-5 scatter within a pass, so the two are
  equivalent. Epoch 527 is now the checkpoint in `working_points.py`; t stays
  pending.
- **Plateau:** yes, for this schedule.
  - Mean validation loss per pass changed by -0.00076, -0.00046, -0.00028 and
    -0.00018 over the last four passes (-0.07% in the last one).
  - The learning rate is at its 1e-5 floor.
  - Training and validation loss are equal (0.2552 / 0.2559), so there is no
    overfitting.
  - A longer schedule could still give a small gain (v1 → v2 gained 0.015 with
    2.5x the data). This was not tried.
- **Figure:** `checkpoint/learning_curve_v1_v2.png`.
- **v2 against v1 (e213), same events:**
  - Truth-candidate graphs (the 1,500 monitor events), track F1 at cuts
    0 / 0.9 / 0.95 / 0.98 / 0.99:
    - v1: 76.0 / 77.2 / 75.9 / 69.9 / 62.8%
    - v2: 77.0 / 78.0 / 76.9 / 71.4 / 65.8%
  - Full chain on v6 peaks: v2 is at least as good in track F1 in every mu bin
    and at every t, within ±0.1 point where it is not better
    (`v1_vs_v2.md`).
  - HS selection at t = 0.98 and mu < 20: 93.8% for v1, 98.3% for v2 (test).
  - Clean/truth is higher for v2 at t ≤ 0.98. At t = 0.99 it is 0.5-1.8 points
    lower above mu = 20.

## 2. Data hygiene

Proved from the trained-on shard list (`runtime_config.yml`) in
`hygiene.json` (`scripts/gnn_v2_freeze_hygiene.py`).

- **What GNN v2 trained on in r16443:** exactly v1's 60,000 training entries,
  all in 25,000-99,999.
- **Test set:** r16443 entries 0-24,999, the note's sample.
  - 0 overlap with GNN v2 training, validation or reserved test.
  - The density models never saw r16443 at all.
- **t-scan set:** the 7,500 GNN validation entries.
  - 0 overlap with training and 0 overlap with the test set.
  - Note: GNN checkpoint selection also used these events, through the
    1,500-event monitor, which is part of this set.
- **Untouched:** the 7,500 GNN reserved-test entries are kept for one
  evaluation after the freeze.
- **Fingerprints:** no r16443 hard-scatter (x, y, z) or full truth-z
  fingerprint occurs in r16638, r16438 or r16633. The last two are density
  training files with the same e/s tags.
- **No stored HS vertex:** two r16443 entries have none (29424 in the
  validation set, 57641 in neither set). They are excluded from HS
  denominators only.

## 3. Setup

- **Density models:** at their `working_points.py` cuts.
  - v6: 0.30 / 0.03.
  - s8: 0.184 / 0.0184.
  - Both: local centroid ±3.
- **GNN:** kNN graph, k = 20, built exactly as in the deck evaluation. Each
  track keeps its top-1 edge if its score is ≥ t.
- **Vertices:** those with ≥ 1 assigned track (drop-empty). Results for ≥ 2
  tracks are also stored.
- **AMVF:** its stored lists, on the same events.
- **Categories, two taxonomies:**
  - Track-purity (`classify_assignments`): Clean / Merged / Split / Fake.
  - Positional: greedy closest-first, 0.17 mm. In the tables these are
    "pos. eff" and "pos. Fake".
- **HS quantities:**
  - HS selection uses the Athena 24.0 rule (`pv_selection`). PV-Finder
    positions come from the 3D adaptive fit. For AMVF the reference is its
    stored primary.
  - HS resolution is the 3D adaptive beam-spot fit of the tracks the GNN
    assigns at t to the peak closest to the HS, using an iterative 3-RMS. It is
    measured on the events where AMVF also reconstructs the HS.
- **Errors:** event bootstrap, 200 replicas. Differences are paired.
- **Code:** `gnn.evaluation.chain_inference` (GPU, git f7f16c7),
  `gnn.evaluation.chain_t_metrics` and `chain_t_summary` (git 471be32), and
  `scripts/gnn_v2_freeze_*.py`.
- **Checkpoint sha256:** recorded in every `metadata.json`.

## 4. Quality gates (all passed)

- **Bookkeeping:** asserted per event for every arm, t and track minimum:
  - Clean + Merged + Split + Fake equals the number of vertices, in both
    taxonomies;
  - found ≤ truth, and correct ≤ assigned ≤ labelled.
- **Identical events:**
  - v6 and s8 use the same entries (asserted).
  - AMVF is evaluated in the same pass.
  - The entry sha256 is in each json.
- **Regression against 11 November:** v1 e213 on v6 peaks, run through the
  new code (`regression_vs_11nov.json`):
  - The histograms are bit-identical.
  - AMVF categories and all HS-selection flags (GNN and AMVF) are identical
    in every event.
  - GNN categories differ in ≤ 6 of 25,000 events, from top-1 ties and
    1e-5 score differences.
- **Independent recomputation** (`crosscheck/`): a second agent wrote
  separate code on the original classifier and the definitions only. It
  matches every count exactly (Clean, Merged, Split, Fake, reco, correct and
  assigned tracks, positional matches) for t = 0, t = 0.98 and AMVF, on all
  events and at PU200. It also matches:
  - the HS selection: 1,824/1,920 and AMVF 23,785/25,000;
  - the HS widths: 9.3175, 9.5878 and 10.2468 um (`crosscheck_compare.json`).
- **AMVF anchors (test, PU200):**
  - stored-primary HS selection 92.92% (1,784/1,920);
  - our-code rule 89.11%;
  - HS resolution 10.25 um;
  - clean-HS 92.5%.
  All four equal the published values.
- **Physics checks:**
  - At t = 0, PU200, positional efficiency is 82.93%. Before the GNN it is
    83.20%; dropping the empty peaks explains the difference.
  - Clean/truth falls with mu at every t, and Merged rises.
  - Track efficiency falls and purity rises with t.
  - The HS resolution is flat in mu, at 9-10 um.

## 5. t options (derived on validation, quoted on test, v6)

Option rules (`scripts/gnn_v2_freeze_options.py`):

- **A:** maximum inclusive Clean/truth.
- **B:** maximum inclusive HS selection.
- **C:** maximum inclusive track F1.
- **D:** the most mu-robust t: the largest worst-case ratio, over the five mu
  bins, of Clean/truth to its best value in that bin.
- **E:** no cut.

All rules are applied on the validation set. Full tables, per mu bin, for both
sets and for s8: `options.md`.

| Option | t | Clean/truth incl. | PU200 | mu 0-20 | Fake/reco incl. | Split/reco incl. | HS sel incl. / PU200 / 0-20 | HS res incl. / PU200 [um] | trk F1 incl. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.98 | **73.83** | **67.73** | 84.22 | 0.17 | 5.95 | 96.79 / 95.00 / 98.28 | 9.70 / 9.59 | 72.09 |
| D | 0.97 | 73.23 | 66.99 | 86.45 | 0.18 | 7.37 | 97.04 / 95.26 / 98.60 | 9.53 / 9.49 | 74.45 |
| B | 0.94 | 68.70 | 61.71 | 88.82 | 0.19 | 9.46 | **97.10** / 95.31 / 99.12 | 9.43 / 9.41 | 76.61 |
| C | 0.92 | 65.84 | 58.39 | **89.04** | 0.19 | 10.08 | 97.04 / 95.31 / 99.28 | 9.40 / 9.36 | **76.77** |
| E | 0 | 56.79 | 47.71 | 86.40 | 0.20 | 11.09 | 95.97 / 92.45 / 99.28 | **9.36 / 9.32** | 75.77 |
| AMVF | - | 40.49 | 30.94 | 77.06 | 0.07 | 21.50 | 95.14 / 92.92 / 98.16 | 9.93 / 10.24 | 74.43 |

Statistical errors on test:
- Clean/truth ±0.06 (inclusive), ±0.12 (PU200), ±0.4 (mu < 20);
- HS selection ±0.12 / ±0.55 / ±0.3;
- HS resolution ±0.08 / ±0.23 um.

### Trade-offs

- **A and D are nearly equal at high mu.** D is 0.6-0.7 points lower at
  PU200. It gains 2.2 points at mu < 20, 2.4 points of track F1, and is
  0.17 um better in HS resolution.
- **B and C give the best HS selection and track F1.** They cost 5-8 points
  of Clean/truth at high mu.
- **The HS resolution gets worse as t rises.** It is best at t = 0. It is
  below AMVF (9.93 um inclusive) for every t ≤ 0.98, and at 0.99 it is
  10.03 um.
- **"Fake ≤ AMVF" depends on the taxonomy.**
  - Track-purity Fake/reco is 0.17-0.20% at every t, against AMVF's 0.07%. No
    t reaches AMVF.
  - Positional Fake/reco is below AMVF (21.4%) at every t: 18.4% at t = 0 and
    14.3% at 0.98.
- **t = 0.99 against 0.98 (test, inclusive):**
  - It is worse in Clean/truth (-2.1 points), positional efficiency (-3.0),
    track F1 (-4.9) and HS selection (-1.1), and in HS resolution (10.03 um,
    above AMVF).
  - It is better only in the fractions per reconstructed vertex: Clean/reco,
    Merged, Split and Fake.

## 6. Low pile-up (mu < 20)

What is non-monotonic:

1. **The t dependence reverses.**
   - At mu < 20, Clean/truth peaks at t = 0.9-0.92 (89.0%) and falls to 84.2%
     at 0.98 and 78.5% at 0.99.
   - Above mu = 50 it rises up to t = 0.98.
   - So at t = 0.98, Clean/truth rises with mu from mu 0-10 (82%) to 20-30
     (86.5%), then falls (`figures/vs_mu_test_v6_v2e527.png`).
2. **Split/reco and positional Fake/reco are high at the lowest mu.**
   - At mu 0-10, Split/reco is 19.5% and positional Fake/reco 26% (t = 0),
     against 11% and 18% above mu = 30.
   - AMVF shows the same: 29% and 32%.
3. **The HS selection drops with t at mu < 20:** from 99.3% at t = 0 to
   98.3% at 0.98 and 96.6% at 0.99.

Cause, measured (`lowmu/`, `scripts/gnn_v2_lowmu_*.py`):

- **The cut removes almost only correct tracks at low mu.**
  - At mu < 10, 97% of top-1 edges are correct, against 71% at mu 150-200.
  - The fraction of correct tracks scoring below 0.99 does not grow at low mu:
    24-34% in every mu bin (33% at mu < 10).
  - So at low mu a higher t mostly removes good tracks, with few wrong ones to
    purify.
  - Vertices with 5-10 tracks at mu < 10 go from 92% Clean at t = 0.9 to 61%
    at 0.99.
- **The GNN is also less confident on small vertices at low mu.**
  - At mu < 10, 47% of correct tracks of 5-10-track truth vertices score
    below 0.99, against 25% at mu 20-50 and 41% at mu 150-200.
  - This is measured on the truth-candidate graphs the GNN was trained on
    (`lowmu_truthgraph_v2e527.json`), so it is a property of the network, not
    of the PV-Finder peaks.
  - The minimum is near mu 20-50. A plausible cause is that mu < 20 events
    supply only about 2% of the tracks in a flat-mu sample (1.95% on the
    monitor), so they carry little weight in the per-edge loss. This is not
    established.
  - v1 was worse: 47% of all correct tracks below 0.99 at mu < 10, against 33%
    for v2. This is very likely the "weird behaviour" Mattia saw.
- **Display.** `lowmu/lowmu_displays_val_v6_v2e527.png` shows isolated,
  well-separated 7- and 11-track vertices at mu ≈ 17. Every one of their
  tracks scores 0.9-0.99, so the vertex vanishes at t = 0.99.
- **Splits at low mu are hard-scatter satellites.**
  - At mu < 20, 60% of Split vertices have the HS as their dominant truth.
    63% of them hold one track, and they sit a median 0.8 mm from the HS peak.
  - These are weak satellite peaks of the density model next to the large HS
    peak (see the satellite-peak study).
  - They are a larger share of the reco list when there are few vertices.
    They are not GNN errors.
- **Nothing pathological beyond this.** Fake/reco is small (≤ 0.4%) and
  there is no failure mode.

## 7. v6 against s8 after GNN v2 (test, paired, s8 minus v6)

| | t = 0.98 incl. | PU200 | mu 0-20 | t = 0.97 incl. | t = 0.94 incl. |
|---|---:|---:|---:|---:|---:|
| Clean/truth [pt] | -0.07 ± 0.02 | -0.11 ± 0.06 | +0.13 ± 0.18 | -0.14 ± 0.02 | -0.28 ± 0.02 |
| Clean/reco [pt] | -0.19 ± 0.02 | -0.12 ± 0.06 | -0.59 ± 0.20 | -0.26 ± 0.02 | -0.43 ± 0.02 |
| Split/reco [pt] | +0.14 ± 0.02 | +0.10 ± 0.04 | +0.43 ± 0.20 | +0.18 ± 0.02 | +0.26 ± 0.02 |
| Fake/reco [pt] | +0.01 ± 0.00 | -0.01 ± 0.01 | +0.04 ± 0.03 | +0.00 | +0.00 |
| HS selection [pt] | -0.24 ± 0.07 | -0.16 ± 0.27 | -0.48 ± 0.23 | -0.10 ± 0.06 | -0.02 ± 0.05 |
| HS resolution [um] | +0.04 ± 0.04 | +0.18 ± 0.13 | +0.32 ± 0.14 | +0.04 ± 0.04 | +0.05 ± 0.03 |

- **v6 stays slightly better after the GNN**, as before it (step 2a).
  - The Clean/truth gap is ≤ 0.3 points at every option t, and ≤ 0.1 at
    t = 0.98.
  - s8 makes about 0.1-0.3 points more splits.
  - At t = 0.98, its HS selection is 0.24 ± 0.07 points lower.
  - The HS resolution is equal within 0.2 um.
- **The trends with mu are identical:** `figures/v6_vs_s8_after_gnn_vs_mu.png`.
- **For the freeze:** s8's loss after the GNN is a fraction of a point on the
  chain headlines. It is 9x faster on CPU.

## Figures and tables

All paths below are under `outputs/11_12_2026_output/gnn_v2_freeze/`.

- `figures/t_scan_val_v6_v2e527.png`: all metrics against t, per mu bin. Also
  for the test set and s8.
- `figures/vs_mu_test_v6_v2e527.png`: all metrics against mu, for t = 0, 0.9,
  0.95, 0.98, 0.99 and AMVF.
- `figures/v6_vs_s8_after_gnn_vs_mu.png`
- `checkpoint/learning_curve_v1_v2.png`
- `lowmu/lowmu_displays_val_v6_v2e527.png`
- `tables.md`: every metric, arm and mu bin, with errors.
- `options.md`, `v1_vs_v2.md`

## Reproduction

```bash
# inside tmux, repo root
SET=test DENSITY=v6 DEVICE=0 bash scripts/run_gnn_v2_freeze_inference.sh   # also s8, val
bash outputs/11_12_2026_output/gnn_v2_freeze/run_metrics.sh test_v6:v2e527 ...
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_summary.py --dir outputs/11_12_2026_output/gnn_v2_freeze
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_options.py --dir outputs/11_12_2026_output/gnn_v2_freeze
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_checkpoint_selection.py --out outputs/11_12_2026_output/gnn_v2_freeze/checkpoint
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_hygiene.py --out outputs/11_12_2026_output/gnn_v2_freeze
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_regression.py --dir outputs/11_12_2026_output/gnn_v2_freeze
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_crosscheck_compare.py --dir outputs/11_12_2026_output/gnn_v2_freeze
PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_lowmu_diagnosis.py --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/val_v6 --out outputs/11_12_2026_output/gnn_v2_freeze/lowmu
```
