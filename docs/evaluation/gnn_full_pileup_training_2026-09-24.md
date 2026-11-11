# GNN training over the full ITk pileup range — 24–25 September 2026

This run tests whether broader training data improve track assignment at low
and intermediate pileup. The previous deck GNN (v4 noaug epoch 107) was trained
on 180,000 all-hadronic ttbar events at μ=190–210. The fullmu run covers
μ=0–200 in the July 2026 ITk sample. It does not mix Run 2 or Run 3 detector
inputs.

## Status

**Training complete.** `outputs/09_24_2026_output/gnn_fullmu_v1/status.json`
records `training_complete` at **2026-09-24T22:47:32Z** after 216 shard
iterations (18 passes). Best validation loss: shard epoch **213**
(val ≈ 0.270436). Final epoch 215 is essentially tied (val ≈ 0.270450).

**Deck checkpoint (25 September):**
`model_weights/ttva_gnn_hllhc_fullmu_v1_20260924/ttva_gat_alleta_k20_fullmu_v1_60k_epoch_213.pyt`

**Deck-matched full-chain eval complete.** Same conventions as the 25 September
opening deck: integral 0.3, top-1 / no score cut, reco \(n_{\mathrm{trk}}\ge2\),
on the diagnostic 25,000 entries (0–24999) and the 1,920-event high-μ table
subset. Outputs live under
`outputs/09_25_2026_output/gnn_fullmu_deck_eval/` and are promoted into
`presentations/mattia/09_25_2026/opening_slides/review/`. Prior v4 noaug
caches are archived in
`presentations/mattia/09_25_2026/opening_slides/review/archive_v4_noaug_epoch107_20260925/`.

| Measurement (185≤μ≤200 unless noted) | v4 noaug e107 | fullmu e213 | AMVF |
|---|---:|---:|---:|
| After-GNN positional efficiency (0.1 mm) | 70.05 ± 0.10% | **71.07 ± 0.10%** | 70.99 ± 0.10% |
| Unmatched reco / event after selection | 12.92 ± 0.08 | **15.05 ± 0.09** | 25.46 ± 0.11 |
| Track efficiency | 68.68 ± 0.06% | **69.06 ± 0.06%** | 68.32 ± 0.06% |
| Assignment purity | 69.45 ± 0.06% | **69.69 ± 0.06%** | 68.65 ± 0.06% |
| Assigned fraction | 98.89 ± 0.01% | **99.10 ± 0.01%** | 99.52 ± 0.01% |
| Clean / reco | 51.78 ± 0.13% | **49.75 ± 0.13%** | 35.17 ± 0.10% |
| Δeff / Δpur vs AMVF (pp) | +0.37 / +0.80 | **+0.75 / +1.05** | — |

Efficiency and purity improve in every 10-μ bin on the diagnostic sample; the
largest gains are near μ 20–50. Clean/reco at high μ drops slightly despite
better track metrics. Entries 0–24999 remain a **diagnostic** comparison only
(held out of fullmu train/val/test). Choosing the operating point on the
validation set and freezing before the reserved test is still outstanding.

Campaign logs and truth-candidate monitors:
`outputs/09_24_2026_output/gnn_fullmu_v1/` (`campaign.log`,
`learning_curve.jsonl`, `validation_pass_01.json` … `_18.json`).

## Source and split

The source is
`data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root`.
Its measured range is **0 ≤ μ ≤ 200**, despite `PU200` in the filename.

| Role | Events | Use |
|---|---:|---|
| Existing deck sample, entries 0–24,999 | 25,000 | Diagnostic comparison only; never training or independent test |
| Training | 60,000 | Gradients; 12 shards of 5,000 graphs |
| Validation | 7,500 | Checkpoint and working-point selection |
| Reserved test | 7,500 | Final comparison after choices are frozen |

The last three sets partition entries 25,000–99,999, stratified in 10-unit
pileup bins using seed **20260924**. There is no entry overlap. All 100,000
complete truth-vertex-z fingerprints are unique. This ROOT format lacks
run/event identifiers, so the audit uses disjoint entries of one production
and those fingerprints; alternative reconstruction tags are not mixed.

The exact index arrays live under
`data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/`. The manifest in
`outputs/09_24_2026_output/gnn_fullmu_v1/split_manifest.json` records the ROOT
SHA-256, index hashes and pileup counts. No test graphs are built during
training. The fixed learning-curve monitor uses 1,500 validation events,
75 from each pileup bin, all drawn from the 7,500-event validation set.

“Full pileup” means coverage of the complete available pileup range. It does
not mean every available collision is used for gradients. Independent
validation and test data are essential. Training sample size and ttbar decay
channel also change relative to v4, so a gain cannot be attributed solely to
pileup coverage.

## Training recipe

- Configuration: `configs/gnn/config_gnn_ttva_hllhc_fullmu_v1.yml`.
- Same GAT architecture, eight track features, two PV features, three edge
  features, k=20 nearest candidate vertices, and `hllhc_alleta` resolution
  preset as v4 without augmentation.
- Pristine truth-vertex candidates, as in v4. Training with actual PV-Finder
  peaks remains a separate experiment.
- Fresh initialization with seed **42**; dynamic weighted binary
  cross-entropy, Adam at 0.001, batch size 32, dropout 0.25, gradient clipping
  at 1, cosine learning rate ending at 0.00001.
- **216 shard iterations = 18 full passes** over 60,000 training events.
  This is **1.08 million event presentations**, the same total as v4's
  180,000 events × six passes. The trainer calls each shard iteration an
  “epoch”; it is not a full pass over this dataset.
- Physical **GPU 2**, exposed as `cuda:0`. Two graph-building processes and
  two data-loader workers. Existing GPU jobs are not stopped or modified;
  the wrapper waits for at least 10 GB free before optimization starts.

The runtime graph cache was
`/dev/shm/matmauro_gnn_fullmu_v1_20260924/`, approximately 40.2 GiB.
`runtime_config.yml` records the resolved RAM paths. Checkpoints are under
`model_weights/ttva_gnn_hllhc_fullmu_v1_20260924/`, named
`ttva_gat_alleta_k20_fullmu_v1_60k_epoch_<N>.pyt` and
`..._fullstate.pth`. Local MLflow experiment `TTVA_GNN`, run
`ttva_gat_alleta_k20_fullmu_v1_60k`.

## What the monitoring metrics mean

The loss is evaluated after each shard iteration. After each complete pass,
`validation_pass_<N>.json` records track efficiency, purity and assignment
counts for cuts 0, 0.5, 0.8, 0.9, 0.95, 0.98 and 0.99 in 10-unit pileup bins.
These counts require at least two assigned tracks per surviving candidate.

Those are **truth-candidate diagnostics**: a correct assignment means the
winning candidate's truth ID equals the track's truth ID. They are not
PV-Finder-chain results, and are not directly comparable to AMVF links or
the deck's dominant-origin definition. Low validation loss alone does not
establish a better TTVA operating point.

The publishable deck comparison above evaluates the best-val checkpoint on
actual PV-Finder peaks at integral 0.3 with the deck conventions. Remaining
work: choose working points on **validation** data, freeze, then evaluate the
reserved **test** set; include efficiency at matched purity and the reverse
comparison in pileup bins for old GNN, new GNN and AMVF. Official ATLAS
fit-weight working points still need information absent from these ntuples.

V4's large clean-vertex yield advantage at a 0.98 cut was not a track-efficiency
advantage. The presentation audit is documented in
[`TTVA_RECONCILIATION.md`](../../presentations/mattia/09_25_2026/opening_slides/TTVA_RECONCILIATION.md).

## Reproduction

Training launch (already finished; do not relaunch under the same name):

```bash
tmux new-session -d -s gnn_fullmu_20260924 \
  -c /data/home/matmauro/codice/PV-Finder \
  'bash scripts/run_gnn_fullmu_20260924.sh'
```

Deck-matched eval (already finished; artifacts under
`outputs/09_25_2026_output/gnn_fullmu_deck_eval/`):

```bash
tmux new-session -d -s fullmu_deck_eval \
  -c /data/home/matmauro/codice/PV-Finder \
  'bash outputs/09_25_2026_output/gnn_fullmu_deck_eval/run_eval.sh'
```

`training_provenance.json` records code hashes and the Python/PyTorch/CUDA
versions used at optimization startup. The inherited standard trainer's
generic resume option is not an exact continuation of this campaign.
