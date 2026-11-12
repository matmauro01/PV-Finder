# Full-pileup GNN v2: the v1 recipe on more data, 11 November 2026

Task D4 of [the 11 November plan](../additional_studies/meeting_2026-11-11_plan.md):
retrain the full-pileup GNN on more data, then freeze it. v1 is documented in
[gnn_full_pileup_training_2026-09-24.md](gnn_full_pileup_training_2026-09-24.md).

## What changes and what does not

Only the training set changes. These are identical to v1:
- architecture (GAT, hidden 32, 4 heads, dropout 0.25)
- loss, Adam at 0.001, cosine schedule to 1e-5, batch 32, gradient clip 1
- kNN with k = 20, and truth-vertex candidates
- the `hllhc_alleta` resolution preset, and the graph builder

The graphs use truth-vertex candidates, so **no PV-Finder density model
enters the training data**. Re-freezing the density model (D3) does not
require new GNN training graphs.

## Data and split

| Set | Source | Events | Use |
|---|---|---:|---|
| Training, v1 part | r16443 entries 25,000-99,999 (v1's 12 shards, unchanged) | 60,000 | gradients |
| Training, new part | r16638 (18 new shards of 4,990) | 89,820 | gradients |
| **Training total** | | **149,820** | 2.5 x v1 |
| Validation | r16443, v1's set, unchanged | 7,500 | checkpoint and working-point choice |
| Validation monitor | subset of the above, v1's, unchanged | 1,500 | learning curve |
| Reserved test | r16443, v1's set, unchanged | 7,500 | final comparison after freezing |
| Extra reserved test | r16638, never built | 9,980 | spare independent test |
| Diagnostic deck sample | r16443 entries 0-24,999 | 25,000 | never training (as in v1) |

- **Independence of r16638.** r16638 is a second production of the same flat-mu
  (0 <= mu <= 200) single-lepton ttbar sample, with 99,800 entries and
  mu flat and uncorrelated with the entry number. Checked in the split step:
  - no r16638 full truth-z fingerprint occurs in r16443
  - no hard-scatter truth vertex (x, y, z) occurs in r16443
  - all r16638 fingerprints are unique
- **Stratification.** The r16638 split is stratified in 10-unit mu bins, seed
  20261111. The reserved 10% has 450-550 events per bin, and training has
  4,050-4,950 per bin.
- **Records.** Entry arrays and hashes are in
  `data/run4_all_etas/ttva_graphs/fullmu_v2_20261111/` and
  `outputs/11_11_2026_output/gnn_fullmu_v2/split_manifest.json`.
- **Evaluation caution.** r16638 is now GNN training data. Full-chain
  evaluations of v2 must use r16443 (entries 0-24,999, or v1's reserved test)
  or the 9,980 reserved r16638 events. PV-Finder-only evaluations on r16638
  (for example the CPU small-model comparison) are unaffected, because the
  density model is not trained on r16638.

## Schedule

- **540 shard epochs = 18 full passes** over 149,820 events, i.e. 2.70 million
  event presentations, 2.5 x v1.
- **Why 18 passes:** this is v1's number of passes. v1's best validation epoch
  was its 213th of 216, still improving at the end of the cosine schedule, so
  there is no sign of overfitting.
- **Shard order:** v1 and r16638 shards are interleaved in the config, so the
  round-robin loader mixes both files throughout each pass.

## Running

- Config: `configs/gnn/config_gnn_ttva_hllhc_fullmu_v2.yml`.
- Campaign: `scripts/gnn_fullmu_v2_campaign.py` (plan, prepare/build, train).
  It reuses v1's `validation_tracks` monitor and `write_json`.
- Wrapper: `scripts/run_gnn_fullmu_v2_20261111.sh`, in tmux `gnn_fullmu_v2`.
- GPU and limits: physical GPU 2, `ulimit -v 64 GiB`, 2 data-loader workers.
- Shards are read from disk; there is no tmpfs cache (v1 staged 40 GiB in
  `/dev/shm`, which would be about 100 GiB here on a shared machine).
- Logs and progress: `outputs/11_11_2026_output/gnn_fullmu_v2/`
  (`campaign.log`, `training.log`, `learning_curve.jsonl`,
  `validation_pass_XX.json` after each pass, `status.json`).
- Checkpoints: `model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/`, one per
  shard epoch.

## Status

- **Finished 2026-11-12 04:01 UTC** (540 shard epochs, final validation-monitor
  loss 0.2558, learning rate at the 1e-5 floor, plateaued). Checkpoint chosen:
  **epoch 527** (lowest validation loss). Checkpoint evidence, t scan and full
  chain: [gnn_v2_freeze_2026-11-12.md](gnn_v2_freeze_2026-11-12.md).

- Graph building finished at 2026-11-12 00:00 UTC.
- **The first launch was stopped and relaunched.**
  - The first launch was trainer pid 98041, started at 00:01 UTC. It was stopped at shard epoch 13, with validation-monitor loss 0.327.
  - Its config, data and code are identical to the kept run; only docs commits differ in git.
  - Why it was stopped: it ran under `ulimit -v 48 GiB`, and the trainer's virtual size was a stable 47.80 GiB, 0.2 GiB from the limit. RSS was 9.2 GiB; the rest is CUDA address-space reservation. Its forked data-loader workers started at 47.76 GiB.
  - Any further allocation, such as the per-pass validation monitor, risked a MemoryError.
  - The limit was raised to 64 GiB, the 28 checkpoints were deleted, and training restarted from scratch at 00:08 UTC.
  - Logs of the aborted launch: `outputs/11_11_2026_output/gnn_fullmu_v2/aborted_run1/` (with a README).
- **Kept run:** started at 2026-11-12 00:08 UTC, at about 20-28 s per shard epoch.
- **ETA: about 3.0-3.3 h, finishing around 2026-11-12 03:20 UTC (22:20 EST on 11 November).**
- First shard epochs (train / validation-monitor loss), close to v1 at the same number of events seen. Kept run vs aborted launch differ at the 0.01 level (non-deterministic GPU scatter operations):

| Shard epoch | v2 train | v2 val | aborted launch val | v1 val |
|---:|---:|---:|---:|---:|
| 0 | 0.879 | 0.558 | 0.553 | 0.554 |
| 1 | 0.520 | 0.497 | 0.492 | 0.486 |
| 2 | 0.469 | 0.469 | 0.452 | 0.469 |
| 3 | 0.444 | 0.436 | 0.415 | |

## Freezing (after training)

1. Choose the checkpoint on the validation monitor loss (as in v1).
2. Choose one HL-LHC working point `t` on the 7,500 validation events with the
   frozen density model (plan D1).
3. Evaluate the reserved test once, against v1 and AMVF.
