# Run 2 Data

Real ATLAS Run 2 proton-proton collision data (2018, √s = 13 TeV, ZeroBias trigger).

## Location

```
data/run2/Run2_Data/user.rgarg.data18_13TeV.00364076.physics_ZeroBias.AOD1_EXT0/
  user.rgarg.49035490.EXT0._000001.ATLAS_PVFinderData_Run3Data.root  (44,584 events)
  user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root  (98,247 events)
  user.rgarg.49035490.EXT0._000003.ATLAS_PVFinderData_Run3Data.root  (98,114 events)
  user.rgarg.49035490.EXT0._000004.ATLAS_PVFinderData_Run3Data.root  (57,694 events)
```

Total: 298,639 events. File naming uses `Run3Data` suffix but content is Run 2.

## Pileup

| Stat | Value |
|------|-------|
| Mean μ | 32.5 |
| Median μ | 31.2 |
| Range | 3.3 – 66.7 |
| Distribution | Bell-shaped, peaked at μ ≈ 25–30 |

Compared to MC (flat, mean 39.9) and Run 3 (bimodal, median 59.0).

## Beam Position

BeamPosZ ranges -5.37 to +0.08 mm (mean -2.47 mm). Beam correction **is required** — subtract BeamPosZ from AMVF vertex z to align with model predictions.

## ROOT Branches

Same structure as Run 3. Key branches used for evaluation:

| Branch | Type | Used for |
|--------|------|----------|
| `RecoTrack_z0`, `_d0`, `_ErrD0`, `_ErrZ0`, `_ErrD0Z0` | `vector<float>` | Model input (track features) |
| `RecoVertex_z` | `vector<float>` | AMVF reco vertices (reference) |
| `RecoVertex_nTracks` | `vector<float>` | nTracks filter (≥ 2) |
| `BeamPosZ` | `float` | Beam correction |
| `ActualNumOfInt` | `float` | Pileup (μ) |

Also has `TruthVertex_*` branches — but for real data these are not MC truth. Use `RecoVertex_*` as reference.

## Evaluation

Use `run_eval_pvf_run3.py` (same script as Run 3 — identical ROOT format):

```bash
python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run2/Run2_Data/.../user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root \
    --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --max-events 2500 --output-dir outputs/eval_pvf_run2 --device 0
```

## Baseline Results (T2KDE+K2H, 2500 events, AMVF ntrks ≥ 2)

| Metric | Run 2 | MC (comparison) |
|--------|-------|-----------------|
| Overall Efficiency | 87.3% | 85.9% |
| σ_vtx_vtx | 0.303 mm | 0.333 mm |
| Truth PVs/evt | 22.5 | 22.8 |
| Pred PVs/evt | 23.0 | 23.5 |
