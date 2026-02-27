# Evaluation — Vertex Finding

Evaluation of PV-Finder vertex finding on MC test data.

## Script

`src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py`

Three pipeline modes (mutually exclusive):

| Flag | Pipeline |
|------|----------|
| _(default)_ | Analytical KDE_A_z (h5) → K2H (UNet_1000) |
| `--full-pipeline` | Raw tracks → T2KDE (MaskedDNN) → K2H |
| `--e2e-model` | Raw tracks → trackstoHists_UNet_1000 (no KDE stage) |

## How to Run

```bash
source venv/bin/activate

# E2E model, full test set, ROOT truth (recommended — matches mattia_finder):
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --h5 /share/lazy/qibinlei/recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5 \
    --e2e-model model_weights/tracks2hist_1channel_200epochs_epoch_191_fullstate.pth \
    --root /share/lazy/rocky/ATLAS_data/Latest_Sept2023/ATLAS_PVFinderData_TruthMatched.root \
    --qibin configs/qibin_test_main_indices_v2.p \
    --indices configs/test_main_indices_2550evt.p \
    --output-dir outputs/eval_tracks2hist_1ch_e191_root \
    --device 0

# K2H stage-2 only, no ROOT:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --h5 ... --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --indices configs/test_main_indices_2550evt.p \
    --output-dir outputs/eval_k2h --device 0
```

## Test Set

- **File:** `configs/test_main_indices_2550evt.p`
- **Events:** h5 indices 48450–50999 (2550 events = last 5% of 51000)
- **Subevents:** 581400–611999 (30600 subevents, split [0.7, 0.25, 0.05])
- Matches `mattia_finder` test set exactly.

## Key Design Decisions

### Peak-finding thresholds

| Parameter | Value | Used for |
|-----------|-------|---------|
| `threshold` | `1e-2` | Min bin height to start a peak |
| `INTEGRAL_THRESHOLD` | `0.2` | Min peak area — for performance metrics |
| `INTEGRAL_THRESHOLD_RES` | `0.5` | Min peak area — for σ_vtx_vtx only |
| `min_width` | `3` bins | Min peak width |

Matches `evaluate_model.py` (0.2) and `calculate_sigma.py` (0.5) in `mattia_finder`.

### ROOT truth vs h5 truth

Without `--root`: truth PVs from h5 `pv` field — **no nTracks filter**. All truth PVs included, which inflates `reco_merged` count.

With `--root` + `--qibin`: truth PVs from `ATLAS_PVFinderData_TruthMatched.root`, filtered to **nTracks ≥ 2**. Matches `mattia_finder` exactly.

### h5 ↔ ROOT event index mapping

The h5 file uses reindexed ("pubindices") event ordering — h5 event `i` ≠ ROOT event `i`. The correct ROOT index for the `k`-th sequential test event is `qibin[k]`, stored in `configs/qibin_test_main_indices_v2.p` (copied from `mattia_finder/config/`).

### Pileup filter for summary

Summary statistics (clean/merged/split/fake averages) are computed only over events with **55 ≤ ActualNumOfInt ≤ 65** (from ROOT). This matches `mattia_finder`'s `plot_tracks2hist.py` convention. Overall efficiency across all events is also printed.

### Pairwise Δz for σ_vtx_vtx

`pv_locations_updated_res` returns PVs sorted ascending in z, so all pairwise differences `pvs[i]-pvs[j]` for `i<j` are negative. Both `+dz` and `-dz` are added to make the distribution symmetric before fitting the sigmoid.

## Outputs

| File | Contents |
|------|----------|
| `resolution_plot.png` | Pairwise Δz histogram + sigmoid fit → σ_vtx_vtx |
| `performance_plot.png` | Clean/merged/split/fake fractions and efficiency vs pileup |
| `stats_histogram.png` | Avg count/event for clean/merged/split/fake vs pileup (all events, mattia_finder style) |
| `eval_results.pkl` | All per-event results, pred/truth PV positions, fit params |

## Model Checkpoints

| Model | File |
|-------|------|
| E2E ep191 (tracks→hist) | `model_weights/tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` |
| K2H ep190 | `model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth` |
| T2KDE ep130 | `model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth` |

The E2E checkpoint was extracted from the mattia_finder MLflow artifact (`.pyt` full model → state dict) using the `pvfinder` conda env, since the `.pyt` format embeds the `model` module path.

## Outstanding Issues

1. **No nTracks per truth PV in h5** — the flat h5 `pv` field has only z positions, no track counts. The nTracks≥2 filter therefore requires the ROOT file. Consider adding a `--no-root` fast mode that notes this caveat explicitly.

2. **σ_vtx_vtx fit quality** — the sigmoid fit sometimes has large uncertainty (±0.24 mm on a 0.34 mm value). More events or a tighter fit range would help.

3. **Pileup proxy inconsistency** — performance plot x-axis uses N truth PVs (from h5/ROOT), while mattia_finder uses `ActualNumOfInt`. These differ: `ActualNumOfInt` is the number of pp interactions, not reconstructed PVs. Should switch plot x-axis to `ActualNumOfInt` when ROOT is available.

4. **E2E checkpoint extraction** — `tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` was manually extracted. Other epoch checkpoints have not been extracted. Automate if needed.

5. **Split file limit** — `run_eval_pvf.py` is at exactly 500 lines (the pre-commit limit). Any further feature additions will require splitting the file (e.g., extract plotting helpers to `plots_pvf.py`).
