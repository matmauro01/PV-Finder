# Training — Vertex Finding

Source: `src/pv_finder/training/`
Configs: `configs/vertex_finding/`

## Two Training Strategies

### Strategy A: Two-phase (separate)

Train Phase 1 and Phase 2 independently, then combine.

```
1. Train MaskedDNN:   tracks → KDE_A_z         (train_tracks_to_kde.py)
2. Train UNet_1000:   KDE_A_z → histogram      (train_kde_to_hist.py)
3. Initialize combined model from Phase 1 + 2   (initialize_combined_weights.py)
4. Fine-tune end-to-end                         (train_tracks_to_hist.py)
```

### Strategy B: End-to-end with pretrained init

Train trackstoHists_UNet_1000 directly, initialized from Strategy A weights.

```
1. (Optional) Initialize from pretrained weights
2. Train end-to-end: tracks → histogram         (train_tracks_to_hist.py)
```

### Strategy C: End-to-end without KDEs (MLP warmup)

No KDE supervision at all. Two-phase approach to avoid degenerate solutions.

```
1. Phase 1: Train MLP only on histogram targets, UNet frozen  (50 epochs)
2. Phase 2: Train MLP + UNet end-to-end on histograms         (400 epochs)
```

Script: `train_mlp_hist_then_e2e.py`
Config: `config_mlp_hist_e2e.yml`

Run:
```bash
python src/pv_finder/training/train_mlp_hist_then_e2e.py \
    -c configs/vertex_finding/config_mlp_hist_e2e.yml
```

Why two phases: training MLP+UNet jointly from random init on histogram MSE finds
degenerate solutions (peaks at bin 0). Warming up the MLP first gives it a reasonable
spatial mapping before the UNet co-adapts.

## Configs

| Config | Phase | Key settings |
|--------|-------|-------------|
| `config_T2KDE_A_z_reproduction.yml` | 1 | 200 epochs, lr=0.001, batch=128 |
| `config_KDE2HIST_matmauro.yml` | 2 | 200 epochs, lr=0.0001, batch=128 |
| `config_T2HIST_matmauro.yml` | 3 | 100 epochs, lr=0.001, MSE loss |
| `config_T2HIST_combined_200epochs.yml` | 3 | 400 epochs, initialized from Phase 1+2 |
| `config_mlp_hist_e2e.yml` | C | 50+400 epochs, no KDE, MLP warmup |

## MLflow

Experiment tracking via MLflow. Logs: loss, validation loss, efficiency, FPR, model weights.

Tracking URI: `file:<repo_root>/mlruns`

## Performance Targets

| Metric | 1-channel (A_z) |
|--------|-----------------|
| Resolution (sigma) | ~0.34 mm |
| Efficiency | ~93% |
| False positive rate | ~1.85 |

## Files

| File | Purpose |
|------|---------|
| `train_tracks_to_kde.py` | Phase 1 training script |
| `train_kde_to_hist.py` | Phase 2 training script |
| `train_tracks_to_hist.py` | Phase 3 / end-to-end training script |
| `initialize_combined_weights.py` | Combines Phase 1 + 2 weights |
| `train_mlp_hist_then_e2e.py` | Strategy C: KDE-free end-to-end with MLP warmup |
| `training.py` | `trainNet()` loop, GPU selection |
