# Training — Vertex Finding

Source: `src/pv_finder/training/`
Configs: `configs/vertex_finding/`

## Training Strategies

### Strategy A: Two-phase (separate), then fine-tune

Train MaskedDNN and UNet independently, combine weights, then fine-tune end-to-end.

```
1. Train MaskedDNN:   tracks → KDE_A_z         (train_tracks_to_kde.py)
2. Train UNet_1000:   KDE_A_z → histogram      (train_kde_to_hist.py)
3. Initialize combined model from Phase 1 + 2   (initialize_combined_weights.py)
4. Fine-tune end-to-end                         (train_tracks_to_hist.py)
```

**Step 1 — Tracks to KDE** (MaskedDNN, 200 epochs):
```bash
python -m pv_finder.training.train_tracks_to_kde \
    -c configs/vertex_finding/config_T2KDE_A_z_reproduction.yml
```

**Step 2 — KDE to Histogram** (UNet, 200 epochs, independent of Step 1):
```bash
python -m pv_finder.training.train_kde_to_hist \
    -c configs/vertex_finding/config_KDE2HIST_matmauro.yml
```

**Step 3 — Initialize combined model** from Step 1 + 2 weights:
```bash
python src/pv_finder/training/initialize_combined_weights.py \
    --t2kde model_weights/<best_t2kde>.pyt \
    --kde2hist model_weights/<best_kde2hist>.pyt \
    --output model_weights/initialized_t2hist.pth \
    --n-latent 1 --n-features 7 --dropout 0.25
```

**Step 4 — Fine-tune end-to-end** (400 epochs, MSE loss):
```bash
python -m pv_finder.training.train_tracks_to_hist \
    -c configs/vertex_finding/config_T2HIST_combined_200epochs.yml
```

This config points to the Step 1/2 weights via `pretrained_tracks2kde` and
`pretrained_kde2hist` keys and initializes automatically (no separate Step 3 needed).
Alternatively, use `config_T2HIST_matmauro.yml` with a pre-built `.pth` from Step 3.

### Strategy B: End-to-end without KDEs (MLP warmup)

No KDE supervision at all. Two-phase approach to avoid degenerate solutions.

```
1. Phase 1: Train MLP only on histogram targets, UNet frozen  (50 epochs)
2. Phase 2: Train MLP + UNet end-to-end on histograms         (400 epochs)
```

Run:
```bash
python -m pv_finder.training.train_mlp_hist_then_e2e \
    -c configs/vertex_finding/config_mlp_hist_e2e.yml
```

Why two phases: training MLP+UNet jointly from random init on histogram MSE finds
degenerate solutions (peaks at bin 0). Warming up the MLP first gives it a reasonable
spatial mapping before the UNet co-adapts.

## Configs

| Config | Strategy | Key settings |
|--------|----------|-------------|
| `config_T2KDE_A_z_reproduction.yml` | A step 1 | 200 epochs, lr=0.001, batch=128 |
| `config_T2KDE_400ep_03_24_2026.yml` | A step 1 | 400 epochs, lr=0.001, batch=128 |
| `config_KDE2HIST_matmauro.yml` | A step 2 | 200 epochs, lr=0.0001, batch=128 |
| `config_KDE2HIST_400ep_03_24_2026.yml` | A step 2 | 400 epochs, lr=0.0001, batch=128 |
| `config_T2HIST_matmauro.yml` | A step 4 | 100 epochs, lr=0.001, MSE loss, pre-built .pth |
| `config_T2HIST_combined_200epochs.yml` | A step 4 | 400 epochs, auto-init from step 1+2 weights |
| `config_T2HIST_400ep_03_24_2026.yml` | A step 4 | 400 epochs, T2KDE ep100 + K2H ep150, MSE, Qi Bin repro |
| `config_mlp_hist_e2e.yml` | B | 50+400 epochs, no KDE, MLP warmup |
| `config_KDE2HIST_v2.yml` | A step 2 (v2) | 200 epochs, lr=0.0001, UNet_1000_v2 |

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
| `train_tracks_to_kde.py` | Strategy A step 1: tracks to KDE |
| `train_kde_to_hist.py` | Strategy A step 2: KDE to histogram |
| `initialize_combined_weights.py` | Strategy A step 3: combine step 1+2 weights |
| `train_tracks_to_hist.py` | Strategy A step 4: end-to-end fine-tuning |
| `train_mlp_hist_then_e2e.py` | Strategy B: KDE-free end-to-end with MLP warmup |
| `training.py` | `trainNet()` loop, GPU selection |
