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

### Strategy B (HLLHC PU200): stabilized LR + grad clipping

Same phase structure as Strategy B, but with the training recipe tuned for the much
higher pileup of HLLHC PU200 data. The original recipe (`lr=1e-3` throughout Phase 2)
diverges mid-training around Phase 2 epoch 100, slowly recovers, and produces a
noticeably worse model — the per-event MSE gradient scales up with pileup so the
same LR is effectively much hotter at μ≈200.

Fixes (all in `train_hllhc_e2e.py`):

- **Phase 2 LR: `1e-3 → 1e-4`** (main fix).
- **Phase 2 LR schedule**: 5-epoch linear warmup from `0.01·lr → lr`, then cosine
  decay to `eta_min = lr × 0.01`. Fresh Adam at Phase 2 start (Phase 1 optimizer
  state is discarded).
- **Gradient clipping** (`max_grad_norm=1.0`) on both phases. Insurance against
  single-batch gradient spikes on outlier events.
- **Modestly larger model**: `n_UNetChannels` 64 → 96, `l_HiddenNodes`
  `[100]*5 → [128]*5`. Total parameters 359K → 681K (≈1.9×).

Phase 1 LR stays at `1e-3` because Phase 1 (MLP warmup with UNet frozen) converges
fine at v1 settings.

Run:
```bash
python -m pv_finder.training.train_hllhc_e2e \
    -c configs/vertex_finding/config_hllhc_pu200_e2e_v2.yml
```

`train_hllhc_e2e.py` is a separate script (not a config flag on
`train_mlp_hist_then_e2e.py`) because Phase 2 needs a custom loop to inject grad
clipping and the scheduler without modifying the shared `trainNet` helper. It
imports the Phase 1 MLP-only forward pass (`forward_mlp_hist`, `_squeeze_hist`)
from `train_mlp_hist_then_e2e.py` to avoid duplication.

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
| `config_hllhc_pu200_e2e.yml` | B (HLLHC v1) | 50+400 epochs, HLLHC PU200 — **diverged**, kept for reference |
| `config_hllhc_pu200_e2e_v2.yml` | B (HLLHC v2) | 50+400 epochs, Phase 2 lr=1e-4 + warmup + cosine + grad clip, 96-ch UNet, 128-node MLP |
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
| `train_hllhc_e2e.py` | Strategy B (HLLHC): adds Phase 2 LR warmup + cosine decay + grad clipping |
| `training.py` | `trainNet()` loop, GPU selection |
