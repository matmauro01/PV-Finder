# Vertex Finding Models

Source: `src/pv_finder/models/`

## Architecture

Three-phase pipeline, each with its own model:

```
Phase 1: Tracks → KDE        (MaskedDNN)
Phase 2: KDE → Histogram     (UNet_1000)
Phase 3: Tracks → Histogram  (trackstoHists_UNet_1000, combines Phase 1 + 2)
```

### MaskedDNN (Phase 1)

Fully-connected network. Maps 7 track features to a 1000-bin KDE.
Masks out invalid tracks (z_0 <= -240). Output is summed per-track contributions.

- Input: (batch, 7, num_tracks)
- Output: (batch, 1000)
- Layers: 5 FC layers, 100 hidden nodes each, LeakyReLU + Softplus output

### UNet_1000 (Phase 2)

1-D UNet with skip connections. Refines a KDE into the final vertex histogram.

- Input: (batch, n_features, 1000)
- Output: (batch, 1000)
- 4-level encoder-decoder with ConvBNrelu blocks

### trackstoHists_UNet_1000 (Phase 3)

End-to-end model: MLP (same as MaskedDNN) feeding into UNet (same as UNet_1000).
Can be initialized from pretrained Phase 1 + 2 weights via `initialize_combined_weights.py`.

- Input: (batch, 7, num_tracks)
- Output: (batch, 1000)

## Loss Functions

- **MSELoss**: standard, used in Phase 3 (recommended)
- **Asymmetric loss** (`alt_loss_A.py`): from LHCb PV-Finder, emphasizes false positive suppression. Used in Phase 2.

## Standard Configuration

- KDE channel: **A_z only** (out_idx=0, n_latent_channels=1)
- 4-channel variant exists but A_z is the production standard

## Files

| File | Contents |
|------|----------|
| `autoencoder_models.py` | All model classes (1628 lines, needs splitting) |
| `alt_loss_A.py` | Asymmetric loss function |
