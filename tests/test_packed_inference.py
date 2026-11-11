"""PackedTracksToHist must reproduce TracksToHist_v2 on padded input."""

import torch

from pv_finder.models.packed_inference import PackedTracksToHist, pack_event
from pv_finder.training.hllhc_helpers import build_model


def _model(width: int, latent: int) -> torch.nn.Module:
    cfg = {
        "models_config": {
            "model_type": "v2",
            "n_input_features": 7,
            "n_latent_channels": latent,
            "dropout": 0.25,
            "n_unet_channels": width,
            "l_hidden_nodes": [32] * 5,
        }
    }
    torch.manual_seed(0)
    model = build_model(cfg, torch.device("cpu")).eval()
    # Non-trivial BatchNorm statistics so eval mode is actually exercised.
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return model


def _padded_event(n_slots: int = 90) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(12, 7, n_slots, generator=g)
    x[:, 1, :] *= 50.0  # z0 in mm
    n_real = torch.randint(0, n_slots, (12,), generator=g)
    n_real[3] = 0  # one empty sub-event
    n_real[5] = n_slots  # one full sub-event
    for s, n in enumerate(n_real.tolist()):
        x[s, :, n:] = -999999.0  # padding sentinel, as in the h5 files
    return x


def test_packed_matches_padded() -> None:
    for width, latent in [(16, 1), (24, 4)]:
        model = _model(width, latent)
        x = _padded_event()
        with torch.no_grad():
            ref = model(x)
            tracks, idx = pack_event(x)
            got = PackedTracksToHist(model)(tracks, idx)
        assert got.shape == ref.shape == (12, 1000)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_pack_event_drops_padding_only() -> None:
    x = _padded_event()
    tracks, idx = pack_event(x)
    n_real = int((x[:, 1, :] > -240.0).sum())
    assert tracks.shape == (n_real, 7)
    assert idx.shape == (n_real,)
    assert not (tracks[:, 1] <= -240.0).any()
    assert int(idx.max()) <= 11 and 3 not in idx.tolist()
