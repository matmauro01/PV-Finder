"""The MaskedDNN padding crop must be bit-exact against the pre-crop model.

``_crop_and_zero_padding`` trims the all-padding tail of a tracks batch and
zeroes the padding sentinel. Both are meant to be pure optimisations: the model
multiplies every per-track contribution by the mask before summing, so padded
slots contribute exactly 0.0 either way.

"Meant to be" is not good enough for a change that sits in the forward pass of
the production checkpoint, so these tests compare against a reference
implementation of the *original* forward pass, on the real v4b weights and real
HDF5 batches when they are available.

Run: ``pytest tests/test_masked_dnn_crop.py -v``
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from pv_finder.models.autoencoder_models import MaskedDNN, _crop_and_zero_padding

MASK_VAL = -240.0
DATA_SENTINEL = -999999.0
H5_POOL = (
    "data/run4/PU200_corrected_h5/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.h5"
)
V4B_CKPT = (
    "model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup"
    "_phase2_epoch_3_fullstate.pth"
)


def reference_forward(model: MaskedDNN, x: torch.Tensor) -> torch.Tensor:
    """The MaskedDNN forward pass exactly as it was before the crop was added."""
    leaky = nn.LeakyReLU(model.LeakyReLU_param)
    softplus = nn.Softplus()
    n_evts = x.shape[0]
    mask = x[:, 1, :] > model.maskVal
    f2 = mask.float().unsqueeze(2).expand(-1, -1, model.output_size)
    h = x.transpose(1, 2)
    h = leaky(model.linear1(h))
    h = leaky(model.linear2(h))
    h = leaky(model.linear3(h))
    h = leaky(model.linear4(h))
    h = leaky(model.linear5(h))
    h = model.linear6(h)
    if not model.allow_negative_output:
        h = softplus(h)
    h = h.view(n_evts, -1, model.output_size)
    return torch.mul(torch.sum(torch.mul(f2, h), dim=1), model.predScaleFactor)


def make_model(output_size: int = 64, seed: int = 0) -> MaskedDNN:
    torch.manual_seed(seed)
    return MaskedDNN(
        input_size=7,
        hidden_nodes=[16] * 5,
        output_size=output_size,
        leaky_param=0.01,
        maskVal=MASK_VAL,
        predScaleFactor=0.001,
    ).eval()


def synthetic_batch(
    batch: int = 8, n_tracks: int = 64, seed: int = 1, left_packed: bool = True
) -> torch.Tensor:
    """A padded tracks batch with a realistic mix of occupancies, incl. empties."""
    g = torch.Generator().manual_seed(seed)
    x = torch.full((batch, 7, n_tracks), DATA_SENTINEL)
    counts = torch.randint(0, n_tracks // 2, (batch,), generator=g)
    counts[0] = 0  # an empty sub-event
    counts[1] = n_tracks  # a full one, so nothing can be cropped away
    for i, c in enumerate(counts.tolist()):
        if c == 0:
            continue
        vals = torch.randn(7, c, generator=g)
        vals[1] = torch.rand(c, generator=g) * 40.0 - 200.0  # z0 inside the window
        if left_packed:
            x[i, :, :c] = vals
        else:  # scatter the real tracks to random columns
            cols = torch.randperm(n_tracks, generator=g)[:c]
            x[i][:, cols] = vals
    return x


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------


def test_crop_keeps_every_real_track():
    x = synthetic_batch()
    mask = x[:, 1, :] > MASK_VAL
    x_c, mask_c = _crop_and_zero_padding(x, mask)
    assert mask_c.sum() == mask.sum()
    assert torch.equal(
        x_c[mask_c.unsqueeze(1).expand_as(x_c)],
        x[:, :, : x_c.shape[2]][mask_c.unsqueeze(1).expand_as(x_c)],
    )


def test_crop_zeroes_padding():
    x = synthetic_batch()
    mask = x[:, 1, :] > MASK_VAL
    x_c, mask_c = _crop_and_zero_padding(x, mask)
    padded = ~mask_c.unsqueeze(1).expand_as(x_c)
    assert torch.all(x_c[padded] == 0.0)


def test_crop_actually_shrinks_when_it_can():
    x = torch.full((4, 7, 128), DATA_SENTINEL)
    x[0, 1, :10] = 0.0  # ten real tracks in the first 10 columns only
    mask = x[:, 1, :] > MASK_VAL
    x_c, _ = _crop_and_zero_padding(x, mask)
    assert x_c.shape[2] == 10


def test_all_empty_batch_is_survivable():
    x = torch.full((4, 7, 128), DATA_SENTINEL)
    mask = x[:, 1, :] > MASK_VAL
    x_c, mask_c = _crop_and_zero_padding(x, mask)
    assert x_c.shape == (4, 7, 1)
    assert not mask_c.any()
    assert torch.all(x_c == 0.0)


# --------------------------------------------------------------------------
# Bit-exactness of the full forward pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("left_packed", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_bit_exact_synthetic(left_packed: bool, seed: int):
    model = make_model(seed=seed)
    x = synthetic_batch(seed=seed + 10, left_packed=left_packed)
    with torch.no_grad():
        got, want = model(x), reference_forward(model, x)
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_forward_bit_exact_all_empty():
    model = make_model()
    x = torch.full((4, 7, 128), DATA_SENTINEL)
    with torch.no_grad():
        assert torch.equal(model(x), reference_forward(model, x))


def test_gradients_bit_exact():
    """The crop must not perturb the backward pass either."""
    x = synthetic_batch(seed=99)
    grads = []
    for fwd in (lambda m, t: m(t), reference_forward):
        model = make_model(seed=7)
        out = fwd(model, x)
        out.pow(2).mean().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Against the real production weights and real data
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(H5_POOL) and os.path.exists(V4B_CKPT)),
    reason="production checkpoint or HDF5 pool not available",
)
def test_forward_bit_exact_v4b_on_real_data():
    import h5py

    from pv_finder.training.hllhc_helpers import build_model

    configs = {
        "models_config": {
            "model_type": "v2",
            "n_input_features": 7,
            "n_latent_channels": 4,
            "dropout": 0.25,
            "n_unet_channels": 280,
            "l_hidden_nodes": [128] * 5,
        }
    }
    model = build_model(configs, torch.device("cpu")).eval()
    ckpt = torch.load(V4B_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    with h5py.File(H5_POOL, "r") as fh:
        x = torch.from_numpy(fh["tracks"][:64]).float()

    # Compare the MLP stage (the only part the crop touches) bit-for-bit, then
    # the end-to-end histogram, which is what training actually optimises.
    with torch.no_grad():
        got_mlp, want_mlp = model.t2kde(x), reference_forward(model.t2kde, x)
        assert torch.equal(got_mlp, want_mlp), (got_mlp - want_mlp).abs().max().item()
        hist = model(x)
    assert hist.shape == (64, 1000)
    assert torch.isfinite(hist).all()
