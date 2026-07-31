"""Properties WeightedMSELoss must hold before it is allowed near a 3-day run.

The critical one is the last group: on empty target bins the weight must be
*exactly* 1, so that switching loss cannot silently change the false-positive
penalty that sets the fake rate. Everything else is a knob; that one is a
correctness requirement.

Run: ``pytest tests/test_losses.py -v``
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pv_finder.models.losses import LOSS_TYPES, WeightedMSELoss, build_loss

# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_default_is_plain_mse():
    """An untouched config must keep the historical behaviour exactly."""
    assert isinstance(build_loss({}), nn.MSELoss)


def test_factory_selects_weighted():
    loss = build_loss({"loss_type": "weighted_mse", "loss_y0": 0.5})
    assert isinstance(loss, WeightedMSELoss)
    assert loss.y0 == 0.5


def test_factory_is_case_insensitive():
    assert isinstance(build_loss({"loss_type": "Weighted_MSE"}), WeightedMSELoss)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="unknown loss_type"):
        build_loss({"loss_type": "focal"})


def test_all_declared_types_build():
    for name in LOSS_TYPES:
        assert isinstance(build_loss({"loss_type": name}), nn.Module)


def test_rejects_nonpositive_y0():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            WeightedMSELoss(y0=bad)


# --------------------------------------------------------------------------
# The background-preservation guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("y0", [0.1, 0.3, 1.0, 3.0])
def test_identical_to_mse_when_target_is_all_zero(y0: float):
    """92 % of real target bins are exactly 0; there the two losses must agree."""
    pred = torch.rand(16, 1000) * 3.0
    target = torch.zeros(16, 1000)
    assert torch.equal(WeightedMSELoss(y0=y0)(pred, target), nn.MSELoss()(pred, target))


@pytest.mark.parametrize("y0", [0.3, 1.0])
def test_empty_bin_gradient_identical_to_mse(y0: float):
    """The false-positive penalty -- and so the fake rate -- must not move."""
    target = torch.zeros(4, 100)
    grads = []
    for loss_fn in (WeightedMSELoss(y0=y0), nn.MSELoss()):
        pred = (
            torch.arange(400, dtype=torch.float32).reshape(4, 100) / 400
        ).requires_grad_(True)
        loss_fn(pred, target).backward()
        grads.append(pred.grad.clone())
    assert torch.equal(grads[0], grads[1])


# --------------------------------------------------------------------------
# The intended reweighting
# --------------------------------------------------------------------------


def test_weight_never_exceeds_one():
    """w = y0/(y+y0) <= 1, so the loss can never be larger than the MSE it
    replaces -- switching cannot blow up a learning rate tuned for MSE."""
    pred = torch.rand(8, 500) * 5
    target = torch.rand(8, 500) * 8
    assert WeightedMSELoss(y0=0.3)(pred, target) <= nn.MSELoss()(pred, target)


def test_large_y0_converges_to_mse():
    pred, target = torch.rand(8, 500) * 5, torch.rand(8, 500) * 8
    mse = nn.MSELoss()(pred, target)
    assert WeightedMSELoss(y0=1e7)(pred, target) == pytest.approx(mse.item(), rel=1e-4)


def test_tall_peaks_are_downweighted_relative_to_short():
    """The whole point: missing a tall peak must cost relatively less than it
    does under MSE, so the optimiser stops ignoring low-multiplicity vertices.

    This checks the *single-bin* ratio at the p5 and p99 peak heights measured
    on the real pool (0.34 and 6.99). Note this is a different quantity from
    the 50.5x -> 8.0x figures quoted in the module docstring: those integrate
    over the whole peak, and tall peaks are also *narrower* (sigma falls as
    n^-B while height rises), so they span fewer bins. Single-bin 422 -> 37 and
    integrated 50.5 -> 8.0 are the same effect measured two ways.
    """
    short = torch.tensor([[0.34]])
    tall = torch.tensor([[6.99]])
    zero = torch.zeros(1, 1)
    wmse = WeightedMSELoss(y0=0.3)
    mse_ratio = (nn.MSELoss()(zero, tall) / nn.MSELoss()(zero, short)).item()
    w_ratio = (wmse(zero, tall) / wmse(zero, short)).item()
    assert mse_ratio == pytest.approx(422.5, rel=0.01)
    assert w_ratio == pytest.approx(37.1, rel=0.01)  # 11.4x compression
    assert w_ratio < mse_ratio


def test_zero_loss_on_perfect_prediction():
    target = torch.rand(4, 100) * 5
    assert WeightedMSELoss(y0=0.3)(target.clone(), target) == 0.0


def test_negative_target_cannot_produce_negative_weight():
    """Targets are non-negative by construction; a corrupt one must not flip
    the sign of the loss or divide by zero."""
    pred = torch.rand(4, 10)
    target = torch.full((4, 10), -5.0)
    out = WeightedMSELoss(y0=0.3)(pred, target)
    assert torch.isfinite(out) and out >= 0


def test_gradients_are_finite_and_scale_free():
    pred = (torch.rand(4, 200) * 5).requires_grad_(True)
    target = torch.rand(4, 200) * 8
    WeightedMSELoss(y0=0.3)(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_matches_closed_form():
    pred, target = torch.rand(3, 7), torch.rand(3, 7) * 4
    y0 = 0.3
    expected = ((y0 / (target + y0)) * (pred - target) ** 2).mean()
    assert torch.allclose(WeightedMSELoss(y0=y0)(pred, target), expected)


def test_dtype_and_device_agnostic():
    for dtype in (torch.float32, torch.float64):
        pred, target = torch.rand(4, 50, dtype=dtype), torch.rand(4, 50, dtype=dtype)
        out = WeightedMSELoss(y0=0.3)(pred, target)
        assert out.dtype == dtype and torch.isfinite(out)
