"""Loss functions for histogram regression, and the factory the trainers use.

Background
----------
The training target is a 1000-bin histogram in which each truth PV appears as a
Gaussian whose width follows ``sigma(n) = A*n^-B + C`` and whose amplitude is
scaled by ``max(1, 0.15/sigma)`` (``root_to_h5._build_truth_histogram``). Peak
height is therefore a steeply increasing function of the vertex track
multiplicity: measured on ``PU200_corrected_h5`` (6000 sub-events, 46 662
peaks) the 5th and 99th percentiles of peak height are 0.34 and 6.99, a 20.5x
spread.

Under plain MSE the loss incurred by *missing a peak entirely* scales with the
square of its amplitude, so the optimiser cares far more about the tall
(high-multiplicity) peaks it already finds than about the short ones where the
misses actually live. Measured on the same sample, grouping peaks into height
quintiles and computing the loss of predicting exactly zero over each peak:

===================  ==============  ==================  =================
height band          <height>        miss-loss, MSE      miss-loss, w-MSE
                                                         (``y0 = 0.3``)
===================  ==============  ==================  =================
0.05 - 0.51          0.38             1.0x                1.0x
0.51 - 0.86          0.68             2.5x                1.9x
0.86 - 1.54          1.18             6.2x                3.1x
1.54 - 2.68          2.04            14.8x                4.8x
2.68 - 16.5          4.20            **50.5x**            **8.0x**
===================  ==============  ==================  =================

``WeightedMSELoss`` compresses that 50x tilt to 8x.

Design constraint: do not disturb the background
------------------------------------------------
92.4 % of target bins are exactly zero, and the penalty the model pays for
putting amplitude in an empty bin is what controls the fake rate. A naive
inverse-variance weight ``1/(y+y0)`` gives empty bins a weight of ``1/y0``
(3.33 at ``y0=0.3``), i.e. it silently triples the false-positive penalty at the
same time as it rebalances the peaks — two changes at once, and the second one
is not the one we want.

``WeightedMSELoss`` therefore uses the *normalised* form

.. math:: w(y) = \\frac{y_0}{y + y_0}

so that ``w(0) = 1`` **exactly**. Empty bins are treated bit-identically to
plain MSE, and the only thing that changes is that tall peaks are down-weighted
relative to short ones. ``y0 -> inf`` recovers plain MSE exactly.

Because ``w <= 1`` everywhere, the loss (and hence the gradient scale) is never
larger than the MSE it replaces, so switching to it cannot destabilise a
learning rate that was tuned for MSE. It does make the loss *smaller*, which
lowers the effective step size somewhat -- see the note in
``docs/training/vertex_finding.md``.

Caveat that the numbers above cannot settle
-------------------------------------------
Re-weighting the peaks changes the amplitude scale the network converges to, so
**every downstream operating point (height floor, integral thresholds, the GBT
fake gate, the TTVA augmentation quantiles) must be re-tuned** for a model
trained this way. This is an A/B candidate, not a drop-in replacement: the
default in every trainer remains ``mse``.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["WeightedMSELoss", "build_loss", "LOSS_TYPES"]

LOSS_TYPES = ("mse", "weighted_mse")


class WeightedMSELoss(nn.Module):
    """MSE with per-bin weight ``y0 / (target + y0)``.

    Down-weights tall (high track-multiplicity) target peaks relative to short
    ones while leaving empty bins weighted exactly 1, i.e. identical to
    ``nn.MSELoss`` on the 92 % of bins that are zero.

    Args:
        y0: Amplitude scale at which the weight falls to 1/2. Smaller values
            flatten the peak-height tilt more aggressively. ``y0 = 0.3`` sits
            just below the 5th-percentile peak height (0.34), which compresses
            the measured 50.5x miss-loss tilt to 8.0x. Must be positive.

    Shape:
        Both arguments are broadcast together; the result is the mean over all
        elements, matching ``nn.MSELoss(reduction="mean")``.
    """

    def __init__(self, y0: float = 0.3) -> None:
        super().__init__()
        if not y0 > 0:
            raise ValueError(f"y0 must be positive, got {y0}")
        self.y0 = float(y0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target >= 0 by construction (Gaussian-CDF amplitudes), so the
        # denominator cannot vanish; clamp defensively anyway so a corrupt
        # target can never produce a negative weight or a division by zero.
        weight = self.y0 / (target.clamp_min(0.0) + self.y0)
        return (weight * (prediction - target) ** 2).mean()

    def extra_repr(self) -> str:
        return f"y0={self.y0}"


def build_loss(configs: dict) -> nn.Module:
    """Build the training loss from a config dict.

    Reads ``loss_type`` (default ``"mse"``, i.e. unchanged behaviour) and, for
    ``"weighted_mse"``, ``loss_y0`` (default 0.3).

    Raises:
        ValueError: if ``loss_type`` is not one of :data:`LOSS_TYPES`.
    """
    loss_type = str(configs.get("loss_type", "mse")).lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "weighted_mse":
        return WeightedMSELoss(y0=float(configs.get("loss_y0", 0.3)))
    raise ValueError(
        f"unknown loss_type {loss_type!r}; expected one of {list(LOSS_TYPES)}"
    )
