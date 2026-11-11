"""Packed-track inference wrapper for TracksToHist_v2 (CPU deployment).

The training model takes a padded ``(n_sub, 7, n_slots)`` tensor and runs the
per-track MLP, including the output softplus over ``1000 * n_latent`` bins, on
every slot. Padded slots are multiplied by zero before the sum over tracks, so
that work is wasted: at PU200 an event has ~1,400 real tracks but its 12
sub-events padded to the widest one have ~5,500 slots.

This wrapper is mathematically identical to ``TracksToHist_v2.forward`` but
takes only the real tracks of one event, packed into ``(n_tracks, 7)``, plus
the sub-event index of each track. Per-track contributions are scatter-added
into their sub-event. The only difference from the padded model is the
floating-point summation order. It uses the trained weights unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pv_finder.models.unet_v2 import TracksToHist_v2

N_BINS = 1000


class PackedTracksToHist(nn.Module):
    """Inference-only view of a trained ``TracksToHist_v2`` on packed tracks."""

    def __init__(self, model: TracksToHist_v2, n_sub: int = 12):
        super().__init__()
        self.mlp = model.t2kde
        self.k2h = model.k2h
        self.n_latent = model.n_latent
        self.n_sub = n_sub

    def forward(self, tracks: torch.Tensor, sub_idx: torch.Tensor) -> torch.Tensor:
        """Map packed tracks to the per-sub-event histogram.

        Args:
            tracks: ``(n_tracks, 7)`` real tracks only, features in the order
                of the training tensor's channel axis.
            sub_idx: ``(n_tracks,)`` int64 sub-event index in ``[0, n_sub)``.

        Returns:
            ``(n_sub, 1000)`` histogram, as ``TracksToHist_v2`` would return
            for the padded ``(n_sub, 7, n_slots)`` tensor.
        """
        m = self.mlp
        slope = m.LeakyReLU_param
        h = F.leaky_relu(m.linear1(tracks), slope)
        h = F.leaky_relu(m.linear2(h), slope)
        h = F.leaky_relu(m.linear3(h), slope)
        h = F.leaky_relu(m.linear4(h), slope)
        h = F.leaky_relu(m.linear5(h), slope)
        out = m.linear6(h)
        if not m.allow_negative_output:
            out = F.softplus(out)
        width = out.shape[1]
        kde = torch.zeros(self.n_sub, width, dtype=out.dtype, device=out.device)
        kde = kde.scatter_add(0, sub_idx.unsqueeze(1).expand(-1, width), out)
        kde = kde * m.predScaleFactor
        return self.k2h(kde.view(self.n_sub, self.n_latent, N_BINS))


def pack_event(
    x: torch.Tensor, mask_val: float = -240.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a padded ``(n_sub, 7, n_slots)`` event to packed inputs.

    A slot is a real track when its z0 (channel 1) exceeds ``mask_val``,
    the same rule ``MaskedDNN`` uses.
    """
    mask = x[:, 1, :] > mask_val
    sub_idx = torch.nonzero(mask)[:, 0]
    tracks = x.transpose(1, 2)[mask]
    return tracks.contiguous(), sub_idx
