"""Mid-epoch validation helpers for sub-epoch MLflow telemetry.

Used by train_hllhc_e2e.py to log Phase-2 ``p2_val_loss_step``,
``p2_eff_step`` and ``p2_fpr_step`` without paying the cost of a full
val sweep. Two tiers:

  - ``mini_val_mse``: forward-only on a cached batch list. ~2 s on
    sneezy at bs=256, 30 batches; ~1.5 % overhead if called every
    2000 train steps.
  - ``mini_val_full``: forward + per-item ``efficiency()``. ~25 s
    on the same cache (CPU vertex-matching dominates); ~2 % overhead
    if called every ~20 000 train steps, ~4 % at 10 000.

The cached batches are sampled once at the start of the run from the
val loader and held in CPU memory so the val-loss curve is stable
(not jittered by reshuffling per call).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def prefetch_val_batches(loader, n: int) -> list:
    """Pre-cache the first ``n`` batches of ``loader`` into a list."""
    cache: list = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        cache.append(batch)
    return cache


@torch.no_grad()
def mini_val_mse(
    model: nn.Module, batches: Sequence, loss_fn: nn.Module, device: torch.device
) -> float:
    """MSE-only val pass over the cached batch list. Returns mean loss."""
    model.eval()
    total = 0.0
    for inputs, labels in batches:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        total += loss_fn(model(inputs), labels[:, 0, :]).item()
    model.train()
    return total / max(len(batches), 1)


@torch.no_grad()
def mini_val_full(
    model: nn.Module, batches: Sequence, loss_fn: nn.Module, device: torch.device
):
    """MSE + per-item efficiency() pass.

    Returns ``(mean_loss, eff)`` where ``eff`` is a ``ValueSet`` accumulator
    you can read ``.eff_rate`` / ``.fp_rate`` from. Slow because the
    matching loop is per-item CPU work.
    """
    # Local imports to avoid pulling pv_finder.training.training (which has its
    # own heavy module-level state) into modules that just want the cheap MSE.
    from pv_finder.training.training import PARAM_EFF, efficiency
    from pv_finder.utils.efficiency import ValueSet

    model.eval()
    total = 0.0
    eff = ValueSet(0, 0, 0, 0)
    for inputs, labels in batches:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        out = model(inputs)
        total += loss_fn(out, labels[:, 0, :]).item()
        for label, output in zip(labels[:, 0, :].cpu().numpy(), out.cpu().numpy()):
            eff += efficiency(label, output, **PARAM_EFF)
    model.train()
    return total / max(len(batches), 1), eff
