"""Helpers for the HLLHC PU200 E2E training script.

This module contains the small, focused utilities used by
``train_hllhc_e2e.py``:

  * ``build_model`` — instantiate the v1 or v2 architecture from a config.
  * ``build_phase2_scheduler`` — build a Phase 2 LR schedule
    (linear warmup -> cosine, or cosine warm restarts).
  * ``clip_and_step`` — clip gradients (optional) and step the optimizer.
  * Per-architecture dispatch helpers: ``is_v2``, ``get_param_groups_v2``,
    ``forward_mlp_v2``, ``phase1_forward``.
  * ``load_model_state`` — load only ``model_state`` from a fullstate ckpt.
  * ``tv_loss`` — 1-D total-variation regulariser.

Splitting these out keeps ``train_hllhc_e2e.py`` under the project's
500-line pre-commit cap and makes the helpers individually testable.
"""

from __future__ import annotations

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

from pv_finder.models.autoencoder_models import MaskedDNN, trackstoHists_UNet_1000
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2
from pv_finder.training.train_mlp_hist_then_e2e import (
    _squeeze_hist,
    forward_mlp_hist,
)

__all__ = [
    "build_model",
    "build_phase2_scheduler",
    "clip_and_step",
    "forward_mlp_v2",
    "get_param_groups_v2",
    "is_v2",
    "load_model_state",
    "phase1_forward",
    "train_phase1_epoch",
    "tv_loss",
    "validate_phase1_epoch",
]


# --- Model construction ---
def build_model(configs: dict, device: torch.device) -> nn.Module:
    """Instantiate the v1 (``trackstoHists_UNet_1000``) or v2 model from config.

    The model is moved to ``device`` before being returned. Selection is driven
    by ``configs["models_config"]["model_type"]`` (``"v1"`` or ``"v2"``).
    """
    mc = configs["models_config"]
    model_type = mc.get("model_type", "v1")
    if model_type == "v2":
        n_latent = mc.get("n_latent_channels", 1)
        t2kde = MaskedDNN(
            input_size=mc["n_input_features"],
            hidden_nodes=list(mc.get("l_hidden_nodes", [100] * 5)),
            output_size=1000 * n_latent,
            leaky_param=0.01,
            maskVal=-240.0,
            predScaleFactor=0.001,
        )
        k2h = UNet_1000_v2(
            n=mc.get("n_unet_channels", 64),
            n_features=n_latent,
            dropout_p=mc["dropout"],
        )
        model = TracksToHist_v2(t2kde, k2h)
    else:
        model = trackstoHists_UNet_1000(
            n_InputFeatures=mc["n_input_features"],
            n_LatentChannels=mc["n_latent_channels"],
            dropout=mc["dropout"],
            n_UNetChannels=mc.get("n_unet_channels", 64),
            l_HiddenNodes=list(mc.get("l_hidden_nodes", [100] * 5)),
        )
    return model.to(device)


# --- Phase 2 scheduler ---
def build_phase2_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    eta_min_frac: float,
    warm_restarts_t0: int = 0,
    steps_per_epoch: int = 0,
    warmup_steps: int = 0,
):
    """Build the Phase 2 learning-rate schedule.

    Behaviour:
      * ``warm_restarts_t0 > 0`` -> ``CosineAnnealingWarmRestarts``.
      * ``warmup_epochs > 0`` -> linear warmup followed by ``CosineAnnealingLR``.
      * otherwise -> plain ``CosineAnnealingLR`` over ``total_epochs``.

    ``eta_min`` is ``base_lr * eta_min_frac``.

    Scheduling unit
    ---------------
    When ``steps_per_epoch <= 0`` (default) the schedule is built in **epoch**
    units and the caller must ``scheduler.step()`` once per epoch — the legacy
    behaviour, preserved exactly.

    When ``steps_per_epoch > 0`` the schedule is built in **step (batch)** units
    and the caller must ``scheduler.step()`` once per batch. This makes a short
    warmup actually ramp (e.g. ``warmup_steps=3000`` -> smooth 0.01·lr → lr over
    3000 batches) instead of a per-epoch ``LinearLR(total_iters=1)`` that holds
    the whole first epoch at 0.01·lr and then jumps 100x at the epoch boundary.
    ``warmup_steps`` (if > 0) sets the warmup length directly in steps; otherwise
    it defaults to ``warmup_epochs * steps_per_epoch``.
    """
    eta_min = base_lr * eta_min_frac
    per_step = steps_per_epoch > 0
    unit = steps_per_epoch if per_step else 1

    if warm_restarts_t0 > 0:
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, warm_restarts_t0 * unit), T_mult=1, eta_min=eta_min
        )

    if per_step and warmup_steps > 0:
        w_iters = warmup_steps
    elif warmup_epochs > 0:
        w_iters = warmup_epochs * unit
    else:
        w_iters = 0

    if w_iters > 0:
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=w_iters
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs * unit - w_iters), eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[w_iters]
        )
    return CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs * unit), eta_min=eta_min
    )


# --- Shared training-step utilities ---
def clip_and_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_norm: float | None,
) -> None:
    """Clip ``model``'s gradients (if ``max_norm`` is set) then step optimizer."""
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()


def tv_loss(pred: torch.Tensor) -> torch.Tensor:
    """1-D total-variation penalty along the last axis of ``pred``."""
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))


# --- Per-architecture dispatch ---
def is_v2(model: nn.Module) -> bool:
    """Return ``True`` if ``model`` is a v2 (``TracksToHist_v2``) instance."""
    return isinstance(model, TracksToHist_v2)


def get_param_groups_v2(
    model: TracksToHist_v2,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(mlp_params, unet_params)`` for the v2 architecture."""
    return list(model.t2kde.parameters()), list(model.k2h.parameters())


def forward_mlp_v2(model: TracksToHist_v2, inputs: torch.Tensor) -> torch.Tensor:
    """Run the v2 MLP (``t2kde``) and reshape to ``(B, n_latent, 1000)``."""
    kde = model.t2kde(inputs)  # (B, n_latent * 1000)
    return kde.view(kde.shape[0], model.n_latent, 1000)


def phase1_forward(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Phase 1 MLP-only forward, dispatching by model type (v1 vs v2)."""
    if is_v2(model):
        return _squeeze_hist(forward_mlp_v2(model, inputs))
    return _squeeze_hist(forward_mlp_hist(model, inputs))


# --- Checkpoint loading ---
def load_model_state(path: str, model: nn.Module) -> int:
    """Load ``model_state`` only from a fullstate ckpt at ``path``.

    Optimizer state is intentionally skipped: this is used to resume Phase 2
    from a Phase 1 checkpoint, where the optimizer differs.

    Returns the integer epoch recorded in the checkpoint, or ``-1`` if absent.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", -1))


# --- Phase 1 epoch loops (MLP warmup, UNet frozen) ---
def train_phase1_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    max_norm: float | None,
    log_every: int = 500,
) -> float:
    """Run one Phase 1 (MLP-only, UNet frozen) epoch; returns mean batch loss."""
    model.train()
    total = 0.0
    win_sum, win_n = 0.0, 0
    n_per = len(loader)
    pbar = tqdm(
        loader, desc=f"P1 E{epoch_idx}", leave=False, ncols=100, mininterval=2.0
    )
    for step, (inputs, target_split) in enumerate(pbar):
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        optimizer.zero_grad()
        pred = phase1_forward(model, inputs)
        loss = loss_fn(pred, target_hist)
        loss.backward()
        clip_and_step(model, optimizer, max_norm)
        batch_loss = loss.item()
        total += batch_loss
        win_sum += batch_loss
        win_n += 1
        gstep = (epoch_idx - 1) * n_per + step + 1
        if log_every > 0 and gstep % log_every == 0:
            mlflow.log_metric("p1_train_loss_step", win_sum / win_n, step=gstep)
            mlflow.log_metric("p1_lr", optimizer.param_groups[0]["lr"], step=gstep)
            win_sum, win_n = 0.0, 0
        pbar.set_postfix({"loss": f"{total / (step + 1):.6f}"})
    return total / len(loader)


@torch.no_grad()
def validate_phase1_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Validation pass through the MLP-only Phase 1 forward."""
    model.eval()
    total = 0.0
    for inputs, target_split in loader:
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        pred = phase1_forward(model, inputs)
        total += loss_fn(pred, target_hist).item()
    return total / len(loader)
