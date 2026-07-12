"""GNN training loop for Track-to-Vertex Association.

BCEWithLogitsLoss with per-batch dynamic pos_weight to handle class imbalance.
Migrated from atlas_pvfinder/tracks_to_vertex/model/training_graph.py.
"""

from __future__ import annotations

import sys
import time
from collections import namedtuple

import torch
import torch.nn as nn

from pv_finder.training.training import (
    get_gradient_momentum_angle,
    get_gradient_norm,
    get_momentum_norm,
    get_update_norm,
)
from pv_finder.utils.utilities import get_device_from_model, import_progress_bar

GNNResults = namedtuple("GNNResults", ["epoch", "cost", "val", "time"])


def trainNet(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    val_loader,
    n_epochs: int,
    *,
    notebook: bool | None = None,
    epoch_start: int = 0,
):
    """Run GNN training, yielding GNNResults after each epoch."""
    if not notebook:
        print("{:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"n_epochs: {n_epochs}\n"
            f"batch_size: {train_loader.batch_size} events\n"
            f"dataset_train: {len(train_loader)} batches\n"
            f"dataset_val: {len(val_loader)} batches\n"
            f"optimizer: {optimizer}\nmodel: {model}"
        )
        print("=" * 80)

    progress = import_progress_bar(notebook)
    device = get_device_from_model(model)
    print("Model is in device: ", device)
    print(f"Number of batches: train = {len(train_loader)}, val = {len(val_loader)}")

    epoch_iterator = progress(
        range(epoch_start, n_epochs),
        desc="Epochs",
        postfix="train=start, val=start",
        dynamic_ncols=True,
        position=0,
        file=sys.stderr,
    )

    for epoch in epoch_iterator:
        training_start_time = time.time()

        total_train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)

        total_val_loss = _validate_one_epoch(
            model, val_loader, device, progress=progress
        )
        val_epoch = total_val_loss / len(val_loader)
        time_epoch = time.time() - training_start_time

        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Gradient diagnostics
        gradient_norm = get_gradient_norm(model)
        update_norm = get_update_norm(optimizer)
        momentum_norm = get_momentum_norm(optimizer)
        grad_momentum_angle = get_gradient_momentum_angle(optimizer)

        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, "
            f"took {time_epoch:.5} s"
        )
        write(
            f"  Gradient Norm: {gradient_norm:.4f}, Update Norm: {update_norm:.4f}, "
            f"Momentum Norm: {momentum_norm:.4f}, Angle: {grad_momentum_angle:.4f}"
        )

        yield GNNResults(epoch, cost_epoch, val_epoch, time_epoch)


def _compute_bce_loss(
    outputs: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """BCEWithLogitsLoss with dynamic pos_weight computed from batch labels."""
    num_zeros = (labels == 0).sum()
    num_ones = (labels == 1).sum()
    pos_weight = torch.tensor(
        [num_zeros / num_ones], dtype=torch.float32, device=device
    )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, labels)


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    progress,
) -> float:
    """Single training epoch. Returns total loss summed over batches."""
    total_loss = 0.0
    model.train()
    loader = progress(
        loader,
        postfix="train=start",
        desc="Training",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )
    for batch in loader:
        batch = batch.to(device)
        labels = batch[("track", "to", "pv")].y.to(device).float()

        optimizer.zero_grad()
        outputs = model(batch)
        loss = _compute_bce_loss(outputs, labels, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss.data.item():.4g}"
    return total_loss


def _validate_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    progress,
) -> float:
    """Single validation epoch. Returns total loss summed over batches."""
    total_loss = 0.0
    model.eval()
    loader = progress(
        loader,
        postfix="val=start",
        desc="Validation",
        mininterval=0.5,
        dynamic_ncols=True,
        position=1,
        leave=False,
        file=sys.stderr,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch[("track", "to", "pv")].y.to(device).float()
            val_outputs = model(batch)
            loss = _compute_bce_loss(val_outputs, labels, device)
            total_loss += loss.data.item()
    return total_loss
