# copied directly from https://gitlab.cern.ch/LHCb-Reco-Dev/pv-finder/-/blob/master/model/training.py
import os
import sys
import time
from collections import namedtuple

import torch

from pv_finder.utils.efficiency import (  # rudimentary efficiency function for gauging progress while training
    ValueSet,
    efficiency,
)
from pv_finder.utils.utilities import (
    get_device_from_model,
    import_progress_bar,
)

Results = namedtuple("Results", ["epoch", "cost", "val", "time", "eff_val"])

PARAM_EFF = {
    "difference": 5.0,
    "threshold": 1e-2,
    "integral_threshold": 0.5,
    "min_width": 3,
}


def select_gpu(selection=None):
    """
    Select a GPU if availale.

    selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")

    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("CPU")
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def get_update_norm(optimizer):
    total_norm = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                update = -group["lr"] * p.grad  # Update step
                param_norm = update.norm(2)
                total_norm += param_norm.item() ** 2
    return total_norm**0.5


def get_momentum_norm(optimizer):
    total_norm = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                momentum_norm = optimizer.state[p]["exp_avg"].norm(2)
                total_norm += momentum_norm.item() ** 2
    return total_norm**0.5


def get_gradient_momentum_angle(optimizer):
    dot_product = 0.0
    grad_norm = 0.0
    momentum_norm = 0.0

    for group in optimizer.param_groups:
        for p in group["params"]:
            if (
                p.grad is not None
                and p in optimizer.state
                and "exp_avg" in optimizer.state[p]
            ):
                grad = p.grad.detach().flatten()
                momentum = optimizer.state[p]["exp_avg"].detach().flatten()

                dot_product += (grad * momentum).sum().item()
                grad_norm += grad.norm(2).item() ** 2
                momentum_norm += momentum.norm(2).item() ** 2

    grad_norm = grad_norm**0.5
    momentum_norm = momentum_norm**0.5

    if grad_norm == 0 or momentum_norm == 0:
        return 0  # Avoid division by zero

    return dot_product / (grad_norm * momentum_norm)  # Cosine similarity


def trainNet(
    model,
    optimizer,
    loss,
    train_loader,
    val_loader,
    n_epochs,
    out_idx,
    *,
    notebook=None,
    epoch_start=0,
):
    """
    If notebook = None, no progress bar will be drawn. If False, this will be a terminal progress bar.
    """

    # Print all of the hyperparameters of the training iteration
    if not notebook:
        print("{:=^80}".format(" HYPERPARAMETERS "))
        print(
            f"""\
n_epochs: {n_epochs}
batch_size: {train_loader.batch_size} events
dataset_train: {len(train_loader)} events
dataset_val: {len(val_loader)} events
loss: {loss}
optimizer: {optimizer}
model: {model}"""
        )
        print("=" * 80)

    # Set up notebook or regular progress bar (or none)
    progress = import_progress_bar(notebook)

    # Get the current device
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

    # Loop for n_epochs
    for epoch in epoch_iterator:
        # print("Epoch: ", epoch)
        training_start_time = time.time()

        # Run the training step
        total_train_loss = train(
            model, loss, train_loader, optimizer, device, out_idx, progress=progress
        )
        cost_epoch = total_train_loss / len(train_loader)

        # At the end of the epoch, do a pass on the validation set
        total_val_loss, cur_val_eff = validate(
            model, loss, val_loader, device, out_idx, progress=progress
        )
        val_epoch = total_val_loss / len(val_loader)

        # Record total time
        time_epoch = time.time() - training_start_time

        # Pretty print a description
        if hasattr(epoch_iterator, "postfix"):
            epoch_iterator.postfix = f"train={cost_epoch:.4}, val={val_epoch:.4}"

        # Compute gradient-related metrics
        gradient_norm = get_gradient_norm(model)
        update_norm = get_update_norm(optimizer)
        momentum_norm = get_momentum_norm(optimizer)
        grad_momentum_angle = get_gradient_momentum_angle(optimizer)

        # Redirect stdout if needed to avoid clash with progress bar
        write = getattr(progress, "write", print)
        write(
            f"Epoch {epoch}: train={cost_epoch:.6}, val={val_epoch:.6}, took {time_epoch:.5} s"
        )
        write(f"  Validation {cur_val_eff}")
        write(
            f"  Gradient Norm: {gradient_norm:.4f}, Update Norm: {update_norm:.4f}, Momentum Norm: {momentum_norm:.4f}, Angle: {grad_momentum_angle:.4f}"
        )

        yield Results(epoch, cost_epoch, val_epoch, time_epoch, cur_val_eff)


def train(model, loss, loader, optimizer, device, out_idx, progress):
    total_loss = 0.0

    # switch to train mode
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
    for inputs, labels in loader:
        if inputs.device != device:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
        # Set the parameter gradients to zero
        optimizer.zero_grad()
        # Forward pass, backward pass, optimize
        outputs = model(inputs)

        # print("Outputs Shape: ", outputs.shape)
        # print("Label Shape: ", labels.shape)
        # print("These are your outputs: ", outputs)
        # print("These are your labels: ", labels[:,out_idx,:])
        loss_output = loss(outputs, labels[:, out_idx, :])
        loss_output.backward()
        optimizer.step()

        total_loss += loss_output.data.item()
        if hasattr(loader, "postfix"):
            loader.postfix = f"train={loss_output.data.item():.4g}"

    return total_loss


def validate(model, loss, loader, device, out_idx, progress):
    total_loss = 0
    eff = ValueSet(0, 0, 0, 0)

    # switch to evaluate mode
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
        for inputs, labels in loader:
            if inputs.device != device:
                inputs, labels = inputs.to(device).float(), labels.to(device).float()

            # Forward pass
            val_outputs = model(inputs)

            loss_output = loss(val_outputs, labels[:, out_idx, :])

            total_loss += loss_output.data.item()

            for label, output in zip(
                labels[:, 0, :].cpu().numpy(), val_outputs.cpu().numpy()
            ):
                eff += efficiency(label, output, **PARAM_EFF)
    return total_loss, eff
